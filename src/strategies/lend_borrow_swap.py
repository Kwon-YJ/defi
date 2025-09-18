from __future__ import annotations

"""
Lending/Borrowing + Swap 조합 전략 탐색기

개요
- 담보 토큰 A를 예치(Aave) → 차입 토큰 B를 대출 → B를 DEX에서 A로 스왑
- (옵션) 추가 루프/상환은 제외하고, 단일 조합의 기대치/제약을 산출

설계 원칙
- 온체인 수집기는 기존 collectors를 재사용(AaveV2Collector, UniswapV2Collector 등)
- 가격/슬리피지는 DEX 풀 리저브 기반 근사(amount_out_uniswap_v2)
- 보수적 안전 마진(safety_factor)와 eMode/LTV를 반영한 최대 차입 한도 계산
- 실행 비용(가스)은 대략치로 반환; 최종 실행 전에 별도 추정 필요

주의
- 본 모듈은 포지션을 닫지 않는 조합(예치+차입+스왑)을 평가합니다.
  즉, 차입 잔액은 남아 있으며, 전략적으로 롤링/루프/상환을 별도 절차로 수행해야 합니다.
  Bellman-Ford 기반 음의 사이클(완전 청산형)과는 별개 기능입니다.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from web3 import Web3

from src.lending_collectors import AaveV2Collector
from src.dex_data_collector import UniswapV2Collector, SushiSwapCollector
from src.dex_uniswap_v3_collector import UniswapV3Collector
from src.dex_curve_collector import CurveStableSwapCollector
from src.dex_balancer_collector import BalancerWeightedCollector
from src.slippage import amount_out_uniswap_v2, amount_out_balancer_weighted, amount_out_cpmm
from src.erc20_utils import get_decimals, normalize_reserves
from src.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class LendBorrowSwapPlan:
    collateral_symbol: str
    borrow_symbol: str
    deposit_amount_collateral: float
    max_borrow_amount: float
    chosen_dex: str
    expected_received_collateral: float
    safety_factor: float
    ltv_used: float
    borrow_rate_info: Dict
    gas_cost_estimate_eth: float


class LendBorrowSwapStrategy:
    def __init__(self, w3: Web3, tokens: Dict[str, str]):
        self.w3 = w3
        self.tokens = tokens  # symbol -> address
        # collectors
        self.aave = AaveV2Collector(w3)
        self.v2 = UniswapV2Collector(w3)
        self.sushi = SushiSwapCollector(w3)
        self.v3 = UniswapV3Collector(w3)
        self.curve = CurveStableSwapCollector(w3)
        self.balancer = BalancerWeightedCollector(w3)

    def _price_ratio_via_v2(self, base: str, quote: str) -> Optional[float]:
        """UniswapV2/Sushi 기준 base->quote 환율(수수료 전 근사)"""
        try:
            pair = self.v2.get_pair_address(base, quote)
            if not pair:
                return None
            r0, r1, _ = self.v2.get_pool_reserves(pair)
            t0, t1 = self.v2.get_pool_tokens(pair)
            if not t0 or not t1 or r0 == 0 or r1 == 0:
                return None
            d0 = get_decimals(self.w3, t0, 18)
            d1 = get_decimals(self.w3, t1, 18)
            nr0, nr1 = normalize_reserves(r0, d0, r1, d1)
            if base.lower() == t0.lower():
                return float(nr1) / float(nr0)
            else:
                return float(nr0) / float(nr1)
        except Exception:
            return None

    def _best_swap_out_to_collateral(self, borrow_token: str, collateral_token: str, amount_in: float) -> Tuple[str, float]:
        """여러 DEX 후보 중 B->A 스왑 아웃 최댓값과 DEX 레이블 반환"""
        best_dex = 'uniswap_v2'
        best_out = 0.0

        # 1) Uniswap V2
        try:
            pair = self.v2.get_pair_address(borrow_token, collateral_token)
            if pair:
                r0, r1, _ = self.v2.get_pool_reserves(pair)
                t0, t1 = self.v2.get_pool_tokens(pair)
                if r0 > 0 and r1 > 0 and t0 and t1:
                    d0 = get_decimals(self.w3, t0, 18)
                    d1 = get_decimals(self.w3, t1, 18)
                    nr0, nr1 = normalize_reserves(r0, d0, r1, d1)
                    fee = 0.003
                    if borrow_token.lower() == t0.lower() and collateral_token.lower() == t1.lower():
                        out = amount_out_uniswap_v2(amount_in, float(nr0), float(nr1), fee)
                    else:
                        out = amount_out_uniswap_v2(amount_in, float(nr1), float(nr0), fee)
                    if out > best_out:
                        best_out, best_dex = out, 'uniswap_v2'
        except Exception:
            pass

        # 2) SushiSwap (V2 계열)
        try:
            pair = self.sushi.get_pair_address(borrow_token, collateral_token)
            if pair:
                r0, r1, _ = self.sushi.get_pool_reserves(pair)
                t0, t1 = self.sushi.get_pool_tokens(pair)
                if r0 > 0 and r1 > 0 and t0 and t1:
                    d0 = get_decimals(self.w3, t0, 18)
                    d1 = get_decimals(self.w3, t1, 18)
                    nr0, nr1 = normalize_reserves(r0, d0, r1, d1)
                    fee = 0.003
                    if borrow_token.lower() == t0.lower() and collateral_token.lower() == t1.lower():
                        out = amount_out_uniswap_v2(amount_in, float(nr0), float(nr1), fee)
                    else:
                        out = amount_out_uniswap_v2(amount_in, float(nr1), float(nr0), fee)
                    if out > best_out:
                        best_out, best_dex = out, 'sushiswap'
        except Exception:
            pass

        # 3) Uniswap V3 (근사 CPMM)
        try:
            # fee 우선순위 0.3% → 0.05% → 1%
            for fee_tier in (3000, 500, 10000):
                pool = self.v3.get_pool_address(borrow_token, collateral_token, fee_tier)
                if not pool:
                    continue
                state = self.v3.get_pool_core_state(pool)
                if not state:
                    continue
                # pseudo reserves (r0/r1) 사용
                r0 = float(state.get('pseudo_r0', 0.0) or 0.0)
                r1 = float(state.get('pseudo_r1', 0.0) or 0.0)
                if r0 <= 0 or r1 <= 0:
                    # 근사치가 없으면 건너뜀
                    continue
                fee = float(fee_tier) / 1_000_000.0
                t0 = state['token0']; t1 = state['token1']
                if borrow_token.lower() == t0.lower() and collateral_token.lower() == t1.lower():
                    out = amount_out_cpmm(amount_in, r0, r1, fee)
                else:
                    out = amount_out_cpmm(amount_in, r1, r0, fee)
                if out > best_out:
                    best_out, best_dex = out, f'uniswap_v3_{fee_tier}'
        except Exception:
            pass

        # 4) Balancer Weighted (근사)
        try:
            pool = self.balancer.find_pool_for_pair(borrow_token, collateral_token)
            if pool:
                eff_rate, fee_frac, wi, wj, bi = self.balancer.effective_rate_for_fraction(pool, borrow_token, collateral_token)
                # bi는 in token balance, w는 가중치; out ~ bi*(1 - (bi/(bi+amount))^(wi/wj)) 근사 대신 고정 비율 사용
                out = amount_in * float(eff_rate) * (1.0 - float(fee_frac))
                if out > best_out:
                    best_out, best_dex = out, 'balancer'
        except Exception:
            pass

        # 5) Curve Stable (주로 스테이블 쌍)
        try:
            found = self.curve.find_pool_for_pair(borrow_token, collateral_token)
            if found:
                pool, i, j = found
                price = self.curve.get_price(pool, i, j, borrow_token, collateral_token)
                fee = float(self.curve.get_pool_params(pool).get('fee', 0.0) or 0.0)
                out = amount_in * float(price) * (1.0 - fee)
                if out > best_out:
                    best_out, best_dex = out, 'curve'
        except Exception:
            pass

        return best_dex, float(best_out)

    def build_plan(self,
                   collateral_symbol: str,
                   borrow_symbol: str,
                   deposit_amount_collateral: float,
                   safety_factor: float = 0.9) -> Optional[LendBorrowSwapPlan]:
        """담보/차입/스왑 조합의 기대 결과를 계산하여 플랜을 생성.

        safety_factor: 0~1 (담보대 비율의 보수적 사용 비율)
        """
        try:
            col = self.tokens.get(collateral_symbol)
            bor = self.tokens.get(borrow_symbol)
            if not col or not bor:
                logger.error("토큰 심볼/주소 매핑을 찾을 수 없습니다")
                return None

            # Aave 담보 파라미터
            cfg = self.aave.get_reserve_configuration(col) or {}
            rates = self.aave.get_reserve_rates(col) or {}
            # liquidationThreshold (bps)를 사용하여 보수적 LTV로 근사
            lt_raw = cfg.get('liquidationThreshold')
            try:
                ltv_used = float(lt_raw) / 10000.0 if lt_raw is not None else 0.5
            except Exception:
                ltv_used = 0.5
            ltv_used = max(0.0, min(1.0, ltv_used * float(safety_factor)))

            # 가격 비율: collateral -> borrow
            p_cb = self._price_ratio_via_v2(col, bor)
            p_bc = self._price_ratio_via_v2(bor, col)
            if p_cb is None or p_bc is None or p_cb <= 0 or p_bc <= 0:
                logger.warning("V2 기준 환율 조회 실패")
                return None

            # 최대 차입량: deposit * ltv_used (가치기준)
            # 가치 변환: A 단위 → B 단위: amount_A * (price_B_per_A)
            max_borrow_amount = float(deposit_amount_collateral) * float(p_cb) * float(ltv_used)

            # 스왑: B -> A 최적 DEX 선택
            dex, received_A = self._best_swap_out_to_collateral(bor, col, max_borrow_amount)

            # 가스비 근사 (ETH 단위): swap + borrow + supply 정도만 포함(보수적)
            # 실제 실행은 컨트랙트/라우팅에 따라 상이함
            gas_eth = (150_000 + 220_000 + 120_000) * (20e9) / 1e18  # rough default 20 gwei

            plan = LendBorrowSwapPlan(
                collateral_symbol=collateral_symbol,
                borrow_symbol=borrow_symbol,
                deposit_amount_collateral=float(deposit_amount_collateral),
                max_borrow_amount=float(max_borrow_amount),
                chosen_dex=dex,
                expected_received_collateral=float(received_A),
                safety_factor=float(safety_factor),
                ltv_used=float(ltv_used),
                borrow_rate_info={
                    'variableBorrowRate': rates.get('variableBorrowRate'),
                    'stableBorrowRate': rates.get('stableBorrowRate'),
                    'liquidityRate': rates.get('liquidityRate'),
                },
                gas_cost_estimate_eth=float(gas_eth),
            )
            return plan
        except Exception as e:
            logger.error(f"플랜 생성 실패: {e}")
            return None


def format_plan(plan: LendBorrowSwapPlan) -> str:
    """사람이 읽을 수 있는 플랜 요약 문자열."""
    if not plan:
        return "플랜 없음"
    return (
        f"[Lend+Borrow+Swap]\n"
        f"- Collateral: {plan.collateral_symbol}, Deposit: {plan.deposit_amount_collateral:.6f}\n"
        f"- Borrow: {plan.borrow_symbol}, MaxBorrow≈ {plan.max_borrow_amount:.6f} (LTV_used={plan.ltv_used:.2f}, safety={plan.safety_factor:.2f})\n"
        f"- Swap(B->A) via {plan.chosen_dex}: Receive≈ {plan.expected_received_collateral:.6f} {plan.collateral_symbol}\n"
        f"- BorrowRates: var={plan.borrow_rate_info.get('variableBorrowRate')}, stable={plan.borrow_rate_info.get('stableBorrowRate')}\n"
        f"- Gas(rough): ~{plan.gas_cost_estimate_eth:.6f} ETH\n"
    )

