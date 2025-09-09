from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable
from itertools import combinations
from web3 import Web3

from src.logger import setup_logger
from src.market_graph import DeFiMarketGraph
from src.dex_data_collector import UniswapV2Collector, SushiSwapCollector
from src.dex_uniswap_v3_collector import UniswapV3Collector
from src.dex_curve_collector import CurveStableSwapCollector
from src.lending_collectors import CompoundCollector, AaveV2Collector
from src.synthetix_collectors import SynthetixCollector
from src.maker_collectors import MakerCollector
from src.dex_balancer_collector import BalancerWeightedCollector
from src.dydx_collectors import DyDxCollector
from src.yearn_collectors import YearnV2Collector
from src.uniswap_v2_lp_collector import UniswapV2LPCollector
from src.erc20_utils import get_decimals, normalize_reserves
from src.yearn_collectors import YearnV2Collector
from src.edge_meta import set_edge_meta
from src.gas_utils import estimate_gas_cost_usd_for_dex, set_edge_gas_cost
from src.synth_tokens import lp_v2, lp_v3, lp_curve, bpt, debt_aave, debt_compound
from src.slippage import amount_out_uniswap_v2
from config.config import config
from config.config import config
from src.constants import ETH_NATIVE_ADDRESS

logger = setup_logger(__name__)


class ProtocolAction:
    """Protocol action interface for scalable integration (target: 96 actions)."""

    name: str = "base"
    enabled: bool = False

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        """Update market graph for this action. Returns number of edges updated/added."""
        raise NotImplementedError


def _weth_wrap_update(graph: DeFiMarketGraph, tokens: Dict[str, str]) -> int:
    """ETH(native) <-> WETH wrap/unwrap 엣지 추가.

    - 1:1 비율, 수수료 0.0, 충분한 가상 유동성으로 모델링
    - pool_address는 비어둠(이벤트 구독 주소 필터에 포함되지 않도록)
    """
    try:
        weth = tokens.get('WETH') or tokens.get('weth')
        if not weth:
            return 0
        base = 1_000.0
        graph.add_trading_pair(
            token0=ETH_NATIVE_ADDRESS,
            token1=weth,
            dex='weth_wrap',
            pool_address='',
            reserve0=base,
            reserve1=base,
            fee=0.0,
        )
        try:
            set_edge_meta(
                graph.graph, ETH_NATIVE_ADDRESS, weth, dex='weth_wrap', pool_address='',
                fee_tier=None, source='synthetic', confidence=0.99,
                extra={'native': True, 't0': ETH_NATIVE_ADDRESS, 't1': weth}
            )
        except Exception:
            pass
        return 2
    except Exception:
        return 0


class _SwapPairsMixin:
    def _major_pairs(self, tokens: Dict[str, str]) -> List[Tuple[str, str]]:
        addrs = list(tokens.values())
        return list(combinations(addrs, 2))


class UniswapV2SwapAction(ProtocolAction, _SwapPairsMixin):
    name = "uniswap_v2.swap"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = UniswapV2Collector(w3)
        self.fee = 0.003

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        for token0, token1 in self._major_pairs(tokens):
            try:
                pair_address = await self.collector.get_pair_address(token0, token1)
                if not pair_address:
                    continue
                r0, r1, _ = await self.collector.get_pool_reserves(pair_address)
                if r0 == 0 or r1 == 0:
                    continue
                t0, t1 = await self.collector.get_pool_tokens(pair_address)
                if not t0 or not t1:
                    continue
                d0 = get_decimals(w3, t0, 18)
                d1 = get_decimals(w3, t1, 18)
                nr0, nr1 = normalize_reserves(r0, d0, r1, d1)
                # amount-dependent effective rate for reference trade fraction
                try:
                    f = float(getattr(config, 'slippage_trade_fraction', 0.01))
                except Exception:
                    f = 0.01
                amt_in = max(1e-12, float(nr0) * max(1e-6, min(f, 0.5)))
                out = amount_out_uniswap_v2(amt_in, float(nr0), float(nr1), self.fee)
                eff_rate = (out / amt_in) if amt_in > 0 and out > 0 else (float(nr1) / float(nr0)) * (1.0 - self.fee)
                # set pseudo reserves to match effective rate: rate_pre_fee = eff_rate / (1-fee)
                pre_fee_rate = eff_rate / max(1e-9, (1.0 - self.fee))
                base = 100.0
                graph.add_trading_pair(
                    token0=t0,
                    token1=t1,
                    dex='uniswap_v2',
                    pool_address=pair_address,
                    reserve0=base,
                    reserve1=base * float(pre_fee_rate),
                    fee=self.fee,
                )
                set_edge_meta(
                    graph.graph, t0, t1, dex='uniswap_v2', pool_address=pair_address,
                    fee_tier=None, source='onchain', confidence=0.98,
                    extra={'t0': t0, 't1': t1, 'r0': float(nr0), 'r1': float(nr1), 'eff_rate': float(eff_rate), 'ref_fraction': float(f)}
                )
                updated += 2
            except Exception as e:
                logger.debug(f"UniswapV2 update failed {token0[:6]}-{token1[:6]}: {e}")
        return updated


class SushiSwapSwapAction(ProtocolAction, _SwapPairsMixin):
    name = "sushiswap.swap"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = SushiSwapCollector(w3)
        self.fee = 0.003

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        for token0, token1 in self._major_pairs(tokens):
            try:
                pair_address = await self.collector.get_pair_address(token0, token1)
                if not pair_address:
                    continue
                r0, r1, _ = await self.collector.get_pool_reserves(pair_address)
                if r0 == 0 or r1 == 0:
                    continue
                t0, t1 = await self.collector.get_pool_tokens(pair_address)
                if not t0 or not t1:
                    continue
                d0 = get_decimals(w3, t0, 18)
                d1 = get_decimals(w3, t1, 18)
                nr0, nr1 = normalize_reserves(r0, d0, r1, d1)
                try:
                    f = float(getattr(config, 'slippage_trade_fraction', 0.01))
                except Exception:
                    f = 0.01
                amt_in = max(1e-12, float(nr0) * max(1e-6, min(f, 0.5)))
                out = amount_out_uniswap_v2(amt_in, float(nr0), float(nr1), self.fee)
                eff_rate = (out / amt_in) if amt_in > 0 and out > 0 else (float(nr1) / float(nr0)) * (1.0 - self.fee)
                pre_fee_rate = eff_rate / max(1e-9, (1.0 - self.fee))
                base = 100.0
                graph.add_trading_pair(
                    token0=t0,
                    token1=t1,
                    dex='sushiswap',
                    pool_address=pair_address,
                    reserve0=base,
                    reserve1=base * float(pre_fee_rate),
                    fee=self.fee,
                )
                set_edge_meta(
                    graph.graph, t0, t1, dex='sushiswap', pool_address=pair_address,
                    fee_tier=None, source='onchain', confidence=0.98,
                    extra={'t0': t0, 't1': t1, 'r0': float(nr0), 'r1': float(nr1), 'eff_rate': float(eff_rate), 'ref_fraction': float(f)}
                )
                updated += 2
            except Exception as e:
                logger.debug(f"SushiSwap update failed {token0[:6]}-{token1[:6]}: {e}")
        return updated


# Skeletons for future expansion (disabled by default)
class UniswapV3SwapAction(ProtocolAction, _SwapPairsMixin):
    name = "uniswap_v3.swap"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = UniswapV3Collector(w3)
        from src.data_storage import DataStorage
        self.storage = DataStorage()
        self.w_ema = float(getattr(config, 'v3_score_weight_ema', 1.0))
        self.w_liq = float(getattr(config, 'v3_score_weight_liq', 0.001))
        from src.data_storage import DataStorage
        self.storage = DataStorage()

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        for token0, token1 in self._major_pairs(tokens):
            try:
                edges = await self.collector.build_edges_for_pair(token0, token1)
                if not edges:
                    continue
                # fee tier별 통계 기반 점수 산정 및 선택 마킹
                scored = []
                for e in edges:
                    # 밴드 히스토그램 기반 가중치로 pseudo-liquidity 보정
                    try:
                        bw = await self.storage.get_v3_band_weight(e['pool'], e.get('tick_lower'), e.get('tick_upper'))
                        scale = max(0.25, float(bw))
                        e['reserve0'] = float(e['reserve0']) * scale
                        e['reserve1'] = float(e['reserve1']) * scale
                        e['band_weight'] = float(bw)
                    except Exception:
                        e['band_weight'] = 1.0
                    stats = await self.storage.get_v3_fee_stats(e['pool'], e.get('fee_tier', 0))
                    ema0 = float(stats.get('ema_fee0', 0.0)) if stats else 0.0
                    ema1 = float(stats.get('ema_fee1', 0.0)) if stats else 0.0
                    liq_proxy = max(1e-9, min(float(e['reserve0']), float(e['reserve1'])))
                    score = self.w_ema * (ema0 + ema1) + self.w_liq * liq_proxy
                    scored.append((score, e))
                scored.sort(key=lambda x: x[0], reverse=True)
                best_pool = scored[0][1]['pool'] if scored else None
                for score, e in scored:
                    # 유사 리저브를 통한 환율 반영 (add_trading_pair 내부에서 수수료 감안)
                    graph.add_trading_pair(
                        token0=e['token0'],
                        token1=e['token1'],
                        dex='uniswap_v3',
                        pool_address=e['pool'], edge_key=f"uniswap_v3:{e.get('fee_tier')}",
                        reserve0=float(e['reserve0']),
                        reserve1=float(e['reserve1']),
                        fee=float(e['fee_fraction']),
                    )
                    set_edge_meta(
                        graph.graph, e['token0'], e['token1'], dex='uniswap_v3', pool_address=e['pool'],
                        fee_tier=e.get('fee_tier'), source='onchain', confidence=0.95,
                        extra={
                            't0': e['token0'], 't1': e['token1'],
                            'r0': float(e['reserve0']), 'r1': float(e['reserve1']),
                            'tick': e.get('tick'),
                            'tick_lower': e.get('tick_lower'),
                            'tick_upper': e.get('tick_upper'),
                            'tick_spacing': e.get('tick_spacing'),
                            'sqrtPriceX96': e.get('sqrtPriceX96'),
                            'dex_key': f"uniswap_v3:{e.get('fee_tier')}",
                            'fee_label': e.get('fee_label'),
                            'tier_score': float(score),
                            'selected': bool(best_pool and e['pool'] == best_pool),
                            'band_weight': float(e.get('band_weight', 1.0)),
                        }
                    )
                    updated += 2
            except Exception as e:
                logger.debug(f"UniswapV3 update failed {token0[:6]}-{token1[:6]}: {e}")
        return updated


class CurveStableSwapAction(ProtocolAction, _SwapPairsMixin):
    name = "curve.stableswap"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = CurveStableSwapCollector(w3)
        self.fee = 0.0004  # kept for fallback; get_dy is net-of-fee so we set edge fee=0

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        # 레지스트리 동기화(주기적)
        try:
            self.collector.refresh_registry_pools(ttl_sec=1800)
        except Exception:
            pass
        updated = 0
        base_liq = 200.0
        for token0, token1 in self._major_pairs(tokens):
            try:
                found = self.collector.find_pool_for_pair(token0, token1)
                if not found:
                    continue
                pool, i, j = found
                price01 = self.collector.get_price(pool, i, j, token0, token1)
                if price01 <= 0:
                    continue
                # Liquidity proxy scaling using LP totalSupply (best-effort)
                try:
                    lp_addr = self.collector.get_lp_token(pool)
                    lp_dec = self.collector.get_lp_decimals(lp_addr) if lp_addr else 18
                    lp_ts_raw = self.collector.get_lp_total_supply(lp_addr) if lp_addr else 0
                    lp_ts = float(lp_ts_raw) / float(10 ** lp_dec) if lp_dec else float(lp_ts_raw) / 1e18
                    # scale in [0.5, 2.0] using sqrt(ts) normalized by heuristic 1e6
                    import math
                    scale = math.sqrt(max(0.0, lp_ts) / 1e6)
                    scale = max(0.5, min(2.0, float(scale)))
                except Exception:
                    scale = 1.0
                    lp_addr = None
                    lp_ts = 0.0
                reserve0 = base_liq * float(scale)
                reserve1 = reserve0 * price01
                graph.add_trading_pair(
                    token0=token0,
                    token1=token1,
                    dex='curve',
                    pool_address=pool, edge_key=f"curve:{i}-{j}",
                    reserve0=reserve0,
                    reserve1=reserve1,
                    fee=0.0,
                )
                # Fetch pool params and decimals info for metadata
                params = {}
                try:
                    params = self.collector.get_pool_params(pool) or {}
                except Exception:
                    params = {}
                try:
                    d0 = self.collector._decimals(token0)
                    d1 = self.collector._decimals(token1)
                except Exception:
                    d0 = d1 = 18
                set_edge_meta(
                    graph.graph, token0, token1, dex='curve', pool_address=pool,
                    fee_tier=None, source='onchain', confidence=0.92,
                    extra={
                        'stableswap': True,
                        'amp': params.get('A', None),
                        'pool_fee': params.get('fee', None),
                        'admin_fee': params.get('admin_fee', None),
                        'dec_in': d0,
                        'dec_out': d1,
                        'lp_token': lp_addr,
                        'lp_total_supply': float(lp_ts),
                        'liq_scale': float(scale),
                    }
                )
                updated += 2
            except Exception as e:
                logger.debug(f"Curve update failed {token0[:6]}-{token1[:6]}: {e}")
        return updated


class BalancerWeightedSwapAction(ProtocolAction, _SwapPairsMixin):
    name = "balancer.weighted_swap"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = BalancerWeightedCollector(w3)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        for token0, token1 in self._major_pairs(tokens):
            try:
                pool = self.collector.find_pool_for_pair(token0, token1)
                if not pool:
                    continue
                # read weights, balances and fee; compute effective rate at reference trade size
                eff_rate, fee_frac, wi, wj, bi = self.collector.effective_rate_for_fraction(pool, token0, token1)
                if eff_rate <= 0 or wi <= 0 or wj <= 0 or bi <= 0:
                    continue
                # set reserves so that exchange_rate ~= effective average price at ref size
                reserve0 = 150.0
                reserve1 = reserve0 * float(eff_rate)
                graph.add_trading_pair(
                    token0=token0,
                    token1=token1,
                    dex='balancer',
                    pool_address=pool,
                    reserve0=reserve0,
                    reserve1=reserve1,
                    fee=float(fee_frac),
                )
                set_edge_meta(
                    graph.graph, token0, token1, dex='balancer', pool_address=pool,
                    fee_tier=None, source='onchain', confidence=0.9,
                    extra={'w_in': wi, 'w_out': wj, 'b_in': bi, 'ref_fraction': float(getattr(config, 'slippage_trade_fraction', 0.01)), 'eff_rate': float(eff_rate)}
                )
                updated += 2
            except Exception as e:
                logger.debug(f"Balancer update failed {token0[:6]}-{token1[:6]}: {e}")
        return updated


class AaveSupplyBorrowAction(ProtocolAction):
    name = "aave.supply_borrow"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = AaveV2Collector(w3)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 100.0
        for sym, underlying in tokens.items():
            try:
                atoken = self.collector.get_atoken(underlying)
                if not atoken:
                    continue
                rate = self.collector.get_deposit_rate_underlying_to_atoken(atoken)  # ~1.0
                reserve0 = base_liq
                reserve1 = base_liq * rate
                graph.add_trading_pair(
                    token0=underlying,
                    token1=atoken,
                    dex='aave',
                    pool_address=atoken,
                    reserve0=reserve0,
                    reserve1=reserve1,
                    fee=0.0,
                )
                # Fetch risk and rate params for metadata
                cfg = self.collector.get_reserve_configuration(underlying) or {}
                rates = self.collector.get_reserve_rates(underlying) or {}
                emode = self.collector.get_emode_category(underlying)
                set_edge_meta(
                    graph.graph, underlying, atoken, dex='aave', pool_address=atoken,
                    fee_tier=None, source='onchain', confidence=0.9,
                    extra={
                        'ltv': cfg.get('ltv'),
                        'liquidationThreshold': cfg.get('liquidationThreshold'),
                        'liquidationBonus': cfg.get('liquidationBonus'),
                        'reserveFactor': cfg.get('reserveFactor'),
                        'borrowingEnabled': cfg.get('borrowingEnabled'),
                        'stableBorrowRateEnabled': cfg.get('stableBorrowRateEnabled'),
                        'liquidityRate': rates.get('liquidityRate'),
                        'variableBorrowRate': rates.get('variableBorrowRate'),
                        'stableBorrowRate': rates.get('stableBorrowRate'),
                        'eModeCategory': emode,
                    }
                )
                updated += 2
            except Exception as e:
                logger.debug(f"Aave update failed {sym}: {e}")
        return updated


class CompoundSupplyBorrowAction(ProtocolAction):
    name = "compound.supply_borrow"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class MakerCdpAction(ProtocolAction):
    name = "maker.cdp"
    enabled = True

    def __init__(self, w3: Web3):
        self.w3 = w3

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        # Support WETH/DAI and WBTC/DAI minting using Vat/Jug ilk params when possible
        sym_to_addr = tokens or {}
        dai = sym_to_addr.get('DAI') or sym_to_addr.get('dai')
        if not dai:
            return 0
        updated = 0
        try:
            weth = sym_to_addr.get('WETH') or sym_to_addr.get('weth')
            if weth:
                collector = MakerCollector(self.w3, weth=weth, dai=dai)
                # Prefer VAT/JUG spot via ETH-A
                rate_ilk = collector.mintable_dai_per_collateral_via_ilk('ETH-A', safety_factor=0.95)
                if rate_ilk <= 0:
                    # fallback to price-based approximation
                    rate_ilk = collector.mintable_dai_per_weth(collateral_ratio=1.5, safety_factor=0.95)
                # Apply stability fee penalty over hold seconds
                hold_sec = int(getattr(config, 'maker_hold_seconds', 3600))
                params = collector.get_ilk_params('ETH-A')
                duty = float(params.get('duty', 0.0) or 0.0)
                if hold_sec > 0 and duty > 0:
                    try:
                        penalty = (1.0 + duty) ** hold_sec - 1.0
                        rate_ilk = rate_ilk / (1.0 + penalty)
                    except Exception:
                        pass
                if rate_ilk > 0:
                        base_liq = 200.0
                        graph.add_trading_pair(
                            token0=weth,
                            token1=dai,
                            dex='maker',
                            pool_address='maker_cdp_weth',
                            reserve0=base_liq,
                            reserve1=base_liq * rate_ilk,
                            fee=0.0005,
                        )
                        set_edge_meta(graph.graph, weth, dai, dex='maker', pool_address='maker_cdp_weth',
                                      fee_tier=None, source='onchain', confidence=0.8,
                                  extra={'ilk': 'ETH-A', 'spot': params.get('spot'), 'duty': params.get('duty'), 'line': params.get('line'), 'holdSeconds': hold_sec})
                        updated += 2
            # WBTC support if present
            wbtc = sym_to_addr.get('WBTC') or sym_to_addr.get('wbtc')
            if wbtc:
                collector2 = MakerCollector(self.w3, weth=weth or '', dai=dai)
                rate_ilk_btc = collector2.mintable_dai_per_collateral_via_ilk('WBTC-A', safety_factor=0.95)
                hold_sec = int(getattr(config, 'maker_hold_seconds', 3600))
                params2 = collector2.get_ilk_params('WBTC-A')
                duty2 = float(params2.get('duty', 0.0) or 0.0)
                if hold_sec > 0 and duty2 > 0:
                    try:
                        penalty2 = (1.0 + duty2) ** hold_sec - 1.0
                        rate_ilk_btc = rate_ilk_btc / (1.0 + penalty2)
                    except Exception:
                        pass
                if rate_ilk_btc > 0:
                        base_liq = 200.0
                        graph.add_trading_pair(
                            token0=wbtc,
                            token1=dai,
                            dex='maker',
                            pool_address='maker_cdp_wbtc',
                            reserve0=base_liq,
                            reserve1=base_liq * rate_ilk_btc,
                            fee=0.0005,
                        )
                        set_edge_meta(graph.graph, wbtc, dai, dex='maker', pool_address='maker_cdp_wbtc',
                                      fee_tier=None, source='onchain', confidence=0.8,
                                  extra={'ilk': 'WBTC-A', 'spot': params2.get('spot'), 'duty': params2.get('duty'), 'line': params2.get('line'), 'holdSeconds': hold_sec})
                        updated += 2
        except Exception as e:
            logger.debug(f"Maker CDP update failed: {e}")
        return updated


class YearnVaultAction(ProtocolAction):
    name = "yearn.vault"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = YearnV2Collector(w3)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 120.0
        for sym, underlying in tokens.items():
            try:
                yv = self.collector.get_vault(underlying)
                if not yv:
                    continue
                shares_per_underlying = self.collector.get_shares_per_underlying(underlying, yv)
                reserve0 = base_liq
                reserve1 = base_liq * shares_per_underlying
                graph.add_trading_pair(
                    token0=underlying,
                    token1=yv,
                    dex='yearn',
                    pool_address=yv,
                    reserve0=reserve0,
                    reserve1=reserve1,
                    fee=0.0,
                )
                set_edge_meta(graph.graph, underlying, yv, dex='yearn', pool_address=yv,
                              fee_tier=None, source='onchain', confidence=0.85)
                updated += 2
            except Exception as e:
                logger.debug(f"Yearn update failed {sym}: {e}")
        return updated


class SynthetixExchangeAction(ProtocolAction):
    name = "synthetix.exchange"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = SynthetixCollector(w3)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        try:
            synths = self.collector.get_synths()
            sUSD = synths["sUSD"]
            sETH = synths["sETH"]
            sBTC = synths.get("sBTC")
            base_liq = 200.0
            updated = 0
            # target c-ratio (issuanceRatio) and total debt
            ir = self.collector.get_issuance_ratio()
            total_debt = self.collector.get_total_debt()
            # sETH -> sUSD
            price = self.collector.price_susd_per_seth()
            # per-synth fee: use conservative max of src/dst fees
            fee_src = self.collector.get_exchange_fee_for('sETH') or self.collector.get_exchange_fee()
            fee_dst = self.collector.get_exchange_fee_for('sUSD') or self.collector.get_exchange_fee()
            fee = max(float(fee_src), float(fee_dst))
            if price > 0:
                graph.add_trading_pair(
                    token0=sETH,
                    token1=sUSD,
                    dex='synthetix',
                    pool_address='synthetix_exchange_seth_susd',
                    reserve0=base_liq,
                    reserve1=base_liq * price,
                    fee=fee,
                )
                set_edge_meta(
                    graph.graph, sETH, sUSD, dex='synthetix', pool_address='synthetix_exchange_seth_susd',
                    fee_tier=None, source='onchain', confidence=0.88,
                    extra={'issuanceRatio': ir, 'targetCRatio': (1.0 / ir) if ir else None, 'totalDebt': total_debt,
                           'fee_src': float(fee_src), 'fee_dst': float(fee_dst)}
                )
                updated += 2
            # sBTC -> sUSD route
            if sBTC:
                p_btc = self.collector.price_susd_per_sbtc()
                fee_src2 = self.collector.get_exchange_fee_for('sBTC') or self.collector.get_exchange_fee()
                fee2 = max(float(fee_src2), float(fee_dst))
                if p_btc > 0:
                    graph.add_trading_pair(
                        token0=sBTC,
                        token1=sUSD,
                        dex='synthetix',
                        pool_address='synthetix_exchange_sbtc_susd',
                        reserve0=base_liq,
                        reserve1=base_liq * p_btc,
                        fee=fee2,
                    )
                    set_edge_meta(
                        graph.graph, sBTC, sUSD, dex='synthetix', pool_address='synthetix_exchange_sbtc_susd',
                        fee_tier=None, source='onchain', confidence=0.86,
                        extra={'issuanceRatio': ir, 'targetCRatio': (1.0 / ir) if ir else None, 'totalDebt': total_debt,
                               'fee_src': float(fee_src2), 'fee_dst': float(fee_dst)}
                    )
                    updated += 2
            return updated
        except Exception as e:
            logger.debug(f"Synthetix update failed: {e}")
            return 0


class DyDxMarginAction(ProtocolAction, _SwapPairsMixin):
    name = "dydx.margin"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = DyDxCollector(w3)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 80.0
        for token0, token1 in self._major_pairs(tokens):
            try:
                price01, fee = self.collector.get_price_and_fee(token0, token1)
                if price01 <= 0:
                    continue
                reserve0 = base_liq
                reserve1 = base_liq * price01
                graph.add_trading_pair(
                    token0=token0,
                    token1=token1,
                    dex='dydx',
                    pool_address='dydx_margin',
                    reserve0=reserve0,
                    reserve1=reserve1,
                    fee=float(fee),
                )
                # Attach risk params
                rp = self.collector.get_risk_params()
                set_edge_meta(
                    graph.graph, token0, token1, dex='dydx', pool_address='dydx_margin',
                    fee_tier=None, source='approx', confidence=0.75,
                    extra={'initialMargin': rp.get('initialMargin'), 'maintenanceMargin': rp.get('maintenanceMargin'), 'maxLeverage': rp.get('maxLeverage')}
                )
                updated += 2
            except Exception as e:
                logger.debug(f"dYdX update failed {token0[:6]}-{token1[:6]}: {e}")
        return updated


# --- Additional major-function scaffolding (disabled by default) ---
class UniswapV2AddLiquidityAction(ProtocolAction, _SwapPairsMixin):
    name = "uniswap_v2.add_liquidity"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = UniswapV2LPCollector(w3)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 100.0
        for token0, token1 in self._major_pairs(tokens):
            try:
                pair = await self.collector.get_pair_address(token0, token1)
                if not pair:
                    continue
                r0, r1, _ = await self.collector.get_pool_reserves(pair)
                if r0 == 0 or r1 == 0:
                    continue
                ts = await self.collector.get_total_supply(pair)
                if ts == 0:
                    continue
                t0, t1 = await self.collector.get_pool_tokens(pair)
                if not t0 or not t1:
                    continue
                d0 = get_decimals(w3, t0, 18)
                d1 = get_decimals(w3, t1, 18)
                # Read LP token decimals dynamically (fallback 18)
                lp_dec = await self.collector.get_lp_decimals(pair) or 18
                nr0, nr1 = normalize_reserves(r0, d0, r1, d1)
                ts_norm = float(ts) / (10 ** lp_dec)
                # LP minted per unit token (exact under proportional deposit):
                # L = min(a0*ts/r0, a1*ts/r1); with proportional a1/a0=r1/r0 => L = a0*ts/r0 = a1*ts/r1
                lp_per_t0 = (ts_norm / nr0) if nr0 > 0 else 0.0
                lp_per_t1 = (ts_norm / nr1) if nr1 > 0 else 0.0
                # token0 -> LP
                graph.add_trading_pair(
                    token0=t0,
                    token1=lp_v2(pair),
                    dex='uniswap_v2_lp_add',
                    pool_address=pair,
                    reserve0=base_liq,
                    reserve1=base_liq * lp_per_t0,
                    fee=0.0,
                )
                set_edge_meta(graph.graph, t0, lp_v2(pair), dex='uniswap_v2_lp_add', pool_address=pair,
                              fee_tier=None, source='approx', confidence=0.8)
                updated += 2
                # token1 -> LP
                graph.add_trading_pair(
                    token0=t1,
                    token1=lp_v2(pair),
                    dex='uniswap_v2_lp_add',
                    pool_address=pair,
                    reserve0=base_liq,
                    reserve1=base_liq * lp_per_t1,
                    fee=0.0,
                )
                set_edge_meta(graph.graph, t1, lp_v2(pair), dex='uniswap_v2_lp_add', pool_address=pair,
                              fee_tier=None, source='approx', confidence=0.8)
                updated += 2
            except Exception as e:
                logger.debug(f"UniswapV2 addLiquidity failed {token0[:6]}-{token1[:6]}: {e}")
        return updated


class UniswapV2RemoveLiquidityAction(ProtocolAction, _SwapPairsMixin):
    name = "uniswap_v2.remove_liquidity"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = UniswapV2LPCollector(w3)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 100.0
        for token0, token1 in self._major_pairs(tokens):
            try:
                pair = await self.collector.get_pair_address(token0, token1)
                if not pair:
                    continue
                r0, r1, _ = await self.collector.get_pool_reserves(pair)
                if r0 == 0 and r1 == 0:
                    continue
                ts = await self.collector.get_total_supply(pair)
                if ts == 0:
                    continue
                t0, t1 = await self.collector.get_pool_tokens(pair)
                if not t0 or not t1:
                    continue
                d0 = get_decimals(w3, t0, 18)
                d1 = get_decimals(w3, t1, 18)
                lp_dec = await self.collector.get_lp_decimals(pair) or 18
                nr0, nr1 = normalize_reserves(r0, d0, r1, d1)
                ts_norm = float(ts) / (10 ** lp_dec)
                t0_per_lp = nr0 / ts_norm if ts_norm > 0 else 0.0
                t1_per_lp = nr1 / ts_norm if ts_norm > 0 else 0.0
                # LP -> token0
                graph.add_trading_pair(
                    token0=lp_v2(pair),
                    token1=t0,
                    dex='uniswap_v2_lp_remove',
                    pool_address=pair,
                    reserve0=base_liq,
                    reserve1=base_liq * t0_per_lp,
                    fee=0.0,
                )
                set_edge_meta(graph.graph, lp_v2(pair), t0, dex='uniswap_v2_lp_remove', pool_address=pair,
                              fee_tier=None, source='approx', confidence=0.8)
                updated += 2
                # LP -> token1
                graph.add_trading_pair(
                    token0=lp_v2(pair),
                    token1=t1,
                    dex='uniswap_v2_lp_remove',
                    pool_address=pair,
                    reserve0=base_liq,
                    reserve1=base_liq * t1_per_lp,
                    fee=0.0,
                )
                set_edge_meta(graph.graph, lp_v2(pair), t1, dex='uniswap_v2_lp_remove', pool_address=pair,
                              fee_tier=None, source='approx', confidence=0.8)
                updated += 2
            except Exception as e:
                logger.debug(f"UniswapV2 removeLiquidity failed {token0[:6]}-{token1[:6]}: {e}")
        return updated


class UniswapV3AddLiquidityAction(ProtocolAction):
    name = "uniswap_v3.add_liquidity"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = UniswapV3Collector(w3)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 80.0
        for token0, token1 in list(combinations(tokens.values(), 2)):
            for fee in self.collector.FEE_TIERS:
                try:
                    pool = await self.collector.get_pool_address(token0, token1, fee)
                    if not pool:
                        continue
                    state = await self.collector.get_pool_core_state(pool)
                    if not state:
                        continue
                    t0 = state['token0']; t1 = state['token1']
                    lp_token = lp_v3(pool)
                    # Tick-range: 현재 tick 기준 한 칸 밴드
                    ts = int(state.get('tickSpacing', 60) or 60)
                    cur_tick = int(state.get('tick', 0) or 0)
                    tick_lower = (cur_tick // ts) * ts
                    tick_upper = tick_lower + ts
                    # 정규화된 sqrt 가격들 계산
                    sqrtP = self.collector.sqrt_from_x96_normalized(state['sqrtPriceX96'], state['dec0'], state['dec1'])
                    sqrtA = self.collector.sqrt_from_tick_normalized(tick_lower, state['dec0'], state['dec1'])
                    sqrtB = self.collector.sqrt_from_tick_normalized(tick_upper, state['dec0'], state['dec1'])
                    if sqrtP <= 0 or sqrtA <= 0 or sqrtB <= 0:
                        continue
                    # L per 1 token
                    lp_per_t0 = max(0.0, float(self.collector.liquidity_per_token0(sqrtP, sqrtA, sqrtB)))
                    lp_per_t1 = max(0.0, float(self.collector.liquidity_per_token1(sqrtP, sqrtA, sqrtB)))
                    # token0 -> LP
                    graph.add_trading_pair(
                        token0=t0,
                        token1=lp_token,
                        dex='uniswap_v3_lp_add',
                        pool_address=pool,
                        reserve0=base_liq,
                        reserve1=base_liq * lp_per_t0,
                        fee=0.0,
                    )
                    set_edge_meta(
                        graph.graph, t0, lp_token, dex='uniswap_v3_lp_add', pool_address=pool,
                        fee_tier=fee, source='approx', confidence=0.8,
                        extra={'tick': cur_tick, 'tick_lower': tick_lower, 'tick_upper': tick_upper, 'tick_spacing': ts, 't0': t0, 't1': t1,
                               'sqrtP': sqrtP, 'sqrtA': sqrtA, 'sqrtB': sqrtB, 'lp_per_t0': lp_per_t0}
                    )
                    updated += 2
                    # token1 -> LP
                    if lp_per_t1 > 0:
                        graph.add_trading_pair(
                            token0=t1,
                            token1=lp_token,
                            dex='uniswap_v3_lp_add',
                            pool_address=pool,
                            reserve0=base_liq,
                            reserve1=base_liq * lp_per_t1,
                            fee=0.0,
                        )
                        set_edge_meta(
                            graph.graph, t1, lp_token, dex='uniswap_v3_lp_add', pool_address=pool,
                            fee_tier=fee, source='approx', confidence=0.8,
                            extra={'tick': cur_tick, 'tick_lower': tick_lower, 'tick_upper': tick_upper, 'tick_spacing': ts, 't0': t0, 't1': t1,
                                   'sqrtP': sqrtP, 'sqrtA': sqrtA, 'sqrtB': sqrtB, 'lp_per_t1': lp_per_t1}
                        )
                        updated += 2
                except Exception as e:
                    logger.debug(f"UniswapV3 addLiquidity failed {token0[:6]}-{token1[:6]}: {e}")
        return updated


class UniswapV3RemoveLiquidityAction(ProtocolAction):
    name = "uniswap_v3.remove_liquidity"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = UniswapV3Collector(w3)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 80.0
        for token0, token1 in list(combinations(tokens.values(), 2)):
            for fee in self.collector.FEE_TIERS:
                try:
                    pool = await self.collector.get_pool_address(token0, token1, fee)
                    if not pool:
                        continue
                    state = await self.collector.get_pool_core_state(pool)
                    if not state:
                        continue
                    t0 = state['token0']; t1 = state['token1']
                    lp_token = lp_v3(pool)
                    ts = int(state.get('tickSpacing', 60) or 60)
                    cur_tick = int(state.get('tick', 0) or 0)
                    tick_lower = (cur_tick // ts) * ts
                    tick_upper = tick_lower + ts
                    sqrtP = self.collector.sqrt_from_x96_normalized(state['sqrtPriceX96'], state['dec0'], state['dec1'])
                    sqrtA = self.collector.sqrt_from_tick_normalized(tick_lower, state['dec0'], state['dec1'])
                    sqrtB = self.collector.sqrt_from_tick_normalized(tick_upper, state['dec0'], state['dec1'])
                    if sqrtP <= 0 or sqrtA <= 0 or sqrtB <= 0:
                        continue
                    t0_per_lp, t1_per_lp = self.collector.amounts_per_liquidity(sqrtP, sqrtA, sqrtB)
                    t0_per_lp = max(0.0, float(t0_per_lp))
                    t1_per_lp = max(0.0, float(t1_per_lp))
                    # LP -> token0
                    graph.add_trading_pair(
                        token0=lp_token,
                        token1=t0,
                        dex='uniswap_v3_lp_remove',
                        pool_address=pool,
                        reserve0=base_liq,
                        reserve1=base_liq * t0_per_lp,
                        fee=0.0,
                    )
                    set_edge_meta(
                        graph.graph, lp_token, t0, dex='uniswap_v3_lp_remove', pool_address=pool,
                        fee_tier=fee, source='approx', confidence=0.8,
                        extra={'tick': cur_tick, 'tick_lower': tick_lower, 'tick_upper': tick_upper, 'tick_spacing': ts, 't0': t0, 't1': t1,
                               'sqrtP': sqrtP, 'sqrtA': sqrtA, 'sqrtB': sqrtB, 't0_per_lp': t0_per_lp}
                    )
                    updated += 2
                    # LP -> token1
                    graph.add_trading_pair(
                        token0=lp_token,
                        token1=t1,
                        dex='uniswap_v3_lp_remove',
                        pool_address=pool,
                        reserve0=base_liq,
                        reserve1=base_liq * t1_per_lp,
                        fee=0.0,
                    )
                    set_edge_meta(
                        graph.graph, lp_token, t1, dex='uniswap_v3_lp_remove', pool_address=pool,
                        fee_tier=fee, source='approx', confidence=0.8,
                        extra={'tick': cur_tick, 'tick_lower': tick_lower, 'tick_upper': tick_upper, 'tick_spacing': ts, 't0': t0, 't1': t1,
                               'sqrtP': sqrtP, 'sqrtA': sqrtA, 'sqrtB': sqrtB, 't1_per_lp': t1_per_lp}
                    )
                    updated += 2
                except Exception as e:
                    logger.debug(f"UniswapV3 removeLiquidity failed {token0[:6]}-{token1[:6]}: {e}")
        return updated


class UniswapV3CollectFeesAction(ProtocolAction):
    name = "uniswap_v3.collect_fees"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = UniswapV3Collector(w3)
        from src.data_storage import DataStorage
        self.storage = DataStorage()

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 20.0
        for token0, token1 in list(combinations(tokens.values(), 2)):
            for fee in self.collector.FEE_TIERS:
                try:
                    pool = await self.collector.get_pool_address(token0, token1, fee)
                    if not pool:
                        continue
                    state = await self.collector.get_pool_core_state(pool)
                    if not state:
                        continue
                    t0 = state['token0']; t1 = state['token1']
                    dec0 = state['dec0']; dec1 = state['dec1']
                    lp_token = lp_v3(pool)
                    # 현재 활성 밴드 한 칸
                    ts = int(state.get('tickSpacing', 60) or 60)
                    cur_tick = int(state.get('tick', 0) or 0)
                    tick_lower = (cur_tick // ts) * ts
                    tick_upper = tick_lower + ts
                    # 해당 밴드에서 직전 대비 1 L 당 수취된 수수료 추정
                    fee0_per_L, fee1_per_L = await self.collector.estimate_fees_per_L(
                        pool, dec0, dec1, tick_lower, tick_upper, cur_tick
                    )
                    # 통계 업데이트 (EMA)
                    try:
                        await self.storage.upsert_v3_fee_stats(pool, int(state.get('fee', 0)), fee0_per_L, fee1_per_L)
                    except Exception:
                        pass
                    # LP -> token0 (fees)
                    graph.add_trading_pair(
                        token0=lp_token,
                        token1=t0,
                        dex='uniswap_v3_fee_collect',
                        pool_address=pool, edge_key=f"uniswap_v3:{fee}",
                        reserve0=base_liq,
                        reserve1=base_liq * float(fee0_per_L),
                        fee=0.0,
                    )
                    set_edge_meta(
                        graph.graph, lp_token, t0, dex='uniswap_v3_fee_collect', pool_address=pool,
                        fee_tier=fee, source='onchain', confidence=0.7,
                        extra={'tick': cur_tick, 'tick_lower': tick_lower, 'tick_upper': tick_upper,
                               'tick_spacing': ts, 't0': t0, 't1': t1, 'fee0_per_L': float(fee0_per_L)}
                    )
                    updated += 2
                    # LP -> token1 (fees)
                    graph.add_trading_pair(
                        token0=lp_token,
                        token1=t1,
                        dex='uniswap_v3_fee_collect',
                        pool_address=pool, edge_key=f"uniswap_v3:{fee}",
                        reserve0=base_liq,
                        reserve1=base_liq * float(fee1_per_L),
                        fee=0.0,
                    )
                    set_edge_meta(
                        graph.graph, lp_token, t1, dex='uniswap_v3_fee_collect', pool_address=pool,
                        fee_tier=fee, source='onchain', confidence=0.7,
                        extra={'tick': cur_tick, 'tick_lower': tick_lower, 'tick_upper': tick_upper,
                               'tick_spacing': ts, 't0': t0, 't1': t1, 'fee1_per_L': float(fee1_per_L)}
                    )
                    updated += 2
                except Exception as e:
                    logger.debug(f"UniswapV3 collectFees failed {token0[:6]}-{token1[:6]}: {e}")
        return updated


class CurveAddLiquidityAction(ProtocolAction):
    name = "curve.add_liquidity"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = CurveStableSwapCollector(w3)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 150.0
        try:
            # Refresh registry pools (lazy)
            self.collector.refresh_registry_pools(ttl_sec=1800)
            pools = self.collector.pools or self.collector.FALLBACK_POOLS
            for p in pools:
                pool = p['address']
                coins = self.collector.get_pool_coins(pool)
                n = len(coins)
                if n == 0:
                    continue
                lp_addr = self.collector.get_lp_token(pool) or pool
                lp_token = lp_curve(pool)
                lp_dec = self.collector.get_lp_decimals(lp_addr)
                lp_supply = self.collector.get_lp_total_supply(lp_addr)
                # For each configured token that is a pool coin
                for _, addr in tokens.items():
                    try:
                        if addr.lower() not in [c.lower() for c in coins]:
                            continue
                        i = self.collector.coin_index(pool, addr)
                        if i is None:
                            continue
                        # amounts array with 1 unit of token i
                        d = self.collector._decimals(addr)
                        dx = int((10 ** d))  # 1 token
                        amts = [0] * n
                        amts[int(i)] = dx
                        minted = self.collector.calc_token_amount(pool, amts, True)  # LP units (typically 1e18)
                        denom = float(10 ** lp_dec) if lp_dec else 1e18
                        lp_per_token = (float(minted) / denom)
                        if lp_per_token <= 0:
                            continue
                        graph.add_trading_pair(
                            token0=addr,
                            token1=lp_token,
                            dex='curve_lp_add',
                            pool_address=pool, edge_key=f"curve_lp_add:{i}",
                            reserve0=base_liq,
                            reserve1=base_liq * lp_per_token,
                            fee=0.0,
                        )
                        set_edge_meta(graph.graph, addr, lp_token, dex='curve_lp_add', pool_address=pool,
                                      fee_tier=None, source='onchain', confidence=0.9,
                                      extra={'n_coins': n, 'coin_index': int(i), 'lp_token': lp_addr, 'lp_decimals': lp_dec, 'lp_total_supply': float(lp_supply)})
                        updated += 2
                    except Exception:
                        continue
        except Exception as e:
            logger.debug(f"Curve addLiquidity update failed: {e}")
        return updated


class CurveRemoveLiquidityAction(ProtocolAction):
    name = "curve.remove_liquidity"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = CurveStableSwapCollector(w3)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 150.0
        try:
            self.collector.refresh_registry_pools(ttl_sec=1800)
            pools = self.collector.pools or self.collector.FALLBACK_POOLS
            for p in pools:
                pool = p['address']
                lp_addr = self.collector.get_lp_token(pool) or pool
                lp_token = lp_curve(pool)
                lp_dec = self.collector.get_lp_decimals(lp_addr)
                lp_supply = self.collector.get_lp_total_supply(lp_addr)
                coins = self.collector.get_pool_coins(pool)
                n = len(coins)
                if n == 0:
                    continue
                for i, coin in enumerate(coins):
                    try:
                        # Withdraw 1 LP (1e18) into coin i
                        denom = int(10 ** lp_dec) if lp_dec else int(1e18)
                        out = self.collector.calc_withdraw_one_coin(pool, denom, int(i))
                        if out <= 0:
                            continue
                        dec = self.collector._decimals(coin)
                        coin_per_lp = float(out) / float(10 ** dec)
                        graph.add_trading_pair(
                            token0=lp_token,
                            token1=coin,
                            dex='curve_lp_remove',
                            pool_address=pool, edge_key=f"curve_lp_remove:{i}",
                            reserve0=base_liq,
                            reserve1=base_liq * coin_per_lp,
                            fee=0.0,
                        )
                        set_edge_meta(graph.graph, lp_token, coin, dex='curve_lp_remove', pool_address=pool,
                                      fee_tier=None, source='onchain', confidence=0.9,
                                      extra={'n_coins': n, 'coin_index': int(i), 'lp_token': lp_addr, 'lp_decimals': lp_dec, 'lp_total_supply': float(lp_supply)})
                        updated += 2
                    except Exception:
                        continue
        except Exception as e:
            logger.debug(f"Curve removeLiquidity update failed: {e}")
        return updated


class BalancerJoinPoolAction(ProtocolAction):
    name = "balancer.join_pool"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = BalancerWeightedCollector(w3)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 120.0
        for token0, token1 in list(combinations(tokens.values(), 2)):
            try:
                pool = self.collector.find_pool_for_pair(token0, token1)
                if not pool:
                    continue
                lp_token = bpt(pool)
                # token0 -> LP (1 token0) — single-asset join (with fee on non-proportional part)
                d0 = self.collector._decimals(token0)
                bpt_per_t0 = self.collector.bpt_out_per_token_in(pool, token0, 1.0)
                if bpt_per_t0 > 0:
                    graph.add_trading_pair(
                        token0=token0,
                        token1=lp_token,
                        dex='balancer_lp_join',
                        pool_address=pool,
                        reserve0=base_liq,
                        reserve1=base_liq * float(bpt_per_t0),
                        fee=0.0,
                    )
                    set_edge_meta(graph.graph, token0, lp_token, dex='balancer_lp_join', pool_address=pool,
                                  fee_tier=None, source='onchain', confidence=0.88,
                                  extra={'dec_in': d0})
                    updated += 2
                # token1 -> LP (1 token1) — single-asset join
                d1 = self.collector._decimals(token1)
                bpt_per_t1 = self.collector.bpt_out_per_token_in(pool, token1, 1.0)
                if bpt_per_t1 > 0:
                    graph.add_trading_pair(
                        token0=token1,
                        token1=lp_token,
                        dex='balancer_lp_join',
                        pool_address=pool,
                        reserve0=base_liq,
                        reserve1=base_liq * float(bpt_per_t1),
                        fee=0.0,
                    )
                    set_edge_meta(graph.graph, token1, lp_token, dex='balancer_lp_join', pool_address=pool,
                                  fee_tier=None, source='onchain', confidence=0.88,
                                  extra={'dec_in': d1})
                    updated += 2

                # Proportional join branch (no swap fee): BPT per 1 token = ts / balance_i
                try:
                    bpt_dec, ts_raw = self.collector.get_bpt_info(pool)
                    ts = float(ts_raw) / float(10 ** bpt_dec) if bpt_dec else float(ts_raw) / 1e18
                    # token0 proportional
                    wi0, _, bi0, _ = self.collector.get_weights_and_balances(pool, token0, token0)
                    if ts > 0 and bi0 > 0:
                        bpt_per_t0_prop = ts / float(bi0)
                        graph.add_trading_pair(
                            token0=token0,
                            token1=lp_token,
                            dex='balancer_lp_join_prop',
                            pool_address=pool,
                            reserve0=base_liq,
                            reserve1=base_liq * float(bpt_per_t0_prop),
                            fee=0.0,
                        )
                        set_edge_meta(graph.graph, token0, lp_token, dex='balancer_lp_join_prop', pool_address=pool,
                                      fee_tier=None, source='onchain', confidence=0.85,
                                      extra={'proportional': True, 'dec_in': d0})
                        updated += 2
                    # token1 proportional
                    wi1, _, bi1, _ = self.collector.get_weights_and_balances(pool, token1, token1)
                    if ts > 0 and bi1 > 0:
                        bpt_per_t1_prop = ts / float(bi1)
                        graph.add_trading_pair(
                            token0=token1,
                            token1=lp_token,
                            dex='balancer_lp_join_prop',
                            pool_address=pool,
                            reserve0=base_liq,
                            reserve1=base_liq * float(bpt_per_t1_prop),
                            fee=0.0,
                        )
                        set_edge_meta(graph.graph, token1, lp_token, dex='balancer_lp_join_prop', pool_address=pool,
                                      fee_tier=None, source='onchain', confidence=0.85,
                                      extra={'proportional': True, 'dec_in': d1})
                        updated += 2
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"Balancer join_pool update failed: {e}")
        return updated


class BalancerExitPoolAction(ProtocolAction):
    name = "balancer.exit_pool"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = BalancerWeightedCollector(w3)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 120.0
        for token0, token1 in list(combinations(tokens.values(), 2)):
            try:
                pool = self.collector.find_pool_for_pair(token0, token1)
                if not pool:
                    continue
                lp_token = bpt(pool)
                # LP -> token0 (1 BPT) — single-asset exit (with fee on non-proportional part)
                t0_per_bpt = self.collector.token_out_per_bpt_in(pool, token0, 1.0)
                if t0_per_bpt > 0:
                    graph.add_trading_pair(
                        token0=lp_token,
                        token1=token0,
                        dex='balancer_lp_exit',
                        pool_address=pool,
                        reserve0=base_liq,
                        reserve1=base_liq * float(t0_per_bpt),
                        fee=0.0,
                    )
                    set_edge_meta(graph.graph, lp_token, token0, dex='balancer_lp_exit', pool_address=pool,
                                  fee_tier=None, source='onchain', confidence=0.88)
                    updated += 2
                # LP -> token1 (1 BPT) — single-asset exit
                t1_per_bpt = self.collector.token_out_per_bpt_in(pool, token1, 1.0)
                if t1_per_bpt > 0:
                    graph.add_trading_pair(
                        token0=lp_token,
                        token1=token1,
                        dex='balancer_lp_exit',
                        pool_address=pool,
                        reserve0=base_liq,
                        reserve1=base_liq * float(t1_per_bpt),
                        fee=0.0,
                    )
                    set_edge_meta(graph.graph, lp_token, token1, dex='balancer_lp_exit', pool_address=pool,
                                  fee_tier=None, source='onchain', confidence=0.88)
                    updated += 2

                # Proportional exit branch (no swap fee): token per 1 BPT = balance_i / ts
                try:
                    bpt_dec, ts_raw = self.collector.get_bpt_info(pool)
                    ts = float(ts_raw) / float(10 ** bpt_dec) if bpt_dec else float(ts_raw) / 1e18
                    wi0, _, bi0, _ = self.collector.get_weights_and_balances(pool, token0, token0)
                    if ts > 0 and bi0 > 0:
                        t0_per_bpt_prop = float(bi0) / ts
                        graph.add_trading_pair(
                            token0=lp_token,
                            token1=token0,
                            dex='balancer_lp_exit_prop',
                            pool_address=pool,
                            reserve0=base_liq,
                            reserve1=base_liq * float(t0_per_bpt_prop),
                            fee=0.0,
                        )
                        set_edge_meta(graph.graph, lp_token, token0, dex='balancer_lp_exit_prop', pool_address=pool,
                                      fee_tier=None, source='onchain', confidence=0.85,
                                      extra={'proportional': True})
                        updated += 2
                    wi1, _, bi1, _ = self.collector.get_weights_and_balances(pool, token1, token1)
                    if ts > 0 and bi1 > 0:
                        t1_per_bpt_prop = float(bi1) / ts
                        graph.add_trading_pair(
                            token0=lp_token,
                            token1=token1,
                            dex='balancer_lp_exit_prop',
                            pool_address=pool,
                            reserve0=base_liq,
                            reserve1=base_liq * float(t1_per_bpt_prop),
                            fee=0.0,
                        )
                        set_edge_meta(graph.graph, lp_token, token1, dex='balancer_lp_exit_prop', pool_address=pool,
                                      fee_tier=None, source='onchain', confidence=0.85,
                                      extra={'proportional': True})
                        updated += 2
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"Balancer exit_pool update failed: {e}")
        return updated


class AaveBorrowAction(ProtocolAction):
    name = "aave.borrow"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = AaveV2Collector(w3)
        self.fee = 0.0005  # borrowing cost approximation

    def _debt_token_id(self, underlying: str) -> str:
        return debt_aave(underlying)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 100.0
        for sym, underlying in tokens.items():
            try:
                addrs = self.collector.get_reserve_tokens(underlying)
                if not addrs:
                    # fallback to atoken-only support
                    if not self.collector.get_atoken(underlying):
                        continue
                    # variable debt synth fallback
                    debt_var = debt_aave(underlying)
                    graph.add_trading_pair(
                        token0=debt_var,
                        token1=underlying,
                        dex='aave_borrow',
                        pool_address=f"aave_borrow_{underlying[:6]}",
                        reserve0=base_liq,
                        reserve1=base_liq * 1.0,
                        fee=self.fee,
                    )
                    set_edge_meta(graph.graph, debt_var, underlying, dex='aave_borrow', pool_address=f"aave_borrow_{underlying[:6]}",
                                  fee_tier=None, source='approx', confidence=0.75,
                                  extra={'debt_type': 'variable', 'fallback': True})
                    updated += 2
                    continue
                a_token, stable_debt, variable_debt = addrs
                cfg = self.collector.get_reserve_configuration(underlying) or {}
                rates = self.collector.get_reserve_rates(underlying) or {}
                emode = self.collector.get_emode_category(underlying)
                # Borrow variable
                if isinstance(variable_debt, str) and int(variable_debt, 16) != 0:
                    graph.add_trading_pair(
                        token0=variable_debt,
                        token1=underlying,
                        dex='aave_borrow_variable',
                        pool_address=f"aave_borrow_var_{underlying[:6]}",
                        reserve0=base_liq,
                        reserve1=base_liq * 1.0,
                        fee=self.fee,
                    )
                    set_edge_meta(
                        graph.graph, variable_debt, underlying, dex='aave_borrow_variable', pool_address=f"aave_borrow_var_{underlying[:6]}",
                        fee_tier=None, source='onchain', confidence=0.9,
                        extra={'debt_type': 'variable', 'ltv': cfg.get('ltv'), 'liquidationThreshold': cfg.get('liquidationThreshold'),
                               'variableBorrowRate': rates.get('variableBorrowRate'), 'eModeCategory': emode}
                    )
                    updated += 2
                # Borrow stable
                if isinstance(stable_debt, str) and int(stable_debt, 16) != 0:
                    graph.add_trading_pair(
                        token0=stable_debt,
                        token1=underlying,
                        dex='aave_borrow_stable',
                        pool_address=f"aave_borrow_st_{underlying[:6]}",
                        reserve0=base_liq,
                        reserve1=base_liq * 1.0,
                        fee=self.fee,
                    )
                    set_edge_meta(
                        graph.graph, stable_debt, underlying, dex='aave_borrow_stable', pool_address=f"aave_borrow_st_{underlying[:6]}",
                        fee_tier=None, source='onchain', confidence=0.9,
                        extra={'debt_type': 'stable', 'ltv': cfg.get('ltv'), 'liquidationThreshold': cfg.get('liquidationThreshold'),
                               'stableBorrowRate': rates.get('stableBorrowRate'), 'eModeCategory': emode}
                    )
                    updated += 2
            except Exception as e:
                logger.debug(f"Aave borrow update failed {sym}: {e}")
        return updated


class AaveRepayAction(ProtocolAction):
    name = "aave.repay"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = AaveV2Collector(w3)
        self.fee = 0.0  # repay treated as neutral in this model

    def _debt_token_id(self, underlying: str) -> str:
        return f"aave_vdebt:{underlying}"

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 100.0
        for sym, underlying in tokens.items():
            try:
                addrs = self.collector.get_reserve_tokens(underlying)
                if not addrs:
                    if not self.collector.get_atoken(underlying):
                        continue
                    debt_var = f"aave_vdebt:{underlying}"
                    graph.add_trading_pair(
                        token0=underlying,
                        token1=debt_var,
                        dex='aave_repay_variable',
                        pool_address=f"aave_repay_var_{underlying[:6]}",
                        reserve0=base_liq,
                        reserve1=base_liq * 1.0,
                        fee=self.fee,
                    )
                    set_edge_meta(
                        graph.graph, underlying, debt_var, dex='aave_repay_variable', pool_address=f"aave_repay_var_{underlying[:6]}",
                        fee_tier=None, source='approx', confidence=0.75,
                        extra={'debt_type': 'variable', 'fallback': True}
                    )
                    updated += 2
                    continue
                a_token, stable_debt, variable_debt = addrs
                cfg = self.collector.get_reserve_configuration(underlying) or {}
                rates = self.collector.get_reserve_rates(underlying) or {}
                emode = self.collector.get_emode_category(underlying)
                # Repay variable
                if isinstance(variable_debt, str) and int(variable_debt, 16) != 0:
                    graph.add_trading_pair(
                        token0=underlying,
                        token1=variable_debt,
                        dex='aave_repay_variable',
                        pool_address=f"aave_repay_var_{underlying[:6]}",
                        reserve0=base_liq,
                        reserve1=base_liq * 1.0,
                        fee=self.fee,
                    )
                    set_edge_meta(
                        graph.graph, underlying, variable_debt, dex='aave_repay_variable', pool_address=f"aave_repay_var_{underlying[:6]}",
                        fee_tier=None, source='onchain', confidence=0.9,
                        extra={'debt_type': 'variable', 'ltv': cfg.get('ltv'), 'liquidationThreshold': cfg.get('liquidationThreshold'),
                               'variableBorrowRate': rates.get('variableBorrowRate'), 'eModeCategory': emode}
                    )
                    updated += 2
                # Repay stable
                if isinstance(stable_debt, str) and int(stable_debt, 16) != 0:
                    graph.add_trading_pair(
                        token0=underlying,
                        token1=stable_debt,
                        dex='aave_repay_stable',
                        pool_address=f"aave_repay_st_{underlying[:6]}",
                        reserve0=base_liq,
                        reserve1=base_liq * 1.0,
                        fee=self.fee,
                    )
                    set_edge_meta(
                        graph.graph, underlying, stable_debt, dex='aave_repay_stable', pool_address=f"aave_repay_st_{underlying[:6]}",
                        fee_tier=None, source='onchain', confidence=0.9,
                        extra={'debt_type': 'stable', 'ltv': cfg.get('ltv'), 'liquidationThreshold': cfg.get('liquidationThreshold'),
                               'stableBorrowRate': rates.get('stableBorrowRate'), 'eModeCategory': emode}
                    )
                    updated += 2
            except Exception as e:
                logger.debug(f"Aave repay update failed {sym}: {e}")
        return updated


class CompoundBorrowAction(ProtocolAction):
    name = "compound.borrow"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = CompoundCollector(w3)
        self.fee = 0.0005  # borrowing cost approximation

    def _debt_token_id(self, underlying: str) -> str:
        return debt_compound(underlying)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 100.0
        for sym, underlying in tokens.items():
            try:
                ctoken = self.collector.get_ctoken(underlying)
                if not ctoken:
                    continue
                debt = self._debt_token_id(underlying)
                # interest penalty approximation
                hold_blocks = int(getattr(config, 'interest_hold_blocks', 100))
                ip = self.collector.approx_interest_penalty(ctoken, hold_blocks)
                eff = 1.0 / (1.0 + ip) if ip > 0 else 1.0
                graph.add_trading_pair(
                    token0=debt,
                    token1=underlying,
                    dex='compound_borrow',
                    pool_address=f"compound_borrow_{underlying[:6]}",
                    reserve0=base_liq,
                    reserve1=base_liq * float(eff),
                    fee=0.0,
                )
                rates = self.collector.get_rates_per_block(ctoken) or {}
                cf = self.collector.get_collateral_factor(ctoken)
                set_edge_meta(
                    graph.graph, debt, underlying, dex='compound_borrow', pool_address=f"compound_borrow_{underlying[:6]}",
                    fee_tier=None, source='onchain', confidence=0.85,
                    extra={'collateralFactor': cf, 'borrowRatePerBlock': rates.get('borrowRatePerBlock'),
                           'supplyRatePerBlock': rates.get('supplyRatePerBlock'), 'interestPenalty': float(ip), 'holdBlocks': hold_blocks}
                )
                updated += 2
            except Exception as e:
                logger.debug(f"Compound borrow update failed {sym}: {e}")
        return updated


class CompoundRepayAction(ProtocolAction):
    name = "compound.repay"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = CompoundCollector(w3)
        self.fee = 0.0

    def _debt_token_id(self, underlying: str) -> str:
        return f"compound_debt:{underlying}"

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 100.0
        for sym, underlying in tokens.items():
            try:
                ctoken = self.collector.get_ctoken(underlying)
                if not ctoken:
                    continue
                mi = self.collector.get_market_info(ctoken) or {}
                if not mi.get('isListed', True):
                    continue
                debt = self._debt_token_id(underlying)
                graph.add_trading_pair(
                    token0=underlying,
                    token1=debt,
                    dex='compound_repay',
                    pool_address=f"compound_repay_{underlying[:6]}",
                    reserve0=base_liq,
                    reserve1=base_liq * 1.0,
                    fee=self.fee,
                )
                rates = self.collector.get_rates_per_block(ctoken) or {}
                closef = self.collector.get_close_factor()
                liqinc = self.collector.get_liquidation_incentive()
                cf = mi.get('collateralFactor') if mi else self.collector.get_collateral_factor(ctoken)
                set_edge_meta(
                    graph.graph, underlying, debt, dex='compound_repay', pool_address=f"compound_repay_{underlying[:6]}",
                    fee_tier=None, source='onchain', confidence=0.85,
                    extra={'collateralFactor': cf, 'borrowRatePerBlock': rates.get('borrowRatePerBlock'),
                           'supplyRatePerBlock': rates.get('supplyRatePerBlock'), 'isListed': mi.get('isListed'),
                           'closeFactor': closef, 'liquidationIncentive': liqinc}
                )
                updated += 2
            except Exception as e:
                logger.debug(f"Compound repay update failed {sym}: {e}")
        return updated


class SynthetixMintAction(ProtocolAction):
    name = "synthetix.mint"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = SynthetixCollector(w3)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        try:
            synths = self.collector.get_synths()
            sUSD = synths["sUSD"]
            SNX = synths["SNX"]
            rate = self.collector.mintable_susd_per_snx(collateral_ratio=5.0, safety_factor=0.95)
            if rate <= 0:
                return 0
            base_liq = 120.0
            graph.add_trading_pair(
                token0=SNX,
                token1=sUSD,
                dex='synthetix_mint',
                pool_address='synthetix_mint_snx_susd',
                reserve0=base_liq,
                reserve1=base_liq * rate,
                fee=0.0,
            )
            set_edge_meta(graph.graph, SNX, sUSD, dex='synthetix_mint', pool_address='synthetix_mint_snx_susd',
                          fee_tier=None, source='approx', confidence=0.75)
            gc = estimate_gas_cost_usd_for_dex(w3, 'synthetix')
            set_edge_gas_cost(graph.graph, SNX, sUSD, dex='synthetix_mint', pool_address='synthetix_mint_snx_susd', gas_cost=gc)
            return 2
        except Exception as e:
            logger.debug(f"Synthetix mint update failed: {e}")
            return 0


class SynthetixBurnAction(ProtocolAction):
    name = "synthetix.burn"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = SynthetixCollector(w3)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        try:
            synths = self.collector.get_synths()
            sUSD = synths["sUSD"]
            SNX = synths["SNX"]
            rate = self.collector.unlockable_snx_per_susd(collateral_ratio=5.0, safety_factor=0.95)
            if rate <= 0:
                return 0
            base_liq = 120.0
            graph.add_trading_pair(
                token0=sUSD,
                token1=SNX,
                dex='synthetix_burn',
                pool_address='synthetix_burn_susd_snx',
                reserve0=base_liq,
                reserve1=base_liq * rate,
                fee=0.0,
            )
            set_edge_meta(graph.graph, sUSD, SNX, dex='synthetix_burn', pool_address='synthetix_burn_susd_snx',
                          fee_tier=None, source='approx', confidence=0.75)
            gc = estimate_gas_cost_usd_for_dex(w3, 'synthetix')
            set_edge_gas_cost(graph.graph, sUSD, SNX, dex='synthetix_burn', pool_address='synthetix_burn_susd_snx', gas_cost=gc)
            return 2
        except Exception as e:
            logger.debug(f"Synthetix burn update failed: {e}")
            return 0


class DyDxOpenPositionAction(ProtocolAction):
    name = "dydx.open_position"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class DyDxClosePositionAction(ProtocolAction):
    name = "dydx.close_position"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class MakerPsmSwapAction(ProtocolAction):
    name = "maker.psm_swap"
    enabled = True

    def __init__(self, w3: Web3):
        self.w3 = w3
        self.fee = 0.0001  # fallback ~0.01%

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        # USDC <-> DAI near 1:1 swap via PSM (approximate)
        usdc = tokens.get('USDC') or tokens.get('usdc')
        dai = tokens.get('DAI') or tokens.get('dai')
        if not usdc or not dai:
            return 0
        base_liq = 500.0
        try:
            # Read tin/tout from PSM if configured
            from config.config import config
            from src.maker_collectors import MakerCollector
            psm_addr = getattr(config, 'maker_psm_usdc', '')
            tin = tout = self.fee
            if psm_addr:
                try:
                    mk = MakerCollector(self.w3, weth=usdc, dai=dai)
                    fees = mk.get_psm_fees(psm_addr)
                    if fees:
                        tin, tout = fees
                except Exception:
                    pass
            # Use conservative fee as max(tin, tout) due to symmetric edge limitation
            fee_use = max(float(tin or 0.0), float(tout or 0.0), float(self.fee))
            graph.add_trading_pair(
                token0=usdc,
                token1=dai,
                dex='maker_psm',
                pool_address=psm_addr or 'maker_psm_usdc_dai',
                reserve0=base_liq,
                reserve1=base_liq,
                fee=fee_use,
            )
            set_edge_meta(graph.graph, usdc, dai, dex='maker_psm', pool_address=psm_addr or 'maker_psm_usdc_dai',
                          fee_tier=None, source='onchain' if psm_addr else 'approx', confidence=0.95,
                          extra={'psm_tin': float(tin), 'psm_tout': float(tout), 'psm_address': psm_addr or None})
            return 2
        except Exception as e:
            logger.debug(f"Maker PSM update failed: {e}")
            return 0


class StablecoinAggregatorAction(ProtocolAction):
    name = "stable.aggregator"
    enabled = True

    def __init__(self):
        # 보수적 기본 수수료 (집계 라우팅/브릿지, CEX-온체인 조합 근사)
        self.fee = 0.0005
        self.base_liq = 1_000.0

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        usdc = tokens.get('USDC') or tokens.get('usdc')
        usdt = tokens.get('USDT') or tokens.get('usdt')
        dai = tokens.get('DAI') or tokens.get('dai')
        stables = [t for t in (usdc, usdt, dai) if t]
        if len(stables) < 2:
            return 0
        pairs = []
        if usdc and usdt:
            pairs.append((usdc, usdt, 'stable_agg_usdc_usdt'))
        if usdc and dai:
            pairs.append((usdc, dai, 'stable_agg_usdc_dai'))
        if usdt and dai:
            pairs.append((usdt, dai, 'stable_agg_usdt_dai'))
        for a, b, key in pairs:
            try:
                graph.add_trading_pair(
                    token0=a,
                    token1=b,
                    dex='stable_agg',
                    pool_address=key,
                    reserve0=self.base_liq,
                    reserve1=self.base_liq,
                    fee=self.fee,
                )
                set_edge_meta(graph.graph, a, b, dex='stable_agg', pool_address=key,
                              fee_tier=None, source='approx', confidence=0.85,
                              extra={'stable': True})
                updated += 2
            except Exception as e:
                logger.debug(f"Stable aggregator add failed {key}: {e}")
        return updated


class CompoundSupplyBorrowAction(ProtocolAction):
    name = "compound.supply_borrow"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = CompoundCollector(w3)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 100.0
        for sym, underlying in tokens.items():
            try:
                ctoken = self.collector.get_ctoken(underlying)
                if not ctoken:
                    continue
                mi = self.collector.get_market_info(ctoken) or {}
                if not mi.get('isListed', True):
                    continue
                rate = self.collector.get_deposit_rate_underlying_to_ctoken(ctoken, underlying)  # precise cToken per underlying
                reserve0 = base_liq
                reserve1 = base_liq * rate
                graph.add_trading_pair(
                    token0=underlying,
                    token1=ctoken,
                    dex='compound',
                    pool_address=ctoken,
                    reserve0=reserve0,
                    reserve1=reserve1,
                    fee=0.0,
                )
                cf = mi.get('collateralFactor') if mi else self.collector.get_collateral_factor(ctoken)
                rates = self.collector.get_rates_per_block(ctoken) or {}
                closef = self.collector.get_close_factor()
                liqinc = self.collector.get_liquidation_incentive()
                set_edge_meta(
                    graph.graph, underlying, ctoken, dex='compound', pool_address=ctoken,
                    fee_tier=None, source='onchain', confidence=0.9,
                    extra={'collateralFactor': cf, 'borrowRatePerBlock': rates.get('borrowRatePerBlock'),
                           'supplyRatePerBlock': rates.get('supplyRatePerBlock'), 'isListed': mi.get('isListed'),
                           'closeFactor': closef, 'liquidationIncentive': liqinc}
                )
                updated += 2
            except Exception as e:
                logger.debug(f"Compound update failed {sym}: {e}")
        return updated


class ActionRegistry:
    def __init__(self):
        self.actions: Dict[str, ProtocolAction] = {}

    def register(self, action: ProtocolAction):
        if action.name in self.actions:
            logger.warning(f"Action already registered: {action.name}, overriding")
        self.actions[action.name] = action

    def list_enabled(self) -> List[ProtocolAction]:
        return [a for a in self.actions.values() if getattr(a, 'enabled', False)]

    async def update_all(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                         block_number: Optional[int] = None) -> int:
        total = 0
        for action in self.list_enabled():
            try:
                updated = await action.update_graph(graph, w3, tokens, block_number)
                total += updated
            except Exception as e:
                logger.error(f"Action {action.name} failed: {e}")
        return total


class WETHWrapAction(ProtocolAction):
    name = "weth.wrap"
    enabled = True

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        return _weth_wrap_update(graph, tokens)


def register_default_actions(w3: Web3) -> ActionRegistry:
    reg = ActionRegistry()
    # Enabled core swaps
    reg.register(UniswapV2SwapAction(w3))
    reg.register(SushiSwapSwapAction(w3))
    reg.register(WETHWrapAction())

    # Disabled skeletons for scalability toward 96 actions
    reg.register(UniswapV3SwapAction(w3))
    reg.register(CurveStableSwapAction(w3))
    reg.register(CurveAddLiquidityAction(w3))
    reg.register(CurveRemoveLiquidityAction(w3))
    reg.register(BalancerWeightedSwapAction(w3))
    reg.register(BalancerJoinPoolAction(w3))
    reg.register(BalancerExitPoolAction(w3))
    reg.register(UniswapV2AddLiquidityAction(w3))
    reg.register(UniswapV2RemoveLiquidityAction(w3))
    reg.register(UniswapV3AddLiquidityAction(w3))
    reg.register(UniswapV3RemoveLiquidityAction(w3))
    reg.register(UniswapV3CollectFeesAction(w3))
    reg.register(AaveSupplyBorrowAction(w3))
    reg.register(AaveBorrowAction(w3))
    reg.register(AaveRepayAction(w3))
    reg.register(CompoundSupplyBorrowAction(w3))
    reg.register(CompoundBorrowAction(w3))
    reg.register(CompoundRepayAction(w3))
    reg.register(MakerCdpAction(w3))
    reg.register(MakerPsmSwapAction(w3))
    reg.register(StablecoinAggregatorAction())
    reg.register(YearnVaultAction(w3))
    reg.register(SynthetixExchangeAction(w3))
    reg.register(SynthetixMintAction(w3))
    reg.register(SynthetixBurnAction(w3))
    reg.register(DyDxMarginAction(w3))
    # Additional placeholders to reach and organize toward 96 actions
    class KyberSwapAction(ProtocolAction):
        name = "kyber.swap"
        enabled = False
        async def update_graph(self, *a, **k) -> int: return 0

    class OneInchSwapAction(ProtocolAction):
        name = "1inch.swap"
        enabled = False
        async def update_graph(self, *a, **k) -> int: return 0

    class ParaSwapAction(ProtocolAction):
        name = "paraswap.swap"
        enabled = False
        async def update_graph(self, *a, **k) -> int: return 0

    class BancorAction(ProtocolAction):
        name = "bancor.pool"
        enabled = False
        async def update_graph(self, *a, **k) -> int: return 0

    class MStableAction(ProtocolAction):
        name = "mstable.swap"
        enabled = False
        async def update_graph(self, *a, **k) -> int: return 0

    class BalancerV2Action(ProtocolAction):
        name = "balancer_v2.swap"
        enabled = False
        async def update_graph(self, *a, **k) -> int: return 0

    class AaveV3Action(ProtocolAction):
        name = "aave_v3.lend"
        enabled = False
        async def update_graph(self, *a, **k) -> int: return 0

    class CompoundV3Action(ProtocolAction):
        name = "compound_v3.lend"
        enabled = False
        async def update_graph(self, *a, **k) -> int: return 0

    class GMXAction(ProtocolAction):
        name = "gmx.perp"
        enabled = False
        async def update_graph(self, *a, **k) -> int: return 0

    class CurveMetaPoolAction(ProtocolAction):
        name = "curve.metapool"
        enabled = False
        async def update_graph(self, *a, **k) -> int: return 0

    reg.register(KyberSwapAction())
    reg.register(OneInchSwapAction())
    reg.register(ParaSwapAction())
    reg.register(BancorAction())
    reg.register(MStableAction())
    reg.register(BalancerV2Action())
    reg.register(AaveV3Action())
    reg.register(CompoundV3Action())
    reg.register(GMXAction())
    reg.register(CurveMetaPoolAction())
    return reg
