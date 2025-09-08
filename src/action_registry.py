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

logger = setup_logger(__name__)


class ProtocolAction:
    """Protocol action interface for scalable integration (target: 96 actions)."""

    name: str = "base"
    enabled: bool = False

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        """Update market graph for this action. Returns number of edges updated/added."""
        raise NotImplementedError


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
                graph.add_trading_pair(
                    token0=t0,
                    token1=t1,
                    dex='uniswap_v2',
                    pool_address=pair_address,
                    reserve0=float(nr0),
                    reserve1=float(nr1),
                    fee=self.fee,
                )
                set_edge_meta(graph.graph, t0, t1, dex='uniswap_v2', pool_address=pair_address,
                              fee_tier=None, source='onchain', confidence=0.98)
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
                graph.add_trading_pair(
                    token0=t0,
                    token1=t1,
                    dex='sushiswap',
                    pool_address=pair_address,
                    reserve0=float(nr0),
                    reserve1=float(nr1),
                    fee=self.fee,
                )
                set_edge_meta(graph.graph, t0, t1, dex='sushiswap', pool_address=pair_address,
                              fee_tier=None, source='onchain', confidence=0.98)
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

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        for token0, token1 in self._major_pairs(tokens):
            try:
                edges = await self.collector.build_edges_for_pair(token0, token1)
                if not edges:
                    continue
                for e in edges:
                    # 유사 리저브를 통한 환율 반영 (add_trading_pair 내부에서 수수료 감안)
                    graph.add_trading_pair(
                        token0=e['token0'],
                        token1=e['token1'],
                        dex='uniswap_v3',
                        pool_address=e['pool'],
                        reserve0=float(e['reserve0']),
                        reserve1=float(e['reserve1']),
                        fee=float(e['fee_fraction']),
                    )
                    set_edge_meta(graph.graph, e['token0'], e['token1'], dex='uniswap_v3', pool_address=e['pool'],
                                  fee_tier=e.get('fee_tier'), source='onchain', confidence=0.95)
                    updated += 2
            except Exception as e:
                logger.debug(f"UniswapV3 update failed {token0[:6]}-{token1[:6]}: {e}")
        return updated


class CurveStableSwapAction(ProtocolAction, _SwapPairsMixin):
    name = "curve.stableswap"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = CurveStableSwapCollector(w3)
        self.fee = 0.0004  # 0.04% typical for stables

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
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
                reserve0 = base_liq
                reserve1 = base_liq * price01
                graph.add_trading_pair(
                    token0=token0,
                    token1=token1,
                    dex='curve',
                    pool_address=pool,
                    reserve0=reserve0,
                    reserve1=reserve1,
                    fee=self.fee,
                )
                set_edge_meta(graph.graph, token0, token1, dex='curve', pool_address=pool,
                              fee_tier=None, source='onchain', confidence=0.9)
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
        base_liq = 150.0
        for token0, token1 in self._major_pairs(tokens):
            try:
                pool = self.collector.find_pool_for_pair(token0, token1)
                if not pool:
                    continue
                price01, fee_frac = self.collector.get_spot_price_and_fee(pool, token0, token1)
                if price01 <= 0:
                    continue
                reserve0 = base_liq
                reserve1 = base_liq * price01
                graph.add_trading_pair(
                    token0=token0,
                    token1=token1,
                    dex='balancer',
                    pool_address=pool,
                    reserve0=reserve0,
                    reserve1=reserve1,
                    fee=float(fee_frac),
                )
                set_edge_meta(graph.graph, token0, token1, dex='balancer', pool_address=pool,
                              fee_tier=None, source='onchain', confidence=0.9)
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
                set_edge_meta(graph.graph, underlying, atoken, dex='aave', pool_address=atoken,
                              fee_tier=None, source='approx', confidence=0.85)
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
        # Focus on WETH -> DAI minting
        sym_to_addr = tokens or {}
        weth = sym_to_addr.get('WETH') or sym_to_addr.get('weth')
        dai = sym_to_addr.get('DAI') or sym_to_addr.get('dai')
        if not weth or not dai:
            return 0
        try:
            collector = MakerCollector(self.w3, weth=weth, dai=dai)
            rate = collector.mintable_dai_per_weth(collateral_ratio=1.5, safety_factor=0.95)
            if rate <= 0:
                return 0
            base_liq = 200.0
            # Add pair representing CDP minting capacity
            graph.add_trading_pair(
                token0=weth,
                token1=dai,
                dex='maker',
                pool_address='maker_cdp_weth',
                reserve0=base_liq,
                reserve1=base_liq * rate,
                fee=0.0005,  # represent minor costs
            )
            set_edge_meta(graph.graph, weth, dai, dex='maker', pool_address='maker_cdp_weth',
                          fee_tier=None, source='approx', confidence=0.7)
            return 2
        except Exception as e:
            logger.debug(f"Maker CDP update failed: {e}")
            return 0


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
            price = self.collector.price_susd_per_seth()
            fee = self.collector.get_exchange_fee()
            if price <= 0:
                return 0
            base_liq = 200.0
            graph.add_trading_pair(
                token0=sETH,
                token1=sUSD,
                dex='synthetix',
                pool_address='synthetix_exchange_seth_susd',
                reserve0=base_liq,
                reserve1=base_liq * price,
                fee=fee,
            )
            set_edge_meta(graph.graph, sETH, sUSD, dex='synthetix', pool_address='synthetix_exchange_seth_susd',
                          fee_tier=None, source='onchain', confidence=0.85)
            return 2
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
                set_edge_meta(graph.graph, token0, token1, dex='dydx', pool_address='dydx_margin',
                              fee_tier=None, source='approx', confidence=0.7)
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
                # LP tokens are 18 decimals by convention
                lp_dec = 18
                nr0, nr1 = normalize_reserves(r0, d0, r1, d1)
                ts_norm = float(ts) / (10 ** lp_dec)
                # LP minted per unit token (split 50/50 assumption)
                lp_per_t0 = (ts_norm / nr0) * 0.5 if nr0 > 0 else 0.0
                lp_per_t1 = (ts_norm / nr1) * 0.5 if nr1 > 0 else 0.0
                # token0 -> LP
                graph.add_trading_pair(
                    token0=t0,
                    token1=pair,
                    dex='uniswap_v2_lp_add',
                    pool_address=pair,
                    reserve0=base_liq,
                    reserve1=base_liq * lp_per_t0,
                    fee=0.0,
                )
                set_edge_meta(graph.graph, t0, pair, dex='uniswap_v2_lp_add', pool_address=pair,
                              fee_tier=None, source='approx', confidence=0.8)
                updated += 2
                # token1 -> LP
                graph.add_trading_pair(
                    token0=t1,
                    token1=pair,
                    dex='uniswap_v2_lp_add',
                    pool_address=pair,
                    reserve0=base_liq,
                    reserve1=base_liq * lp_per_t1,
                    fee=0.0,
                )
                set_edge_meta(graph.graph, t1, pair, dex='uniswap_v2_lp_add', pool_address=pair,
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
                lp_dec = 18
                nr0, nr1 = normalize_reserves(r0, d0, r1, d1)
                ts_norm = float(ts) / (10 ** lp_dec)
                t0_per_lp = nr0 / ts_norm if ts_norm > 0 else 0.0
                t1_per_lp = nr1 / ts_norm if ts_norm > 0 else 0.0
                # LP -> token0
                graph.add_trading_pair(
                    token0=pair,
                    token1=t0,
                    dex='uniswap_v2_lp_remove',
                    pool_address=pair,
                    reserve0=base_liq,
                    reserve1=base_liq * t0_per_lp,
                    fee=0.0,
                )
                set_edge_meta(graph.graph, pair, t0, dex='uniswap_v2_lp_remove', pool_address=pair,
                              fee_tier=None, source='approx', confidence=0.8)
                updated += 2
                # LP -> token1
                graph.add_trading_pair(
                    token0=pair,
                    token1=t1,
                    dex='uniswap_v2_lp_remove',
                    pool_address=pair,
                    reserve0=base_liq,
                    reserve1=base_liq * t1_per_lp,
                    fee=0.0,
                )
                set_edge_meta(graph.graph, pair, t1, dex='uniswap_v2_lp_remove', pool_address=pair,
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
                    price01 = self.collector.price_from_sqrtX96(state['sqrtPriceX96'], state['dec0'], state['dec1'])
                    if price01 <= 0:
                        continue
                    t0 = state['token0']; t1 = state['token1']
                    lp_token = f"{pool}-v3lp"
                    lp_per_t0 = 1.0
                    lp_per_t1 = (1.0 / price01) if price01 > 0 else 0.0
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
                    set_edge_meta(graph.graph, t0, lp_token, dex='uniswap_v3_lp_add', pool_address=pool,
                                  fee_tier=fee, source='approx', confidence=0.75)
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
                        set_edge_meta(graph.graph, t1, lp_token, dex='uniswap_v3_lp_add', pool_address=pool,
                                      fee_tier=fee, source='approx', confidence=0.75)
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
                    price01 = self.collector.price_from_sqrtX96(state['sqrtPriceX96'], state['dec0'], state['dec1'])
                    if price01 <= 0:
                        continue
                    t0 = state['token0']; t1 = state['token1']
                    lp_token = f"{pool}-v3lp"
                    t0_per_lp = 1.0
                    t1_per_lp = price01
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
                    set_edge_meta(graph.graph, lp_token, t0, dex='uniswap_v3_lp_remove', pool_address=pool,
                                  fee_tier=fee, source='approx', confidence=0.75)
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
                    set_edge_meta(graph.graph, lp_token, t1, dex='uniswap_v3_lp_remove', pool_address=pool,
                                  fee_tier=fee, source='approx', confidence=0.75)
                    updated += 2
                except Exception as e:
                    logger.debug(f"UniswapV3 removeLiquidity failed {token0[:6]}-{token1[:6]}: {e}")
        return updated


class UniswapV3CollectFeesAction(ProtocolAction):
    name = "uniswap_v3.collect_fees"
    enabled = True

    def __init__(self, w3: Web3):
        self.collector = UniswapV3Collector(w3)

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 20.0
        fee_yield = 1e-4  # very small unit fees per LP
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
                    lp_token = f"{pool}-v3lp"
                    # LP -> token0 (fees)
                    graph.add_trading_pair(
                        token0=lp_token,
                        token1=t0,
                        dex='uniswap_v3_fee_collect',
                        pool_address=pool,
                        reserve0=base_liq,
                        reserve1=base_liq * fee_yield,
                        fee=0.0,
                    )
                    set_edge_meta(graph.graph, lp_token, t0, dex='uniswap_v3_fee_collect', pool_address=pool,
                                  fee_tier=fee, source='approx', confidence=0.6)
                    updated += 2
                    # LP -> token1 (fees)
                    graph.add_trading_pair(
                        token0=lp_token,
                        token1=t1,
                        dex='uniswap_v3_fee_collect',
                        pool_address=pool,
                        reserve0=base_liq,
                        reserve1=base_liq * fee_yield,
                        fee=0.0,
                    )
                    set_edge_meta(graph.graph, lp_token, t1, dex='uniswap_v3_fee_collect', pool_address=pool,
                                  fee_tier=fee, source='approx', confidence=0.6)
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
            # Iterate known pools (e.g., 3pool) and add token -> LP edges for member coins
            for p in self.collector.POOLS:
                pool = p['address']
                coins = [c.lower() for c in p['coins']]
                lp_token = f"{pool}-curvelp"
                for _, addr in tokens.items():
                    if addr.lower() in coins:
                        # Approximate 1 LP per 1 unit of stable for add liquidity
                        graph.add_trading_pair(
                            token0=addr,
                            token1=lp_token,
                            dex='curve_lp_add',
                            pool_address=pool,
                            reserve0=base_liq,
                            reserve1=base_liq * 1.0,
                            fee=0.0,
                        )
                        set_edge_meta(graph.graph, addr, lp_token, dex='curve_lp_add', pool_address=pool,
                                      fee_tier=None, source='approx', confidence=0.85)
                        updated += 2
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
            for p in self.collector.POOLS:
                pool = p['address']
                coins = [c.lower() for c in p['coins']]
                lp_token = f"{pool}-curvelp"
                for coin in coins:
                    # LP -> coin (single-asset withdraw) approximated at 1:1
                    graph.add_trading_pair(
                        token0=lp_token,
                        token1=coin,
                        dex='curve_lp_remove',
                        pool_address=pool,
                        reserve0=base_liq,
                        reserve1=base_liq * 1.0,
                        fee=0.0,
                    )
                    set_edge_meta(graph.graph, lp_token, coin, dex='curve_lp_remove', pool_address=pool,
                                  fee_tier=None, source='approx', confidence=0.85)
                    updated += 2
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
                price01, _ = self.collector.get_spot_price_and_fee(pool, token0, token1)
                if price01 <= 0:
                    continue
                lp_token = f"{pool}-balp"
                # token0 -> LP (unit)
                graph.add_trading_pair(
                    token0=token0,
                    token1=lp_token,
                    dex='balancer_lp_join',
                    pool_address=pool,
                    reserve0=base_liq,
                    reserve1=base_liq * 1.0,
                    fee=0.0,
                )
                set_edge_meta(graph.graph, token0, lp_token, dex='balancer_lp_join', pool_address=pool,
                              fee_tier=None, source='approx', confidence=0.85)
                updated += 2
                # token1 -> LP (scaled by inverse price)
                inv = (1.0 / price01) if price01 > 0 else 0.0
                if inv > 0:
                    graph.add_trading_pair(
                        token0=token1,
                        token1=lp_token,
                        dex='balancer_lp_join',
                        pool_address=pool,
                        reserve0=base_liq,
                        reserve1=base_liq * inv,
                        fee=0.0,
                    )
                    set_edge_meta(graph.graph, token1, lp_token, dex='balancer_lp_join', pool_address=pool,
                                  fee_tier=None, source='approx', confidence=0.85)
                    updated += 2
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
                price01, _ = self.collector.get_spot_price_and_fee(pool, token0, token1)
                if price01 <= 0:
                    continue
                lp_token = f"{pool}-balp"
                # LP -> token0 (unit)
                graph.add_trading_pair(
                    token0=lp_token,
                    token1=token0,
                    dex='balancer_lp_exit',
                    pool_address=pool,
                    reserve0=base_liq,
                    reserve1=base_liq * 1.0,
                    fee=0.0,
                )
                set_edge_meta(graph.graph, lp_token, token0, dex='balancer_lp_exit', pool_address=pool,
                              fee_tier=None, source='approx', confidence=0.85)
                updated += 2
                # LP -> token1 (scaled by price)
                graph.add_trading_pair(
                    token0=lp_token,
                    token1=token1,
                    dex='balancer_lp_exit',
                    pool_address=pool,
                    reserve0=base_liq,
                    reserve1=base_liq * price01,
                    fee=0.0,
                )
                set_edge_meta(graph.graph, lp_token, token1, dex='balancer_lp_exit', pool_address=pool,
                              fee_tier=None, source='approx', confidence=0.85)
                updated += 2
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
        return f"aave_vdebt:{underlying}"

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 100.0
        for sym, underlying in tokens.items():
            try:
                # Only consider assets that have an aToken mapping (supported on Aave)
                if not self.collector.get_atoken(underlying):
                    continue
                debt = self._debt_token_id(underlying)
                # Borrow: incur 1 debt unit to receive ~1 underlying
                graph.add_trading_pair(
                    token0=debt,
                    token1=underlying,
                    dex='aave_borrow',
                    pool_address=f"aave_borrow_{underlying[:6]}",
                    reserve0=base_liq,
                    reserve1=base_liq * 1.0,
                    fee=self.fee,
                )
                set_edge_meta(graph.graph, debt, underlying, dex='aave_borrow', pool_address=f"aave_borrow_{underlying[:6]}",
                              fee_tier=None, source='approx', confidence=0.8)
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
                if not self.collector.get_atoken(underlying):
                    continue
                debt = self._debt_token_id(underlying)
                # Repay: spend underlying to extinguish 1 debt unit
                graph.add_trading_pair(
                    token0=underlying,
                    token1=debt,
                    dex='aave_repay',
                    pool_address=f"aave_repay_{underlying[:6]}",
                    reserve0=base_liq,
                    reserve1=base_liq * 1.0,
                    fee=self.fee,
                )
                set_edge_meta(graph.graph, underlying, debt, dex='aave_repay', pool_address=f"aave_repay_{underlying[:6]}",
                              fee_tier=None, source='approx', confidence=0.8)
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
        return f"compound_debt:{underlying}"

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        updated = 0
        base_liq = 100.0
        for sym, underlying in tokens.items():
            try:
                if not self.collector.get_ctoken(underlying):
                    continue
                debt = self._debt_token_id(underlying)
                graph.add_trading_pair(
                    token0=debt,
                    token1=underlying,
                    dex='compound_borrow',
                    pool_address=f"compound_borrow_{underlying[:6]}",
                    reserve0=base_liq,
                    reserve1=base_liq * 1.0,
                    fee=self.fee,
                )
                set_edge_meta(graph.graph, debt, underlying, dex='compound_borrow', pool_address=f"compound_borrow_{underlying[:6]}",
                              fee_tier=None, source='approx', confidence=0.8)
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
                if not self.collector.get_ctoken(underlying):
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
                set_edge_meta(graph.graph, underlying, debt, dex='compound_repay', pool_address=f"compound_repay_{underlying[:6]}",
                              fee_tier=None, source='approx', confidence=0.8)
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
        self.fee = 0.0001  # ~0.01%

    async def update_graph(self, graph: DeFiMarketGraph, w3: Web3, tokens: Dict[str, str],
                           block_number: Optional[int] = None) -> int:
        # USDC <-> DAI near 1:1 swap via PSM (approximate)
        usdc = tokens.get('USDC') or tokens.get('usdc')
        dai = tokens.get('DAI') or tokens.get('dai')
        if not usdc or not dai:
            return 0
        base_liq = 500.0
        try:
            graph.add_trading_pair(
                token0=usdc,
                token1=dai,
                dex='maker_psm',
                pool_address='maker_psm_usdc_dai',
                reserve0=base_liq,
                reserve1=base_liq,
                fee=self.fee,
            )
            set_edge_meta(graph.graph, usdc, dai, dex='maker_psm', pool_address='maker_psm_usdc_dai',
                          fee_tier=None, source='approx', confidence=0.95)
            return 2
        except Exception as e:
            logger.debug(f"Maker PSM update failed: {e}")
            return 0


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
                rate = self.collector.get_deposit_rate_underlying_to_ctoken(ctoken)  # cToken per underlying
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
                set_edge_meta(graph.graph, underlying, ctoken, dex='compound', pool_address=ctoken,
                              fee_tier=None, source='onchain', confidence=0.9)
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


def register_default_actions(w3: Web3) -> ActionRegistry:
    reg = ActionRegistry()
    # Enabled core swaps
    reg.register(UniswapV2SwapAction(w3))
    reg.register(SushiSwapSwapAction(w3))

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
