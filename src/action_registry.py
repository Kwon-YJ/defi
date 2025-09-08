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
from src.yearn_collectors import YearnV2Collector

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
                graph.add_trading_pair(
                    token0=token0,
                    token1=token1,
                    dex='uniswap_v2',
                    pool_address=pair_address,
                    reserve0=float(r0),
                    reserve1=float(r1),
                    fee=self.fee,
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
                graph.add_trading_pair(
                    token0=token0,
                    token1=token1,
                    dex='sushiswap',
                    pool_address=pair_address,
                    reserve0=float(r0),
                    reserve1=float(r1),
                    fee=self.fee,
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
                # LP minted per token (approx; require both tokens, split 50/50)
                lp_per_t0 = (ts / r0) * 0.5
                lp_per_t1 = (ts / r1) * 0.5
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
                t0_per_lp = r0 / ts
                t1_per_lp = r1 / ts
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
                    updated += 2
                except Exception as e:
                    logger.debug(f"UniswapV3 collectFees failed {token0[:6]}-{token1[:6]}: {e}")
        return updated


class CurveAddLiquidityAction(ProtocolAction):
    name = "curve.add_liquidity"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class CurveRemoveLiquidityAction(ProtocolAction):
    name = "curve.remove_liquidity"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class BalancerJoinPoolAction(ProtocolAction):
    name = "balancer.join_pool"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class BalancerExitPoolAction(ProtocolAction):
    name = "balancer.exit_pool"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class AaveBorrowAction(ProtocolAction):
    name = "aave.borrow"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class AaveRepayAction(ProtocolAction):
    name = "aave.repay"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class CompoundBorrowAction(ProtocolAction):
    name = "compound.borrow"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class CompoundRepayAction(ProtocolAction):
    name = "compound.repay"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class SynthetixMintAction(ProtocolAction):
    name = "synthetix.mint"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class SynthetixBurnAction(ProtocolAction):
    name = "synthetix.burn"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
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
    reg.register(BalancerWeightedSwapAction(w3))
    reg.register(UniswapV2AddLiquidityAction(w3))
    reg.register(UniswapV2RemoveLiquidityAction(w3))
    reg.register(UniswapV3AddLiquidityAction(w3))
    reg.register(UniswapV3RemoveLiquidityAction(w3))
    reg.register(UniswapV3CollectFeesAction(w3))
    reg.register(AaveSupplyBorrowAction(w3))
    reg.register(CompoundSupplyBorrowAction(w3))
    reg.register(MakerCdpAction(w3))
    reg.register(MakerPsmSwapAction(w3))
    reg.register(YearnVaultAction(w3))
    reg.register(SynthetixExchangeAction(w3))
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
