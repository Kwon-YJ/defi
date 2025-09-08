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
from src.dex_balancer_collector import BalancerWeightedCollector

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
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class YearnVaultAction(ProtocolAction):
    name = "yearn.vault"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class SynthetixExchangeAction(ProtocolAction):
    name = "synthetix.exchange"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class DyDxMarginAction(ProtocolAction):
    name = "dydx.margin"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
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
    reg.register(AaveSupplyBorrowAction(w3))
    reg.register(CompoundSupplyBorrowAction(w3))
    reg.register(MakerCdpAction())
    reg.register(YearnVaultAction())
    reg.register(SynthetixExchangeAction())
    reg.register(DyDxMarginAction())
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
