from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable
from itertools import combinations
from web3 import Web3

from src.logger import setup_logger
from src.market_graph import DeFiMarketGraph
from src.dex_data_collector import UniswapV2Collector, SushiSwapCollector

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
class UniswapV3SwapAction(ProtocolAction):
    name = "uniswap_v3.swap"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class CurveStableSwapAction(ProtocolAction):
    name = "curve.stableswap"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class BalancerWeightedSwapAction(ProtocolAction):
    name = "balancer.weighted_swap"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


class AaveSupplyBorrowAction(ProtocolAction):
    name = "aave.supply_borrow"
    enabled = False
    async def update_graph(self, *args, **kwargs) -> int:
        return 0


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
    reg.register(UniswapV3SwapAction())
    reg.register(CurveStableSwapAction())
    reg.register(BalancerWeightedSwapAction())
    reg.register(AaveSupplyBorrowAction())
    reg.register(CompoundSupplyBorrowAction())
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
