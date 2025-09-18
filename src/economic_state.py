from typing import Dict, Optional
from src.logger import setup_logger
from src.market_graph import DeFiMarketGraph
from src.data_storage import DataStorage
from config.config import config
from src.edge_meta import set_edge_meta

logger = setup_logger(__name__)


class EconomicStateExploitation:
    """Injects synthetic edges that exploit current economic state anomalies.

    Initial scope:
    - Stablecoin depeg handling via Maker PSM synthetic edges (USDC<->DAI)
      When USDC and DAI drift beyond configured stable max deviation, introduce
      high-liquidity, low-fee edges representing PSM conversions in both directions.
    """

    def __init__(self, graph: DeFiMarketGraph, storage: Optional[DataStorage] = None):
        self.graph = graph
        self.storage = storage or DataStorage()

    async def inject(self, tokens: Dict[str, str]) -> int:
        """Examine current state and inject edges. Returns number of edges added.
        tokens: mapping of symbol->address
        """
        added = 0
        try:
            usdc = tokens.get('USDC') or tokens.get('usdc')
            dai = tokens.get('DAI') or tokens.get('dai')
            if not (usdc and dai):
                return 0
            # Read prices from storage populated by PriceFeed
            p_usdc = await self.storage.get_token_price(usdc)
            p_dai = await self.storage.get_token_price(dai)
            if not (p_usdc and p_dai):
                return 0
            pu = float(p_usdc.get('price_usd') or 0.0)
            pd = float(p_dai.get('price_usd') or 0.0)
            if pu <= 0 or pd <= 0:
                return 0
            ratio = pu / pd
            dev = abs(ratio - 1.0)
            thresh = float(getattr(config, 'price_stable_max_dev', 0.03))
            if dev <= thresh:
                return 0
            # Derive PSM fees
            try:
                from src.maker_collectors import MakerCollector
                from web3 import Web3
                w3 = Web3()  # offline object; fee calls require RPC only if address present
                mc = MakerCollector(w3, '', dai, usdc)
                tin_tout = None
                if getattr(config, 'maker_psm_usdc', ''):
                    tin_tout = mc.get_psm_fees(getattr(config, 'maker_psm_usdc'))
            except Exception:
                tin_tout = None
            tin, tout = 0.001, 0.001  # defaults
            if tin_tout and isinstance(tin_tout, tuple) and len(tin_tout) == 2:
                tin = max(0.0, float(tin_tout[0]))
                tout = max(0.0, float(tin_tout[1]))

            # Add directional edges representing PSM conversions
            # Use generous liquidity; real routing will size via local search and slippage models
            liq = 1_000_000.0
            gas = 0.0005  # ~low ETH equivalent (approx)
            pool_addr = getattr(config, 'maker_psm_usdc', '') or 'psm:usdc-dai'
            # USDC -> DAI (fee = tin)
            self.graph.add_directed_edge(
                from_token=usdc,
                to_token=dai,
                dex='maker_psm',
                pool_address=pool_addr,
                exchange_rate=max(0.0, 1.0 - float(tin)),
                liquidity=liq,
                fee=float(tin),
                gas_cost=gas,
                edge_key='psm_in',
            )
            set_edge_meta(self.graph.graph, usdc, dai, dex='maker_psm', pool_address=pool_addr,
                          fee_tier=None, source='approx' if not getattr(config, 'maker_psm_usdc', '') else 'onchain',
                          confidence=0.95,
                          extra={'psm_in': True, 't0': usdc, 't1': dai})
            added += 1
            # DAI -> USDC (fee = tout)
            self.graph.add_directed_edge(
                from_token=dai,
                to_token=usdc,
                dex='maker_psm',
                pool_address=pool_addr,
                exchange_rate=max(0.0, 1.0 - float(tout)),
                liquidity=liq,
                fee=float(tout),
                gas_cost=gas,
                edge_key='psm_out',
            )
            set_edge_meta(self.graph.graph, dai, usdc, dex='maker_psm', pool_address=pool_addr,
                          fee_tier=None, source='approx' if not getattr(config, 'maker_psm_usdc', '') else 'onchain',
                          confidence=0.95,
                          extra={'psm_out': True, 't0': dai, 't1': usdc})
            added += 1
            logger.info(f"Economic exploit: Maker PSM edges injected (dev={dev:.4f}, tin={tin}, tout={tout})")
        except Exception as e:
            logger.debug(f"Economic state injection failed: {e}")
            return 0
        return added

