from typing import Dict, Optional, List, Tuple
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
        try:
            stables: List[Tuple[str, str]] = []
            # gather available stable tokens in current token set
            stable_syms = ['USDC', 'USDT', 'DAI', 'TUSD', 'SUSD', 'sUSD']
            sym_to_addr: Dict[str, str] = {}
            for s in stable_syms:
                a = tokens.get(s) or tokens.get(s.lower())
                if a:
                    sym_to_addr[s.upper()] = a
            # create unique symbol pairs
            syms = list(sym_to_addr.keys())
            for i in range(len(syms)):
                for j in range(i + 1, len(syms)):
                    stables.append((syms[i], syms[j]))
            if not stables:
                return 0
            # load prices
            prices: Dict[str, float] = {}
            for s, addr in sym_to_addr.items():
                p = await self.storage.get_token_price(addr)
                if p and p.get('price_usd'):
                    prices[s] = float(p['price_usd'])
            if not prices:
                return 0
            thresh = float(getattr(config, 'price_stable_max_dev', 0.03))
            added = 0
            # helper: add PSM for USDC/DAI
            async def add_psm(usdc_addr: str, dai_addr: str, dev: float):
                nonlocal added
                tin, tout = 0.001, 0.001
                try:
                    from src.maker_collectors import MakerCollector
                    from web3 import Web3
                    mc = MakerCollector(Web3(), '', dai_addr, usdc_addr)
                    if getattr(config, 'maker_psm_usdc', ''):
                        fees = mc.get_psm_fees(getattr(config, 'maker_psm_usdc'))
                        if fees and len(fees) == 2:
                            tin = max(0.0, float(fees[0])); tout = max(0.0, float(fees[1]))
                except Exception:
                    pass
                liq = 1_000_000.0
                gas = 0.0005
                pool_addr = getattr(config, 'maker_psm_usdc', '') or 'psm:usdc-dai'
                self.graph.add_directed_edge(usdc_addr, dai_addr, dex='maker_psm', pool_address=pool_addr,
                                             exchange_rate=max(0.0, 1.0 - tin), liquidity=liq, fee=tin, gas_cost=gas, edge_key='psm_in')
                set_edge_meta(self.graph.graph, usdc_addr, dai_addr, dex='maker_psm', pool_address=pool_addr,
                              fee_tier=None, source='approx' if not getattr(config, 'maker_psm_usdc', '') else 'onchain',
                              confidence=0.95, extra={'psm_in': True, 't0': usdc_addr, 't1': dai_addr})
                added += 1
                self.graph.add_directed_edge(dai_addr, usdc_addr, dex='maker_psm', pool_address=pool_addr,
                                             exchange_rate=max(0.0, 1.0 - tout), liquidity=liq, fee=tout, gas_cost=gas, edge_key='psm_out')
                set_edge_meta(self.graph.graph, dai_addr, usdc_addr, dex='maker_psm', pool_address=pool_addr,
                              fee_tier=None, source='approx' if not getattr(config, 'maker_psm_usdc', '') else 'onchain',
                              confidence=0.95, extra={'psm_out': True, 't0': dai_addr, 't1': usdc_addr})
                added += 1
                logger.info(f"Economic exploit: Maker PSM injected (dev={dev:.4f}, tin={tin}, tout={tout})")

            # helper: add Curve stable edge approximation if pool exists
            def add_curve_edge(a_addr: str, b_addr: str, pool: str) -> None:
                nonlocal added
                fee = 0.0004
                try:
                    from src.dex_curve_collector import CurveStableSwapCollector
                    from web3 import Web3
                    cc = CurveStableSwapCollector(Web3())
                    # attempt to read normalized pool fee/admin_fee
                    try:
                        params = cc.get_pool_params(pool)
                        f = params.get('fee')
                        if f is not None and f > 0:
                            fee = float(f)
                    except Exception:
                        pass
                except Exception:
                    pass
                liq = 2_000_000.0
                gas = 0.0006
                # A->B and B->A near 1:1 minus fee
                self.graph.add_directed_edge(a_addr, b_addr, dex='curve_psm', pool_address=pool,
                                             exchange_rate=max(0.0, 1.0 - fee), liquidity=liq, fee=fee, gas_cost=gas, edge_key='curve_psm_ab')
                set_edge_meta(self.graph.graph, a_addr, b_addr, dex='curve_psm', pool_address=pool,
                              fee_tier=None, source='approx', confidence=0.9, extra={'stableswap': True, 't0': a_addr, 't1': b_addr})
                added += 1
                self.graph.add_directed_edge(b_addr, a_addr, dex='curve_psm', pool_address=pool,
                                             exchange_rate=max(0.0, 1.0 - fee), liquidity=liq, fee=fee, gas_cost=gas, edge_key='curve_psm_ba')
                set_edge_meta(self.graph.graph, b_addr, a_addr, dex='curve_psm', pool_address=pool,
                              fee_tier=None, source='approx', confidence=0.9, extra={'stableswap': True, 't0': b_addr, 't1': a_addr})
                added += 1

            # iterate candidate pairs
            for s0, s1 in stables:
                a0 = sym_to_addr[s0]
                a1 = sym_to_addr[s1]
                p0 = prices.get(s0)
                p1 = prices.get(s1)
                if not (p0 and p1 and p0 > 0 and p1 > 0):
                    continue
                dev = abs((p0 / p1) - 1.0)
                if dev <= thresh:
                    continue
                # prefer Maker PSM for USDC/DAI
                if {s0, s1} == {'USDC', 'DAI'}:
                    await add_psm(sym_to_addr['USDC'], sym_to_addr['DAI'], dev)
                    continue
                # else use Curve stables if available
                try:
                    from src.dex_curve_collector import CurveStableSwapCollector
                    from web3 import Web3
                    cc = CurveStableSwapCollector(Web3())
                    found = cc.find_pool_for_pair(a0, a1)
                    if found:
                        pool, _i, _j = found
                        add_curve_edge(a0, a1, pool)
                        logger.info(f"Economic exploit: Curve stabilization edges injected for {s0}/{s1} (dev={dev:.4f})")
                except Exception:
                    continue
            return added
        except Exception as e:
            logger.debug(f"Economic state injection failed: {e}")
            return 0
