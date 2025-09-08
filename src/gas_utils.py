from typing import Optional

from web3 import Web3
import networkx as nx


_DEX_GAS_LIMITS = {
    # Swaps
    'uniswap_v2': 150_000,
    'sushiswap': 150_000,
    'uniswap_v3': 180_000,
    'curve': 200_000,
    'balancer': 250_000,
    'synthetix': 220_000,
    'dydx': 250_000,
    # LP / Pool ops
    'uniswap_v2_lp_add': 200_000,
    'uniswap_v2_lp_remove': 180_000,
    'uniswap_v3_lp_add': 240_000,
    'uniswap_v3_lp_remove': 220_000,
    'uniswap_v3_fee_collect': 100_000,
    'curve_lp_add': 250_000,
    'curve_lp_remove': 240_000,
    'balancer_lp_join': 220_000,
    'balancer_lp_exit': 220_000,
    # Lending
    'aave': 180_000,
    'compound': 160_000,
    'yearn': 180_000,
    # Borrow / Repay
    'aave_borrow': 300_000,
    'aave_repay': 200_000,
    'compound_borrow': 220_000,
    'compound_repay': 200_000,
    # Maker
    'maker': 160_000,
    'maker_psm': 120_000,
}


def _get_base_fee_gwei(w3: Optional[Web3]) -> Optional[float]:
    try:
        if not w3 or not w3.is_connected():
            return None
        latest = w3.eth.get_block('latest')
        base = latest.get('baseFeePerGas')
        if base is None:
            return None
        return float(base) / 1e9
    except Exception:
        return None


def estimate_effective_gas_price_gwei(w3: Optional[Web3], priority_gwei: float = 2.0, fallback_gwei: float = 20.0) -> float:
    base = _get_base_fee_gwei(w3)
    if base is None:
        return fallback_gwei
    return base + max(0.0, priority_gwei)


def dex_gas_limit(dex: str) -> int:
    return _DEX_GAS_LIMITS.get(dex.lower(), 150_000)


def estimate_gas_cost_usd_for_dex(w3: Optional[Web3], dex: str, *, eth_price_usd: float = 2000.0,
                                  priority_gwei: float = 2.0, fallback_gwei: float = 20.0) -> float:
    gas_price_gwei = estimate_effective_gas_price_gwei(w3, priority_gwei, fallback_gwei)
    gas_price_wei = gas_price_gwei * 1e9
    gl = dex_gas_limit(dex)
    cost_eth = (gl * gas_price_wei) / 1e18
    return cost_eth * eth_price_usd


def set_edge_gas_cost(nx_graph: nx.Graph, token0: str, token1: str, *, dex: Optional[str] = None,
                      pool_address: Optional[str] = None, gas_cost: float = 0.0) -> None:
    """Override gas_cost on edges between token0 and token1 (both directions).

    Filters by dex and pool_address when provided. Supports DiGraph and MultiDiGraph.
    """
    def _apply(u: str, v: str):
        try:
            if not nx_graph.has_edge(u, v):
                return
            ed = nx_graph.get_edge_data(u, v)
            if isinstance(ed, dict) and 'exchange_rate' in ed:
                if dex and ed.get('dex') and ed.get('dex') != dex:
                    return
                if pool_address and ed.get('pool_address') and ed.get('pool_address') != pool_address:
                    return
                ed['gas_cost'] = float(gas_cost)
                return
            if isinstance(ed, dict):
                for _, data in ed.items():
                    if not isinstance(data, dict):
                        continue
                    if dex and data.get('dex') and data.get('dex') != dex:
                        continue
                    if pool_address and data.get('pool_address') and data.get('pool_address') != pool_address:
                        continue
                    data['gas_cost'] = float(gas_cost)
        except Exception:
            return

    _apply(token0, token1)
    _apply(token1, token0)

