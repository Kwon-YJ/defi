from typing import List
from src.market_graph import TradingEdge, ArbitrageOpportunity
from config.config import config


def _edge_sandwich_risk(edge: TradingEdge) -> float:
    dex = (edge.dex or '').lower()
    base = 0.2
    if dex in ('uniswap_v2', 'sushiswap'):
        base = 0.25
    elif dex == 'uniswap_v3':
        base = 0.2
    elif dex == 'curve' or dex == 'curve_psm':
        base = 0.1
    elif dex == 'balancer':
        base = 0.12
    elif dex == 'maker_psm':
        base = 0.05
    # Liquidity mitigates risk (simple saturation function)
    liq = max(1e-9, float(edge.liquidity or 0.0))
    liq_factor = 1.0 / (1.0 + (liq / 500.0))  # 500 units as scale
    fee = max(0.0, float(edge.fee or 0.0))
    fee_factor = max(0.8, 1.0 - fee * 10)  # higher fee slightly reduces sandwich incentive
    r = base * liq_factor * fee_factor
    return max(0.0, min(0.95, r))


def _path_sandwich_risk(edges: List[TradingEdge]) -> float:
    if not edges:
        return 0.0
    p = 1.0
    for e in edges:
        r = _edge_sandwich_risk(e)
        p *= (1.0 - r)
    return max(0.0, min(0.95, 1.0 - p))


def _estimate_bribe(net_profit_eth: float) -> float:
    try:
        pct = float(getattr(config, 'mev_bribe_pct_of_profit', 0.1))
        minb = float(getattr(config, 'mev_min_bribe_eth', 0.003))
    except Exception:
        pct, minb = 0.1, 0.003
    return max(minb, max(0.0, pct) * max(0.0, net_profit_eth))


def _priority_fee_gwei() -> float:
    try:
        return float(getattr(config, 'mev_priority_fee_gwei', 3.0))
    except Exception:
        return 3.0


def annotate_expected_value(opp: ArbitrageOpportunity) -> ArbitrageOpportunity:
    """Compute MEV-aware expected value and annotate the opportunity in-place."""
    risk = _path_sandwich_risk(opp.edges)
    bribe = _estimate_bribe(float(getattr(opp, 'net_profit', 0.0)))
    ev = max(0.0, float(opp.net_profit) - bribe) * (1.0 - risk)
    try:
        opp.sandwich_risk = float(risk)
        opp.mev_bribe_est = float(bribe)
        opp.priority_fee_suggested = float(_priority_fee_gwei())
        opp.expected_value = float(ev)
    except Exception:
        pass
    return opp

