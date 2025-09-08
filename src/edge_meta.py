from typing import Optional, Dict, Any
import networkx as nx
from src.token_risk import is_fee_on_transfer, is_rebase


def set_edge_meta(nx_graph: nx.Graph,
                  token0: str,
                  token1: str,
                  *,
                  dex: Optional[str] = None,
                  pool_address: Optional[str] = None,
                  fee_tier: Optional[int] = None,
                  source: str = "unknown",
                  confidence: float = 0.8,
                  extra: Optional[Dict[str, Any]] = None) -> None:
    """Attach standardized metadata to edges between token0 and token1 (both directions).

    Metadata fields:
      - contract: original contract/pool address (if any)
      - fee_tier: e.g., 500/3000/10000 for Uniswap V3 (None otherwise)
      - source: 'onchain' | 'approx' | 'fallback' | 'event' | 'unknown'
      - confidence: 0.0 ~ 1.0
    """
    meta: Dict[str, Any] = {
        'contract': pool_address,
        'fee_tier': fee_tier,
        'source': source,
        'confidence': confidence,
    }
    if extra:
        try:
            meta.update(extra)
        except Exception:
            pass

    # Risk flags (fee-on-transfer / rebase)
    try:
        t0 = extra.get('t0') if extra else None
        t1 = extra.get('t1') if extra else None
        fot = any(is_fee_on_transfer(t) for t in (t0, t1) if t)
        rbs = any(is_rebase(t) for t in (t0, t1) if t)
        meta['risk_fot'] = fot
        meta['risk_rebase'] = rbs
        # Lower confidence slightly for risky tokens
        if fot:
            meta['confidence'] = max(0.0, float(meta.get('confidence', 0.8)) - 0.1)
        if rbs:
            meta['confidence'] = max(0.0, float(meta.get('confidence', 0.8)) - 0.05)
    except Exception:
        pass

    def _apply(u: str, v: str):
        try:
            if not nx_graph.has_edge(u, v):
                return
            ed = nx_graph.get_edge_data(u, v)
            # DiGraph
            if isinstance(ed, dict) and 'exchange_rate' in ed:
                # Optional filtering by dex/pool
                if dex and ed.get('dex') and ed.get('dex') != dex:
                    return
                if pool_address and ed.get('pool_address') and ed.get('pool_address') != pool_address:
                    return
                ed['meta'] = meta
                return
            # MultiDiGraph: ed is key->data
            if isinstance(ed, dict):
                for _, data in ed.items():
                    if not isinstance(data, dict):
                        continue
                    if dex and data.get('dex') and data.get('dex') != dex:
                        continue
                    if pool_address and data.get('pool_address') and data.get('pool_address') != pool_address:
                        continue
                    data['meta'] = meta
        except Exception:
            return

    _apply(token0, token1)
    _apply(token1, token0)
