from typing import Tuple, Dict, Any
import networkx as nx
from src.logger import setup_logger

logger = setup_logger(__name__)


def prune_graph(nx_graph: nx.Graph, min_liquidity: float = 0.1, keep_top_k: int = 2) -> Dict[str, int]:
    """Prune inefficient edges from a (Multi)DiGraph.

    - Remove edges with liquidity below `min_liquidity`.
    - For each (u,v) pair, keep only `keep_top_k` edges with highest exchange_rate.

    Returns a dict with pruning stats.
    """
    removed_low_liquidity = 0
    removed_dominated = 0

    if isinstance(nx_graph, nx.MultiDiGraph):
        # First pass: remove low-liquidity edges
        to_remove = []
        for u, v, key, data in nx_graph.edges(keys=True, data=True):
            if data.get('liquidity', float('inf')) < min_liquidity:
                to_remove.append((u, v, key))
        for u, v, key in to_remove:
            nx_graph.remove_edge(u, v, key)
        removed_low_liquidity += len(to_remove)

        # Second pass: keep top-K by exchange_rate per (u,v)
        # Build map (u,v) -> list[(key, data)]
        uv_map: Dict[Tuple[str, str], list] = {}
        for u, v, key, data in nx_graph.edges(keys=True, data=True):
            uv_map.setdefault((u, v), []).append((key, data))

        for (u, v), items in uv_map.items():
            if len(items) <= keep_top_k:
                continue
            items_sorted = sorted(items, key=lambda kv: kv[1].get('exchange_rate', 0.0), reverse=True)
            # keep first K
            for key, _ in items_sorted[keep_top_k:]:
                try:
                    nx_graph.remove_edge(u, v, key)
                    removed_dominated += 1
                except Exception:
                    continue

    else:
        # DiGraph: only low-liquidity pruning applies
        to_remove = []
        for u, v, data in nx_graph.edges(data=True):
            if data.get('liquidity', float('inf')) < min_liquidity:
                to_remove.append((u, v))
        for u, v in to_remove:
            nx_graph.remove_edge(u, v)
        removed_low_liquidity += len(to_remove)

    stats = {
        'removed_low_liquidity': removed_low_liquidity,
        'removed_dominated': removed_dominated,
        'remaining_edges': nx_graph.number_of_edges(),
    }
    if removed_low_liquidity or removed_dominated:
        logger.info(
            f"Graph pruned: low_liq={removed_low_liquidity}, dominated={removed_dominated}, "
            f"remaining={stats['remaining_edges']}"
        )
    return stats

