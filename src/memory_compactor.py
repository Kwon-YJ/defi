from typing import Dict, Set
import networkx as nx
from src.logger import setup_logger
from config.config import config

logger = setup_logger(__name__)


def compact_graph_attributes(nx_graph: nx.Graph) -> Dict[str, int]:
    """Reduce memory footprint by removing redundant/heavy attributes.

    - Remove 'weight' from edges (computed on-the-fly as -log(exchange_rate)).
    Returns stats with removed counts.
    """
    removed_weight = 0
    compacted_meta = 0
    keep_keys: Set[str] = set()
    try:
        if getattr(config, 'memory_compact_meta', False):
            raw = getattr(config, 'memory_meta_keep_keys', '') or ''
            keep_keys = {k.strip() for k in raw.split(',') if k.strip()}
    except Exception:
        keep_keys = set()

    def _maybe_compact(data: Dict) -> None:
        nonlocal compacted_meta
        if 'weight' in data:
            try:
                del data['weight']
            except Exception:
                pass
        # compact meta dict by whitelist
        if keep_keys and isinstance(data.get('meta'), dict):
            m = data.get('meta')
            try:
                to_del = [k for k in m.keys() if k not in keep_keys]
                for k in to_del:
                    del m[k]
                compacted_meta += 1
            except Exception:
                pass

    if isinstance(nx_graph, nx.MultiDiGraph):
        for _, _, _, data in nx_graph.edges(keys=True, data=True):
            if 'weight' in data:
                try:
                    del data['weight']
                except Exception:
                    pass
                removed_weight += 1
            _maybe_compact(data)
    else:
        for _, _, data in nx_graph.edges(data=True):
            if 'weight' in data:
                try:
                    del data['weight']
                except Exception:
                    pass
                removed_weight += 1
            _maybe_compact(data)

    if removed_weight:
        logger.debug(f"Graph compacted: removed weight on {removed_weight} edges")

    return {
        'removed_weight': removed_weight,
        'compacted_meta': compacted_meta,
        'edges': nx_graph.number_of_edges(),
    }
