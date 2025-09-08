from typing import Dict
import networkx as nx
from src.logger import setup_logger

logger = setup_logger(__name__)


def compact_graph_attributes(nx_graph: nx.Graph) -> Dict[str, int]:
    """Reduce memory footprint by removing redundant/heavy attributes.

    - Remove 'weight' from edges (computed on-the-fly as -log(exchange_rate)).
    Returns stats with removed counts.
    """
    removed_weight = 0

    if isinstance(nx_graph, nx.MultiDiGraph):
        for _, _, _, data in nx_graph.edges(keys=True, data=True):
            if 'weight' in data:
                del data['weight']
                removed_weight += 1
    else:
        for _, _, data in nx_graph.edges(data=True):
            if 'weight' in data:
                del data['weight']
                removed_weight += 1

    if removed_weight:
        logger.debug(f"Graph compacted: removed weight on {removed_weight} edges")

    return {
        'removed_weight': removed_weight,
        'edges': nx_graph.number_of_edges(),
    }

