import pytest

from src.market_graph import DeFiMarketGraph
from src.bellman_ford_arbitrage import BellmanFordArbitrage


def build_profitable_triangle() -> DeFiMarketGraph:
    g = DeFiMarketGraph()
    # Tokens
    g.add_token("WETH", "WETH")
    g.add_token("USDC", "USDC")
    g.add_token("DAI", "DAI")

    # Set reserves such that WETH->USDC->DAI->WETH yields > 1 product
    # Fees default to 0.3%
    # WETH/USDC (UniV2)
    g.add_trading_pair("WETH", "USDC", "uniswap_v2", "0xpool_uni", 100.0, 210000.0)
    # USDC/DAI (Sushi) ~ 1:1
    g.add_trading_pair("USDC", "DAI", "sushiswap", "0xpool_sushi", 200000.0, 200000.0)
    # DAI/WETH (Curve-like, but we will label as curve)
    g.add_trading_pair("DAI", "WETH", "curve", "0xpool_curve", 200000.0, 100.0)

    # Reduce gas cost impact for deterministic unit test
    for (u, v) in ("WETH", "USDC"), ("USDC", "DAI"), ("DAI", "WETH"):
        if g.graph.has_edge(u, v):
            try:
                g.graph[u][v]['gas_cost'] = 0.0
            except Exception:
                # MultiDiGraph case (not used in current tests)
                for k in list(g.graph[u][v].keys()):
                    g.graph[u][v][k]['gas_cost'] = 0.0
    return g


def test_find_multihop_three_protocols():
    graph = build_profitable_triangle()
    bf = BellmanFordArbitrage(graph)

    opps = bf.find_multihop_opportunities(
        source_token="WETH",
        min_protocols=3,
        max_hops=4,
        top_k=5,
    )

    assert isinstance(opps, list)
    assert len(opps) >= 1
    best = opps[0]
    # Path should start and end at WETH and have at least 3 edges
    assert best.path[0] == "WETH"
    assert best.path[-1] == "WETH"
    assert len(best.edges) >= 3
    # Ensure at least 3 distinct protocols are used in the route
    assert len(set(e.dex for e in best.edges)) >= 3

