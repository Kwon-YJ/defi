#!/usr/bin/env python3
import os
import time
from itertools import combinations

def main():
    # 활성화 플래그를 먼저 설정한 후 모듈을 임포트
    os.environ['USE_PAPER_25_ASSETS'] = '1'

    from src.market_graph import DeFiMarketGraph
    from src.block_graph_updater import BlockGraphUpdater
    from src.graph_pruner import prune_graph
    from src.memory_compactor import compact_graph_attributes

    g = DeFiMarketGraph()
    updater = BlockGraphUpdater(g)

    tokens = updater.tokens
    n = len(tokens)
    pairs = list(combinations(tokens.values(), 2))
    print(f"paper-25 assets loaded: {n}")
    print(f"pair combinations (nC2): {len(pairs)}")
    # 간단 벤치마크: 액션 기반 업데이트 1회 (네트워크 의존 → 실패 시 스킵)
    t0 = time.time()
    try:
        # update_via_actions는 내부적으로 네트워크 호출 가능하므로 예외를 잡아 통계만 출력
        import asyncio
        asyncio.run(updater.update_via_actions())
    except Exception as e:
        print(f"update_via_actions skipped (reason: {e})")
    # 프루닝/컴팩션 파라미터 (튜닝 기본값)
    prune_graph(g.graph, min_liquidity=0.1, keep_top_k=2)
    compact_graph_attributes(g.graph)
    dt = time.time() - t0
    try:
        edges = g.graph.number_of_edges()
        nodes = g.graph.number_of_nodes()
    except Exception:
        edges = nodes = 0
    print(f"benchmark: nodes={nodes} edges={edges} elapsed={dt:.3f}s")
    print("tuning: prune keep_top_k=2, min_liquidity=0.1, MultiDiGraph enabled")

if __name__ == '__main__':
    main()

