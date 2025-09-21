#!/usr/bin/env python3
import os
import sys
import time
from itertools import combinations

def main():
    # 활성화 플래그를 먼저 설정한 후 모듈을 임포트
    os.environ['USE_PAPER_25_ASSETS'] = '1'
    # 벤치마크에서는 논문 25자산만 사용하도록 추가 토큰 포함 비활성화
    os.environ['INCLUDE_MAJOR_TOKENS'] = '0'
    os.environ['INCLUDE_DEFI_TOKENS'] = '0'
    os.environ['INCLUDE_SYNTH_TOKENS'] = '0'
    os.environ['INCLUDE_EXTRA_TOKENS'] = '0'
    # 과도한 RPC 호출 방지: 동시성 축소
    os.environ['GRAPH_BUILD_CONCURRENCY'] = '8'

    # Ensure project root is on sys.path so `src` package is importable
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.market_graph import DeFiMarketGraph
    from src.block_graph_updater import BlockGraphUpdater
    from src.graph_pruner import prune_graph
    from src.memory_compactor import compact_graph_attributes

    g = DeFiMarketGraph()
    updater = BlockGraphUpdater(g)
    # 벤치마크용: 핵심 액션만 유지 (V2/Sushi/Wrap)
    try:
        allowed = {'uniswap_v2.swap', 'sushiswap.swap', 'weth.wrap'}
        reg = updater.registry
        reg.actions = {k: v for k, v in reg.actions.items() if k in allowed}
    except Exception:
        pass

    tokens = updater.tokens
    n = len(tokens)
    pairs = list(combinations(tokens.values(), 2))
    print(f"paper-25 assets loaded: {n}")
    print(f"pair combinations (nC2): {len(pairs)}")
    # 간단 벤치마크: 액션 기반 업데이트 1회 (네트워크 의존 → 실패 시 스킵)
    t0 = time.time()
    try:
        # RPC 연결 불가 시 네트워크 의존 단계는 스킵
        if hasattr(updater, 'w3') and getattr(updater.w3, 'is_connected', lambda: False)():
            import asyncio
            asyncio.run(updater.update_via_actions())
        else:
            print("update_via_actions skipped (reason: no RPC connection)")
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
