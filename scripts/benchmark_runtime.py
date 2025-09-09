#!/usr/bin/env python3
import os
import time
import argparse
import asyncio
from statistics import mean

from src.market_graph import DeFiMarketGraph
from src.block_graph_updater import BlockGraphUpdater
from src.graph_pruner import prune_graph
from src.memory_compactor import compact_graph_attributes


async def run_once(updater: BlockGraphUpdater) -> float:
    t0 = time.perf_counter()
    await updater.update_via_actions()
    prune_graph(updater.graph.graph, min_liquidity=0.1, keep_top_k=2)
    compact_graph_attributes(updater.graph.graph)
    return time.perf_counter() - t0


async def main():
    ap = argparse.ArgumentParser(description='Measure runtime per-block build/update')
    ap.add_argument('-n', '--runs', type=int, default=5, help='number of runs')
    ap.add_argument('--paper25', action='store_true', help='use paper 25 assets')
    ap.add_argument('--include-majors', action='store_true')
    ap.add_argument('--include-defi', action='store_true')
    ap.add_argument('--include-extra', action='store_true')
    args = ap.parse_args()

    if args.paper25:
        os.environ['USE_PAPER_25_ASSETS'] = '1'
    if args.include_majors:
        os.environ['INCLUDE_MAJOR_TOKENS'] = '1'
    if args.include_defi:
        os.environ['INCLUDE_DEFI_TOKENS'] = '1'
    if args.include_extra:
        os.environ['INCLUDE_EXTRA_TOKENS'] = '1'

    g = DeFiMarketGraph()
    updater = BlockGraphUpdater(g)

    # warmup
    dt0 = await run_once(updater)
    print(f"warmup: {dt0:.3f}s, nodes={g.graph.number_of_nodes()} edges={g.graph.number_of_edges()}")

    times = []
    for i in range(max(1, args.runs)):
        dt = await run_once(updater)
        times.append(dt)
        print(f"run {i+1}/{args.runs}: {dt:.3f}s")

    print(f"avg: {mean(times):.3f}s, min: {min(times):.3f}s, max: {max(times):.3f}s")


if __name__ == '__main__':
    asyncio.run(main())

