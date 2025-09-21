#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Dict
import os
import sys

from web3 import Web3

"""Ensure project root on sys.path when running as a script."""
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import config
from src.block_graph_updater import BlockGraphUpdater
from src.strategies.lend_borrow_swap import LendBorrowSwapStrategy, format_plan


def load_default_tokens() -> Dict[str, str]:
    # BlockGraphUpdater와 동일한 기본 토큰 셋 구성
    return {
        'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
        'USDC': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
        'DAI':  '0x6B175474E89094C44Da98b954EedeAC495271d0F',
        'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
    }


def main():
    ap = argparse.ArgumentParser(description='Lending/Borrowing + Swap 조합 전략 산출')
    ap.add_argument('--collateral', default='WETH', help='담보 토큰 심볼 (기본: WETH)')
    ap.add_argument('--borrow', default='USDC', help='차입 토큰 심볼 (기본: USDC)')
    ap.add_argument('--amount', type=float, default=1.0, help='예치량 (담보 토큰 단위, 기본: 1.0)')
    ap.add_argument('--safety', type=float, default=0.9, help='안전 마진 (0~1, 기본: 0.9)')
    args = ap.parse_args()

    try:
        w3 = Web3(Web3.HTTPProvider(config.ethereum_mainnet_rpc, request_kwargs={'timeout': 8}))
    except Exception:
        w3 = Web3(Web3.HTTPProvider(config.ethereum_mainnet_rpc))
    tokens = load_default_tokens()
    strat = LendBorrowSwapStrategy(w3, tokens)
    plan = strat.build_plan(args.collateral, args.borrow, args.amount, safety_factor=args.safety)
    print(format_plan(plan))


if __name__ == '__main__':
    main()
