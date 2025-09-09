#!/usr/bin/env python3
import os
import asyncio
import argparse
from web3 import Web3
from src.price_feed import PriceFeed
from src.token_manager import TokenManager
from config.config import config


async def main():
    ap = argparse.ArgumentParser(description='Historical price backfilling (Uniswap V2 spot-based)')
    ap.add_argument('--from-block', type=int, required=True)
    ap.add_argument('--to-block', type=int, required=False)
    ap.add_argument('--step', type=int, default=100)
    ap.add_argument('--include-synths', action='store_true')
    args = ap.parse_args()

    rpc = config.ethereum_mainnet_rpc
    if not rpc:
        raise SystemExit('ETHEREUM_MAINNET_RPC not configured')
    w3 = Web3(Web3.HTTPProvider(rpc))
    pf = PriceFeed(w3)

    # Build token set
    tm = TokenManager()
    tokens = {}
    # base set
    for sym in ('WETH','USDC','DAI','USDT'):
        a = tm.get_address_by_symbol(sym)
        if a:
            tokens[sym] = a
    # extras to reach ~25 assets if configured
    for sym in ('WBTC','LINK','UNI','SUSHI','COMP','AAVE','CRV','BAL','YFI','MKR','LDO','1INCH','GRT','MATIC','ZRX','LRC','REN','TUSD','PAXG'):
        a = tm.get_address_by_symbol(sym)
        if a:
            tokens.setdefault(sym, a)
    if args.include_synths:
        for sym in ('sUSD','sETH'):
            a = tm.get_address_by_symbol(sym)
            if a:
                tokens.setdefault(sym, a)

    # iterate blocks
    to_block = args.to_block or w3.eth.block_number
    updated_total = 0
    for blk in range(args.from_block, to_block + 1, max(1, args.step)):
        u = await pf.update_prices_for_at_block(tokens, blk)
        updated_total += u
        print(f"block {blk}: updated {u} prices")
    print(f"done. total updates: {updated_total}")


if __name__ == '__main__':
    asyncio.run(main())

