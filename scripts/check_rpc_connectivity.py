#!/usr/bin/env python3
import os
from web3 import Web3


def main():
    rpc = os.getenv('ETHEREUM_MAINNET_RPC') or os.getenv('ALCHEMY_HTTP')
    if not rpc:
        print("ETHEREUM_MAINNET_RPC not set")
        return 1
    try:
        w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={'timeout': 8}))
    except Exception:
        w3 = Web3(Web3.HTTPProvider(rpc))
    if not w3.is_connected():
        print("RPC not reachable")
        return 2
    bn = w3.eth.block_number
    print(f"connected: block={bn}")
    # Quick Uniswap V2 getPair(WETH, USDC)
    try:
        factory = w3.eth.contract(
            address=Web3.to_checksum_address("0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"),
            abi=[{"constant": True, "inputs": [
                {"name": "tokenA", "type": "address"},
                {"name": "tokenB", "type": "address"}
            ], "name": "getPair", "outputs": [{"name": "pair", "type": "address"}], "type": "function"}],
        )
        weth = Web3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
        usdc = Web3.to_checksum_address("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48")
        pair = factory.functions.getPair(weth, usdc).call()
        print(f"uniswap_v2 WETH/USDC pair={pair}")
    except Exception as e:
        print(f"getPair failed: {e}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
