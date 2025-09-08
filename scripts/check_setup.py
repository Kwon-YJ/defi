#!/usr/bin/env python3
"""
환경 설정 검증 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import asyncio
from web3 import Web3
from src.token_manager import TokenManager
from src.dex_data_collector import UniswapV2Collector
from config.config import config

async def check_rpc_connection():
    """RPC 연결 확인"""
    try:
        w3 = Web3(Web3.HTTPProvider(config.ethereum_mainnet_rpc))
        latest_block = w3.eth.get_block('latest')
        print(f"✅ RPC 연결 성공: 블록 #{latest_block.number}")
        return True
    except Exception as e:
        print(f"❌ RPC 연결 실패: {e}")
        return False

async def check_token_manager():
    """토큰 매니저 확인"""
    try:
        token_manager = TokenManager()
        weth_info = await token_manager.get_token_info(
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        )
        if weth_info:
            print(f"✅ 토큰 매니저 동작: {weth_info.symbol}")
            return True
        else:
            print("❌ 토큰 정보 조회 실패")
            return False
    except Exception as e:
        print(f"❌ 토큰 매니저 오류: {e}")
        return False

async def check_dex_collector():
    """DEX 수집기 확인"""
    try:
        w3 = Web3(Web3.HTTPProvider(config.ethereum_mainnet_rpc))
        collector = UniswapV2Collector(w3)
        
        # WETH-USDC 풀 조회
        weth = w3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
        usdc = w3.to_checksum_address("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48")
        
        pair_address = await collector.get_pair_address(weth, usdc)
        if pair_address:
            print(f"✅ DEX 수집기 동작: {pair_address}")
            return True
        else:
            print("❌ 풀 주소 조회 실패")
            return False
    except Exception as e:
        print(f"❌ DEX 수집기 오류: {e}")
        return False

async def main():
    """메인 검증 함수"""
    print("🔍 DeFi Arbitrage Validator 환경 설정 검증\n")
    
    checks = [
        ("RPC 연결", check_rpc_connection),
        ("토큰 매니저", check_token_manager),
        ("DEX 수집기", check_dex_collector)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"검증 중: {name}")
        result = await check_func()
        results.append(result)
        print()
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"📊 검증 결과: {success_count}/{total_count} 성공")
    
    if success_count == total_count:
        print("🎉 모든 검증 통과! 시스템 사용 준비 완료")
    else:
        print("⚠️  일부 검증 실패. 설정을 확인하세요.")

if __name__ == "__main__":
    asyncio.run(main())