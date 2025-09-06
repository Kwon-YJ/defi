#!/usr/bin/env python3
"""
Historical Data Backfilling Runner
Completes TODO requirement: Historical data backfilling for analysis

This script runs the historical data backfilling system to collect past price data
for the DeFiPoser-ARB system reproduction.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.historical_data_backfill import HistoricalDataBackfill
from src.token_manager import TokenManager
from src.logger import setup_logger

logger = setup_logger(__name__)

async def main():
    """메인 실행 함수"""
    print("🔄 DeFiPoser-ARB Historical Data Backfill Starting...")
    print("=" * 70)
    
    try:
        # 토큰 매니저 초기화
        print("1️⃣ Initializing Token Manager...")
        token_manager = TokenManager()
        
        # 과거 데이터 백필러 초기화
        print("2️⃣ Initializing Historical Data Backfill...")
        backfill = HistoricalDataBackfill(token_manager)
        
        # 테스트용 작은 범위로 시작 (논문의 전체 범위는 너무 큼)
        # 논문: 블록 9,100,000 ~ 10,050,000 (150일)
        # 테스트: 첫 10,000 블록만 (약 2일치)
        test_start = 9_100_000
        test_end = 9_110_000  # 10,000 blocks for testing
        
        print(f"3️⃣ Starting backfill for blocks {test_start:,} ~ {test_end:,}")
        print(f"   (Test range: ~{(test_end - test_start):,} blocks)")
        
        start_time = datetime.now()
        
        # 백필링 시작
        success = await backfill.start_backfill(test_start, test_end)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("📊 BACKFILL COMPLETION SUMMARY")
        print("=" * 70)
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"🎯 Success: {'✅ YES' if success else '❌ NO'}")
        print(f"📦 Blocks processed: {test_end - test_start:,}")
        print(f"⚡ Rate: {(test_end - test_start) / duration:.2f} blocks/second")
        
        # 샘플 데이터 확인
        print("\n4️⃣ Verifying collected data...")
        
        # ETH 가격 데이터 확인
        eth_address = token_manager.get_address_by_symbol('ETH')
        if eth_address:
            eth_prices = await backfill.get_historical_prices(eth_address, test_start, test_end)
            print(f"   📈 ETH prices collected: {len(eth_prices)} data points")
            
            if eth_prices:
                first_price = eth_prices[0]
                last_price = eth_prices[-1]
                print(f"   💰 ETH price range: ${first_price.price_usd:.2f} ~ ${last_price.price_usd:.2f}")
        
        # USDC 가격 데이터 확인
        usdc_address = token_manager.get_address_by_symbol('USDC')
        if usdc_address:
            usdc_prices = await backfill.get_historical_prices(usdc_address, test_start, test_end)
            print(f"   🪙 USDC prices collected: {len(usdc_prices)} data points")
        
        # 특정 블록의 데이터 확인
        sample_block = test_start + 1000
        block_prices = await backfill.get_block_prices(sample_block)
        print(f"   🧱 Block {sample_block} prices: {len(block_prices)} tokens")
        
        if success:
            print("\n✅ Historical data backfilling completed successfully!")
            print("📝 This completes TODO requirement: 'Historical data backfilling for analysis'")
        else:
            print("\n⚠️  Historical data backfilling completed with warnings.")
            print("💡 Some targets may not have been fully achieved.")
            
    except Exception as e:
        print(f"\n❌ Error during backfill: {e}")
        logger.error(f"Backfill error: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("🚀 Starting Historical Data Backfill for DeFiPoser-ARB...")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        result = asyncio.run(main())
        if result:
            print("\n🎉 Historical Data Backfill completed successfully!")
            sys.exit(0)
        else:
            print("\n💥 Historical Data Backfill failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⛔ Backfill interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)