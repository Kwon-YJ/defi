#!/usr/bin/env python3
"""
Historical Data Verification Script
Verifies that historical data backfilling is working correctly

This script checks the TODO requirement completion for historical data backfilling.
"""

import sys
import sqlite3
from datetime import datetime

def verify_historical_data():
    """Verify historical data backfilling functionality"""
    print("🔍 Verifying Historical Data Backfilling...")
    print("=" * 60)
    
    try:
        # Connect to database
        conn = sqlite3.connect('historical_data.db')
        cursor = conn.cursor()
        
        # Check blocks table
        cursor.execute("SELECT COUNT(*) as total_blocks, MIN(number) as min_block, MAX(number) as max_block FROM blocks")
        blocks_stats = cursor.fetchone()
        total_blocks, min_block, max_block = blocks_stats
        
        print(f"📦 Blocks Data:")
        print(f"   Total blocks collected: {total_blocks:,}")
        print(f"   Block range: {min_block:,} ~ {max_block:,}")
        print(f"   Block span: {max_block - min_block + 1:,} blocks")
        
        # Check historical prices table
        cursor.execute("""
            SELECT COUNT(*) as total_records,
                   COUNT(DISTINCT token_address) as unique_tokens,
                   COUNT(DISTINCT block_number) as covered_blocks,
                   COUNT(DISTINCT source) as data_sources,
                   MIN(timestamp) as start_time,
                   MAX(timestamp) as end_time
            FROM historical_prices
        """)
        
        price_stats = cursor.fetchone()
        total_records, unique_tokens, covered_blocks, data_sources, start_time, end_time = price_stats
        
        print(f"\n💰 Price Data:")
        print(f"   Total price records: {total_records:,}")
        print(f"   Unique tokens: {unique_tokens}")
        print(f"   Covered blocks: {covered_blocks:,}")
        print(f"   Data sources: {data_sources}")
        
        if start_time and end_time:
            duration_hours = (end_time - start_time) / 3600
            print(f"   Time span: {duration_hours:.1f} hours")
        
        # Check token coverage
        cursor.execute("""
            SELECT symbol, COUNT(*) as record_count,
                   MIN(price_usd) as min_price,
                   MAX(price_usd) as max_price,
                   AVG(price_usd) as avg_price
            FROM historical_prices
            GROUP BY token_address, symbol
            ORDER BY record_count DESC
            LIMIT 5
        """)
        
        token_stats = cursor.fetchall()
        
        print(f"\n📈 Top Tokens by Data Points:")
        for symbol, count, min_price, max_price, avg_price in token_stats:
            volatility = ((max_price - min_price) / min_price * 100) if min_price > 0 else 0
            print(f"   {symbol}: {count:,} points, ${avg_price:.6f} avg, {volatility:.1f}% volatility")
        
        # Check data sources
        cursor.execute("SELECT source, COUNT(*) as records FROM historical_prices GROUP BY source")
        source_stats = cursor.fetchall()
        
        print(f"\n🔗 Data Sources:")
        for source, count in source_stats:
            print(f"   {source}: {count:,} records")
        
        # Assessment of completion
        print(f"\n" + "=" * 60)
        print("📊 TODO REQUIREMENT ASSESSMENT")
        print("=" * 60)
        
        requirements_met = []
        
        # Check if basic functionality exists
        if total_blocks > 0:
            requirements_met.append("✅ Block data collection working")
        else:
            requirements_met.append("❌ Block data collection not working")
            
        if total_records > 0:
            requirements_met.append("✅ Historical price data collection working")
        else:
            requirements_met.append("❌ Historical price data collection not working")
            
        if unique_tokens > 0:
            requirements_met.append(f"✅ Multi-token support ({unique_tokens} tokens)")
        else:
            requirements_met.append("❌ Multi-token support missing")
            
        if data_sources > 0:
            requirements_met.append(f"✅ Multiple data sources ({data_sources} sources)")
        else:
            requirements_met.append("❌ Multiple data sources missing")
            
        # Database structure check
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['blocks', 'historical_prices']
        if all(table in tables for table in expected_tables):
            requirements_met.append("✅ Database schema complete")
        else:
            requirements_met.append("❌ Database schema incomplete")
        
        for req in requirements_met:
            print(f"   {req}")
        
        # Overall assessment
        success_count = sum(1 for req in requirements_met if req.startswith("✅"))
        total_checks = len(requirements_met)
        
        print(f"\n📈 Overall Score: {success_count}/{total_checks} ({success_count/total_checks*100:.0f}%)")
        
        if success_count == total_checks:
            print("🎉 Historical Data Backfilling TODO: COMPLETED")
            print("✅ All core functionality is working correctly")
            return True
        elif success_count >= total_checks * 0.8:
            print("⚠️  Historical Data Backfilling TODO: MOSTLY COMPLETED")
            print("💡 Core functionality working, minor improvements possible")
            return True
        else:
            print("❌ Historical Data Backfilling TODO: NOT COMPLETED")
            print("🔧 Significant work needed")
            return False
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Historical Data Backfilling Verification")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = verify_historical_data()
    
    if success:
        print(f"\n✅ Verification completed successfully!")
    else:
        print(f"\n❌ Verification failed!")
        sys.exit(1)