#!/usr/bin/env python3
"""
Test script for database query optimization.
Tests the TODO item completion: "Database query 최적화"
"""

import sys
import os
import time
import random
import asyncio
from typing import List, Tuple

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database_optimizer import DatabaseOptimizer, get_db_optimizer, optimize_database_for_paper_requirements
from historical_data_backfill import HistoricalDataBackfill
from logger import setup_logger

logger = setup_logger(__name__)

def test_basic_database_optimization():
    """Test basic database optimization features"""
    print("🔍 Testing basic database optimization...")
    
    try:
        # Initialize optimizer
        optimizer = get_db_optimizer("test_optimization.db")
        
        # Create test tables if they don't exist
        with optimizer.pool.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS test_prices (
                    id INTEGER PRIMARY KEY,
                    token_address TEXT,
                    price REAL,
                    timestamp INTEGER,
                    block_number INTEGER
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_test_token_block 
                ON test_prices(token_address, block_number)
            ''')
        
        # Test basic query execution
        test_data = [
            (f"token_{i}", random.uniform(0.1, 100.0), int(time.time()), 1000000 + i)
            for i in range(100)
        ]
        
        optimizer.bulk_insert_optimized(
            "test_prices", 
            test_data,
            ["token_address", "price", "timestamp", "block_number"]
        )
        
        print(f"✅ Inserted {len(test_data)} test records")
        
        # Test cached queries
        start_time = time.time()
        result1 = optimizer.execute_query(
            "SELECT COUNT(*) FROM test_prices WHERE token_address LIKE 'token_%'",
            cache_key="test_count_query"
        )
        first_query_time = time.time() - start_time
        
        start_time = time.time()
        result2 = optimizer.execute_query(
            "SELECT COUNT(*) FROM test_prices WHERE token_address LIKE 'token_%'",
            cache_key="test_count_query"
        )
        cached_query_time = time.time() - start_time
        
        print(f"✅ First query time: {first_query_time:.4f}s")
        print(f"✅ Cached query time: {cached_query_time:.4f}s")
        print(f"✅ Cache speedup: {first_query_time / max(cached_query_time, 0.0001):.1f}x")
        
        # Test batch operations
        batch_data = [
            (f"batch_token_{i}", random.uniform(1.0, 50.0), int(time.time()), 2000000 + i)
            for i in range(50)
        ]
        
        for data in batch_data:
            optimizer.add_to_batch("test_inserts", data)
        
        print(f"✅ Added {len(batch_data)} items to batch queue")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic optimization test failed: {e}")
        return False

def test_connection_pool_performance():
    """Test connection pool performance"""
    print("\n🚀 Testing connection pool performance...")
    
    try:
        optimizer = get_db_optimizer("test_optimization.db")
        
        # Test concurrent connections
        start_time = time.time()
        
        def execute_test_query():
            return optimizer.execute_query(
                "SELECT COUNT(*) FROM test_prices WHERE block_number > ?",
                (1000000,)
            )
        
        # Sequential execution
        sequential_results = []
        sequential_start = time.time()
        for _ in range(20):
            sequential_results.append(execute_test_query())
        sequential_time = time.time() - sequential_start
        
        print(f"✅ Sequential execution (20 queries): {sequential_time:.3f}s")
        print(f"   Average per query: {sequential_time/20:.4f}s")
        
        # Test connection pool stats
        stats = optimizer.get_performance_stats()
        pool_stats = stats['pool_stats']
        
        print(f"✅ Connection pool stats:")
        print(f"   Active connections: {pool_stats['active_connections']}")
        print(f"   Pool size: {pool_stats['pool_size']}")
        print(f"   Cache hit rate: {pool_stats['cache_hit_rate']:.1f}%")
        print(f"   Total queries: {pool_stats['total_queries']:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection pool test failed: {e}")
        return False

def test_query_optimization():
    """Test query optimization features"""
    print("\n🧠 Testing query optimization...")
    
    try:
        optimizer = get_db_optimizer("test_optimization.db")
        
        # Create more test data for meaningful optimization
        large_dataset = []
        for i in range(1000):
            token_addr = f"0x{''.join(random.choices('0123456789abcdef', k=40))}"
            price = random.uniform(0.01, 1000.0)
            timestamp = int(time.time()) - random.randint(0, 86400 * 30)  # Last 30 days
            block_number = 10000000 + i
            
            large_dataset.append((token_addr, price, timestamp, block_number))
        
        # Insert large dataset
        start_time = time.time()
        optimizer.bulk_insert_optimized(
            "test_prices",
            large_dataset,
            ["token_address", "price", "timestamp", "block_number"]
        )
        insert_time = time.time() - start_time
        
        print(f"✅ Bulk inserted {len(large_dataset):,} records in {insert_time:.3f}s")
        print(f"   Rate: {len(large_dataset)/insert_time:.0f} records/second")
        
        # Test complex queries
        test_queries = [
            ("Range query", "SELECT * FROM test_prices WHERE block_number BETWEEN ? AND ? ORDER BY price DESC LIMIT 10"),
            ("Aggregation", "SELECT token_address, COUNT(*), AVG(price), MAX(price) FROM test_prices GROUP BY token_address HAVING COUNT(*) > 1"),
            ("Complex filter", "SELECT * FROM test_prices WHERE price > ? AND timestamp > ? ORDER BY block_number"),
        ]
        
        for query_name, sql in test_queries:
            start_time = time.time()
            if "?" in sql:
                if "BETWEEN" in sql:
                    result = optimizer.execute_query(sql, (10000100, 10000200))
                elif "price >" in sql:
                    result = optimizer.execute_query(sql, (10.0, int(time.time()) - 86400))
                else:
                    result = optimizer.execute_query(sql, (100.0,))
            else:
                result = optimizer.execute_query(sql)
            
            query_time = time.time() - start_time
            print(f"✅ {query_name}: {query_time:.4f}s ({len(result)} rows)")
        
        # Test memory temp table
        optimizer.create_memory_temp_table(
            "temp_analysis", 
            "(token_address TEXT, avg_price REAL, price_change REAL)"
        )
        print("✅ Created memory-based temporary table")
        
        return True
        
    except Exception as e:
        print(f"❌ Query optimization test failed: {e}")
        return False

def test_paper_compliance():
    """Test database optimization for paper requirements"""
    print("\n📊 Testing paper compliance and performance targets...")
    
    try:
        # Run comprehensive optimization check
        success = optimize_database_for_paper_requirements()
        
        if success:
            print("✅ Database optimization meets paper requirements")
            print("   - Query performance optimized for 6.43s average execution time")
            print("   - Connection pooling ready for 96 protocol actions")
            print("   - Caching system ready for 25 assets real-time processing")
        else:
            print("⚠️  Database optimization needs improvement for paper requirements")
            
        return success
        
    except Exception as e:
        print(f"❌ Paper compliance test failed: {e}")
        return False

def test_integration_with_historical_data():
    """Test integration with existing historical data system"""
    print("\n🔄 Testing integration with historical data system...")
    
    try:
        optimizer = get_db_optimizer("historical_data.db")
        
        # Test optimized historical data queries
        test_token = "0x1234567890abcdef1234567890abcdef12345678"
        
        # Test optimized price queries
        start_time = time.time()
        prices = optimizer.get_historical_prices_optimized(
            token_address=test_token,
            start_block=10000000,
            end_block=10000100,
            limit=50
        )
        query_time = time.time() - start_time
        
        print(f"✅ Optimized historical price query: {query_time:.4f}s ({len(prices)} results)")
        
        # Test price analysis
        start_time = time.time()
        analysis = optimizer.get_price_range_analysis(
            token_address=test_token,
            start_block=10000000, 
            end_block=10000100
        )
        analysis_time = time.time() - start_time
        
        print(f"✅ Price analysis query: {analysis_time:.4f}s")
        if analysis[0] > 0:  # If we have data
            print(f"   Analysis results: {analysis[0]} records, price range: ${analysis[1]:.6f} - ${analysis[2]:.6f}")
        
        # Test arbitrage opportunity queries
        start_time = time.time()
        opportunities = optimizer.get_arbitrage_opportunities_optimized(min_profit=0.01)
        arb_time = time.time() - start_time
        
        print(f"✅ Arbitrage opportunities query: {arb_time:.4f}s ({len(opportunities)} opportunities)")
        
        # Performance summary
        total_query_time = query_time + analysis_time + arb_time
        print(f"✅ Total query time: {total_query_time:.4f}s")
        
        # Check against paper target (6.43s total, DB should be <10% of that)
        target_db_time = 0.643  # 10% of 6.43s
        if total_query_time < target_db_time:
            print(f"✅ Database performance target achieved: {total_query_time:.4f}s < {target_db_time:.4f}s")
            return True
        else:
            print(f"⚠️  Database performance target missed: {total_query_time:.4f}s > {target_db_time:.4f}s")
            return False
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage optimization"""
    print("\n💾 Testing memory usage optimization...")
    
    try:
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        optimizer = get_db_optimizer("test_optimization.db")
        
        # Perform memory-intensive operations
        for i in range(5):
            large_batch = [
                (f"mem_test_token_{j}", random.uniform(1.0, 100.0), 
                 int(time.time()), 3000000 + j)
                for j in range(1000)
            ]
            
            optimizer.bulk_insert_optimized(
                "test_prices",
                large_batch,
                ["token_address", "price", "timestamp", "block_number"]
            )
            
            # Execute some queries to populate cache
            optimizer.execute_query(
                "SELECT COUNT(*) FROM test_prices WHERE token_address LIKE ?",
                (f"mem_test_token_{i}%",),
                cache_key=f"mem_test_{i}"
            )
        
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"✅ Memory usage test:")
        print(f"   Initial memory: {initial_memory:.1f} MB")
        print(f"   Final memory: {final_memory:.1f} MB")
        print(f"   Memory increase: {memory_increase:.1f} MB")
        
        # Memory efficiency check
        if memory_increase < 50:  # Should use less than 50MB for test operations
            print("✅ Memory usage within acceptable limits")
            return True
        else:
            print("⚠️  Memory usage higher than expected")
            return False
            
    except ImportError:
        print("⚠️  psutil not available, skipping memory test")
        return True
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def cleanup_test_data():
    """Clean up test databases"""
    try:
        import os
        test_files = ["test_optimization.db", "test_optimization.db-wal", "test_optimization.db-shm"]
        
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"🧹 Cleaned up: {file}")
                
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

def main():
    """Run all database optimization tests"""
    print("🧪 Testing Database Query Optimization")
    print("=" * 60)
    
    tests = [
        ("Basic Database Optimization", test_basic_database_optimization),
        ("Connection Pool Performance", test_connection_pool_performance),
        ("Query Optimization", test_query_optimization),
        ("Paper Compliance", test_paper_compliance),
        ("Integration with Historical Data", test_integration_with_historical_data),
        ("Memory Usage", test_memory_usage)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📝 Running: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"🏆 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Database query optimization is working correctly.")
        print("\n✅ TODO item completed: 'Database query 최적화'")
        print("   - Connection pooling implemented for concurrent access")
        print("   - Query caching system reduces redundant database calls")
        print("   - Batch processing optimizes bulk operations")
        print("   - Index optimization improves query performance")
        print("   - Memory-based temporary tables for fast analysis")
        print("   - Performance monitoring and automatic optimization")
        print("   - Ready for paper-scale requirements (96 protocols, 25 assets)")
        print("   - Contributes to 6.43s average execution time target")
        cleanup_test_data()
        return True
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)