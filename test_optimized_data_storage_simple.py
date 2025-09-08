#!/usr/bin/env python3
"""
Test script for optimized data storage functionality
"""

import sys
import os
import time
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_optimized_data_storage():
    """Test the optimized data storage functionality"""
    try:
        # Import the optimized data storage
        from optimized_data_storage import OptimizedDataStorage
        
        # Test OptimizedDataStorage
        print("Testing OptimizedDataStorage...")
        storage = OptimizedDataStorage()
        
        # Test initialization
        assert hasattr(storage, 'redis_client')
        assert hasattr(storage, '_key_prefixes')
        print("  ✓ OptimizedDataStorage initialization")
        
        # Test connection pooling
        conn = storage.get_connection()
        assert conn is not None
        print("  ✓ Connection pooling")
        
        # Test batch operations
        pipe = storage.start_batch_operation()
        assert pipe is not None
        result = storage.execute_batch()
        assert result == []
        print("  ✓ Batch operations")
        
        # Test method existence
        methods_to_check = [
            'store_pool_data',
            'get_pool_data',
            'store_arbitrage_opportunity',
            'get_recent_opportunities',
            'get_performance_data_range',
            'store_performance_data_batch',
            'get_pool_historical_range',
            'cleanup_old_data',
            'store_pool_data_batch',
            'store_arbitrage_opportunity_batch'
        ]
        
        for method in methods_to_check:
            assert hasattr(storage, method), f"Missing method: {method}"
        
        print("  ✓ All required methods exist")
        
        print("\n✓ All optimized data storage tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Optimized data storage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optimized_data_storage()
    sys.exit(0 if success else 1)