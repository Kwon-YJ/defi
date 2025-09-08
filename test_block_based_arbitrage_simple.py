#!/usr/bin/env python3
"""
Test script for block-based arbitrage detection functionality
"""

import sys
import os

# Set environment variables for testing
os.environ['LOG_LEVEL'] = 'INFO'
os.environ['REDIS_URL'] = 'redis://localhost:6379'

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_block_based_arbitrage_detector():
    """Test the block-based arbitrage detection functionality"""
    try:
        # Import the block-based arbitrage detector
        from block_based_arbitrage_detector import BlockBasedArbitrageDetector, ArbitrageDetector
        
        # Test BlockBasedArbitrageDetector
        print("Testing BlockBasedArbitrageDetector...")
        block_detector = BlockBasedArbitrageDetector()
        
        # Test initialization
        assert hasattr(block_detector, 'market_graph')
        assert hasattr(block_detector, 'bellman_ford')
        assert hasattr(block_detector, 'protocol_manager')
        assert hasattr(block_detector, 'token_manager')
        assert hasattr(block_detector, 'storage')
        assert hasattr(block_detector, 'performance_monitor')
        print("  ✓ BlockBasedArbitrageDetector initialization")
        
        # Test ArbitrageDetector
        print("Testing ArbitrageDetector...")
        arbitrage_detector = ArbitrageDetector()
        
        # Test initialization
        assert hasattr(arbitrage_detector, 'market_graph')
        assert hasattr(arbitrage_detector, 'bellman_ford')
        assert hasattr(arbitrage_detector, 'protocol_manager')
        assert hasattr(arbitrage_detector, 'token_manager')
        assert hasattr(arbitrage_detector, 'storage')
        assert hasattr(arbitrage_detector, 'performance_monitor')
        assert hasattr(arbitrage_detector, 'block_detector')
        print("  ✓ ArbitrageDetector initialization")
        
        # Test method existence
        block_methods = [
            'start_block_based_detection',
            '_subscribe_to_new_blocks',
            '_handle_websocket_message',
            '_process_new_block',
            '_update_market_data',
            '_process_opportunities',
            'stop_detection'
        ]
        
        for method in block_methods:
            assert hasattr(block_detector, method), f"Missing method in BlockBasedArbitrageDetector: {method}"
        print("  ✓ BlockBasedArbitrageDetector methods")
        
        arbitrage_methods = [
            'start_detection',
            'start_block_based_detection',
            '_update_market_data',
            '_process_opportunities',
            'stop_detection'
        ]
        
        for method in arbitrage_methods:
            assert hasattr(arbitrage_detector, method), f"Missing method in ArbitrageDetector: {method}"
        print("  ✓ ArbitrageDetector methods")
        
        # Test WebSocket URL
        assert block_detector.ws_url is not None
        assert isinstance(block_detector.ws_url, str)
        print("  ✓ WebSocket URL configuration")
        
        print("\n✓ All block-based arbitrage detection tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Block-based arbitrage detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_block_based_arbitrage_detector()
    sys.exit(0 if success else 1)