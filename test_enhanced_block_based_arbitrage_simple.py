#!/usr/bin/env python3
"""
Test script for enhanced block-based arbitrage detection functionality with Ethereum block time guarantee
"""

import sys
import os

# Set environment variables for testing
os.environ['LOG_LEVEL'] = 'INFO'
os.environ['REDIS_URL'] = 'redis://localhost:6379'

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_enhanced_block_based_arbitrage_detector():
    """Test the enhanced block-based arbitrage detection functionality"""
    try:
        # Import the enhanced block-based arbitrage detector
        from enhanced_block_based_arbitrage_detector import EnhancedBlockBasedArbitrageDetector, EnhancedArbitrageDetector
        
        # Test EnhancedBlockBasedArbitrageDetector
        print("Testing EnhancedBlockBasedArbitrageDetector...")
        enhanced_block_detector = EnhancedBlockBasedArbitrageDetector()
        
        # Test initialization
        assert hasattr(enhanced_block_detector, 'market_graph')
        assert hasattr(enhanced_block_detector, 'bellman_ford')
        assert hasattr(enhanced_block_detector, 'protocol_manager')
        assert hasattr(enhanced_block_detector, 'token_manager')
        assert hasattr(enhanced_block_detector, 'storage')
        assert hasattr(enhanced_block_detector, 'performance_monitor')
        print("  ✓ EnhancedBlockBasedArbitrageDetector initialization")
        
        # Test Ethereum block time constants
        assert enhanced_block_detector.ETHEREUM_BLOCK_TIME == 13.5
        assert enhanced_block_detector.TARGET_PROCESSING_TIME == 6.43
        assert enhanced_block_detector.WARNING_THRESHOLD == 10.0
        assert enhanced_block_detector.CRITICAL_THRESHOLD == 12.0
        assert enhanced_block_detector.processing_timeout == 12.0
        print("  ✓ Ethereum block time constants")
        
        # Test EnhancedArbitrageDetector
        print("Testing EnhancedArbitrageDetector...")
        enhanced_arbitrage_detector = EnhancedArbitrageDetector()
        
        # Test initialization
        assert hasattr(enhanced_arbitrage_detector, 'market_graph')
        assert hasattr(enhanced_arbitrage_detector, 'bellman_ford')
        assert hasattr(enhanced_arbitrage_detector, 'protocol_manager')
        assert hasattr(enhanced_arbitrage_detector, 'token_manager')
        assert hasattr(enhanced_arbitrage_detector, 'storage')
        assert hasattr(enhanced_arbitrage_detector, 'performance_monitor')
        assert hasattr(enhanced_arbitrage_detector, 'enhanced_block_detector')
        print("  ✓ EnhancedArbitrageDetector initialization")
        
        # Test method existence
        enhanced_block_methods = [
            'start_block_based_detection',
            '_subscribe_to_new_blocks',
            '_handle_websocket_message',
            '_process_new_block_with_timeout',
            '_process_new_block',
            '_update_market_data_with_timeout',
            '_update_market_data',
            '_process_opportunities_with_timeout',
            '_process_opportunities',
            'stop_detection',
            '_track_processing_time',
            '_check_ethereum_block_time_compliance',
            '_log_performance_metrics',
            'get_performance_statistics'
        ]
        
        for method in enhanced_block_methods:
            assert hasattr(enhanced_block_detector, method), f"Missing method in EnhancedBlockBasedArbitrageDetector: {method}"
        print("  ✓ EnhancedBlockBasedArbitrageDetector methods")
        
        enhanced_arbitrage_methods = [
            'start_detection',
            'start_enhanced_block_based_detection',
            '_update_market_data',
            '_process_opportunities',
            'stop_detection'
        ]
        
        for method in enhanced_arbitrage_methods:
            assert hasattr(enhanced_arbitrage_detector, method), f"Missing method in EnhancedArbitrageDetector: {method}"
        print("  ✓ EnhancedArbitrageDetector methods")
        
        # Test performance tracking
        processing_times = [1.0, 2.0, 3.0, 4.0, 5.0]
        for time in processing_times:
            enhanced_block_detector._track_processing_time(time)
        
        assert len(enhanced_block_detector.block_processing_times) == 5
        print("  ✓ Performance tracking")
        
        # Test performance statistics
        stats = enhanced_block_detector.get_performance_statistics()
        assert isinstance(stats, dict)
        print("  ✓ Performance statistics")
        
        print("\n✓ All enhanced block-based arbitrage detection tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced block-based arbitrage detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_block_based_arbitrage_detector()
    sys.exit(0 if success else 1)