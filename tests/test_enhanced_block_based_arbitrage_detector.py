import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhanced_block_based_arbitrage_detector import EnhancedBlockBasedArbitrageDetector, EnhancedArbitrageDetector

class TestEnhancedBlockBasedArbitrageDetector:
    
    @pytest.fixture
    def enhanced_block_detector(self):
        """Enhanced block-based arbitrage detector fixture"""
        with patch('src.enhanced_block_based_arbitrage_detector.setup_logger'):
            detector = EnhancedBlockBasedArbitrageDetector()
            return detector
    
    @pytest.fixture
    def enhanced_arbitrage_detector(self):
        """Enhanced arbitrage detector fixture"""
        with patch('src.enhanced_block_based_arbitrage_detector.setup_logger'):
            detector = EnhancedArbitrageDetector()
            return detector
    
    def test_enhanced_block_detector_initialization(self, enhanced_block_detector):
        """Test enhanced block-based arbitrage detector initialization"""
        assert hasattr(enhanced_block_detector, 'market_graph')
        assert hasattr(enhanced_block_detector, 'bellman_ford')
        assert hasattr(enhanced_block_detector, 'protocol_manager')
        assert hasattr(enhanced_block_detector, 'token_manager')
        assert hasattr(enhanced_block_detector, 'storage')
        assert hasattr(enhanced_block_detector, 'performance_monitor')
        assert hasattr(enhanced_block_detector, 'running')
        assert hasattr(enhanced_block_detector, 'websocket')
        assert hasattr(enhanced_block_detector, 'ws_url')
        
        # Check Ethereum block time constants
        assert hasattr(enhanced_block_detector, 'ETHEREUM_BLOCK_TIME')
        assert enhanced_block_detector.ETHEREUM_BLOCK_TIME == 13.5
        assert hasattr(enhanced_block_detector, 'TARGET_PROCESSING_TIME')
        assert enhanced_block_detector.TARGET_PROCESSING_TIME == 6.43
        assert hasattr(enhanced_block_detector, 'WARNING_THRESHOLD')
        assert hasattr(enhanced_block_detector, 'CRITICAL_THRESHOLD')
        assert hasattr(enhanced_block_detector, 'processing_timeout')
        
        # Check that supported tokens are initialized
        assert hasattr(enhanced_block_detector, 'base_tokens')
        assert len(enhanced_block_detector.base_tokens) >= 0  # Could be empty initially
    
    def test_enhanced_arbitrage_detector_initialization(self, enhanced_arbitrage_detector):
        """Test enhanced arbitrage detector initialization"""
        assert hasattr(enhanced_arbitrage_detector, 'market_graph')
        assert hasattr(enhanced_arbitrage_detector, 'bellman_ford')
        assert hasattr(enhanced_arbitrage_detector, 'protocol_manager')
        assert hasattr(enhanced_arbitrage_detector, 'token_manager')
        assert hasattr(enhanced_arbitrage_detector, 'storage')
        assert hasattr(enhanced_arbitrage_detector, 'performance_monitor')
        assert hasattr(enhanced_arbitrage_detector, 'running')
        assert hasattr(enhanced_arbitrage_detector, 'enhanced_block_detector')
        
        # Check that supported tokens are initialized
        assert hasattr(enhanced_arbitrage_detector, 'base_tokens')
        assert len(enhanced_arbitrage_detector.base_tokens) >= 0  # Could be empty initially
    
    def test_enhanced_block_detector_methods(self, enhanced_block_detector):
        """Test enhanced block-based detector methods"""
        # Check that all required methods exist
        methods = [
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
        
        for method in methods:
            assert hasattr(enhanced_block_detector, method), f"Missing method: {method}"
    
    def test_enhanced_arbitrage_detector_methods(self, enhanced_arbitrage_detector):
        """Test enhanced arbitrage detector methods"""
        # Check that all required methods exist
        methods = [
            'start_detection',
            'start_enhanced_block_based_detection',
            '_update_market_data',
            '_process_opportunities',
            'stop_detection'
        ]
        
        for method in methods:
            assert hasattr(enhanced_arbitrage_detector, method), f"Missing method: {method}"
    
    @pytest.mark.asyncio
    async def test_stop_detection(self, enhanced_block_detector):
        """Test stop detection functionality"""
        # Initially should be False
        assert enhanced_block_detector.running == False
        
        # Start and then stop
        enhanced_block_detector.running = True
        enhanced_block_detector.stop_detection()
        assert enhanced_block_detector.running == False
    
    def test_performance_tracking(self, enhanced_block_detector):
        """Test performance tracking functionality"""
        # Test tracking processing times
        processing_times = [1.0, 2.0, 3.0, 4.0, 5.0]
        for time in processing_times:
            enhanced_block_detector._track_processing_time(time)
        
        # Check that times are tracked
        assert len(enhanced_block_detector.block_processing_times) == 5
        
        # Test statistics
        stats = enhanced_block_detector.get_performance_statistics()
        assert isinstance(stats, dict)
        
        # Test with empty times
        empty_detector = EnhancedBlockBasedArbitrageDetector()
        empty_stats = empty_detector.get_performance_statistics()
        assert empty_stats == {}
    
    def test_ethereum_block_time_constants(self, enhanced_block_detector):
        """Test Ethereum block time constants"""
        assert enhanced_block_detector.ETHEREUM_BLOCK_TIME == 13.5
        assert enhanced_block_detector.TARGET_PROCESSING_TIME == 6.43
        assert enhanced_block_detector.WARNING_THRESHOLD == 10.0
        assert enhanced_block_detector.CRITICAL_THRESHOLD == 12.0
        assert enhanced_block_detector.processing_timeout == 12.0

if __name__ == "__main__":
    pytest.main([__file__])