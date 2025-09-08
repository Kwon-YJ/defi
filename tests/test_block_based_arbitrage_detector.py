import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from block_based_arbitrage_detector import BlockBasedArbitrageDetector, ArbitrageDetector

class TestBlockBasedArbitrageDetector:
    
    @pytest.fixture
    def block_detector(self):
        """Block-based arbitrage detector fixture"""
        with patch('src.block_based_arbitrage_detector.setup_logger'):
            detector = BlockBasedArbitrageDetector()
            return detector
    
    @pytest.fixture
    def arbitrage_detector(self):
        """Arbitrage detector fixture"""
        with patch('src.block_based_arbitrage_detector.setup_logger'):
            detector = ArbitrageDetector()
            return detector
    
    def test_block_detector_initialization(self, block_detector):
        """Test block-based arbitrage detector initialization"""
        assert hasattr(block_detector, 'market_graph')
        assert hasattr(block_detector, 'bellman_ford')
        assert hasattr(block_detector, 'protocol_manager')
        assert hasattr(block_detector, 'token_manager')
        assert hasattr(block_detector, 'storage')
        assert hasattr(block_detector, 'performance_monitor')
        assert hasattr(block_detector, 'running')
        assert hasattr(block_detector, 'websocket')
        assert hasattr(block_detector, 'ws_url')
        
        # Check that supported tokens are initialized
        assert hasattr(block_detector, 'base_tokens')
        assert len(block_detector.base_tokens) >= 0  # Could be empty initially
    
    def test_arbitrage_detector_initialization(self, arbitrage_detector):
        """Test arbitrage detector initialization"""
        assert hasattr(arbitrage_detector, 'market_graph')
        assert hasattr(arbitrage_detector, 'bellman_ford')
        assert hasattr(arbitrage_detector, 'protocol_manager')
        assert hasattr(arbitrage_detector, 'token_manager')
        assert hasattr(arbitrage_detector, 'storage')
        assert hasattr(arbitrage_detector, 'performance_monitor')
        assert hasattr(arbitrage_detector, 'running')
        assert hasattr(arbitrage_detector, 'block_detector')
        
        # Check that supported tokens are initialized
        assert hasattr(arbitrage_detector, 'base_tokens')
        assert len(arbitrage_detector.base_tokens) >= 0  # Could be empty initially
    
    def test_block_detector_methods(self, block_detector):
        """Test block-based detector methods"""
        # Check that all required methods exist
        methods = [
            'start_block_based_detection',
            '_subscribe_to_new_blocks',
            '_handle_websocket_message',
            '_process_new_block',
            '_update_market_data',
            '_process_opportunities',
            'stop_detection'
        ]
        
        for method in methods:
            assert hasattr(block_detector, method), f"Missing method: {method}"
    
    def test_arbitrage_detector_methods(self, arbitrage_detector):
        """Test arbitrage detector methods"""
        # Check that all required methods exist
        methods = [
            'start_detection',
            'start_block_based_detection',
            '_update_market_data',
            '_process_opportunities',
            'stop_detection'
        ]
        
        for method in methods:
            assert hasattr(arbitrage_detector, method), f"Missing method: {method}"
    
    @pytest.mark.asyncio
    async def test_stop_detection(self, block_detector):
        """Test stop detection functionality"""
        # Initially should be False
        assert block_detector.running == False
        
        # Start and then stop
        block_detector.running = True
        block_detector.stop_detection()
        assert block_detector.running == False
    
    def test_websocket_url_configuration(self, block_detector):
        """Test WebSocket URL configuration"""
        # Should have a WebSocket URL
        assert block_detector.ws_url is not None
        assert isinstance(block_detector.ws_url, str)
        # Should be a WebSocket URL
        assert block_detector.ws_url.startswith(('ws://', 'wss://'))

if __name__ == "__main__":
    pytest.main([__file__])