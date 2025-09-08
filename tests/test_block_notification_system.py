import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from block_notification_system import (
    BlockNotificationSystem, 
    EnhancedNotificationHandler, 
    BlockNotification, 
    BlockNotificationType
)

class TestBlockNotificationSystem:
    
    @pytest.fixture
    def notification_system(self):
        """Block notification system fixture"""
        with patch('src.block_notification_system.setup_logger'):
            system = BlockNotificationSystem()
            return system
    
    def test_notification_system_initialization(self, notification_system):
        """Test block notification system initialization"""
        assert hasattr(notification_system, 'node_configs')
        assert hasattr(notification_system, 'subscribers')
        assert hasattr(notification_system, 'running')
        assert hasattr(notification_system, 'websocket_connections')
        assert hasattr(notification_system, 'http_sessions')
        assert hasattr(notification_system, 'notification_stats')
        
        # Check that notification stats are initialized
        assert 'total_notifications' in notification_system.notification_stats
        assert 'by_type' in notification_system.notification_stats
        assert 'by_source' in notification_system.notification_stats
        assert 'errors' in notification_system.notification_stats
    
    def test_notification_types(self):
        """Test block notification types"""
        # Check that all notification types exist
        expected_types = [
            BlockNotificationType.NEW_BLOCK,
            BlockNotificationType.BLOCK_REORG,
            BlockNotificationType.PENDING_TRANSACTION,
            BlockNotificationType.TRANSACTION_CONFIRMED
        ]
        
        for notification_type in expected_types:
            assert notification_type in BlockNotificationType
    
    def test_block_notification_dataclass(self):
        """Test block notification dataclass"""
        notification = BlockNotification(
            notification_type=BlockNotificationType.NEW_BLOCK,
            block_number=12345678,
            block_hash="0x1234567890abcdef",
            timestamp=datetime.now(),
            data={"test": "data"},
            source="test_source"
        )
        
        assert notification.notification_type == BlockNotificationType.NEW_BLOCK
        assert notification.block_number == 12345678
        assert notification.block_hash == "0x1234567890abcdef"
        assert notification.source == "test_source"
        assert "test" in notification.data
    
    def test_subscribe_functionality(self, notification_system):
        """Test subscription functionality"""
        # Test subscribe method
        callback = Mock()
        notification_system.subscribe(BlockNotificationType.NEW_BLOCK, callback)
        
        assert BlockNotificationType.NEW_BLOCK in notification_system.subscribers
        assert callback in notification_system.subscribers[BlockNotificationType.NEW_BLOCK]
        
        # Test subscribe_all method
        all_callback = Mock()
        notification_system.subscribe_all(all_callback)
        
        # Should be subscribed to all notification types
        for notification_type in BlockNotificationType:
            assert all_callback in notification_system.subscribers[notification_type]
    
    def test_statistics_tracking(self, notification_system):
        """Test statistics tracking"""
        # Test initial statistics
        stats = notification_system.get_statistics()
        assert 'total_notifications' in stats
        assert 'notifications_by_type' in stats
        assert 'notifications_by_source' in stats
        assert 'errors' in stats
        assert 'average_latency' in stats
        assert 'total_latency_samples' in stats
        
        # Test that initial values are correct
        assert stats['total_notifications'] == 0
        assert stats['errors'] == 0
        assert stats['average_latency'] == 0
        assert stats['total_latency_samples'] == 0
    
    def test_stop_functionality(self, notification_system):
        """Test stop functionality"""
        # Initially should be False
        assert notification_system.running == False
        
        # Start and then stop
        notification_system.running = True
        notification_system.stop()
        assert notification_system.running == False

class TestEnhancedNotificationHandler:
    
    @pytest.fixture
    def mock_arbitrage_detector(self):
        """Mock arbitrage detector fixture"""
        return Mock()
    
    @pytest.fixture
    def notification_handler(self, mock_arbitrage_detector):
        """Enhanced notification handler fixture"""
        with patch('src.block_notification_system.setup_logger'):
            handler = EnhancedNotificationHandler(mock_arbitrage_detector)
            return handler
    
    def test_notification_handler_initialization(self, notification_handler, mock_arbitrage_detector):
        """Test notification handler initialization"""
        assert hasattr(notification_handler, 'arbitrage_detector')
        assert hasattr(notification_handler, 'notification_system')
        assert hasattr(notification_handler, 'processing_queue')
        assert hasattr(notification_handler, 'processing_task')
        
        # Check that the arbitrage detector is correctly assigned
        assert notification_handler.arbitrage_detector == mock_arbitrage_detector
    
    def test_notification_handler_methods(self, notification_handler):
        """Test notification handler methods"""
        methods = [
            'initialize',
            'start_notification_handling',
            '_handle_new_block_notification',
            '_handle_pending_transaction_notification',
            '_process_notification_queue',
            '_process_new_block',
            'get_notification_statistics',
            'cleanup'
        ]
        
        for method in methods:
            assert hasattr(notification_handler, method), f"Missing method: {method}"

if __name__ == "__main__":
    pytest.main([__file__])