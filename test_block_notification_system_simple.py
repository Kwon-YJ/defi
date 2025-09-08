#!/usr/bin/env python3
"""
Test script for enhanced block notification system functionality
"""

import sys
import os
from unittest.mock import Mock

# Set environment variables for testing
os.environ['LOG_LEVEL'] = 'INFO'
os.environ['REDIS_URL'] = 'redis://localhost:6379'

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_block_notification_system():
    """Test the enhanced block notification system functionality"""
    try:
        # Import the block notification system
        from block_notification_system import (
            BlockNotificationSystem, 
            EnhancedNotificationHandler, 
            BlockNotification, 
            BlockNotificationType
        )
        
        # Test BlockNotificationSystem
        print("Testing BlockNotificationSystem...")
        notification_system = BlockNotificationSystem()
        
        # Test initialization
        assert hasattr(notification_system, 'node_configs')
        assert hasattr(notification_system, 'subscribers')
        assert hasattr(notification_system, 'running')
        assert hasattr(notification_system, 'websocket_connections')
        assert hasattr(notification_system, 'http_sessions')
        assert hasattr(notification_system, 'notification_stats')
        print("  ✓ BlockNotificationSystem initialization")
        
        # Test notification types
        expected_types = [
            BlockNotificationType.NEW_BLOCK,
            BlockNotificationType.BLOCK_REORG,
            BlockNotificationType.PENDING_TRANSACTION,
            BlockNotificationType.TRANSACTION_CONFIRMED
        ]
        
        for notification_type in expected_types:
            assert notification_type in BlockNotificationType
        print("  ✓ Block notification types")
        
        # Test block notification dataclass
        notification = BlockNotification(
            notification_type=BlockNotificationType.NEW_BLOCK,
            block_number=12345678,
            block_hash="0x1234567890abcdef",
            timestamp=None,  # Will be set to current time
            data={"test": "data"},
            source="test_source"
        )
        
        assert notification.notification_type == BlockNotificationType.NEW_BLOCK
        assert notification.block_number == 12345678
        assert notification.block_hash == "0x1234567890abcdef"
        assert notification.source == "test_source"
        assert "test" in notification.data
        print("  ✓ Block notification dataclass")
        
        # Test subscription functionality
        callback = lambda x: None
        notification_system.subscribe(BlockNotificationType.NEW_BLOCK, callback)
        
        assert BlockNotificationType.NEW_BLOCK in notification_system.subscribers
        assert callback in notification_system.subscribers[BlockNotificationType.NEW_BLOCK]
        print("  ✓ Subscription functionality")
        
        # Test statistics tracking
        stats = notification_system.get_statistics()
        assert 'total_notifications' in stats
        assert 'notifications_by_type' in stats
        assert 'notifications_by_source' in stats
        assert 'errors' in stats
        assert 'average_latency' in stats
        assert 'total_latency_samples' in stats
        print("  ✓ Statistics tracking")
        
        # Test EnhancedNotificationHandler
        print("Testing EnhancedNotificationHandler...")
        mock_arbitrage_detector = Mock()
        notification_handler = EnhancedNotificationHandler(mock_arbitrage_detector)
        
        # Test initialization
        assert hasattr(notification_handler, 'arbitrage_detector')
        assert hasattr(notification_handler, 'notification_system')
        assert hasattr(notification_handler, 'processing_queue')
        assert hasattr(notification_handler, 'processing_task')
        print("  ✓ EnhancedNotificationHandler initialization")
        
        # Test notification handler methods
        handler_methods = [
            'initialize',
            'start_notification_handling',
            '_handle_new_block_notification',
            '_handle_pending_transaction_notification',
            '_process_notification_queue',
            '_process_new_block',
            'get_notification_statistics',
            'cleanup'
        ]
        
        for method in handler_methods:
            assert hasattr(notification_handler, method), f"Missing method in EnhancedNotificationHandler: {method}"
        print("  ✓ EnhancedNotificationHandler methods")
        
        print("\n✓ All block notification system tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Block notification system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_block_notification_system()
    sys.exit(0 if success else 1)