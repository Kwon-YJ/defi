#!/usr/bin/env python3
"""
Test script for enhanced state change detection system functionality
"""

import sys
import os
from unittest.mock import Mock
from datetime import datetime

# Set environment variables for testing
os.environ['LOG_LEVEL'] = 'INFO'
os.environ['REDIS_URL'] = 'redis://localhost:6379'

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_state_change_detection_system():
    """Test the enhanced state change detection system functionality"""
    try:
        # Import the state change detection system
        from state_change_detection import (
            StateChangeDetector, 
            ImmediateResponseSystem, 
            EnhancedStateChangeDetectionSystem,
            StateChange,
            StateChangeType
        )
        
        # Test StateChangeDetector
        print("Testing StateChangeDetector...")
        state_detector = StateChangeDetector()
        
        # Test initialization
        assert hasattr(state_detector, 'storage')
        assert hasattr(state_detector, 'subscribers')
        assert hasattr(state_detector, 'running')
        assert hasattr(state_detector, 'previous_states')
        assert hasattr(state_detector, 'change_thresholds')
        assert hasattr(state_detector, 'detection_stats')
        assert hasattr(state_detector, 'monitored_pools')
        assert hasattr(state_detector, 'monitored_tokens')
        assert hasattr(state_detector, 'monitored_protocols')
        print("  ✓ StateChangeDetector initialization")
        
        # Test state change types
        expected_types = [
            StateChangeType.POOL_RESERVE_CHANGE,
            StateChangeType.TOKEN_PRICE_CHANGE,
            StateChangeType.PROTOCOL_STATE_CHANGE,
            StateChangeType.LIQUIDITY_CHANGE,
            StateChangeType.FEE_CHANGE
        ]
        
        for change_type in expected_types:
            assert change_type in StateChangeType
        print("  ✓ State change types")
        
        # Test state change dataclass
        state_change = StateChange(
            change_type=StateChangeType.POOL_RESERVE_CHANGE,
            entity_id="0x1234567890abcdef",
            timestamp=datetime.now(),
            old_value={"reserve0": 1000000, "reserve1": 2000000},
            new_value={"reserve0": 1100000, "reserve1": 1900000},
            change_percentage=0.05,
            source="test_source",
            metadata={"test": "data"}
        )
        
        assert state_change.change_type == StateChangeType.POOL_RESERVE_CHANGE
        assert state_change.entity_id == "0x1234567890abcdef"
        assert state_change.source == "test_source"
        assert "test" in state_change.metadata
        print("  ✓ State change dataclass")
        
        # Test subscription functionality
        callback = lambda x: None
        state_detector.subscribe(StateChangeType.POOL_RESERVE_CHANGE, callback)
        
        assert StateChangeType.POOL_RESERVE_CHANGE in state_detector.subscribers
        assert callback in state_detector.subscribers[StateChangeType.POOL_RESERVE_CHANGE]
        print("  ✓ Subscription functionality")
        
        # Test statistics tracking
        stats = state_detector.get_statistics()
        assert 'total_changes_detected' in stats
        assert 'changes_by_type' in stats
        assert 'average_response_time' in stats
        assert 'errors' in stats
        assert 'monitored_pools' in stats
        assert 'monitored_tokens' in stats
        assert 'monitored_protocols' in stats
        print("  ✓ Statistics tracking")
        
        # Test ImmediateResponseSystem
        print("Testing ImmediateResponseSystem...")
        mock_state_detector = Mock()
        response_system = ImmediateResponseSystem(mock_state_detector)
        
        # Test initialization
        assert hasattr(response_system, 'state_detector')
        assert hasattr(response_system, 'response_handlers')
        assert hasattr(response_system, 'running')
        assert hasattr(response_system, 'response_priorities')
        assert hasattr(response_system, 'response_thresholds')
        assert hasattr(response_system, 'response_stats')
        print("  ✓ ImmediateResponseSystem initialization")
        
        # Test response priorities
        expected_priorities = [
            StateChangeType.POOL_RESERVE_CHANGE,
            StateChangeType.TOKEN_PRICE_CHANGE,
            StateChangeType.LIQUIDITY_CHANGE,
            StateChangeType.FEE_CHANGE,
            StateChangeType.PROTOCOL_STATE_CHANGE
        ]
        
        for change_type in expected_priorities:
            assert change_type in response_system.response_priorities
        print("  ✓ Response priorities")
        
        # Test response thresholds
        expected_thresholds = [
            StateChangeType.POOL_RESERVE_CHANGE,
            StateChangeType.TOKEN_PRICE_CHANGE,
            StateChangeType.LIQUIDITY_CHANGE,
            StateChangeType.FEE_CHANGE,
            StateChangeType.PROTOCOL_STATE_CHANGE
        ]
        
        for change_type in expected_thresholds:
            assert change_type in response_system.response_thresholds
        print("  ✓ Response thresholds")
        
        # Test EnhancedStateChangeDetectionSystem
        print("Testing EnhancedStateChangeDetectionSystem...")
        enhanced_system = EnhancedStateChangeDetectionSystem()
        
        # Test initialization
        assert hasattr(enhanced_system, 'state_detector')
        assert hasattr(enhanced_system, 'response_system')
        assert hasattr(enhanced_system, 'arbitrage_detector')
        assert hasattr(enhanced_system, 'running')
        print("  ✓ EnhancedStateChangeDetectionSystem initialization")
        
        # Test enhanced system methods
        enhanced_methods = [
            'initialize',
            'start_system',
            'add_monitored_entities',
            'get_system_statistics',
            'cleanup',
            'stop'
        ]
        
        for method in enhanced_methods:
            assert hasattr(enhanced_system, method), f"Missing method in EnhancedStateChangeDetectionSystem: {method}"
        print("  ✓ EnhancedStateChangeDetectionSystem methods")
        
        print("\n✓ All state change detection system tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ State change detection system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_state_change_detection_system()
    sys.exit(0 if success else 1)