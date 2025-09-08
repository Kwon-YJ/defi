import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from state_change_detection import (
    StateChangeDetector, 
    ImmediateResponseSystem, 
    EnhancedStateChangeDetectionSystem,
    StateChange,
    StateChangeType
)

class TestStateChangeDetection:
    
    @pytest.fixture
    def state_detector(self):
        """State change detector fixture"""
        with patch('src.state_change_detection.setup_logger'):
            detector = StateChangeDetector()
            return detector
    
    def test_state_detector_initialization(self, state_detector):
        """Test state change detector initialization"""
        assert hasattr(state_detector, 'storage')
        assert hasattr(state_detector, 'subscribers')
        assert hasattr(state_detector, 'running')
        assert hasattr(state_detector, 'previous_states')
        assert hasattr(state_detector, 'change_thresholds')
        assert hasattr(state_detector, 'detection_stats')
        assert hasattr(state_detector, 'monitored_pools')
        assert hasattr(state_detector, 'monitored_tokens')
        assert hasattr(state_detector, 'monitored_protocols')
        
        # Check that thresholds are initialized
        assert StateChangeType.POOL_RESERVE_CHANGE in state_detector.change_thresholds
        assert StateChangeType.TOKEN_PRICE_CHANGE in state_detector.change_thresholds
    
    def test_state_change_types(self):
        """Test state change types"""
        expected_types = [
            StateChangeType.POOL_RESERVE_CHANGE,
            StateChangeType.TOKEN_PRICE_CHANGE,
            StateChangeType.PROTOCOL_STATE_CHANGE,
            StateChangeType.LIQUIDITY_CHANGE,
            StateChangeType.FEE_CHANGE
        ]
        
        for change_type in expected_types:
            assert change_type in StateChangeType
    
    def test_state_change_dataclass(self):
        """Test state change dataclass"""
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
    
    def test_subscribe_functionality(self, state_detector):
        """Test subscription functionality"""
        # Test subscribe method
        callback = Mock()
        state_detector.subscribe(StateChangeType.POOL_RESERVE_CHANGE, callback)
        
        assert StateChangeType.POOL_RESERVE_CHANGE in state_detector.subscribers
        assert callback in state_detector.subscribers[StateChangeType.POOL_RESERVE_CHANGE]
        
        # Test subscribe_all method
        all_callback = Mock()
        state_detector.subscribe_all(all_callback)
        
        # Should be subscribed to all state change types
        for change_type in StateChangeType:
            assert all_callback in state_detector.subscribers[change_type]
    
    def test_statistics_tracking(self, state_detector):
        """Test statistics tracking"""
        # Test initial statistics
        stats = state_detector.get_statistics()
        assert 'total_changes_detected' in stats
        assert 'changes_by_type' in stats
        assert 'average_response_time' in stats
        assert 'errors' in stats
        assert 'monitored_pools' in stats
        assert 'monitored_tokens' in stats
        assert 'monitored_protocols' in stats
        
        # Test that initial values are correct
        assert stats['total_changes_detected'] == 0
        assert stats['errors'] == 0
        assert stats['average_response_time'] == 0
        assert stats['monitored_pools'] == 0
        assert stats['monitored_tokens'] == 0
        assert stats['monitored_protocols'] == 0
    
    def test_stop_functionality(self, state_detector):
        """Test stop functionality"""
        # Initially should be False
        assert state_detector.running == False
        
        # Start and then stop
        state_detector.running = True
        state_detector.stop()
        assert state_detector.running == False

class TestImmediateResponseSystem:
    
    @pytest.fixture
    def mock_state_detector(self):
        """Mock state detector fixture"""
        return Mock()
    
    @pytest.fixture
    def response_system(self, mock_state_detector):
        """Immediate response system fixture"""
        with patch('src.state_change_detection.setup_logger'):
            system = ImmediateResponseSystem(mock_state_detector)
            return system
    
    def test_response_system_initialization(self, response_system, mock_state_detector):
        """Test response system initialization"""
        assert hasattr(response_system, 'state_detector')
        assert hasattr(response_system, 'response_handlers')
        assert hasattr(response_system, 'running')
        assert hasattr(response_system, 'response_priorities')
        assert hasattr(response_system, 'response_thresholds')
        assert hasattr(response_system, 'response_stats')
        
        # Check that the state detector is correctly assigned
        assert response_system.state_detector == mock_state_detector
    
    def test_response_priorities(self, response_system):
        """Test response priorities"""
        expected_priorities = [
            StateChangeType.POOL_RESERVE_CHANGE,
            StateChangeType.TOKEN_PRICE_CHANGE,
            StateChangeType.LIQUIDITY_CHANGE,
            StateChangeType.FEE_CHANGE,
            StateChangeType.PROTOCOL_STATE_CHANGE
        ]
        
        for change_type in expected_priorities:
            assert change_type in response_system.response_priorities
    
    def test_response_thresholds(self, response_system):
        """Test response thresholds"""
        expected_thresholds = [
            StateChangeType.POOL_RESERVE_CHANGE,
            StateChangeType.TOKEN_PRICE_CHANGE,
            StateChangeType.LIQUIDITY_CHANGE,
            StateChangeType.FEE_CHANGE,
            StateChangeType.PROTOCOL_STATE_CHANGE
        ]
        
        for change_type in expected_thresholds:
            assert change_type in response_system.response_thresholds

class TestEnhancedStateChangeDetectionSystem:
    
    @pytest.fixture
    def enhanced_system(self):
        """Enhanced state change detection system fixture"""
        with patch('src.state_change_detection.setup_logger'):
            system = EnhancedStateChangeDetectionSystem()
            return system
    
    def test_enhanced_system_initialization(self, enhanced_system):
        """Test enhanced system initialization"""
        assert hasattr(enhanced_system, 'state_detector')
        assert hasattr(enhanced_system, 'response_system')
        assert hasattr(enhanced_system, 'arbitrage_detector')
        assert hasattr(enhanced_system, 'running')
        
        # Check that components are initialized
        assert enhanced_system.state_detector is not None
        assert enhanced_system.response_system is not None
    
    def test_enhanced_system_methods(self, enhanced_system):
        """Test enhanced system methods"""
        methods = [
            'initialize',
            'start_system',
            'add_monitored_entities',
            'get_system_statistics',
            'cleanup',
            'stop'
        ]
        
        for method in methods:
            assert hasattr(enhanced_system, method), f"Missing method: {method}"

if __name__ == "__main__":
    pytest.main([__file__])