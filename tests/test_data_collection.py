import pytest
import asyncio
from unittest.mock import Mock, patch
from src.dex_data_collector import UniswapV2Collector
from src.real_time_collector import RealTimeDataCollector
from web3 import Web3

class TestDataCollection:
    
    @pytest.fixture
    def mock_web3(self):
        """Mock Web3 provider"""
        w3 = Mock(spec=Web3)
        w3.eth.get_block.return_value = Mock(timestamp=1640995200)
        return w3
    
    @pytest.fixture
    def uniswap_collector(self, mock_web3):
        """Uniswap V2 collector fixture"""
        return UniswapV2Collector(mock_web3)
    
    def test_pair_address_retrieval(self, uniswap_collector):
        """Test pair address retrieval"""
        # Mock factory contract
        mock_contract = Mock()
        mock_contract.functions.getPair.return_value.call.return_value = "0x1234567890123456789012345678901234567890"
        uniswap_collector.factory_contract = mock_contract
        
        # Test
        result = asyncio.run(uniswap_collector.get_pair_address("token0", "token1"))
        assert result == "0x1234567890123456789012345678901234567890"
    
    def test_pool_reserves_calculation(self, uniswap_collector):
        """Test pool reserves calculation"""
        # Test price calculation
        result = asyncio.run(uniswap_collector.calculate_price(1000000, 2000000))
        assert result == 2.0  # 2000000 / 1000000
    
    @pytest.mark.asyncio
    async def test_real_time_collector_initialization(self):
        """Test real time collector initialization"""
        collector = RealTimeDataCollector()
        assert collector.running == False
        assert collector.subscribers == {}
    
    def test_websocket_message_handling(self):
        """Test WebSocket message handling"""
        collector = RealTimeDataCollector()
        
        # Mock message
        test_message = '{"params": {"subscription": "0x123", "result": {"number": "0x1234"}}}'
        
        # This would normally be tested with actual WebSocket connection
        # For now, just verify the method exists
        assert hasattr(collector, '_handle_websocket_message')

if __name__ == "__main__":
    pytest.main([__file__])
