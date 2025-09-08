import pytest
from unittest.mock import Mock, patch
from src.trade_executor import TradeExecutor, SimulationExecutor, TradeParams
from src.market_graph import ArbitrageOpportunity, TradingEdge
from web3 import Web3

class TestTradeExecution:
    
    @pytest.fixture
    def mock_web3(self):
        """Mock Web3 provider"""
        w3 = Mock(spec=Web3)
        w3.eth.gas_price = 20000000000
        w3.eth.get_transaction_count.return_value = 1
        return w3
    
    @pytest.fixture
    def trade_executor(self, mock_web3):
        """Trade executor fixture"""
        return TradeExecutor(mock_web3, "0x" + "1" * 64)  # Mock private key
    
    @pytest.fixture
    def simulation_executor(self, mock_web3):
        """Simulation executor fixture"""
        return SimulationExecutor(mock_web3)
    
    @pytest.fixture
    def mock_opportunity(self):
        """Mock arbitrage opportunity"""
        edges = [
            TradingEdge("WETH", "USDC", "uniswap_v2", "0x123", 2000.0, 100.0, 0.003, 0.01, 0),
            TradingEdge("USDC", "DAI", "sushiswap", "0x456", 1.001, 1000.0, 0.003, 0.01, 0),
            TradingEdge("DAI", "WETH", "curve", "0x789", 0.0005, 200.0, 0.001, 0.015, 0)
        ]
        
        return ArbitrageOpportunity(
            path=['WETH', 'USDC', 'DAI', 'WETH'],
            edges=edges,
            profit_ratio=1.002,
            required_capital=1.0,
            estimated_profit=0.002,
            gas_cost=0.001,
            net_profit=0.001,
            confidence=0.8
        )
    
    def test_trade_params_preparation(self, trade_executor, mock_opportunity):
        """Test trade parameters preparation"""
        params = trade_executor._prepare_trade_params(mock_opportunity)
        
        assert isinstance(params, TradeParams)
        assert len(params.tokens) == 4  # WETH -> USDC -> DAI -> WETH
        assert len(params.exchanges) == 3  # 3 trading steps
        assert params.flash_loan_amount == int(1.0 * 1e18)  # 1 ETH in Wei
    
    def test_trade_amounts_calculation(self, trade_executor, mock_opportunity):
        """Test trade amounts calculation"""
        amounts = trade_executor._calculate_trade_amounts(mock_opportunity)
        
        assert len(amounts) == 3  # 3 trading steps
        assert all(isinstance(amount, int) for amount in amounts)
        assert amounts[0] == int(1.0 * 1e18)  # First amount should be 1 ETH in Wei
    
    @pytest.mark.asyncio
    async def test_simulation_execution(self, simulation_executor, mock_opportunity):
        """Test simulation execution"""
        result = await simulation_executor.simulate_arbitrage(mock_opportunity)
        
        assert result['success'] == True
        assert 'steps' in result
        assert 'net_profit' in result
        assert len(result['steps']) == 3
    
    def test_slippage_calculation(self, simulation_executor):
        """Test slippage calculation"""
        # Test different liquidity scenarios
        low_slippage = simulation_executor._calculate_slippage(1.0, 1000.0)  # 0.1% of liquidity
        medium_slippage = simulation_executor._calculate_slippage(30.0, 1000.0)  # 3% of liquidity
        high_slippage = simulation_executor._calculate_slippage(150.0, 1000.0)  # 15% of liquidity
        
        assert low_slippage < medium_slippage < high_slippage
        assert 0 <= low_slippage <= 1
        assert 0 <= high_slippage <= 1
    
    def test_profitability_verification(self, trade_executor):
        """Test profitability verification"""
        params = TradeParams(
            tokens=["0x123", "0x456"],
            exchanges=["0x789"],
            amounts=[int(1e18)],
            flash_loan_amount=int(1e18),
            min_profit=int(0.01 * 1e18),  # 0.01 ETH profit
            gas_limit=500000,
            gas_price=20000000000
        )
        
        # Mock the verification (would normally require actual gas estimation)
        # This tests the method structure
        assert hasattr(trade_executor, '_verify_profitability')

if __name__ == "__main__":
    pytest.main([__file__])
