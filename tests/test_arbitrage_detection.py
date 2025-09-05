import pytest
from src.market_graph import DeFiMarketGraph, TradingEdge, ArbitrageOpportunity
from src.bellman_ford_arbitrage import BellmanFordArbitrage

class TestArbitrageDetection:
    
    @pytest.fixture
    def market_graph(self):
        """Market graph fixture"""
        graph = DeFiMarketGraph()
        
        # Add test tokens
        graph.add_token("WETH", "WETH")
        graph.add_token("USDC", "USDC") 
        graph.add_token("DAI", "DAI")
        
        # Add test trading pairs
        graph.add_trading_pair("WETH", "USDC", "uniswap_v2", "0x123", 100.0, 200000.0)
        graph.add_trading_pair("USDC", "DAI", "sushiswap", "0x456", 200000.0, 200000.0)
        graph.add_trading_pair("DAI", "WETH", "curve", "0x789", 200000.0, 99.0)
        
        return graph
    
    @pytest.fixture
    def bellman_ford(self, market_graph):
        """Bellman-Ford arbitrage detector fixture"""
        return BellmanFordArbitrage(market_graph)
    
    def test_graph_creation(self, market_graph):
        """Test market graph creation"""
        stats = market_graph.get_graph_stats()
        assert stats['nodes'] == 3
        assert stats['edges'] == 6  # 3 pairs * 2 directions
        assert stats['tokens'] == 3
    
    def test_trading_edge_creation(self, market_graph):
        """Test trading edge creation"""
        # Check if edges exist
        assert market_graph.graph.has_edge("WETH", "USDC")
        assert market_graph.graph.has_edge("USDC", "WETH")
        
        # Check edge data
        edge_data = market_graph.graph["WETH"]["USDC"]
        assert edge_data['dex'] == "uniswap_v2"
        assert edge_data['pool_address'] == "0x123"
    
    def test_negative_cycle_detection(self, bellman_ford):
        """Test negative cycle detection"""
        # This test would require a graph with actual negative cycles
        # For demonstration, we test the method exists and runs
        opportunities = bellman_ford.find_negative_cycles("WETH")
        assert isinstance(opportunities, list)
    
    def test_profit_ratio_calculation(self, bellman_ford):
        """Test profit ratio calculation"""
        edges = [
            TradingEdge("WETH", "USDC", "uniswap_v2", "0x123", 2000.0, 100.0, 0.003, 0.01, 0),
            TradingEdge("USDC", "DAI", "sushiswap", "0x456", 1.0, 1000.0, 0.003, 0.01, 0),
            TradingEdge("DAI", "WETH", "curve", "0x789", 0.0005, 200.0, 0.001, 0.015, 0)
        ]
        
        profit_ratio = bellman_ford._calculate_profit_ratio(edges)
        expected_ratio = 2000.0 * 1.0 * 0.0005
        assert profit_ratio == expected_ratio
    
    def test_confidence_calculation(self, bellman_ford):
        """Test confidence score calculation"""
        edges = [
            TradingEdge("WETH", "USDC", "uniswap_v2", "0x123", 2000.0, 100.0, 0.003, 0.01, 0),
            TradingEdge("USDC", "WETH", "sushiswap", "0x456", 0.0005, 50.0, 0.003, 0.01, 0)
        ]
        
        confidence = bellman_ford._calculate_confidence(edges)
        assert 0.0 <= confidence <= 1.0

if __name__ == "__main__":
    pytest.main([__file__])
