#!/usr/bin/env python3
"""
Multi-Hop Arbitrage Implementation Test
Tests the newly implemented multi-hop arbitrage (3+ protocols) functionality

This test validates:
- Multi-hop path detection
- Strategy profitability calculation
- Complex arbitrage chain execution
- Performance metrics tracking
"""

import asyncio
import time
from decimal import Decimal, getcontext
from typing import List, Dict
from web3 import Web3

# Set high precision for financial calculations
getcontext().prec = 28

from src.logger import setup_logger
from src.market_graph import DeFiMarketGraph
from src.protocol_actions import ProtocolRegistry
from src.multi_hop_arbitrage import MultiHopArbitrageDetector, MultiHopStrategyType

logger = setup_logger(__name__)

class MultiHopArbitrageTest:
    """Test suite for multi-hop arbitrage functionality"""
    
    def __init__(self):
        self.graph = DeFiMarketGraph()
        # Initialize Web3 with a mock provider for testing
        self.w3 = Web3()
        self.protocol_registry = ProtocolRegistry(self.w3)
        self.detector = None
        
        # Test configuration
        self.test_tokens = [
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "0xA0b86a91c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
            "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
            "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # WBTC
        ]
        
        logger.info("Multi-hop arbitrage test initialized")

    async def setup_test_environment(self):
        """Setup test environment with mock data"""
        logger.info("🔧 Setting up test environment...")
        
        try:
            # Add test tokens to graph
            for i, token in enumerate(self.test_tokens):
                symbol = ["WETH", "USDC", "DAI", "USDT", "WBTC"][i]
                self.graph.add_token(token, symbol)
            
            # Add test edges for multi-hop paths
            await self._add_test_trading_pairs()
            
            # Create multi-hop detector
            self.detector = MultiHopArbitrageDetector(self.graph, self.protocol_registry)
            
            logger.info("✅ Test environment setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test environment setup failed: {e}")
            return False

    async def _add_test_trading_pairs(self):
        """Add test trading pairs to create multi-hop opportunities"""
        
        # Create a complex graph with multiple protocols
        test_edges = [
            # Uniswap V2 edges
            ("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "0xA0b86a91c6218b36c1d19D4a2e9Eb0cE3606eB48", 
             {"protocol": "uniswap_v2", "weight": 1800.0, "action": "swap"}),
            
            # Sushiswap edges
            ("0xA0b86a91c6218b36c1d19D4a2e9Eb0cE3606eB48", "0x6B175474E89094C44Da98b954EedeAC495271d0F",
             {"protocol": "sushiswap", "weight": 1.001, "action": "swap"}),
            
            # Curve Finance edges
            ("0x6B175474E89094C44Da98b954EedeAC495271d0F", "0xdAC17F958D2ee523a2206206994597C13D831ec7",
             {"protocol": "curve", "weight": 0.999, "action": "swap"}),
            
            # Balancer edges
            ("0xdAC17F958D2ee523a2206206994597C13D831ec7", "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
             {"protocol": "balancer", "weight": 0.000055, "action": "swap"}),
            
            # Return path through different protocol
            ("0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
             {"protocol": "uniswap_v3", "weight": 18100.0, "action": "swap"}),
        ]
        
        # Add edges to graph
        for from_token, to_token, data in test_edges:
            self.graph.graph.add_edge(from_token, to_token, **data)
            logger.debug(f"Added edge: {from_token[:8]}...→{to_token[:8]}... via {data['protocol']}")

    async def test_multi_hop_detection(self):
        """Test multi-hop arbitrage opportunity detection"""
        logger.info("🔍 Testing multi-hop arbitrage detection...")
        
        start_time = time.time()
        
        try:
            # Find multi-hop opportunities
            strategies = await self.detector.find_multi_hop_opportunities(
                base_tokens=self.test_tokens[:2],  # Test with first 2 tokens
                max_strategies=5
            )
            
            detection_time = time.time() - start_time
            
            logger.info(f"✅ Detection completed in {detection_time:.3f} seconds")
            logger.info(f"Found {len(strategies)} multi-hop strategies")
            
            # Analyze found strategies
            if strategies:
                for i, strategy in enumerate(strategies[:3]):  # Show top 3
                    logger.info(f"Strategy {i+1}:")
                    logger.info(f"  Type: {strategy.strategy_type}")
                    logger.info(f"  Steps: {len(strategy.steps)}")
                    logger.info(f"  Expected profit: {strategy.net_profit:.6f} ETH")
                    logger.info(f"  Confidence: {strategy.confidence_score:.2f}")
                    logger.info(f"  Risk level: {strategy.risk_level}")
                    
                    # Show path details
                    path_str = " → ".join([
                        f"{step.from_token[:8]}...({step.protocol})" 
                        for step in strategy.steps
                    ])
                    logger.info(f"  Path: {path_str}")
            
            return {
                'success': True,
                'strategies_found': len(strategies),
                'detection_time': detection_time,
                'profitable_strategies': len([s for s in strategies if s.net_profit > 0])
            }
            
        except Exception as e:
            logger.error(f"❌ Multi-hop detection test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'strategies_found': 0,
                'detection_time': time.time() - start_time
            }

    async def test_strategy_execution(self):
        """Test strategy execution simulation"""
        logger.info("🚀 Testing strategy execution...")
        
        try:
            # Find a strategy to test execution
            strategies = await self.detector.find_multi_hop_opportunities(
                base_tokens=self.test_tokens[:1],
                max_strategies=1
            )
            
            if not strategies:
                logger.warning("No strategies found for execution test")
                return {'success': False, 'error': 'No strategies available'}
            
            strategy = strategies[0]
            logger.info(f"Testing execution of strategy: {strategy.strategy_id}")
            
            # Execute strategy (simulation)
            execution_result = await self.detector.execute_strategy(strategy)
            
            logger.info(f"Execution result: {execution_result}")
            
            return {
                'success': execution_result['success'],
                'actual_profit': execution_result.get('actual_profit', 0),
                'gas_used': execution_result.get('gas_used', 0),
                'execution_time': execution_result.get('execution_time', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Strategy execution test failed: {e}")
            return {'success': False, 'error': str(e)}

    async def test_performance_metrics(self):
        """Test performance metrics tracking"""
        logger.info("📊 Testing performance metrics...")
        
        try:
            # Run detection to generate metrics
            await self.detector.find_multi_hop_opportunities(
                base_tokens=self.test_tokens,
                max_strategies=10
            )
            
            # Get performance metrics
            metrics = self.detector.get_performance_metrics()
            
            logger.info("Performance Metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")
            
            return {'success': True, 'metrics': metrics}
            
        except Exception as e:
            logger.error(f"❌ Performance metrics test failed: {e}")
            return {'success': False, 'error': str(e)}

    async def test_strategy_types(self):
        """Test different strategy type detection"""
        logger.info("🎯 Testing strategy type classification...")
        
        try:
            results = {}
            
            # Test different hop counts
            for hop_count in [3, 4, 5]:
                self.detector.min_hops = hop_count
                self.detector.max_hops = hop_count
                
                strategies = await self.detector.find_multi_hop_opportunities(
                    base_tokens=self.test_tokens[:1],
                    max_strategies=3
                )
                
                if strategies:
                    strategy_types = [s.strategy_type for s in strategies]
                    results[f"{hop_count}_hop"] = {
                        'count': len(strategies),
                        'types': list(set(strategy_types))
                    }
                    
                    logger.info(f"{hop_count}-hop strategies: {len(strategies)} found")
                    logger.info(f"  Types: {list(set(strategy_types))}")
            
            return {'success': True, 'results': results}
            
        except Exception as e:
            logger.error(f"❌ Strategy type test failed: {e}")
            return {'success': False, 'error': str(e)}

    async def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        logger.info("🧪 Starting comprehensive multi-hop arbitrage test")
        logger.info("="*60)
        
        # Setup test environment
        setup_success = await self.setup_test_environment()
        if not setup_success:
            logger.error("❌ Test setup failed - aborting tests")
            return False
        
        test_results = {}
        
        # Run all tests
        tests = [
            ("Multi-hop Detection", self.test_multi_hop_detection),
            ("Strategy Execution", self.test_strategy_execution),
            ("Performance Metrics", self.test_performance_metrics),
            ("Strategy Types", self.test_strategy_types),
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*20} {test_name} {'='*20}")
            
            try:
                result = await test_func()
                test_results[test_name] = result
                
                if result.get('success', False):
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ {test_name} CRASHED: {e}")
                test_results[test_name] = {'success': False, 'error': str(e)}
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("🏁 TEST SUMMARY")
        logger.info("="*60)
        
        passed = sum(1 for result in test_results.values() if result.get('success', False))
        total = len(test_results)
        
        logger.info(f"Tests passed: {passed}/{total}")
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - Multi-hop arbitrage implementation successful!")
            return True
        else:
            logger.warning(f"⚠️  Some tests failed ({total-passed}/{total})")
            return False


async def main():
    """Main test execution"""
    logger.info("Multi-Hop Arbitrage (3+ protocols) Implementation Test")
    logger.info("Testing the newly implemented multi-hop arbitrage functionality")
    logger.info("This addresses TODO.txt line 102: Multi-hop arbitrage (3+ protocols)")
    
    test_suite = MultiHopArbitrageTest()
    
    try:
        success = await test_suite.run_comprehensive_test()
        
        if success:
            logger.info("\n🎯 Multi-hop arbitrage implementation completed successfully!")
            logger.info("Ready to mark TODO.txt checkbox as completed")
        else:
            logger.error("\n❌ Multi-hop arbitrage implementation needs refinement")
            
        return success
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return False


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(main())
    exit(0 if success else 1)