"""
Test script for graph building performance optimization
Validates that the optimization achieves the first unchecked TODO item:
"Graph building performance optimization (현재 병목점 제거)"
"""

import sys
import os
import time
import json
from typing import Dict, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.graph_performance_optimizer import GraphBuildingOptimizer, create_optimized_graph
from src.market_graph import DeFiMarketGraph
from src.logger import setup_logger

logger = setup_logger(__name__)

def generate_realistic_test_data() -> tuple[Dict, Dict]:
    """Generate realistic DeFi test data for performance testing"""
    
    # Common DeFi tokens (25 assets as per paper requirement)
    tokens = {
        "0xA0b86a33E6441C6C8320bb5b5c6c3e64f4b7b6bc": {"symbol": "ETH", "decimals": 18},
        "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": {"symbol": "WETH", "decimals": 18},
        "0xA0b86a33E6441C6C8320bb5b5c6c3e64f4b7b6bd": {"symbol": "USDC", "decimals": 6},
        "0xdAC17F958D2ee523a2206206994597C13D831ec7": {"symbol": "USDT", "decimals": 6},
        "0x6B175474E89094C44Da98b954EedeAC495271d0F": {"symbol": "DAI", "decimals": 18},
        "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599": {"symbol": "WBTC", "decimals": 8},
        "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984": {"symbol": "UNI", "decimals": 18},
        "0xd533a949740bb3306d119CC777fa900bA034cd52": {"symbol": "CRV", "decimals": 18},
        "0xba100000625a3754423978a60c9317c58a424e3D": {"symbol": "BAL", "decimals": 18},
        "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9": {"symbol": "AAVE", "decimals": 18},
        "0xc00e94Cb662C3520282E6f5717214004A7f26888": {"symbol": "COMP", "decimals": 18},
        "0x9f8F72aA9304c8B593d555F12eF6589CC3A579A2": {"symbol": "MKR", "decimals": 18},
        "0x0bc529c00C6401aEF6D220BE8C6Ea1667F6Ad93e": {"symbol": "YFI", "decimals": 18},
        "0x6B3595068778DD592e39A122f4f5a5cF09C90fE2": {"symbol": "SUSHI", "decimals": 18},
        "0x1494CA1F11D487c2bBe4543E90080AeBa4BA3C2b": {"symbol": "DPI", "decimals": 18},
        "0x57Ab1ec28D129707052df4dF418D58a2D46d5f51": {"symbol": "sETH", "decimals": 18},
        "0x5e74C9036fb86BD7eCdcb084a0673EFc32eA31cb": {"symbol": "sUSD", "decimals": 18},
        "0x4Fabb145d64652a948d72533023f6E7A623C7C53": {"symbol": "BUSD", "decimals": 18},
        "0x8E870D67F660D95d5be530380D0eC0bd388289E1": {"symbol": "PAXG", "decimals": 18},
        "0x514910771AF9Ca656af840dff83E8264EcF986CA": {"symbol": "LINK", "decimals": 18},
        # cTokens (Compound)
        "0x5d3a536E4D6DbD6114cc1Ead35777bAB948E3643": {"symbol": "cDAI", "decimals": 8},
        "0x4Ddc2D193948926D02f9B1fE9e1daa0718270ED5": {"symbol": "cETH", "decimals": 8},
        "0x39AA39c021dfbaE8faC545936693aC917d5E7563": {"symbol": "cUSDC", "decimals": 8},
        # aTokens (Aave)
        "0x028171bCA77440897B824Ca71D1c56caC55b68A3": {"symbol": "aDAI", "decimals": 18},
        "0x3Ed3B47Dd13EC9a98b44e6204A523E766B225811": {"symbol": "aUSDT", "decimals": 6}
    }
    
    # Generate pool data with realistic reserves and multiple DEX protocols
    pools = {}
    
    token_addresses = list(tokens.keys())
    dex_protocols = ["uniswap_v2", "sushiswap", "curve", "balancer", "uniswap_v3"]
    
    # Create pools for major pairs across different DEXs
    major_pairs = [
        (token_addresses[0], token_addresses[1]),  # ETH-WETH
        (token_addresses[1], token_addresses[2]),  # WETH-USDC
        (token_addresses[2], token_addresses[3]),  # USDC-USDT
        (token_addresses[3], token_addresses[4]),  # USDT-DAI
        (token_addresses[1], token_addresses[5]),  # WETH-WBTC
        (token_addresses[6], token_addresses[1]),  # UNI-WETH
        (token_addresses[7], token_addresses[4]),  # CRV-DAI
    ]
    
    pool_id = 1
    
    # Create multiple pools per pair (multi-DEX support)
    for token0, token1 in major_pairs:
        for dex in dex_protocols[:3]:  # Use top 3 DEXs per pair
            pool_addr = f"0xPool{pool_id:040x}"
            
            # Generate realistic reserves based on token types
            if tokens[token0]["symbol"] in ["USDC", "USDT", "DAI"]:
                reserve0 = 1000000 + (pool_id * 50000)  # 1M+ stablecoins
            else:
                reserve0 = 500 + (pool_id * 25)  # 500+ other tokens
            
            if tokens[token1]["symbol"] in ["USDC", "USDT", "DAI"]:
                reserve1 = 1000000 + (pool_id * 50000)
            else:
                reserve1 = 500 + (pool_id * 25)
            
            pools[pool_addr] = {
                "token0": token0,
                "token1": token1,
                "reserve0": reserve0,
                "reserve1": reserve1,
                "fee": 0.003 if dex == "uniswap_v2" else 0.0025,
                "dex": dex
            }
            
            pool_id += 1
    
    # Add some additional random pairs for scale
    import random
    for _ in range(100):  # Add 100 more random pairs
        token0 = random.choice(token_addresses)
        token1 = random.choice(token_addresses)
        if token0 != token1:
            pool_addr = f"0xPool{pool_id:040x}"
            dex = random.choice(dex_protocols)
            
            pools[pool_addr] = {
                "token0": token0,
                "token1": token1,
                "reserve0": random.uniform(100, 10000),
                "reserve1": random.uniform(100, 10000),
                "fee": random.choice([0.003, 0.0025, 0.005]),
                "dex": dex
            }
            
            pool_id += 1
    
    logger.info(f"Generated test data: {len(tokens)} tokens, {len(pools)} pools")
    return tokens, pools

def test_baseline_performance(tokens: Dict, pools: Dict) -> Dict:
    """Test baseline performance without optimizations"""
    logger.info("Testing baseline performance...")
    
    start_time = time.time()
    
    # Create standard graph
    graph = DeFiMarketGraph()
    
    # Add tokens
    for address, data in tokens.items():
        graph.add_token(address, data["symbol"])
    
    # Add trading pairs
    for pool_addr, data in pools.items():
        graph.add_trading_pair(
            data["token0"], data["token1"], data["dex"],
            pool_addr, data["reserve0"], data["reserve1"], data["fee"]
        )
    
    build_time = time.time() - start_time
    
    # Get graph stats
    stats = graph.get_graph_stats()
    
    result = {
        "build_time": build_time,
        "nodes": stats["nodes"],
        "edges": stats["edges"],
        "edges_per_second": stats["edges"] / build_time if build_time > 0 else 0,
        "multi_graph_stats": stats["multi_graph"]
    }
    
    logger.info(f"Baseline performance: {build_time:.3f}s, {result['edges_per_second']:.0f} edges/s")
    return result

def test_optimized_performance(tokens: Dict, pools: Dict) -> Dict:
    """Test optimized performance"""
    logger.info("Testing optimized performance...")
    
    # Use the optimizer
    optimized_graph, performance_report = create_optimized_graph(
        tokens, pools, target_time=6.43, use_memory_efficient=False
    )
    
    # Get graph stats
    stats = optimized_graph.get_graph_stats()
    
    result = {
        "build_time": performance_report["performance_metrics"]["graph_build_time"],
        "nodes": stats["nodes"],
        "edges": stats["edges"],
        "edges_per_second": performance_report["performance_metrics"]["edge_addition_rate"],
        "memory_usage_mb": performance_report["performance_metrics"]["memory_usage_mb"],
        "optimization_gains": performance_report["optimization_gains"],
        "paper_compliance": performance_report["paper_compliance"],
        "bottlenecks": performance_report["bottleneck_analysis"]
    }
    
    logger.info(f"Optimized performance: {result['build_time']:.3f}s, "
               f"{result['edges_per_second']:.0f} edges/s")
    return result

def validate_optimization_success(baseline: Dict, optimized: Dict) -> Dict:
    """Validate that optimization was successful"""
    
    # Calculate improvements
    time_improvement = max(0, baseline["build_time"] - optimized["build_time"])
    time_improvement_percent = (time_improvement / baseline["build_time"]) * 100 if baseline["build_time"] > 0 else 0
    
    speed_improvement = (optimized["edges_per_second"] / baseline["edges_per_second"]) if baseline["edges_per_second"] > 0 else 1
    
    # Define success criteria
    criteria = {
        "paper_target_achieved": optimized["build_time"] <= 6.43,  # Paper target
        "significant_improvement": time_improvement_percent >= 20,  # At least 20% improvement
        "high_throughput": optimized["edges_per_second"] >= 1000,  # At least 1000 edges/sec
        "reasonable_memory": optimized.get("memory_usage_mb", 0) <= 500  # Under 500MB
    }
    
    all_criteria_met = all(criteria.values())
    
    validation_result = {
        "success": all_criteria_met,
        "criteria": criteria,
        "improvements": {
            "time_saved_seconds": time_improvement,
            "time_improvement_percent": time_improvement_percent,
            "speed_multiplier": speed_improvement,
            "paper_compliance": optimized.get("paper_compliance", {})
        },
        "performance_comparison": {
            "baseline": baseline,
            "optimized": optimized
        }
    }
    
    return validation_result

def main():
    """Main test execution"""
    print("=" * 60)
    print("Graph Building Performance Optimization Test")
    print("=" * 60)
    
    try:
        # Generate test data
        print("\n1. Generating realistic test data...")
        tokens, pools = generate_realistic_test_data()
        
        # Test baseline performance
        print("\n2. Testing baseline performance...")
        baseline_result = test_baseline_performance(tokens, pools)
        
        # Test optimized performance
        print("\n3. Testing optimized performance...")
        optimized_result = test_optimized_performance(tokens, pools)
        
        # Validate optimization success
        print("\n4. Validating optimization success...")
        validation = validate_optimization_success(baseline_result, optimized_result)
        
        # Print results
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        
        print(f"\nBaseline Performance:")
        print(f"  Build Time: {baseline_result['build_time']:.3f}s")
        print(f"  Nodes: {baseline_result['nodes']}")
        print(f"  Edges: {baseline_result['edges']}")
        print(f"  Throughput: {baseline_result['edges_per_second']:.0f} edges/s")
        
        print(f"\nOptimized Performance:")
        print(f"  Build Time: {optimized_result['build_time']:.3f}s")
        print(f"  Nodes: {optimized_result['nodes']}")
        print(f"  Edges: {optimized_result['edges']}")
        print(f"  Throughput: {optimized_result['edges_per_second']:.0f} edges/s")
        print(f"  Memory Usage: {optimized_result.get('memory_usage_mb', 0):.1f}MB")
        
        print(f"\nImprovement Analysis:")
        improvements = validation["improvements"]
        print(f"  Time Saved: {improvements['time_saved_seconds']:.3f}s ({improvements['time_improvement_percent']:.1f}%)")
        print(f"  Speed Multiplier: {improvements['speed_multiplier']:.2f}x")
        
        print(f"\nPaper Compliance:")
        paper_compliance = improvements.get('paper_compliance', {})
        print(f"  Target Time: {paper_compliance.get('target_time', 6.43):.2f}s")
        print(f"  Actual Time: {paper_compliance.get('actual_time', 0):.3f}s")
        print(f"  Target Achieved: {'✅ YES' if paper_compliance.get('target_achieved', False) else '❌ NO'}")
        
        print(f"\nSuccess Criteria:")
        criteria = validation["criteria"]
        for criterion, met in criteria.items():
            print(f"  {criterion.replace('_', ' ').title()}: {'✅ PASS' if met else '❌ FAIL'}")
        
        print(f"\nOverall Result: {'🎉 OPTIMIZATION SUCCESSFUL' if validation['success'] else '⚠️  OPTIMIZATION NEEDS IMPROVEMENT'}")
        
        # Save detailed results to file
        results_file = "optimization_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "test_timestamp": time.time(),
                "baseline_performance": baseline_result,
                "optimized_performance": optimized_result,
                "validation_results": validation
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        return validation["success"]
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)