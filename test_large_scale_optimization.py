"""
Large-scale graph optimization test to validate paper requirements:
- 96 protocol actions
- 25+ assets
- 6.43 second target performance
"""

import sys
import os
import time
import json
import random
import string
from typing import Dict, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.graph_performance_optimizer import GraphBuildingOptimizer, create_optimized_graph
from src.market_graph import DeFiMarketGraph
from src.protocol_actions import ProtocolRegistry, ProtocolAction, ProtocolType
from src.logger import setup_logger

logger = setup_logger(__name__)

def generate_paper_scale_data() -> tuple[Dict, Dict, List]:
    """Generate data matching paper specifications: 96 protocol actions, 25+ assets"""
    
    # 25+ assets as per paper (extended to 70 for full ecosystem coverage)
    major_tokens = {
        # Core assets
        "0xA0b86a33E6441C6C8320bb5b5c6c3e64f4b7b6bc": {"symbol": "ETH", "decimals": 18},
        "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": {"symbol": "WETH", "decimals": 18},
        
        # Stablecoins
        "0xA0b86a33E6441C6C8320bb5b5c6c3e64f4b7b6bd": {"symbol": "USDC", "decimals": 6},
        "0xdAC17F958D2ee523a2206206994597C13D831ec7": {"symbol": "USDT", "decimals": 6},
        "0x6B175474E89094C44Da98b954EedeAC495271d0F": {"symbol": "DAI", "decimals": 18},
        "0x4Fabb145d64652a948d72533023f6E7A623C7C53": {"symbol": "BUSD", "decimals": 18},
        
        # Major tokens
        "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599": {"symbol": "WBTC", "decimals": 8},
        "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984": {"symbol": "UNI", "decimals": 18},
        "0x6B3595068778DD592e39A122f4f5a5cF09C90fE2": {"symbol": "SUSHI", "decimals": 18},
        "0xc00e94Cb662C3520282E6f5717214004A7f26888": {"symbol": "COMP", "decimals": 18},
        "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9": {"symbol": "AAVE", "decimals": 18},
        
        # DeFi ecosystem tokens
        "0xd533a949740bb3306d119CC777fa900bA034cd52": {"symbol": "CRV", "decimals": 18},
        "0xba100000625a3754423978a60c9317c58a424e3D": {"symbol": "BAL", "decimals": 18},
        "0x0bc529c00C6401aEF6D220BE8C6Ea1667F6Ad93e": {"symbol": "YFI", "decimals": 18},
        "0x9f8F72aA9304c8B593d555F12eF6589CC3A579A2": {"symbol": "MKR", "decimals": 18},
        "0x514910771AF9Ca656af840dff83E8264EcF986CA": {"symbol": "LINK", "decimals": 18},
        
        # Compound tokens (cTokens)
        "0x5d3a536E4D6DbD6114cc1Ead35777bAB948E3643": {"symbol": "cDAI", "decimals": 8},
        "0x4Ddc2D193948926D02f9B1fE9e1daa0718270ED5": {"symbol": "cETH", "decimals": 8},
        "0x39AA39c021dfbaE8faC545936693aC917d5E7563": {"symbol": "cUSDC", "decimals": 8},
        "0xf650C3d88D12dB855b8bf7D11Be6C55A4e07dCC9": {"symbol": "cUSDT", "decimals": 8},
        "0xccF4429DB6322D5C611ee964527D42E5d685DD6a": {"symbol": "cWBTC", "decimals": 8},
        
        # Aave tokens (aTokens)
        "0x028171bCA77440897B824Ca71D1c56caC55b68A3": {"symbol": "aDAI", "decimals": 18},
        "0x3Ed3B47Dd13EC9a98b44e6204A523E766B225811": {"symbol": "aUSDT", "decimals": 6},
        "0xBcca60bB61934080951369a648Fb03DF4F96263C": {"symbol": "aUSDC", "decimals": 6},
        "0x9ff58f4fFB29fA2266Ab25e75e2A8b3503311656": {"symbol": "aWBTC", "decimals": 8},
        "0x030bA81f1c18d280636F32af80b9AAd02Cf0854e": {"symbol": "aWETH", "decimals": 18},
        
        # Synthetic assets (Synthetix)
        "0x57Ab1ec28D129707052df4dF418D58a2D46d5f51": {"symbol": "sETH", "decimals": 18},
        "0x5e74C9036fb86BD7eCdcb084a0673EFc32eA31cb": {"symbol": "sUSD", "decimals": 18},
        "0xfE18be6b3Bd88A2D2A7f928d00292E7a9963CfC6": {"symbol": "sBTC", "decimals": 18},
    }
    
    # Add 42 more tokens to reach 70 total (well above paper's 25)
    additional_tokens = {}
    for i in range(42):
        address = f"0x{''.join(random.choices(string.hexdigits.lower(), k=40))}"
        additional_tokens[address] = {
            "symbol": f"TOKEN{i:02d}",
            "decimals": random.choice([6, 8, 18])
        }
    
    all_tokens = {**major_tokens, **additional_tokens}
    
    # Generate large-scale pool data (targeting thousands of pools for realistic load)
    pools = {}
    token_addresses = list(all_tokens.keys())
    
    # Major protocols with realistic distribution
    protocols = [
        ("uniswap_v2", 0.25, 0.003),
        ("uniswap_v3", 0.20, 0.003),
        ("sushiswap", 0.15, 0.0025),
        ("curve", 0.12, 0.0004),
        ("balancer", 0.08, 0.005),
        ("bancor", 0.05, 0.002),
        ("kyber", 0.05, 0.002),
        ("1inch", 0.04, 0.003),
        ("dydx", 0.03, 0.001),
        ("0x", 0.03, 0.0015),
    ]
    
    pool_id = 1
    target_pools = 5000  # Large scale for realistic testing
    
    # Create pools with realistic distribution
    for protocol, weight, base_fee in protocols:
        num_pools = int(target_pools * weight)
        
        for _ in range(num_pools):
            token0 = random.choice(token_addresses)
            token1 = random.choice(token_addresses)
            
            if token0 != token1:
                pool_addr = f"0x{protocol[:6]}{pool_id:034x}"
                
                # Generate realistic reserves based on protocol and tokens
                if protocol == "curve" and all_tokens[token0]["symbol"] in ["USDC", "USDT", "DAI", "BUSD"]:
                    # Stablecoin pools have high liquidity
                    reserve0 = random.uniform(5_000_000, 50_000_000)
                    reserve1 = random.uniform(5_000_000, 50_000_000)
                elif protocol in ["uniswap_v2", "uniswap_v3", "sushiswap"]:
                    # Major AMMs have varied liquidity
                    reserve0 = random.uniform(10_000, 1_000_000)
                    reserve1 = random.uniform(10_000, 1_000_000)
                else:
                    # Other protocols have moderate liquidity
                    reserve0 = random.uniform(1_000, 100_000)
                    reserve1 = random.uniform(1_000, 100_000)
                
                pools[pool_addr] = {
                    "token0": token0,
                    "token1": token1,
                    "reserve0": reserve0,
                    "reserve1": reserve1,
                    "fee": base_fee + random.uniform(-0.0005, 0.0005),
                    "dex": protocol
                }
                
                pool_id += 1
    
    # Generate 96 protocol actions (paper requirement)
    protocol_actions = []
    action_id = 1
    
    # Define action types per protocol
    protocol_action_types = {
        ProtocolType.AMM: ["swap", "add_liquidity", "remove_liquidity"],
        ProtocolType.LENDING: ["lend", "borrow", "repay", "liquidate"],
        ProtocolType.CDP: ["mint", "burn", "liquidate"],
        ProtocolType.DERIVATIVE: ["open_position", "close_position", "settle"],
        ProtocolType.YIELD_FARMING: ["stake", "unstake", "claim_rewards"],
        ProtocolType.STAKING: ["stake", "unstake", "claim_rewards"],
    }
    
    protocol_contracts = {
        "uniswap_v2": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
        "uniswap_v3": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "sushiswap": "0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac",
        "curve": "0x79a8C46DeA5aDa233ABaFFD40F3A0A2B1e5A4F27",
        "balancer": "0xBA12222222228d8Ba445958a75a0704d566BF2C8",
        "compound": "0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B",
        "aave": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
        "makerdao": "0x9f8F72aA9304c8B593d555F12eF6589CC3A579A2",
        "synthetix": "0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F",
        "yearn": "0x0bc529c00C6401aEF6D220BE8C6Ea1667F6Ad93e",
    }
    
    # Generate exactly 96 protocol actions
    protocols_list = list(protocol_contracts.keys())
    actions_per_protocol = 96 // len(protocols_list)
    remaining_actions = 96 % len(protocols_list)
    
    for i, protocol in enumerate(protocols_list):
        num_actions = actions_per_protocol
        if i < remaining_actions:
            num_actions += 1
        
        # Determine protocol type
        if protocol in ["uniswap_v2", "uniswap_v3", "sushiswap", "curve", "balancer"]:
            protocol_type = ProtocolType.AMM
        elif protocol in ["compound", "aave"]:
            protocol_type = ProtocolType.LENDING
        elif protocol == "makerdao":
            protocol_type = ProtocolType.CDP
        elif protocol == "synthetix":
            protocol_type = ProtocolType.DERIVATIVE
        else:
            protocol_type = ProtocolType.STAKING
        
        action_types = protocol_action_types[protocol_type]
        
        for j in range(num_actions):
            action_type = action_types[j % len(action_types)]
            
            protocol_actions.append(ProtocolAction(
                action_id=f"{protocol}_{action_type}_{j}",
                protocol_name=protocol,
                protocol_type=protocol_type,
                action_type=action_type,
                contract_address=protocol_contracts[protocol],
                function_name=action_type,
                input_tokens=["token0"],
                output_tokens=["token1"],
                gas_estimate=random.randint(50000, 300000),
                fee_rate=random.uniform(0.0001, 0.01),
                min_liquidity=1.0,
                abi_fragment={},
                is_active=True
            ))
            
            action_id += 1
    
    logger.info(f"Generated paper-scale data: {len(all_tokens)} tokens, "
               f"{len(pools)} pools, {len(protocol_actions)} protocol actions")
    
    return all_tokens, pools, protocol_actions

def test_paper_scale_performance() -> Dict:
    """Test performance at paper scale"""
    print("\n" + "="*60)
    print("PAPER SCALE PERFORMANCE TEST")
    print("="*60)
    print("Generating paper-scale test data...")
    print("Target: 96 protocol actions, 25+ assets, <6.43s build time")
    
    # Generate paper-scale data
    tokens, pools, protocol_actions = generate_paper_scale_data()
    
    print(f"Generated: {len(tokens)} tokens, {len(pools)} pools, {len(protocol_actions)} protocol actions")
    
    # Test baseline (without optimization)
    print(f"\n1. Testing baseline performance...")
    baseline_start = time.time()
    
    baseline_graph = DeFiMarketGraph()
    
    # Add tokens
    for address, data in tokens.items():
        baseline_graph.add_token(address, data["symbol"])
    
    # Add pools (sample subset to avoid excessive baseline time)
    sample_pools = dict(list(pools.items())[:1000])  # Sample 1000 pools for baseline
    for pool_addr, data in sample_pools.items():
        baseline_graph.add_trading_pair(
            data["token0"], data["token1"], data["dex"],
            pool_addr, data["reserve0"], data["reserve1"], data["fee"]
        )
    
    baseline_time = time.time() - baseline_start
    baseline_stats = baseline_graph.get_graph_stats()
    
    print(f"  Baseline (1000 pools): {baseline_time:.3f}s")
    print(f"  Extrapolated full scale: {(baseline_time * len(pools) / len(sample_pools)):.1f}s")
    
    # Test optimized performance (full scale)
    print(f"\n2. Testing optimized performance (full scale)...")
    
    optimizer = GraphBuildingOptimizer(target_build_time=6.43)
    optimized_start = time.time()
    
    optimized_graph = DeFiMarketGraph()
    metrics = optimizer.optimize_graph_building(optimized_graph, tokens, pools, protocol_actions)
    
    optimized_time = time.time() - optimized_start
    optimized_stats = optimized_graph.get_graph_stats()
    
    print(f"  Optimized (full scale): {optimized_time:.3f}s")
    
    # Validate paper compliance
    paper_compliance = {
        "target_time": 6.43,
        "actual_time": optimized_time,
        "target_achieved": optimized_time <= 6.43,
        "protocol_actions": len(protocol_actions),
        "assets": len(tokens),
        "protocol_target_met": len(protocol_actions) >= 96,
        "assets_target_met": len(tokens) >= 25
    }
    
    return {
        "baseline": {
            "time": baseline_time,
            "extrapolated_time": baseline_time * len(pools) / len(sample_pools),
            "stats": baseline_stats,
            "sample_size": len(sample_pools)
        },
        "optimized": {
            "time": optimized_time,
            "stats": optimized_stats,
            "metrics": metrics,
            "full_scale": True
        },
        "paper_compliance": paper_compliance,
        "test_scale": {
            "tokens": len(tokens),
            "pools": len(pools),
            "protocol_actions": len(protocol_actions)
        }
    }

def main():
    """Main test execution"""
    try:
        # Run paper-scale performance test
        results = test_paper_scale_performance()
        
        print(f"\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        # Paper compliance
        compliance = results["paper_compliance"]
        print(f"\nPaper Compliance:")
        print(f"  Protocol Actions: {compliance['protocol_actions']}/96 {'✅' if compliance['protocol_target_met'] else '❌'}")
        print(f"  Assets: {compliance['assets']}/25 {'✅' if compliance['assets_target_met'] else '❌'}")
        print(f"  Build Time Target: {compliance['actual_time']:.3f}s/{compliance['target_time']}s {'✅' if compliance['target_achieved'] else '❌'}")
        
        # Performance comparison
        baseline = results["baseline"]
        optimized = results["optimized"]
        
        print(f"\nPerformance Comparison:")
        print(f"  Baseline (extrapolated): {baseline['extrapolated_time']:.1f}s")
        print(f"  Optimized (actual): {optimized['time']:.3f}s")
        
        improvement = max(0, baseline['extrapolated_time'] - optimized['time'])
        improvement_pct = (improvement / baseline['extrapolated_time']) * 100 if baseline['extrapolated_time'] > 0 else 0
        
        print(f"  Improvement: {improvement:.1f}s ({improvement_pct:.1f}%)")
        print(f"  Speed multiplier: {baseline['extrapolated_time'] / optimized['time']:.1f}x")
        
        # Graph statistics
        print(f"\nGraph Statistics (Optimized):")
        opt_stats = optimized["stats"]
        print(f"  Nodes: {opt_stats['nodes']}")
        print(f"  Edges: {opt_stats['edges']}")
        print(f"  Multi-graph pairs: {opt_stats['multi_graph']['multi_dex_pairs']}")
        print(f"  Density: {opt_stats['density']:.4f}")
        
        # Success determination
        success = (
            compliance['target_achieved'] and 
            compliance['protocol_target_met'] and 
            compliance['assets_target_met'] and
            improvement_pct > 50  # At least 50% improvement
        )
        
        print(f"\nOverall Result: {'🎉 SUCCESS - Paper targets achieved!' if success else '⚠️  Needs improvement'}")
        
        # Save results
        results_file = f"paper_scale_test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {results_file}")
        
        return success
        
    except Exception as e:
        logger.error(f"Paper scale test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)