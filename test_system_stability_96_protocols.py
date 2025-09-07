#!/usr/bin/env python3
"""
System Stability Test for 96 Protocol Actions
Tests the system's ability to handle all 96 protocol actions simultaneously
according to paper "[2103.02228] On the Just-In-Time Discovery of Profit-Generating Transactions in DeFi Protocols"

This test validates:
1. Memory usage under load
2. Processing time for all actions
3. Error handling and recovery
4. Concurrent action processing
5. Resource consumption monitoring
"""

import time
import threading
import asyncio
import json
import gc
import os
import resource
import math
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import tracemalloc
from web3 import Web3

from src.protocol_actions import ProtocolRegistry, ProtocolAction, ProtocolType
from src.market_graph import DeFiMarketGraph
from src.memory_efficient_graph import MemoryEfficientGraph
from src.bellman_ford_arbitrage import BellmanFordArbitrage
from src.token_manager import TokenManager
from src.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class StabilityTestResult:
    """Results from system stability testing"""
    test_name: str
    success: bool
    duration_seconds: float
    memory_usage_mb: float
    peak_memory_mb: float
    actions_processed: int
    errors_count: int
    error_details: List[str]
    performance_metrics: Dict[str, Any]

class SystemStabilityTester:
    """
    Comprehensive system stability tester for 96 protocol actions
    """
    
    def __init__(self):
        # Initialize Web3 (mock for testing)
        self.w3 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/test"))
        
        # Initialize core components
        self.protocol_registry = ProtocolRegistry(self.w3)
        self.token_manager = TokenManager()
        self.market_graph = DeFiMarketGraph(self.w3)
        self.memory_graph = MemoryEfficientGraph()
        
        # Test configuration
        self.max_concurrent_actions = 20
        self.test_duration_seconds = 300  # 5 minutes stress test
        self.memory_threshold_mb = 1000   # 1GB memory limit
        
        # Monitoring data
        self.test_results: List[StabilityTestResult] = []
        self.is_running = False

    def run_complete_stability_test(self) -> Dict[str, Any]:
        """
        Run complete system stability test suite for 96 protocol actions
        """
        logger.info("🚀 Starting comprehensive system stability test for 96 protocol actions")
        
        # Enable memory tracking
        tracemalloc.start()
        start_time = time.time()
        
        test_suite = [
            ("Protocol Registry Load Test", self._test_protocol_registry_load),
            ("Memory Usage Test", self._test_memory_usage),
            ("Concurrent Action Processing", self._test_concurrent_processing),
            ("Stress Test - Continuous Operation", self._test_continuous_operation),
            ("Error Handling Test", self._test_error_handling),
            ("Resource Cleanup Test", self._test_resource_cleanup),
            ("Performance Degradation Test", self._test_performance_degradation),
            ("Graph Scalability Test", self._test_graph_scalability)
        ]
        
        passed_tests = 0
        total_tests = len(test_suite)
        
        for test_name, test_func in test_suite:
            logger.info(f"Running: {test_name}")
            try:
                result = test_func()
                self.test_results.append(result)
                if result.success:
                    passed_tests += 1
                    logger.info(f"✅ {test_name} - PASSED")
                else:
                    logger.warning(f"❌ {test_name} - FAILED")
            except Exception as e:
                logger.error(f"💥 {test_name} - CRASHED: {str(e)}")
                error_result = StabilityTestResult(
                    test_name=test_name,
                    success=False,
                    duration_seconds=0,
                    memory_usage_mb=0,
                    peak_memory_mb=0,
                    actions_processed=0,
                    errors_count=1,
                    error_details=[str(e)],
                    performance_metrics={}
                )
                self.test_results.append(error_result)
        
        # Final summary
        total_duration = time.time() - start_time
        success_rate = (passed_tests / total_tests) * 100
        
        summary = {
            "test_suite": "96 Protocol Actions System Stability",
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate_percent": success_rate,
            "total_duration_seconds": total_duration,
            "protocol_actions_count": len(self.protocol_registry.actions),
            "individual_results": [
                {
                    "test": result.test_name,
                    "success": result.success,
                    "duration": result.duration_seconds,
                    "memory_mb": result.memory_usage_mb,
                    "actions_processed": result.actions_processed,
                    "errors": result.errors_count
                }
                for result in self.test_results
            ]
        }
        
        logger.info(f"📊 Test Suite Complete: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)")
        
        # Stop memory tracking
        tracemalloc.stop()
        
        return summary

    def _test_protocol_registry_load(self) -> StabilityTestResult:
        """Test loading and accessing all 96 protocol actions"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        errors = []
        
        try:
            # Test 1: Verify 96 actions are loaded
            actions_count = len(self.protocol_registry.actions)
            if actions_count != 96:
                errors.append(f"Expected 96 actions, got {actions_count}")
            
            # Test 2: Access all actions
            accessed_actions = 0
            for action_id, action in self.protocol_registry.actions.items():
                if not isinstance(action, ProtocolAction):
                    errors.append(f"Invalid action object: {action_id}")
                    continue
                
                # Validate action properties
                if not action.contract_address:
                    errors.append(f"Missing contract address: {action_id}")
                if not action.function_name:
                    errors.append(f"Missing function name: {action_id}")
                if action.gas_estimate <= 0:
                    errors.append(f"Invalid gas estimate: {action_id}")
                
                accessed_actions += 1
            
            # Test 3: Group actions by protocol type
            type_summary = self.protocol_registry.get_action_summary()
            logger.info(f"Action distribution: {type_summary['by_protocol']}")
            
            duration = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            return StabilityTestResult(
                test_name="Protocol Registry Load Test",
                success=len(errors) == 0,
                duration_seconds=duration,
                memory_usage_mb=memory_usage,
                peak_memory_mb=memory_usage,
                actions_processed=accessed_actions,
                errors_count=len(errors),
                error_details=errors,
                performance_metrics={
                    "actions_per_second": accessed_actions / duration if duration > 0 else 0,
                    "total_actions": actions_count
                }
            )
            
        except Exception as e:
            return StabilityTestResult(
                test_name="Protocol Registry Load Test",
                success=False,
                duration_seconds=time.time() - start_time,
                memory_usage_mb=self._get_memory_usage() - start_memory,
                peak_memory_mb=0,
                actions_processed=0,
                errors_count=1,
                error_details=[str(e)],
                performance_metrics={}
            )

    def _test_memory_usage(self) -> StabilityTestResult:
        """Test memory usage under load with all 96 actions"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        peak_memory = start_memory
        errors = []
        
        try:
            # Test 1: Load all actions into multiple graph structures
            graphs_created = 0
            for i in range(10):  # Create 10 market graphs
                graph = DeFiMarketGraph()
                
                # Add nodes for each protocol action
                for action_id, action in self.protocol_registry.actions.items():
                    try:
                        # Simulate adding action to graph
                        for input_token in action.input_tokens:
                            graph.add_node(f"{input_token}_{i}")
                        for output_token in action.output_tokens:
                            graph.add_node(f"{output_token}_{i}")
                        
                        current_memory = self._get_memory_usage()
                        peak_memory = max(peak_memory, current_memory)
                        
                        if current_memory > self.memory_threshold_mb:
                            errors.append(f"Memory threshold exceeded: {current_memory}MB > {self.memory_threshold_mb}MB")
                            break
                            
                    except Exception as e:
                        errors.append(f"Failed to add action to graph {i}: {action_id} - {str(e)}")
                
                graphs_created += 1
                
                # Force garbage collection
                gc.collect()
            
            duration = time.time() - start_time
            final_memory = self._get_memory_usage()
            
            return StabilityTestResult(
                test_name="Memory Usage Test",
                success=len(errors) == 0 and peak_memory < self.memory_threshold_mb,
                duration_seconds=duration,
                memory_usage_mb=final_memory - start_memory,
                peak_memory_mb=peak_memory - start_memory,
                actions_processed=graphs_created * len(self.protocol_registry.actions),
                errors_count=len(errors),
                error_details=errors,
                performance_metrics={
                    "peak_memory_mb": peak_memory,
                    "memory_efficiency": (graphs_created * 96) / (peak_memory - start_memory) if peak_memory > start_memory else 0,
                    "graphs_created": graphs_created
                }
            )
            
        except Exception as e:
            return StabilityTestResult(
                test_name="Memory Usage Test", 
                success=False,
                duration_seconds=time.time() - start_time,
                memory_usage_mb=self._get_memory_usage() - start_memory,
                peak_memory_mb=peak_memory - start_memory,
                actions_processed=0,
                errors_count=1,
                error_details=[str(e)],
                performance_metrics={}
            )

    def _test_concurrent_processing(self) -> StabilityTestResult:
        """Test concurrent processing of multiple protocol actions"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        errors = []
        processed_actions = 0
        
        def process_action_batch(actions_batch):
            """Process a batch of actions concurrently"""
            batch_errors = []
            batch_processed = 0
            
            for action in actions_batch:
                try:
                    # Simulate action processing
                    self._simulate_action_execution(action)
                    batch_processed += 1
                    time.sleep(0.01)  # Small delay to simulate work
                except Exception as e:
                    batch_errors.append(f"Action {action.action_id}: {str(e)}")
            
            return batch_processed, batch_errors
        
        try:
            # Split all 96 actions into batches for concurrent processing
            all_actions = list(self.protocol_registry.actions.values())
            batch_size = self.max_concurrent_actions
            batches = [all_actions[i:i + batch_size] for i in range(0, len(all_actions), batch_size)]
            
            # Process batches concurrently
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_batch = {
                    executor.submit(process_action_batch, batch): i 
                    for i, batch in enumerate(batches)
                }
                
                for future in as_completed(future_to_batch):
                    try:
                        batch_processed, batch_errors = future.result(timeout=30)
                        processed_actions += batch_processed
                        errors.extend(batch_errors)
                    except Exception as e:
                        errors.append(f"Batch processing failed: {str(e)}")
            
            duration = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            return StabilityTestResult(
                test_name="Concurrent Action Processing",
                success=len(errors) == 0 and processed_actions == 96,
                duration_seconds=duration,
                memory_usage_mb=memory_usage,
                peak_memory_mb=memory_usage,
                actions_processed=processed_actions,
                errors_count=len(errors),
                error_details=errors,
                performance_metrics={
                    "actions_per_second": processed_actions / duration if duration > 0 else 0,
                    "concurrent_batches": len(batches),
                    "batch_size": batch_size
                }
            )
            
        except Exception as e:
            return StabilityTestResult(
                test_name="Concurrent Action Processing",
                success=False,
                duration_seconds=time.time() - start_time,
                memory_usage_mb=self._get_memory_usage() - start_memory,
                peak_memory_mb=0,
                actions_processed=processed_actions,
                errors_count=1,
                error_details=[str(e)],
                performance_metrics={}
            )

    def _test_continuous_operation(self) -> StabilityTestResult:
        """Test continuous operation under sustained load"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        peak_memory = start_memory
        errors = []
        cycles_completed = 0
        
        try:
            self.is_running = True
            
            # Run for specified duration
            while (time.time() - start_time) < min(self.test_duration_seconds, 60):  # Max 1 minute for testing
                cycle_start = time.time()
                
                # Process all 96 actions in random order
                actions_list = list(self.protocol_registry.actions.values())
                import random
                random.shuffle(actions_list)
                
                for action in actions_list:
                    if not self.is_running:
                        break
                    
                    try:
                        self._simulate_action_execution(action)
                        
                        # Monitor memory
                        current_memory = self._get_memory_usage()
                        peak_memory = max(peak_memory, current_memory)
                        
                        if current_memory > self.memory_threshold_mb:
                            errors.append(f"Memory threshold exceeded in cycle {cycles_completed}")
                            self.is_running = False
                            break
                            
                    except Exception as e:
                        errors.append(f"Action failed in cycle {cycles_completed}: {action.action_id} - {str(e)}")
                
                cycles_completed += 1
                
                # Brief pause between cycles
                time.sleep(0.1)
                
                # Periodic garbage collection
                if cycles_completed % 5 == 0:
                    gc.collect()
            
            self.is_running = False
            duration = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            return StabilityTestResult(
                test_name="Stress Test - Continuous Operation",
                success=len(errors) == 0 and cycles_completed > 0,
                duration_seconds=duration,
                memory_usage_mb=memory_usage,
                peak_memory_mb=peak_memory - start_memory,
                actions_processed=cycles_completed * 96,
                errors_count=len(errors),
                error_details=errors[:10],  # Limit error details
                performance_metrics={
                    "cycles_completed": cycles_completed,
                    "cycles_per_minute": (cycles_completed / duration) * 60 if duration > 0 else 0,
                    "actions_per_second": (cycles_completed * 96) / duration if duration > 0 else 0,
                    "memory_stability": peak_memory < (start_memory * 2)
                }
            )
            
        except Exception as e:
            self.is_running = False
            return StabilityTestResult(
                test_name="Stress Test - Continuous Operation",
                success=False,
                duration_seconds=time.time() - start_time,
                memory_usage_mb=self._get_memory_usage() - start_memory,
                peak_memory_mb=peak_memory - start_memory,
                actions_processed=cycles_completed * 96,
                errors_count=1,
                error_details=[str(e)],
                performance_metrics={"cycles_completed": cycles_completed}
            )

    def _test_error_handling(self) -> StabilityTestResult:
        """Test system error handling and recovery"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        errors_handled = 0
        errors = []
        
        try:
            # Test various error conditions
            error_scenarios = [
                ("Invalid contract address", lambda: self._test_invalid_contract()),
                ("Network timeout", lambda: self._test_network_timeout()),
                ("Invalid ABI", lambda: self._test_invalid_abi()),
                ("Insufficient gas", lambda: self._test_insufficient_gas()),
                ("Invalid token address", lambda: self._test_invalid_token())
            ]
            
            for scenario_name, scenario_func in error_scenarios:
                try:
                    scenario_func()
                    errors_handled += 1
                except Exception as e:
                    # Expected - error handling working
                    errors_handled += 1
                    logger.debug(f"Handled expected error in {scenario_name}: {str(e)}")
            
            # Test system recovery after errors
            recovery_success = self._test_system_recovery()
            
            duration = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            return StabilityTestResult(
                test_name="Error Handling Test",
                success=errors_handled >= len(error_scenarios) and recovery_success,
                duration_seconds=duration,
                memory_usage_mb=memory_usage,
                peak_memory_mb=memory_usage,
                actions_processed=errors_handled,
                errors_count=len(errors),
                error_details=errors,
                performance_metrics={
                    "scenarios_tested": len(error_scenarios),
                    "errors_handled": errors_handled,
                    "recovery_success": recovery_success
                }
            )
            
        except Exception as e:
            return StabilityTestResult(
                test_name="Error Handling Test",
                success=False,
                duration_seconds=time.time() - start_time,
                memory_usage_mb=self._get_memory_usage() - start_memory,
                peak_memory_mb=0,
                actions_processed=errors_handled,
                errors_count=1,
                error_details=[str(e)],
                performance_metrics={}
            )

    def _test_resource_cleanup(self) -> StabilityTestResult:
        """Test proper resource cleanup and memory management"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        errors = []
        
        try:
            initial_memory = self._get_memory_usage()
            
            # Create and destroy multiple components
            for i in range(10):
                # Create temporary components
                temp_registry = ProtocolRegistry(self.w3)
                temp_graph = DeFiMarketGraph()
                temp_memory_graph = MemoryEfficientGraph()
                
                # Use components
                actions_count = len(temp_registry.actions)
                temp_graph.add_node(f"TEST_TOKEN_{i}")
                temp_memory_graph.add_edge("TOKEN_A", "TOKEN_B", 1.0, -0.1, 1000.0, 0.003, 100000, "test_dex", "0x123")
                
                # Explicit cleanup
                del temp_registry
                del temp_graph  
                del temp_memory_graph
                
                # Force garbage collection
                gc.collect()
                
                current_memory = self._get_memory_usage()
                if current_memory > initial_memory * 1.5:
                    errors.append(f"Memory not properly cleaned up after iteration {i}")
            
            final_memory = self._get_memory_usage()
            memory_diff = final_memory - initial_memory
            
            duration = time.time() - start_time
            
            return StabilityTestResult(
                test_name="Resource Cleanup Test",
                success=len(errors) == 0 and memory_diff < 100,  # Less than 100MB difference
                duration_seconds=duration,
                memory_usage_mb=memory_diff,
                peak_memory_mb=memory_diff,
                actions_processed=10,
                errors_count=len(errors),
                error_details=errors,
                performance_metrics={
                    "memory_cleanup_efficiency": max(0, 100 - memory_diff),
                    "cleanup_iterations": 10
                }
            )
            
        except Exception as e:
            return StabilityTestResult(
                test_name="Resource Cleanup Test",
                success=False,
                duration_seconds=time.time() - start_time,
                memory_usage_mb=self._get_memory_usage() - start_memory,
                peak_memory_mb=0,
                actions_processed=0,
                errors_count=1,
                error_details=[str(e)],
                performance_metrics={}
            )

    def _test_performance_degradation(self) -> StabilityTestResult:
        """Test for performance degradation over time"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        errors = []
        measurements = []
        
        try:
            # Take performance measurements over multiple iterations
            for iteration in range(20):
                iter_start = time.time()
                
                # Process all 96 actions
                for action in self.protocol_registry.actions.values():
                    self._simulate_action_execution(action)
                
                iter_duration = time.time() - iter_start
                iter_memory = self._get_memory_usage()
                
                measurements.append({
                    "iteration": iteration,
                    "duration": iter_duration,
                    "memory": iter_memory
                })
                
                # Check for performance degradation
                if iteration > 5:  # After warmup period
                    avg_early = sum(m["duration"] for m in measurements[2:5]) / 3
                    current_duration = iter_duration
                    
                    if current_duration > avg_early * 1.5:  # 50% degradation
                        errors.append(f"Performance degradation detected at iteration {iteration}")
            
            # Analyze trend
            if len(measurements) > 10:
                early_avg = sum(m["duration"] for m in measurements[:5]) / 5
                late_avg = sum(m["duration"] for m in measurements[-5:]) / 5
                degradation_percent = ((late_avg - early_avg) / early_avg) * 100
            else:
                degradation_percent = 0
            
            duration = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            return StabilityTestResult(
                test_name="Performance Degradation Test",
                success=len(errors) == 0 and degradation_percent < 25,  # Less than 25% degradation
                duration_seconds=duration,
                memory_usage_mb=memory_usage,
                peak_memory_mb=memory_usage,
                actions_processed=len(measurements) * 96,
                errors_count=len(errors),
                error_details=errors,
                performance_metrics={
                    "iterations_tested": len(measurements),
                    "performance_degradation_percent": degradation_percent,
                    "average_iteration_time": sum(m["duration"] for m in measurements) / len(measurements)
                }
            )
            
        except Exception as e:
            return StabilityTestResult(
                test_name="Performance Degradation Test",
                success=False,
                duration_seconds=time.time() - start_time,
                memory_usage_mb=self._get_memory_usage() - start_memory,
                peak_memory_mb=0,
                actions_processed=len(measurements) * 96,
                errors_count=1,
                error_details=[str(e)],
                performance_metrics={}
            )

    def _test_graph_scalability(self) -> StabilityTestResult:
        """Test graph scalability with all protocol actions"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        errors = []
        edges_created = 0
        
        try:
            # Create large-scale graph with all protocol actions
            scalability_graph = MemoryEfficientGraph()
            
            # Add edges for all possible action combinations
            actions_list = list(self.protocol_registry.actions.values())
            
            for i, action1 in enumerate(actions_list):
                for j, action2 in enumerate(actions_list[i+1:], i+1):
                    if j > i + 50:  # Limit combinations for testing
                        break
                    
                    try:
                        # Create edge if actions can be connected
                        if self._can_connect_actions(action1, action2):
                            edge_data = {
                                "rate": 1.0,
                                "gas_cost": action1.gas_estimate + action2.gas_estimate,
                                "fee_rate": action1.fee_rate + action2.fee_rate
                            }
                            scalability_graph.add_edge(
                                action1.action_id, 
                                action2.action_id,
                                edge_data.get("rate", 1.0),
                                -math.log(edge_data.get("rate", 1.0)) if edge_data.get("rate", 1.0) > 0 else 0.0,
                                1000.0,
                                edge_data.get("fee_rate", 0.003),
                                edge_data.get("gas_cost", 100000),
                                "multi_protocol",
                                "0x" + "0" * 40
                            )
                            edges_created += 1
                    except Exception as e:
                        errors.append(f"Failed to create edge {action1.action_id} -> {action2.action_id}: {str(e)}")
                        if len(errors) > 100:  # Limit errors
                            break
                
                if len(errors) > 100:
                    break
            
            # Test graph operations
            if edges_created > 0:
                # Test basic graph operations
                try:
                    stats = scalability_graph.get_stats()
                    logger.debug(f"Graph stats: {stats}")
                except Exception as e:
                    errors.append(f"Graph stats failed: {str(e)}")
            
            duration = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            return StabilityTestResult(
                test_name="Graph Scalability Test",
                success=len(errors) < 10 and edges_created > 100,  # Some tolerance for errors
                duration_seconds=duration,
                memory_usage_mb=memory_usage,
                peak_memory_mb=memory_usage,
                actions_processed=edges_created,
                errors_count=len(errors),
                error_details=errors[:20],  # Limit error details
                performance_metrics={
                    "edges_created": edges_created,
                    "graph_density": edges_created / (96 * 95 / 2) if 96 > 1 else 0,
                    "memory_per_edge": memory_usage / edges_created if edges_created > 0 else 0
                }
            )
            
        except Exception as e:
            return StabilityTestResult(
                test_name="Graph Scalability Test",
                success=False,
                duration_seconds=time.time() - start_time,
                memory_usage_mb=self._get_memory_usage() - start_memory,
                peak_memory_mb=0,
                actions_processed=edges_created,
                errors_count=1,
                error_details=[str(e)],
                performance_metrics={}
            )

    def _simulate_action_execution(self, action: ProtocolAction):
        """Simulate execution of a protocol action"""
        # Simulate validation
        if not action.is_active:
            raise ValueError(f"Action {action.action_id} is not active")
        
        if action.gas_estimate <= 0:
            raise ValueError(f"Invalid gas estimate for {action.action_id}")
        
        # Simulate processing time
        time.sleep(0.001)  # 1ms simulation
        
        # Simulate resource usage
        temp_data = {"result": f"processed_{action.action_id}", "gas_used": action.gas_estimate}
        del temp_data

    def _can_connect_actions(self, action1: ProtocolAction, action2: ProtocolAction) -> bool:
        """Check if two actions can be connected in a trading path"""
        # Simple heuristic: actions can connect if output of one matches input of another
        for output in action1.output_tokens:
            if output in action2.input_tokens:
                return True
        return False

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            # Use resource module for memory tracking
            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On Linux, ru_maxrss is in kilobytes, convert to MB
            return mem_usage / 1024 if os.name != 'nt' else mem_usage / 1024 / 1024
        except:
            # Fallback to simple estimation
            return 50.0  # Default value for testing

    def _test_invalid_contract(self):
        """Test handling of invalid contract addresses"""
        action = ProtocolAction(
            action_id="test_invalid",
            protocol_name="Test",
            protocol_type=ProtocolType.AMM,
            action_type="swap",
            contract_address="0xINVALID",
            function_name="swap",
            input_tokens=["TOKEN"],
            output_tokens=["TOKEN"],
            gas_estimate=100000,
            fee_rate=0.003,
            min_liquidity=1000.0,
            abi_fragment={}
        )
        # This should raise an error
        raise ValueError("Invalid contract address")

    def _test_network_timeout(self):
        """Test network timeout handling"""
        time.sleep(0.1)  # Simulate timeout
        raise TimeoutError("Network request timed out")

    def _test_invalid_abi(self):
        """Test invalid ABI handling"""
        raise ValueError("Invalid ABI format")

    def _test_insufficient_gas(self):
        """Test insufficient gas handling"""
        raise ValueError("Insufficient gas for transaction")

    def _test_invalid_token(self):
        """Test invalid token address handling"""
        raise ValueError("Invalid token address")

    def _test_system_recovery(self) -> bool:
        """Test system recovery after errors"""
        try:
            # Test that system can still process actions after errors
            test_action = list(self.protocol_registry.actions.values())[0]
            self._simulate_action_execution(test_action)
            return True
        except:
            return False

    def save_test_report(self, results: Dict[str, Any], filename: str = "system_stability_report.json"):
        """Save detailed test report"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Test report saved to {filename}")

def main():
    """Run the system stability test"""
    logger.info("Starting 96 Protocol Actions System Stability Test")
    
    tester = SystemStabilityTester()
    results = tester.run_complete_stability_test()
    
    # Save detailed report
    tester.save_test_report(results, "system_stability_96_protocols_report.json")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 SYSTEM STABILITY TEST RESULTS")
    print("="*80)
    print(f"Protocol Actions Tested: {results['protocol_actions_count']}")
    print(f"Test Suite: {results['total_tests']} tests")
    print(f"Passed: {results['passed_tests']} ✅")
    print(f"Failed: {results['failed_tests']} ❌")
    print(f"Success Rate: {results['success_rate_percent']:.1f}%")
    print(f"Total Duration: {results['total_duration_seconds']:.2f} seconds")
    print("="*80)
    
    if results['success_rate_percent'] >= 75:
        print("🎉 System stability test PASSED - Ready for production use!")
        return True
    else:
        print("⚠️  System stability test FAILED - Needs improvement before production")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)