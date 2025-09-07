import math
import asyncio
from typing import List, Tuple, Optional
from src.market_graph import TradingEdge, ArbitrageOpportunity
from src.logger import setup_logger

logger = setup_logger(__name__)

class LocalSearch:
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def optimize_arbitrage_opportunity(self, opportunity: ArbitrageOpportunity) -> ArbitrageOpportunity:
        """
        Optimize an arbitrage opportunity using local search (hill climbing).
        This implements the "perform a local search and repeat" step from the paper.
        """
        # Get the best revenue transaction through parameter optimization
        optimized_opportunity = self._hill_climbing_search(opportunity)
        return optimized_opportunity
    
    def _hill_climbing_search(self, opportunity: ArbitrageOpportunity) -> ArbitrageOpportunity:
        """
        Perform hill climbing to optimize the trade amount for maximum profit.
        This is the core of the local search algorithm described in the paper.
        """
        # Start with an initial trade amount (10% of minimum liquidity)
        current_amount = min(edge.liquidity for edge in opportunity.edges) * 0.1
        current_profit = self._calculate_profit_for_amount(opportunity, current_amount)
        
        best_amount = current_amount
        best_profit = current_profit
        
        # Hill climbing iterations
        for i in range(self.max_iterations):
            # Try different step sizes
            step_sizes = [current_amount * 0.1, current_amount * 0.01, current_amount * 0.001]
            
            improved = False
            for step in step_sizes:
                # Try increasing the amount
                new_amount_up = current_amount + step
                new_profit_up = self._calculate_profit_for_amount(opportunity, new_amount_up)
                
                if new_profit_up > current_profit and new_profit_up > best_profit:
                    best_amount = new_amount_up
                    best_profit = new_profit_up
                    current_amount = new_amount_up
                    current_profit = new_profit_up
                    improved = True
                    break
                
                # Try decreasing the amount
                new_amount_down = max(0.001, current_amount - step)  # Minimum amount
                new_profit_down = self._calculate_profit_for_amount(opportunity, new_amount_down)
                
                if new_profit_down > current_profit and new_profit_down > best_profit:
                    best_amount = new_amount_down
                    best_profit = new_profit_down
                    current_amount = new_amount_down
                    current_profit = new_profit_down
                    improved = True
                    break
            
            # If no improvement, reduce step size or stop
            if not improved:
                # Check if we've converged
                if abs(current_profit - best_profit) < self.tolerance:
                    break
        
        # Create optimized opportunity
        return self._create_optimized_opportunity(opportunity, best_amount)
    
    def _calculate_profit_for_amount(self, opportunity: ArbitrageOpportunity, amount: float) -> float:
        """
        Calculate the net profit for a given trade amount.
        """
        # Check liquidity constraints
        for edge in opportunity.edges:
            if amount > edge.liquidity:
                return -float('inf')  # Invalid amount
        
        # Simulate the trade through all edges
        current_amount = amount
        total_gas_cost = 0
        
        for edge in opportunity.edges:
            # Apply the exchange rate
            received_amount = current_amount * edge.exchange_rate
            
            # Deduct fee
            received_amount = received_amount * (1 - edge.fee)
            
            # Update gas cost
            total_gas_cost += edge.gas_cost
            
            # Update amount for next iteration
            current_amount = received_amount
        
        # Calculate profit
        net_profit = current_amount - amount - total_gas_cost
        return net_profit
    
    def _create_optimized_opportunity(self, opportunity: ArbitrageOpportunity, optimal_amount: float) -> ArbitrageOpportunity:
        """
        Create a new arbitrage opportunity with optimized parameters.
        """
        # Recalculate all parameters based on optimal amount
        current_amount = optimal_amount
        total_gas_cost = 0
        total_fee = 0
        product_ratio = 1.0
        
        new_edges = []
        
        for edge in opportunity.edges:
            # Update edge with actual traded amount
            new_edge = TradingEdge(
                from_token=edge.from_token,
                to_token=edge.to_token,
                dex=edge.dex,
                pool_address=edge.pool_address,
                exchange_rate=edge.exchange_rate,
                liquidity=edge.liquidity,
                fee=edge.fee,
                gas_cost=edge.gas_cost,
                weight=edge.weight
            )
            
            # Update calculations
            received_amount = current_amount * edge.exchange_rate
            received_amount = received_amount * (1 - edge.fee)
            
            total_gas_cost += edge.gas_cost
            total_fee += edge.fee
            product_ratio *= edge.exchange_rate
            
            new_edges.append(new_edge)
            current_amount = received_amount
        
        # Calculate final parameters
        estimated_profit = current_amount - optimal_amount
        net_profit = estimated_profit - total_gas_cost
        required_capital = optimal_amount
        
        # Update confidence (might change based on optimal amount)
        confidence = self._calculate_confidence(new_edges, optimal_amount)
        
        return ArbitrageOpportunity(
            path=opportunity.path,
            edges=new_edges,
            profit_ratio=product_ratio,
            required_capital=required_capital,
            estimated_profit=estimated_profit,
            gas_cost=total_gas_cost,
            net_profit=net_profit,
            confidence=confidence
        )
    
    def _calculate_confidence(self, edges: List[TradingEdge], trade_amount: float) -> float:
        """
        Calculate confidence score based on trade amount and liquidity.
        """
        # Liquidity-based confidence
        min_liquidity = min(edge.liquidity for edge in edges)
        liquidity_ratio = min(trade_amount / min_liquidity, 1.0)
        liquidity_score = max(0.0, 1.0 - liquidity_ratio)  # Higher score for smaller trades relative to liquidity
        
        # Path length confidence (shorter is better)
        path_score = max(0.5, 1.0 - (len(edges) - 2) * 0.1)
        
        # DEX diversity confidence
        unique_dexes = len(set(edge.dex for edge in edges))
        diversity_score = min(unique_dexes / len(edges), 1.0)
        
        return (liquidity_score * 0.5 + path_score * 0.3 + diversity_score * 0.2)
    
    async def _optimize_single_start_point(self, opportunity: ArbitrageOpportunity, amount: float) -> ArbitrageOpportunity:
        """
        Optimize from a single starting point asynchronously.
        """
        try:
            # Create a temporary opportunity with this starting amount
            temp_opportunity = self._create_optimized_opportunity(opportunity, amount)
            
            # Optimize from this starting point
            optimized = self._hill_climbing_search(temp_opportunity)
            
            return optimized
        except Exception as e:
            logger.error(f"Error optimizing from start point {amount}: {e}")
            # Return original opportunity if optimization fails
            return opportunity
    
    async def multi_start_search_async(self, opportunity: ArbitrageOpportunity, 
                                     start_points: int = 5) -> ArbitrageOpportunity:
        """
        Perform local search from multiple starting points concurrently and return the best result.
        This implements the "Multiple starting points" requirement from the paper with parallel processing.
        """
        try:
            # Try different starting amounts
            min_liquidity = min(edge.liquidity for edge in opportunity.edges)
            start_amounts = [
                min_liquidity * 0.01,  # 1% of min liquidity
                min_liquidity * 0.05,  # 5% of min liquidity
                min_liquidity * 0.1,   # 10% of min liquidity
                min_liquidity * 0.2,   # 20% of min liquidity
                min_liquidity * 0.5    # 50% of min liquidity
            ]
            
            # Limit to requested number of start points
            start_amounts = start_amounts[:start_points]
            
            # Create tasks for concurrent optimization
            tasks = [
                self._optimize_single_start_point(opportunity, amount)
                for amount in start_amounts
            ]
            
            # Execute all optimizations concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and find the best result
            valid_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Local search task failed: {result}")
                elif isinstance(result, ArbitrageOpportunity):
                    valid_results.append(result)
            
            # Return the best result or original if all failed
            if valid_results:
                best_opportunity = max(valid_results, key=lambda x: x.net_profit)
                return best_opportunity
            else:
                logger.warning("All local search tasks failed, returning original opportunity")
                return opportunity
                
        except Exception as e:
            logger.error(f"Error in multi-start search: {e}")
            return opportunity
    
    def multi_start_search(self, opportunity: ArbitrageOpportunity, 
                          start_points: int = 5) -> ArbitrageOpportunity:
        """
        Perform local search from multiple starting points and return the best result.
        This implements the "Multiple starting points" requirement from the paper.
        """
        # For backward compatibility, we can still use the synchronous version
        best_opportunity = opportunity
        best_profit = opportunity.net_profit
        
        # Try different starting amounts
        min_liquidity = min(edge.liquidity for edge in opportunity.edges)
        start_amounts = [
            min_liquidity * 0.01,  # 1% of min liquidity
            min_liquidity * 0.05,  # 5% of min liquidity
            min_liquidity * 0.1,   # 10% of min liquidity
            min_liquidity * 0.2,   # 20% of min liquidity
            min_liquidity * 0.5    # 50% of min liquidity
        ]
        
        for amount in start_amounts[:start_points]:
            # Create a temporary opportunity with this starting amount
            temp_opportunity = self._create_optimized_opportunity(opportunity, amount)
            
            # Optimize from this starting point
            optimized = self._hill_climbing_search(temp_opportunity)
            
            # Keep track of the best result
            if optimized.net_profit > best_profit:
                best_profit = optimized.net_profit
                best_opportunity = optimized
        
        return best_opportunity