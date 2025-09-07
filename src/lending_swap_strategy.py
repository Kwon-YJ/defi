"""
Lending/Borrowing + Swap Combination Trading Strategy
Implementation for DeFiPoser-ARB advanced trading strategies beyond simple arbitrage
Handles complex multi-protocol transactions: lending → borrowing → swap → repay cycles

Based on the DeFiPoser paper section 5.2: "Complex Trading Strategies"
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from decimal import Decimal, getcontext
from enum import Enum
import math

from web3 import Web3
from eth_account import Account

from src.logger import setup_logger
from src.market_graph import DeFiMarketGraph, ArbitrageOpportunity, TradingEdge
from src.protocol_actions import ProtocolRegistry, ProtocolAction, ProtocolType
from src.flash_loan_manager import FlashLoanManager, FlashLoanProvider
from src.token_manager import TokenInfo
from src.performance_analyzer import PerformanceAnalyzer

# Set high precision for financial calculations
getcontext().prec = 28

logger = setup_logger(__name__)

class StrategyType(Enum):
    """Complex strategy types"""
    LEND_BORROW_SWAP = "lend_borrow_swap"  # Deposit collateral → borrow → swap → repay
    FLASH_LEND_SWAP = "flash_lend_swap"    # Flash loan → lend → borrow different asset → swap → repay
    LEVERAGE_ARBITRAGE = "leverage_arbitrage"  # Use borrowed funds for arbitrage
    YIELD_ARBITRAGE = "yield_arbitrage"    # Exploit yield differences across protocols
    LIQUIDATION_ARBITRAGE = "liquidation_arbitrage"  # MEV from liquidations

@dataclass
class LendingPosition:
    """Lending protocol position"""
    protocol: str
    asset: str
    amount: Decimal
    interest_rate: Decimal
    collateral_factor: Decimal
    health_factor: Decimal
    timestamp: int

@dataclass  
class BorrowingPosition:
    """Borrowing protocol position"""
    protocol: str
    asset: str
    amount: Decimal
    interest_rate: Decimal
    collateral_required: Decimal
    liquidation_threshold: Decimal
    timestamp: int

@dataclass
class ComplexStrategy:
    """Complex multi-protocol trading strategy"""
    strategy_id: str
    strategy_type: StrategyType
    steps: List[Dict]  # Ordered steps: lend, borrow, swap, repay
    expected_profit: Decimal
    required_capital: Decimal
    risk_score: float  # 0-1 scale
    max_slippage: Decimal
    execution_time_estimate: float
    protocols_involved: List[str]
    assets_involved: List[str]

class LendingSwapStrategy:
    """
    Advanced DeFi strategy engine for lending/borrowing + swap combinations
    Implements complex multi-protocol transactions beyond simple arbitrage
    """
    
    def __init__(
        self,
        market_graph: DeFiMarketGraph,
        protocol_registry: ProtocolRegistry,
        web3: Web3,
        account: Account,
        flash_loan_manager: FlashLoanManager = None
    ):
        self.market_graph = market_graph
        self.protocol_registry = protocol_registry
        self.w3 = web3
        self.account = account
        self.flash_loan_manager = flash_loan_manager
        
        # Performance tracking
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Protocol-specific configurations
        self.lending_protocols = {
            'aave_v2': {
                'contract': '0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9',
                'ltv_ratios': {  # Loan-to-Value ratios
                    'WETH': 0.825,
                    'USDC': 0.87,
                    'DAI': 0.87,
                    'WBTC': 0.70
                },
                'interest_rate_model': 'variable'
            },
            'compound': {
                'contract': '0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B',
                'ltv_ratios': {
                    'WETH': 0.75,
                    'USDC': 0.75,
                    'DAI': 0.75,
                    'WBTC': 0.60
                },
                'interest_rate_model': 'fixed'
            },
            'maker_dao': {
                'contract': '0x35D1b3F3D7966A1DFe207aa4514C12a259A0492B',
                'ltv_ratios': {
                    'WETH': 0.74,  # For ETH-A vault
                    'WBTC': 0.70   # For WBTC-A vault
                },
                'stability_fee': Decimal('0.005')  # 0.5% annual
            }
        }
        
        # Strategy configuration
        self.min_profit_threshold = Web3.to_wei(0.01, 'ether')  # 0.01 ETH minimum profit
        self.max_risk_score = 0.7  # Maximum acceptable risk
        self.max_execution_time = 300  # 5 minutes max execution time
        
        # Cache for optimization
        self._lending_rates_cache = {}
        self._borrowing_rates_cache = {}
        self._cache_ttl = 30  # 30 seconds cache TTL
        
    async def discover_lending_swap_opportunities(
        self,
        base_assets: List[str] = None,
        max_strategies: int = 10
    ) -> List[ComplexStrategy]:
        """
        Discover complex lending/borrowing + swap opportunities
        
        Strategy Types:
        1. Collateral Leverage: Deposit → Borrow → Buy more collateral → Repeat
        2. Yield Arbitrage: Lend high yield asset → Borrow low cost → Swap to profitable
        3. Flash Loan Leverage: Flash loan → Multi-protocol leverage → Repay
        """
        start_time = time.time()
        logger.info("Starting lending/swap opportunity discovery")
        
        if not base_assets:
            base_assets = ['WETH', 'USDC', 'DAI', 'WBTC', 'USDT']
        
        strategies = []
        
        try:
            # Update lending/borrowing rates
            await self._update_protocol_rates()
            
            # Strategy 1: Lend-Borrow-Swap cycles
            lend_borrow_strategies = await self._discover_lend_borrow_swap_strategies(base_assets)
            strategies.extend(lend_borrow_strategies)
            
            # Strategy 2: Flash loan enhanced strategies
            if self.flash_loan_manager:
                flash_strategies = await self._discover_flash_enhanced_strategies(base_assets)
                strategies.extend(flash_strategies)
            
            # Strategy 3: Yield arbitrage strategies  
            yield_strategies = await self._discover_yield_arbitrage_strategies(base_assets)
            strategies.extend(yield_strategies)
            
            # Filter and rank strategies
            viable_strategies = await self._filter_viable_strategies(strategies)
            ranked_strategies = self._rank_strategies_by_profit_risk(viable_strategies)
            
            execution_time = time.time() - start_time
            logger.info(f"Discovered {len(ranked_strategies)} viable strategies in {execution_time:.2f}s")
            
            return ranked_strategies[:max_strategies]
            
        except Exception as e:
            logger.error(f"Error discovering lending/swap opportunities: {e}")
            return []
    
    async def _discover_lend_borrow_swap_strategies(self, base_assets: List[str]) -> List[ComplexStrategy]:
        """
        Discover lend → borrow → swap → repay strategies
        
        Example: 
        1. Lend ETH on Aave (collateral)
        2. Borrow USDC against ETH collateral  
        3. Swap USDC to DAI on Uniswap (if profitable rate)
        4. Swap DAI back to USDC 
        5. Repay USDC loan, withdraw ETH
        """
        strategies = []
        
        for collateral_asset in base_assets:
            for borrow_asset in base_assets:
                if collateral_asset == borrow_asset:
                    continue
                    
                # Check all lending protocol combinations
                for lend_protocol in self.lending_protocols.keys():
                    for borrow_protocol in self.lending_protocols.keys():
                        
                        strategy = await self._build_lend_borrow_swap_strategy(
                            collateral_asset=collateral_asset,
                            borrow_asset=borrow_asset,
                            lend_protocol=lend_protocol,
                            borrow_protocol=borrow_protocol
                        )
                        
                        if strategy and strategy.expected_profit > self.min_profit_threshold:
                            strategies.append(strategy)
        
        return strategies
    
    async def _build_lend_borrow_swap_strategy(
        self,
        collateral_asset: str,
        borrow_asset: str, 
        lend_protocol: str,
        borrow_protocol: str
    ) -> Optional[ComplexStrategy]:
        """Build a specific lend-borrow-swap strategy"""
        
        try:
            # Get protocol configurations
            lend_config = self.lending_protocols.get(lend_protocol)
            borrow_config = self.lending_protocols.get(borrow_protocol)
            
            if not lend_config or not borrow_config:
                return None
            
            # Calculate borrowing capacity
            ltv_ratio = min(
                lend_config['ltv_ratios'].get(collateral_asset, 0),
                borrow_config['ltv_ratios'].get(borrow_asset, 0)
            )
            
            if ltv_ratio == 0:
                return None
            
            # Estimate required capital (start with 1 ETH equivalent)
            collateral_amount = Web3.to_wei(1, 'ether')
            max_borrow_amount = int(collateral_amount * ltv_ratio * 0.9)  # 90% of max LTV for safety
            
            # Get current rates
            lending_rate = await self._get_lending_rate(lend_protocol, collateral_asset)
            borrowing_rate = await self._get_borrowing_rate(borrow_protocol, borrow_asset)
            
            # Find profitable swap opportunities with borrowed assets
            swap_opportunities = await self._find_profitable_swaps(borrow_asset, max_borrow_amount)
            
            best_swap_profit = Decimal(0)
            best_swap_path = []
            
            for swap_opp in swap_opportunities:
                if swap_opp.net_profit > best_swap_profit:
                    best_swap_profit = Decimal(str(swap_opp.net_profit))
                    best_swap_path = swap_opp.path
            
            # Calculate strategy profitability
            # Profit = Swap Profit + Lending Yield - Borrowing Cost - Gas Costs
            daily_lending_yield = lending_rate / Decimal(365)
            daily_borrowing_cost = borrowing_rate / Decimal(365)
            
            # Assume 1-day strategy execution
            lending_yield = collateral_amount * daily_lending_yield
            borrowing_cost = max_borrow_amount * daily_borrowing_cost
            estimated_gas_cost = Web3.to_wei(0.02, 'ether')  # ~0.02 ETH for complex tx
            
            net_profit = best_swap_profit + lending_yield - borrowing_cost - estimated_gas_cost
            
            if net_profit <= 0:
                return None
            
            # Build strategy steps
            steps = [
                {
                    'action': 'lend',
                    'protocol': lend_protocol,
                    'asset': collateral_asset,
                    'amount': collateral_amount,
                    'expected_yield': lending_yield
                },
                {
                    'action': 'borrow', 
                    'protocol': borrow_protocol,
                    'asset': borrow_asset,
                    'amount': max_borrow_amount,
                    'interest_rate': borrowing_rate
                },
                {
                    'action': 'swap',
                    'path': best_swap_path,
                    'amount_in': max_borrow_amount,
                    'expected_profit': best_swap_profit
                },
                {
                    'action': 'repay',
                    'protocol': borrow_protocol,
                    'asset': borrow_asset,
                    'amount': max_borrow_amount
                },
                {
                    'action': 'withdraw',
                    'protocol': lend_protocol, 
                    'asset': collateral_asset,
                    'amount': collateral_amount
                }
            ]
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(
                ltv_ratio=ltv_ratio,
                protocols=[lend_protocol, borrow_protocol],
                assets=[collateral_asset, borrow_asset],
                execution_steps=len(steps)
            )
            
            strategy = ComplexStrategy(
                strategy_id=f"lend-borrow-{collateral_asset}-{borrow_asset}-{int(time.time())}",
                strategy_type=StrategyType.LEND_BORROW_SWAP,
                steps=steps,
                expected_profit=net_profit,
                required_capital=collateral_amount,
                risk_score=risk_score,
                max_slippage=Decimal('0.005'),  # 0.5% max slippage
                execution_time_estimate=120.0,  # 2 minutes estimated
                protocols_involved=[lend_protocol, borrow_protocol],
                assets_involved=[collateral_asset, borrow_asset]
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error building lend-borrow-swap strategy: {e}")
            return None
    
    async def _discover_flash_enhanced_strategies(self, base_assets: List[str]) -> List[ComplexStrategy]:
        """
        Discover flash loan enhanced lending/borrowing strategies
        
        Example:
        1. Flash loan large amount of asset A
        2. Lend asset A across multiple protocols for best rates
        3. Borrow asset B using A as collateral
        4. Swap B for profit opportunities
        5. Reverse all positions and repay flash loan
        """
        strategies = []
        
        if not self.flash_loan_manager:
            return strategies
        
        for flash_asset in base_assets:
            for target_asset in base_assets:
                if flash_asset == target_asset:
                    continue
                
                # Try different flash loan amounts
                flash_amounts = [
                    Web3.to_wei(10, 'ether'),   # 10 ETH equivalent
                    Web3.to_wei(50, 'ether'),   # 50 ETH equivalent  
                    Web3.to_wei(100, 'ether')   # 100 ETH equivalent
                ]
                
                for flash_amount in flash_amounts:
                    strategy = await self._build_flash_enhanced_strategy(
                        flash_asset=flash_asset,
                        target_asset=target_asset,
                        flash_amount=flash_amount
                    )
                    
                    if strategy and strategy.expected_profit > self.min_profit_threshold:
                        strategies.append(strategy)
        
        return strategies
    
    async def _build_flash_enhanced_strategy(
        self,
        flash_asset: str,
        target_asset: str,
        flash_amount: int
    ) -> Optional[ComplexStrategy]:
        """Build flash loan enhanced strategy"""
        
        try:
            # Get flash loan fee
            flash_fee = await self.flash_loan_manager.calculate_flash_loan_fee(
                FlashLoanProvider.AAVE_V2, flash_asset, flash_amount
            )
            
            # Find best lending rate for flash asset
            best_lend_protocol, best_lend_rate = await self._find_best_lending_rate(flash_asset)
            
            # Calculate max borrowing capacity
            ltv_ratio = self.lending_protocols[best_lend_protocol]['ltv_ratios'].get(flash_asset, 0)
            max_borrow = int(flash_amount * ltv_ratio * 0.85)  # 85% of max for safety
            
            # Find best borrowing rate for target asset
            best_borrow_protocol, best_borrow_rate = await self._find_best_borrowing_rate(target_asset)
            
            # Look for profitable swaps with borrowed target asset
            swap_opportunities = await self._find_profitable_swaps(target_asset, max_borrow)
            
            if not swap_opportunities:
                return None
            
            best_swap = max(swap_opportunities, key=lambda x: x.net_profit)
            
            # Calculate total profitability
            # Income: Lending yield + Swap profit
            # Costs: Flash loan fee + Borrowing interest + Gas
            
            # Assume 1-hour execution time for more realistic calc
            execution_hours = 1.0
            hourly_lend_rate = best_lend_rate / Decimal(365 * 24)
            hourly_borrow_rate = best_borrow_rate / Decimal(365 * 24)
            
            lending_income = flash_amount * hourly_lend_rate * Decimal(execution_hours)
            borrowing_cost = max_borrow * hourly_borrow_rate * Decimal(execution_hours)
            swap_profit = Decimal(str(best_swap.net_profit))
            gas_costs = Web3.to_wei(0.05, 'ether')  # Higher gas for complex flash loan tx
            
            total_profit = lending_income + swap_profit - borrowing_cost - flash_fee - gas_costs
            
            if total_profit <= 0:
                return None
            
            # Build strategy steps
            steps = [
                {
                    'action': 'flash_loan',
                    'provider': 'aave_v2',
                    'asset': flash_asset,
                    'amount': flash_amount,
                    'fee': flash_fee
                },
                {
                    'action': 'lend',
                    'protocol': best_lend_protocol,
                    'asset': flash_asset, 
                    'amount': flash_amount
                },
                {
                    'action': 'borrow',
                    'protocol': best_borrow_protocol,
                    'asset': target_asset,
                    'amount': max_borrow
                },
                {
                    'action': 'swap',
                    'path': best_swap.path,
                    'amount_in': max_borrow,
                    'expected_profit': best_swap.net_profit
                },
                {
                    'action': 'repay_borrow',
                    'protocol': best_borrow_protocol,
                    'asset': target_asset,
                    'amount': max_borrow
                },
                {
                    'action': 'withdraw_lend',
                    'protocol': best_lend_protocol,
                    'asset': flash_asset,
                    'amount': flash_amount
                },
                {
                    'action': 'repay_flash_loan',
                    'provider': 'aave_v2',
                    'asset': flash_asset,
                    'amount': flash_amount + flash_fee
                }
            ]
            
            risk_score = self._calculate_risk_score(
                ltv_ratio=ltv_ratio,
                protocols=[best_lend_protocol, best_borrow_protocol],
                assets=[flash_asset, target_asset],
                execution_steps=len(steps),
                uses_flash_loan=True
            )
            
            strategy = ComplexStrategy(
                strategy_id=f"flash-lend-{flash_asset}-{target_asset}-{int(time.time())}",
                strategy_type=StrategyType.FLASH_LEND_SWAP,
                steps=steps,
                expected_profit=total_profit,
                required_capital=Web3.to_wei(0.1, 'ether'),  # Just gas money needed
                risk_score=risk_score,
                max_slippage=Decimal('0.01'),  # 1% max slippage for flash loan strategy
                execution_time_estimate=execution_hours * 60,  # Convert to minutes
                protocols_involved=[best_lend_protocol, best_borrow_protocol, 'flash_loan'],
                assets_involved=[flash_asset, target_asset]
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error building flash enhanced strategy: {e}")
            return None
    
    async def _discover_yield_arbitrage_strategies(self, base_assets: List[str]) -> List[ComplexStrategy]:
        """
        Discover yield arbitrage opportunities across different protocols
        
        Example:
        1. Lend USDC on Protocol A (5% APY)  
        2. Borrow USDC on Protocol B (3% APY)
        3. Net 2% APY on borrowed capital
        """
        strategies = []
        
        for asset in base_assets:
            # Get all available rates for this asset
            lending_rates = {}
            borrowing_rates = {}
            
            for protocol in self.lending_protocols.keys():
                lending_rate = await self._get_lending_rate(protocol, asset)
                borrowing_rate = await self._get_borrowing_rate(protocol, asset)
                
                if lending_rate > 0:
                    lending_rates[protocol] = lending_rate
                if borrowing_rate > 0:
                    borrowing_rates[protocol] = borrowing_rate
            
            # Find profitable rate spreads
            for lend_protocol, lend_rate in lending_rates.items():
                for borrow_protocol, borrow_rate in borrowing_rates.items():
                    if lend_protocol == borrow_protocol:
                        continue
                    
                    # Check if lending rate > borrowing rate (profitable spread)
                    rate_spread = lend_rate - borrow_rate
                    
                    if rate_spread > Decimal('0.02'):  # Minimum 2% APY spread
                        strategy = await self._build_yield_arbitrage_strategy(
                            asset=asset,
                            lend_protocol=lend_protocol,
                            borrow_protocol=borrow_protocol,
                            lend_rate=lend_rate,
                            borrow_rate=borrow_rate,
                            rate_spread=rate_spread
                        )
                        
                        if strategy:
                            strategies.append(strategy)
        
        return strategies
    
    async def _build_yield_arbitrage_strategy(
        self,
        asset: str,
        lend_protocol: str,
        borrow_protocol: str,
        lend_rate: Decimal,
        borrow_rate: Decimal,
        rate_spread: Decimal
    ) -> Optional[ComplexStrategy]:
        """Build yield arbitrage strategy"""
        
        try:
            # Calculate optimal position size
            # Start with 10 ETH worth of the asset
            position_size = Web3.to_wei(10, 'ether')
            
            # Calculate annual profit from rate spread
            annual_profit = position_size * rate_spread
            
            # Calculate monthly profit (more realistic timeframe)
            monthly_profit = annual_profit / Decimal(12)
            
            # Account for gas costs
            gas_costs = Web3.to_wei(0.03, 'ether')  # Setup + maintenance costs
            
            net_monthly_profit = monthly_profit - gas_costs
            
            if net_monthly_profit <= 0:
                return None
            
            steps = [
                {
                    'action': 'borrow',
                    'protocol': borrow_protocol,
                    'asset': asset,
                    'amount': position_size,
                    'rate': borrow_rate
                },
                {
                    'action': 'lend',
                    'protocol': lend_protocol,
                    'asset': asset,
                    'amount': position_size,
                    'rate': lend_rate
                },
                {
                    'action': 'maintain_position',
                    'duration_days': 30,  # Hold for 1 month
                    'expected_yield': net_monthly_profit
                },
                {
                    'action': 'unwind_position',
                    'protocols': [lend_protocol, borrow_protocol]
                }
            ]
            
            risk_score = self._calculate_risk_score(
                ltv_ratio=0.0,  # No collateral risk in same-asset strategy
                protocols=[lend_protocol, borrow_protocol],
                assets=[asset],
                execution_steps=len(steps),
                rate_risk=True
            )
            
            strategy = ComplexStrategy(
                strategy_id=f"yield-arb-{asset}-{lend_protocol}-{borrow_protocol}-{int(time.time())}",
                strategy_type=StrategyType.YIELD_ARBITRAGE,
                steps=steps,
                expected_profit=net_monthly_profit,
                required_capital=Web3.to_wei(1, 'ether'),  # Collateral for borrowing
                risk_score=risk_score,
                max_slippage=Decimal('0.001'),  # Low slippage for same asset
                execution_time_estimate=1800,  # 30 minutes setup time
                protocols_involved=[lend_protocol, borrow_protocol],
                assets_involved=[asset]
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error building yield arbitrage strategy: {e}")
            return None
    
    # Helper methods for rate discovery and calculations
    async def _update_protocol_rates(self):
        """Update cached protocol rates"""
        current_time = time.time()
        
        if (current_time - getattr(self, '_last_rate_update', 0)) < self._cache_ttl:
            return  # Use cached rates
        
        logger.info("Updating protocol lending/borrowing rates")
        
        for protocol in self.lending_protocols.keys():
            for asset in ['WETH', 'USDC', 'DAI', 'WBTC', 'USDT']:
                try:
                    lending_rate = await self._fetch_lending_rate(protocol, asset)
                    borrowing_rate = await self._fetch_borrowing_rate(protocol, asset)
                    
                    self._lending_rates_cache[(protocol, asset)] = lending_rate
                    self._borrowing_rates_cache[(protocol, asset)] = borrowing_rate
                    
                except Exception as e:
                    logger.warning(f"Failed to update rates for {protocol}-{asset}: {e}")
        
        self._last_rate_update = current_time
    
    async def _get_lending_rate(self, protocol: str, asset: str) -> Decimal:
        """Get lending rate for protocol/asset pair"""
        cached_rate = self._lending_rates_cache.get((protocol, asset))
        if cached_rate is not None:
            return cached_rate
        
        # Fallback to fetch if not cached
        return await self._fetch_lending_rate(protocol, asset)
    
    async def _get_borrowing_rate(self, protocol: str, asset: str) -> Decimal:
        """Get borrowing rate for protocol/asset pair"""
        cached_rate = self._borrowing_rates_cache.get((protocol, asset))
        if cached_rate is not None:
            return cached_rate
        
        # Fallback to fetch if not cached
        return await self._fetch_borrowing_rate(protocol, asset)
    
    async def _fetch_lending_rate(self, protocol: str, asset: str) -> Decimal:
        """Fetch current lending rate from protocol"""
        # Mock implementation - in production would query actual protocol contracts
        base_rates = {
            'aave_v2': {'WETH': 0.025, 'USDC': 0.03, 'DAI': 0.035, 'WBTC': 0.02},
            'compound': {'WETH': 0.02, 'USDC': 0.025, 'DAI': 0.028, 'WBTC': 0.015},
            'maker_dao': {'WETH': 0.01, 'WBTC': 0.012}
        }
        
        rate = base_rates.get(protocol, {}).get(asset, 0.0)
        return Decimal(str(rate))
    
    async def _fetch_borrowing_rate(self, protocol: str, asset: str) -> Decimal:
        """Fetch current borrowing rate from protocol"""
        # Mock implementation - in production would query actual protocol contracts
        base_rates = {
            'aave_v2': {'WETH': 0.045, 'USDC': 0.055, 'DAI': 0.065, 'WBTC': 0.04},
            'compound': {'WETH': 0.05, 'USDC': 0.06, 'DAI': 0.07, 'WBTC': 0.045},
            'maker_dao': {'DAI': 0.005}  # MakerDAO mints DAI
        }
        
        rate = base_rates.get(protocol, {}).get(asset, 0.0)
        return Decimal(str(rate))
    
    async def _find_profitable_swaps(self, asset: str, amount: int) -> List[ArbitrageOpportunity]:
        """Find profitable swap opportunities for given asset amount"""
        # Use existing arbitrage detector from market graph
        opportunities = []
        
        try:
            # Get arbitrage opportunities starting with the given asset
            if hasattr(self.market_graph, 'find_arbitrage_opportunities'):
                arb_opportunities = await self.market_graph.find_arbitrage_opportunities(
                    source_tokens=[asset],
                    min_profit_threshold=self.min_profit_threshold
                )
                
                # Filter by amount and profitability
                for opp in arb_opportunities:
                    if opp.required_capital <= amount and opp.net_profit > 0:
                        opportunities.append(opp)
            
        except Exception as e:
            logger.error(f"Error finding profitable swaps: {e}")
        
        return opportunities
    
    async def _find_best_lending_rate(self, asset: str) -> Tuple[str, Decimal]:
        """Find protocol offering best lending rate for asset"""
        best_protocol = None
        best_rate = Decimal(0)
        
        for protocol in self.lending_protocols.keys():
            rate = await self._get_lending_rate(protocol, asset)
            if rate > best_rate:
                best_rate = rate
                best_protocol = protocol
        
        return best_protocol, best_rate
    
    async def _find_best_borrowing_rate(self, asset: str) -> Tuple[str, Decimal]:
        """Find protocol offering best (lowest) borrowing rate for asset"""
        best_protocol = None
        best_rate = Decimal('999')  # Start with very high rate
        
        for protocol in self.lending_protocols.keys():
            rate = await self._get_borrowing_rate(protocol, asset)
            if rate > 0 and rate < best_rate:
                best_rate = rate
                best_protocol = protocol
        
        return best_protocol, best_rate
    
    def _calculate_risk_score(
        self,
        ltv_ratio: float,
        protocols: List[str],
        assets: List[str],
        execution_steps: int,
        uses_flash_loan: bool = False,
        rate_risk: bool = False
    ) -> float:
        """Calculate risk score for strategy (0 = low risk, 1 = high risk)"""
        
        risk_score = 0.0
        
        # LTV risk (higher LTV = higher risk)
        risk_score += ltv_ratio * 0.3
        
        # Protocol risk (more protocols = higher risk)  
        risk_score += len(protocols) * 0.1
        
        # Asset risk (more assets = higher risk)
        risk_score += len(assets) * 0.05
        
        # Execution complexity risk
        risk_score += execution_steps * 0.02
        
        # Flash loan risk
        if uses_flash_loan:
            risk_score += 0.2
        
        # Interest rate risk
        if rate_risk:
            risk_score += 0.15
        
        # Cap at 1.0
        return min(risk_score, 1.0)
    
    async def _filter_viable_strategies(self, strategies: List[ComplexStrategy]) -> List[ComplexStrategy]:
        """Filter strategies by viability criteria"""
        viable = []
        
        for strategy in strategies:
            # Check profit threshold
            if strategy.expected_profit < self.min_profit_threshold:
                continue
            
            # Check risk threshold
            if strategy.risk_score > self.max_risk_score:
                continue
            
            # Check execution time
            if strategy.execution_time_estimate > self.max_execution_time:
                continue
            
            # Additional viability checks can be added here
            viable.append(strategy)
        
        return viable
    
    def _rank_strategies_by_profit_risk(self, strategies: List[ComplexStrategy]) -> List[ComplexStrategy]:
        """Rank strategies by profit/risk ratio"""
        
        def profit_risk_score(strategy: ComplexStrategy) -> float:
            """Calculate profit/risk score for ranking"""
            profit_eth = float(Web3.from_wei(int(strategy.expected_profit), 'ether'))
            risk_penalty = strategy.risk_score
            
            # Simple profit/risk ratio with risk penalty
            return profit_eth * (1.0 - risk_penalty)
        
        return sorted(strategies, key=profit_risk_score, reverse=True)

    async def execute_complex_strategy(self, strategy: ComplexStrategy) -> Dict:
        """Execute a complex lending/borrowing + swap strategy"""
        start_time = time.time()
        logger.info(f"Executing complex strategy: {strategy.strategy_id}")
        
        execution_results = {
            'strategy_id': strategy.strategy_id,
            'strategy_type': strategy.strategy_type.value,
            'success': False,
            'steps_completed': 0,
            'actual_profit': 0,
            'gas_used': 0,
            'execution_time': 0,
            'errors': []
        }
        
        try:
            # Execute each step in sequence
            for i, step in enumerate(strategy.steps):
                step_result = await self._execute_strategy_step(step, strategy)
                
                if not step_result['success']:
                    execution_results['errors'].append(f"Step {i+1} failed: {step_result.get('error')}")
                    break
                
                execution_results['steps_completed'] = i + 1
                execution_results['gas_used'] += step_result.get('gas_used', 0)
            
            # If all steps completed successfully
            if execution_results['steps_completed'] == len(strategy.steps):
                execution_results['success'] = True
                execution_results['actual_profit'] = await self._calculate_actual_profit(strategy)
                logger.info(f"Strategy executed successfully: {execution_results['actual_profit']} wei profit")
            
        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            execution_results['errors'].append(str(e))
        
        execution_results['execution_time'] = time.time() - start_time
        return execution_results
    
    async def _execute_strategy_step(self, step: Dict, strategy: ComplexStrategy) -> Dict:
        """Execute individual strategy step"""
        # Mock implementation - in production would execute actual transactions
        logger.info(f"Executing step: {step['action']}")
        
        # Simulate execution time
        await asyncio.sleep(0.1)
        
        return {
            'success': True,
            'gas_used': 50000,  # Mock gas usage
            'transaction_hash': f"0x{'0' * 64}"  # Mock tx hash
        }
    
    async def _calculate_actual_profit(self, strategy: ComplexStrategy) -> int:
        """Calculate actual profit after strategy execution"""
        # Mock implementation - would calculate based on actual transaction results
        # For now, return 90% of expected profit (accounting for slippage, etc.)
        return int(strategy.expected_profit * Decimal('0.9'))