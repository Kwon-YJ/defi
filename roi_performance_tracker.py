#!/usr/bin/env python3
"""
ROI Tracking and Performance Measurement System
Implementation for DeFiPoser-ARB paper reproduction

This system tracks:
- ROI (Return on Investment) metrics
- Performance benchmarks against paper targets
- Weekly and daily revenue tracking
- Success rate analysis
- Capital efficiency metrics
- Risk-adjusted returns
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import statistics
from dataclasses import dataclass
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradeResult:
    timestamp: datetime
    strategy_type: str
    initial_capital: float
    revenue: float
    roi_percent: float
    execution_time: float
    gas_cost: float
    net_profit: float
    assets_involved: List[str]
    protocols_used: List[str]

@dataclass
class PerformanceMetrics:
    total_trades: int
    total_revenue: float
    total_invested: float
    average_roi: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    average_execution_time: float
    weekly_average_revenue: float
    best_trade_revenue: float
    capital_efficiency: float

class ROIPerformanceTracker:
    def __init__(self, db_path: str = "roi_performance.db"):
        self.db_path = db_path
        self.init_database()
        
        # Paper targets from DeFiPoser-ARB research
        self.paper_targets = {
            "weekly_average_revenue": 191.48,  # ETH per week
            "max_single_trade_revenue": 81.31,  # ETH
            "average_execution_time": 6.43,    # seconds
            "total_evaluation_days": 150,       # days
            "minimum_roi_threshold": 10.0,     # percentage
            "target_win_rate": 85.0,           # percentage
        }
        
    def init_database(self):
        """Initialize SQLite database for tracking performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy_type TEXT NOT NULL,
                initial_capital REAL NOT NULL,
                revenue REAL NOT NULL,
                roi_percent REAL NOT NULL,
                execution_time REAL NOT NULL,
                gas_cost REAL NOT NULL,
                net_profit REAL NOT NULL,
                assets_involved TEXT NOT NULL,
                protocols_used TEXT NOT NULL,
                block_number INTEGER,
                transaction_hash TEXT
            )
        """)
        
        # Create daily_metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_metrics (
                date TEXT PRIMARY KEY,
                total_trades INTEGER NOT NULL,
                total_revenue REAL NOT NULL,
                average_roi REAL NOT NULL,
                win_rate REAL NOT NULL,
                execution_time_avg REAL NOT NULL,
                best_trade_revenue REAL NOT NULL,
                total_gas_cost REAL NOT NULL
            )
        """)
        
        # Create weekly_metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weekly_metrics (
                week_start TEXT PRIMARY KEY,
                week_end TEXT NOT NULL,
                total_trades INTEGER NOT NULL,
                total_revenue REAL NOT NULL,
                average_daily_revenue REAL NOT NULL,
                target_achievement_percent REAL NOT NULL,
                sharpe_ratio REAL,
                max_drawdown REAL
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")

    def record_trade(self, trade: TradeResult) -> None:
        """Record a single trade result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades (
                timestamp, strategy_type, initial_capital, revenue, roi_percent,
                execution_time, gas_cost, net_profit, assets_involved, protocols_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.timestamp.isoformat(),
            trade.strategy_type,
            trade.initial_capital,
            trade.revenue,
            trade.roi_percent,
            trade.execution_time,
            trade.gas_cost,
            trade.net_profit,
            json.dumps(trade.assets_involved),
            json.dumps(trade.protocols_used)
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Trade recorded: {trade.revenue:.4f} ETH ROI ({trade.roi_percent:.2f}%)")

    def calculate_roi(self, initial_capital: float, revenue: float, gas_cost: float = 0.0) -> float:
        """Calculate ROI percentage"""
        if initial_capital <= 0:
            return 0.0
        
        net_profit = revenue - gas_cost
        roi = (net_profit / initial_capital) * 100
        return roi

    def get_daily_performance(self, date: datetime) -> Optional[Dict]:
        """Get performance metrics for a specific day"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        date_str = date.strftime("%Y-%m-%d")
        cursor.execute("""
            SELECT * FROM daily_metrics WHERE date = ?
        """, (date_str,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "date": result[0],
                "total_trades": result[1],
                "total_revenue": result[2],
                "average_roi": result[3],
                "win_rate": result[4],
                "execution_time_avg": result[5],
                "best_trade_revenue": result[6],
                "total_gas_cost": result[7]
            }
        return None

    def calculate_daily_metrics(self, date: datetime) -> Dict:
        """Calculate and store daily performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        date_str = date.strftime("%Y-%m-%d")
        next_date_str = (date + timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Get all trades for the day
        cursor.execute("""
            SELECT * FROM trades 
            WHERE timestamp >= ? AND timestamp < ?
        """, (date_str, next_date_str))
        
        trades = cursor.fetchall()
        
        if not trades:
            conn.close()
            return {}
        
        # Calculate metrics
        total_trades = len(trades)
        total_revenue = sum(trade[4] for trade in trades)  # revenue column
        roi_values = [trade[5] for trade in trades]  # roi_percent column
        execution_times = [trade[6] for trade in trades]  # execution_time column
        gas_costs = [trade[7] for trade in trades]  # gas_cost column
        
        average_roi = statistics.mean(roi_values)
        win_rate = (sum(1 for roi in roi_values if roi > 0) / total_trades) * 100
        execution_time_avg = statistics.mean(execution_times)
        best_trade_revenue = max(trade[4] for trade in trades)
        total_gas_cost = sum(gas_costs)
        
        # Store daily metrics
        cursor.execute("""
            INSERT OR REPLACE INTO daily_metrics 
            (date, total_trades, total_revenue, average_roi, win_rate, 
             execution_time_avg, best_trade_revenue, total_gas_cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date_str, total_trades, total_revenue, average_roi, win_rate,
            execution_time_avg, best_trade_revenue, total_gas_cost
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "date": date_str,
            "total_trades": total_trades,
            "total_revenue": total_revenue,
            "average_roi": average_roi,
            "win_rate": win_rate,
            "execution_time_avg": execution_time_avg,
            "best_trade_revenue": best_trade_revenue,
            "total_gas_cost": total_gas_cost
        }

    def calculate_weekly_metrics(self, week_start: datetime) -> Dict:
        """Calculate and store weekly performance metrics"""
        week_end = week_start + timedelta(days=7)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all trades for the week
        cursor.execute("""
            SELECT * FROM trades 
            WHERE timestamp >= ? AND timestamp < ?
        """, (week_start.isoformat(), week_end.isoformat()))
        
        trades = cursor.fetchall()
        
        if not trades:
            return {}
        
        # Calculate weekly metrics
        total_trades = len(trades)
        total_revenue = sum(trade[4] for trade in trades)  # revenue column
        average_daily_revenue = total_revenue / 7
        
        # Calculate target achievement
        target_achievement_percent = (total_revenue / self.paper_targets["weekly_average_revenue"]) * 100
        
        # Calculate Sharpe ratio (simplified)
        roi_values = [trade[5] for trade in trades]
        if len(roi_values) > 1:
            returns_mean = statistics.mean(roi_values)
            returns_std = statistics.stdev(roi_values)
            sharpe_ratio = returns_mean / returns_std if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        cumulative_returns = []
        cumulative = 0
        for trade in trades:
            cumulative += trade[4]  # revenue
            cumulative_returns.append(cumulative)
        
        if cumulative_returns:
            running_max = [max(cumulative_returns[:i+1]) for i in range(len(cumulative_returns))]
            drawdowns = [(cumulative_returns[i] - running_max[i]) / running_max[i] 
                        for i in range(len(cumulative_returns)) if running_max[i] > 0]
            max_drawdown = min(drawdowns) * 100 if drawdowns else 0
        else:
            max_drawdown = 0
        
        # Store weekly metrics
        week_start_str = week_start.strftime("%Y-%m-%d")
        week_end_str = week_end.strftime("%Y-%m-%d")
        
        cursor.execute("""
            INSERT OR REPLACE INTO weekly_metrics 
            (week_start, week_end, total_trades, total_revenue, 
             average_daily_revenue, target_achievement_percent, sharpe_ratio, max_drawdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            week_start_str, week_end_str, total_trades, total_revenue,
            average_daily_revenue, target_achievement_percent, sharpe_ratio, max_drawdown
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "week_start": week_start_str,
            "week_end": week_end_str,
            "total_trades": total_trades,
            "total_revenue": total_revenue,
            "average_daily_revenue": average_daily_revenue,
            "target_achievement_percent": target_achievement_percent,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }

    def generate_performance_report(self, days: int = 30) -> Dict:
        """Generate comprehensive performance report"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all trades in period
        cursor.execute("""
            SELECT * FROM trades 
            WHERE timestamp >= ? AND timestamp <= ?
        """, (start_date.isoformat(), end_date.isoformat()))
        
        trades = cursor.fetchall()
        conn.close()
        
        if not trades:
            return {"error": "No trades found in the specified period"}
        
        # Calculate comprehensive metrics
        total_trades = len(trades)
        revenues = [trade[4] for trade in trades]
        initial_capitals = [trade[3] for trade in trades]
        roi_values = [trade[5] for trade in trades]
        execution_times = [trade[6] for trade in trades]
        gas_costs = [trade[7] for trade in trades]
        
        total_revenue = sum(revenues)
        total_invested = sum(initial_capitals)
        average_roi = statistics.mean(roi_values)
        win_rate = (sum(1 for roi in roi_values if roi > 0) / total_trades) * 100
        average_execution_time = statistics.mean(execution_times)
        best_trade_revenue = max(revenues)
        
        # Calculate weekly average revenue
        weeks_in_period = days / 7
        weekly_average_revenue = total_revenue / weeks_in_period if weeks_in_period > 0 else 0
        
        # Calculate capital efficiency
        total_net_profit = sum(trade[8] for trade in trades)  # net_profit column
        capital_efficiency = (total_net_profit / total_invested) * 100 if total_invested > 0 else 0
        
        # Performance against paper targets
        target_comparison = {
            "weekly_revenue_achievement": (weekly_average_revenue / self.paper_targets["weekly_average_revenue"]) * 100,
            "max_trade_achievement": (best_trade_revenue / self.paper_targets["max_single_trade_revenue"]) * 100,
            "execution_time_efficiency": (self.paper_targets["average_execution_time"] / average_execution_time) * 100 if average_execution_time > 0 else 0,
            "win_rate_achievement": (win_rate / self.paper_targets["target_win_rate"]) * 100,
        }
        
        report = {
            "period": {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "days": days
            },
            "basic_metrics": {
                "total_trades": total_trades,
                "total_revenue": round(total_revenue, 4),
                "total_invested": round(total_invested, 4),
                "average_roi": round(average_roi, 2),
                "win_rate": round(win_rate, 2),
                "average_execution_time": round(average_execution_time, 2),
                "weekly_average_revenue": round(weekly_average_revenue, 4),
                "best_trade_revenue": round(best_trade_revenue, 4),
                "capital_efficiency": round(capital_efficiency, 2)
            },
            "paper_targets": self.paper_targets,
            "target_comparison": {
                key: round(value, 2) for key, value in target_comparison.items()
            },
            "performance_grade": self._calculate_performance_grade(target_comparison)
        }
        
        return report

    def _calculate_performance_grade(self, target_comparison: Dict) -> str:
        """Calculate overall performance grade based on target achievements"""
        scores = list(target_comparison.values())
        average_achievement = statistics.mean(scores)
        
        if average_achievement >= 100:
            return "A+ (Exceeds Paper Targets)"
        elif average_achievement >= 90:
            return "A (Meets Paper Targets)"
        elif average_achievement >= 80:
            return "B+ (Good Performance)"
        elif average_achievement >= 70:
            return "B (Satisfactory Performance)"
        elif average_achievement >= 60:
            return "C+ (Below Target)"
        elif average_achievement >= 50:
            return "C (Needs Improvement)"
        else:
            return "D (Poor Performance)"

    def export_metrics_to_json(self, filepath: str) -> None:
        """Export all performance metrics to JSON file"""
        report = self.generate_performance_report(days=150)  # Full evaluation period
        
        # Add historical data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get weekly metrics
        cursor.execute("SELECT * FROM weekly_metrics ORDER BY week_start")
        weekly_data = cursor.fetchall()
        
        # Get daily metrics for last 30 days
        cursor.execute("SELECT * FROM daily_metrics ORDER BY date DESC LIMIT 30")
        daily_data = cursor.fetchall()
        
        conn.close()
        
        report["weekly_history"] = [
            {
                "week_start": row[0],
                "week_end": row[1],
                "total_trades": row[2],
                "total_revenue": row[3],
                "average_daily_revenue": row[4],
                "target_achievement_percent": row[5],
                "sharpe_ratio": row[6],
                "max_drawdown": row[7]
            }
            for row in weekly_data
        ]
        
        report["daily_history"] = [
            {
                "date": row[0],
                "total_trades": row[1],
                "total_revenue": row[2],
                "average_roi": row[3],
                "win_rate": row[4],
                "execution_time_avg": row[5],
                "best_trade_revenue": row[6],
                "total_gas_cost": row[7]
            }
            for row in daily_data
        ]
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance metrics exported to: {filepath}")

async def main():
    """Main function to demonstrate ROI tracking system"""
    tracker = ROIPerformanceTracker()
    
    # Generate sample trade data for demonstration
    print("🚀 ROI Performance Tracking System Initialized")
    print("=" * 60)
    
    # Simulate some trades
    sample_trades = [
        TradeResult(
            timestamp=datetime.now() - timedelta(days=1),
            strategy_type="arbitrage",
            initial_capital=10.0,
            revenue=0.5,
            roi_percent=tracker.calculate_roi(10.0, 0.5),
            execution_time=5.2,
            gas_cost=0.01,
            net_profit=0.49,
            assets_involved=["ETH", "USDC", "DAI"],
            protocols_used=["Uniswap", "Sushiswap"]
        ),
        TradeResult(
            timestamp=datetime.now() - timedelta(hours=12),
            strategy_type="flash_arbitrage",
            initial_capital=1.0,
            revenue=2.1,
            roi_percent=tracker.calculate_roi(1.0, 2.1),
            execution_time=4.8,
            gas_cost=0.02,
            net_profit=2.08,
            assets_involved=["ETH", "WBTC"],
            protocols_used=["Compound", "Aave", "Uniswap"]
        )
    ]
    
    # Record sample trades
    for trade in sample_trades:
        tracker.record_trade(trade)
    
    # Calculate daily metrics
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    
    daily_metrics = tracker.calculate_daily_metrics(yesterday)
    print(f"📊 Daily Metrics for {yesterday.strftime('%Y-%m-%d')}:")
    if daily_metrics:
        for key, value in daily_metrics.items():
            print(f"  {key}: {value}")
    print()
    
    # Generate performance report
    report = tracker.generate_performance_report(days=7)
    print("📈 Performance Report (Last 7 Days):")
    print(f"  Total Revenue: {report['basic_metrics']['total_revenue']} ETH")
    print(f"  Average ROI: {report['basic_metrics']['average_roi']}%")
    print(f"  Win Rate: {report['basic_metrics']['win_rate']}%")
    print(f"  Weekly Revenue Target Achievement: {report['target_comparison']['weekly_revenue_achievement']}%")
    print(f"  Performance Grade: {report['performance_grade']}")
    print()
    
    # Export metrics
    export_path = f"roi_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    tracker.export_metrics_to_json(export_path)
    
    print(f"✅ ROI Performance Tracking System Complete")
    print(f"📁 Report exported to: {export_path}")
    print(f"🎯 Paper Targets:")
    for key, value in tracker.paper_targets.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())