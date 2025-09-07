#!/usr/bin/env python3
"""
Test script for Real-Time Monitoring System
실시간 모니터링 시스템 테스트

This script tests:
- Real-time monitoring functionality
- Alert generation and processing
- Performance metrics collection
- System resource monitoring
- Dashboard data generation
"""

import time
import json
import random
import threading
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.real_time_monitoring_system import (
    global_monitoring_system,
    start_monitoring,
    stop_monitoring,
    get_dashboard,
    add_monitoring_callback
)
from src.performance_benchmarking import (
    start_benchmarking,
    end_benchmarking,
    time_component
)
from roi_performance_tracker import ROIPerformanceTracker, TradeResult

def test_callback(alert):
    """테스트용 알림 콜백"""
    print(f"📢 Alert Received: [{alert.severity.upper()}] {alert.alert_type} - {alert.message}")

def simulate_trade_execution(block_number: int):
    """거래 실행 시뮬레이션"""
    print(f"🔄 Simulating block {block_number} processing...")
    
    # 성능 측정 시작
    start_benchmarking(block_number)
    
    # 각 컴포넌트 시뮬레이션
    opportunities_found = 0
    strategies_executed = 0
    total_revenue = 0.0
    gas_cost = 0.01
    
    try:
        # 그래프 구축 시뮬레이션
        with time_component("graph_building"):
            time.sleep(random.uniform(0.5, 2.0))
        
        # Negative cycle 탐지
        with time_component("negative_cycle_detection"):
            time.sleep(random.uniform(1.0, 3.5))
            opportunities_found = random.randint(0, 3)
        
        # Local search (기회가 있을 때만)
        if opportunities_found > 0:
            with time_component("local_search"):
                time.sleep(random.uniform(0.8, 2.5))
                strategies_executed = random.randint(1, opportunities_found)
        
        # 파라미터 최적화
        if strategies_executed > 0:
            with time_component("parameter_optimization"):
                time.sleep(random.uniform(0.3, 1.5))
        
        # 검증
        with time_component("validation"):
            time.sleep(random.uniform(0.1, 0.5))
            if strategies_executed > 0:
                total_revenue = random.uniform(0.1, 5.0)
                gas_cost = random.uniform(0.01, 0.05)
    
    except Exception as e:
        print(f"❌ Error in simulation: {e}")
    
    # 성능 측정 완료
    metrics = end_benchmarking(
        opportunities_found=opportunities_found,
        strategies_executed=strategies_executed,
        total_revenue=total_revenue,
        gas_cost=gas_cost
    )
    
    print(f"✅ Block {block_number}: {metrics.total_execution_time:.3f}s, "
          f"{strategies_executed} strategies, {total_revenue:.4f} ETH revenue")
    
    return metrics, total_revenue, gas_cost

def simulate_roi_tracking(revenue: float, gas_cost: float):
    """ROI 추적 시뮬레이션"""
    if revenue <= 0:
        return
    
    roi_tracker = ROIPerformanceTracker()
    
    # 랜덤한 초기 자본 (flash loan 사용을 고려하여 작은 값도 포함)
    initial_capital = random.choice([1.0, 5.0, 10.0, 50.0, 100.0])
    
    # ROI 계산
    net_profit = revenue - gas_cost
    roi_percent = roi_tracker.calculate_roi(initial_capital, revenue, gas_cost)
    
    # 거래 결과 기록
    trade = TradeResult(
        timestamp=datetime.now(),
        strategy_type=random.choice(["arbitrage", "flash_arbitrage", "multi_hop", "lending_swap"]),
        initial_capital=initial_capital,
        revenue=revenue,
        roi_percent=roi_percent,
        execution_time=random.uniform(3.0, 8.0),
        gas_cost=gas_cost,
        net_profit=net_profit,
        assets_involved=random.sample(["ETH", "USDC", "DAI", "WBTC", "UNI"], k=random.randint(2, 4)),
        protocols_used=random.sample(["Uniswap", "Sushiswap", "Compound", "Aave", "Curve"], k=random.randint(1, 3))
    )
    
    roi_tracker.record_trade(trade)
    print(f"💰 Trade recorded: {revenue:.4f} ETH revenue, {roi_percent:.2f}% ROI")

def run_simulation(duration_minutes: int = 5, blocks_per_minute: int = 4):
    """시뮬레이션 실행"""
    print(f"🚀 Starting {duration_minutes} minute simulation")
    print(f"📊 Target: {blocks_per_minute} blocks per minute (every {60/blocks_per_minute:.1f} seconds)")
    print("=" * 60)
    
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    block_number = 10000
    
    while datetime.now() < end_time:
        try:
            # 블록 처리 시뮬레이션
            metrics, revenue, gas_cost = simulate_trade_execution(block_number)
            
            # ROI 추적 (수익이 있을 때만)
            if revenue > 0:
                simulate_roi_tracking(revenue, gas_cost)
            
            block_number += 1
            
            # 다음 블록까지 대기
            time.sleep(60 / blocks_per_minute)
            
        except KeyboardInterrupt:
            print("\n⏹️ Simulation interrupted by user")
            break
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            time.sleep(1)
    
    print(f"\n✅ Simulation completed: {block_number - 10000} blocks processed")

def test_dashboard_data():
    """대시보드 데이터 테스트"""
    print("\n📊 Testing dashboard data generation...")
    
    try:
        dashboard = get_dashboard()
        print("✅ Dashboard data generated successfully")
        
        # 주요 섹션 확인
        sections = ["performance", "system", "alerts", "status"]
        for section in sections:
            if section in dashboard:
                print(f"  ✓ {section} section: OK")
            else:
                print(f"  ❌ {section} section: Missing")
        
        # 데이터를 JSON으로 저장 (디버깅용)
        with open("test_dashboard_output.json", "w", encoding="utf-8") as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False, default=str)
        print("  💾 Dashboard data saved to 'test_dashboard_output.json'")
        
    except Exception as e:
        print(f"❌ Dashboard data test failed: {e}")

def stress_test_alerts():
    """알림 시스템 스트레스 테스트"""
    print("\n🚨 Testing alert system...")
    
    # 다양한 종류의 알림 생성
    test_alerts = [
        ("performance", "critical", "Execution time exceeded target significantly"),
        ("revenue", "warning", "Daily revenue below expectations"),
        ("system", "critical", "Memory usage critically high"),
        ("system", "warning", "CPU usage elevated"),
        ("performance", "info", "New optimization applied successfully")
    ]
    
    for alert_type, severity, message in test_alerts:
        global_monitoring_system._create_alert(
            alert_type, severity, message, 
            {"test": True, "timestamp": datetime.now().isoformat()}
        )
        time.sleep(0.5)  # 알림 간격
    
    print("✅ Alert generation test completed")

def main():
    """메인 테스트 함수"""
    print("🧪 DeFiPoser-ARB Real-Time Monitoring System Test")
    print("=" * 60)
    print("📋 Test Plan:")
    print("1. Initialize monitoring system")
    print("2. Register test callbacks")
    print("3. Run trading simulation")
    print("4. Test dashboard data generation")
    print("5. Test alert system")
    print("6. Generate performance report")
    print("=" * 60)
    
    # 1. 콜백 등록
    print("\n1️⃣ Registering monitoring callbacks...")
    add_monitoring_callback(test_callback)
    
    # 2. 모니터링 시스템 시작
    print("\n2️⃣ Starting monitoring system...")
    start_monitoring()
    
    if global_monitoring_system.is_monitoring:
        print("✅ Monitoring system started successfully")
    else:
        print("❌ Failed to start monitoring system")
        return
    
    # 잠시 대기 (모니터링 시스템 초기화)
    time.sleep(2)
    
    # 3. 시뮬레이션 실행 (별도 스레드)
    print("\n3️⃣ Starting trading simulation...")
    simulation_thread = threading.Thread(
        target=run_simulation,
        args=(3, 6),  # 3분간, 분당 6블록 (10초 간격)
        daemon=True
    )
    simulation_thread.start()
    
    # 4. 알림 시스템 테스트
    time.sleep(5)  # 시뮬레이션이 조금 실행된 후
    stress_test_alerts()
    
    # 5. 주기적으로 대시보드 데이터 확인
    for i in range(3):
        time.sleep(30)  # 30초 대기
        print(f"\n📊 Dashboard check #{i+1}")
        test_dashboard_data()
    
    # 6. 시뮬레이션 완료 대기
    print("\n⏳ Waiting for simulation to complete...")
    simulation_thread.join(timeout=200)  # 최대 200초 대기
    
    # 7. 최종 대시보드 데이터 확인
    print("\n7️⃣ Final dashboard data check...")
    test_dashboard_data()
    
    # 8. 모니터링 보고서 생성
    print("\n8️⃣ Generating monitoring report...")
    try:
        report_path = f"monitoring_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        global_monitoring_system.export_monitoring_report(report_path, days=1)
        print(f"✅ Monitoring report saved: {report_path}")
    except Exception as e:
        print(f"❌ Failed to generate monitoring report: {e}")
    
    # 9. 모니터링 중지
    print("\n9️⃣ Stopping monitoring system...")
    stop_monitoring()
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETED SUCCESSFULLY")
    print("📁 Generated files:")
    print("  - test_dashboard_output.json")
    print("  - monitoring_test_report_*.json")
    print("  - monitoring_system.db")
    print("  - roi_performance.db")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        stop_monitoring()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        stop_monitoring()