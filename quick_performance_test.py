#!/usr/bin/env python3
"""
Quick Performance Benchmarking Test
빠른 성능 벤치마킹 테스트를 위한 스크립트
"""

import asyncio
import time
import random
from src.performance_benchmarking import (
    start_benchmarking, end_benchmarking, time_component,
    get_performance_report
)
from src.logger import setup_logger

logger = setup_logger(__name__)

async def quick_test():
    """빠른 성능 테스트"""
    
    print("⚡ DEFIPOSER-ARB 빠른 성능 테스트")
    print("📊 목표: 평균 6.43초 이하 실행 시간")
    
    # 5개 블록만 처리
    for block_num in range(19000000, 19000005):
        await simulate_block(block_num)
    
    # 성능 보고서 출력
    report = get_performance_report()
    
    if "error" not in report:
        summary = report["summary"]
        print(f"\n📈 성능 결과:")
        print(f"   성공률: {summary['success_rate']:.1%}")
        print(f"   평균 시간: {summary['average_time']:.3f}초")
        print(f"   최고 기록: {summary['fastest_time']:.3f}초")
        print(f"   최악 기록: {summary['slowest_time']:.3f}초")
        
        if summary['success_rate'] >= 0.8:
            print("✅ 논문 기준을 만족합니다!")
        else:
            print("⚠️ 성능 개선이 필요합니다.")
            
        # 권장사항 출력
        print(f"\n💡 권장사항:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
    else:
        print(f"❌ 성능 분석 실패: {report['error']}")

async def simulate_block(block_number: int):
    """블록 처리 시뮬레이션"""
    
    start_benchmarking(block_number)
    
    # 컴포넌트별 처리 시뮬레이션
    with time_component("graph_building"):
        await asyncio.sleep(random.uniform(0.5, 1.5))
    
    with time_component("negative_cycle_detection"):
        await asyncio.sleep(random.uniform(1.0, 3.0))
        
    with time_component("local_search"):
        await asyncio.sleep(random.uniform(0.5, 2.0))
    
    with time_component("parameter_optimization"):
        await asyncio.sleep(random.uniform(0.2, 1.0))
    
    with time_component("validation"):
        await asyncio.sleep(random.uniform(0.1, 0.5))
    
    # 결과 생성
    opportunities = random.randint(0, 5)
    strategies = min(opportunities, random.randint(0, 3))
    revenue = random.uniform(0, 10) if strategies > 0 else 0
    
    # 벤치마킹 완료
    metrics = end_benchmarking(
        opportunities_found=opportunities,
        strategies_executed=strategies,
        total_revenue=revenue,
        gas_cost=0.02
    )
    
    status = "✅" if metrics.total_execution_time <= 6.43 else "❌"
    print(f"블록 {block_number}: {metrics.total_execution_time:.3f}초 {status}")

if __name__ == "__main__":
    try:
        asyncio.run(quick_test())
    except KeyboardInterrupt:
        print("\n🛑 테스트 중단됨")
    except Exception as e:
        logger.error(f"테스트 오류: {e}")