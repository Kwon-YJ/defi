#!/usr/bin/env python3
"""
Performance Benchmarking 테스트 스크립트
DEFIPOSER-ARB의 6.43초 목표 달성을 위한 실행 시간 측정 검증

이 스크립트는 성능 벤치마킹 시스템의 정확성과 유용성을 검증합니다.
"""

import asyncio
import time
import random
from src.performance_benchmarking import (
    start_benchmarking, end_benchmarking, time_component,
    get_performance_report, start_monitoring, PerformanceBenchmarker
)
from src.logger import setup_logger

logger = setup_logger(__name__)

async def simulate_block_processing():
    """실제 블록 처리를 시뮬레이션"""
    
    print("🧪 DEFIPOSER-ARB 성능 벤치마킹 테스트 시작")
    print("📊 목표: 평균 6.43초 이하 실행 시간 달성")
    
    # 실시간 모니터링 시작
    start_monitoring(check_interval=5)  # 5초마다 체크
    
    # 블록 처리 시뮬레이션
    for block_num in range(19000000, 19000020):  # 20개 블록 처리
        await simulate_single_block(block_num)
        
        # 블록 간 대기 (이더리움 평균 블록 시간: 13.5초)
        await asyncio.sleep(1)  # 테스트를 위해 짧게 설정
    
    # 최종 성능 보고서 출력
    await generate_final_report()

async def simulate_single_block(block_number: int):
    """단일 블록 처리 시뮬레이션"""
    
    # 블록 처리 시작
    start_benchmarking(block_number)
    
    logger.info(f"블록 {block_number} 처리 시작")
    
    # 1. 그래프 구축 시뮬레이션 (0.5-2.5초)
    with time_component("graph_building"):
        graph_time = random.uniform(0.5, 2.5)
        await asyncio.sleep(graph_time)
        logger.debug(f"그래프 구축 완료: {graph_time:.3f}초")
    
    # 2. Negative Cycle 탐지 시뮬레이션 (1.0-4.0초)
    with time_component("negative_cycle_detection"):
        cycle_time = random.uniform(1.0, 4.0)
        await asyncio.sleep(cycle_time)
        logger.debug(f"Negative Cycle 탐지 완료: {cycle_time:.3f}초")
    
    # 3. Local Search 시뮬레이션 (0.5-2.0초)
    with time_component("local_search"):
        search_time = random.uniform(0.5, 2.0)
        await asyncio.sleep(search_time)
        logger.debug(f"Local Search 완료: {search_time:.3f}초")
    
    # 4. 파라미터 최적화 시뮬레이션 (0.2-1.5초)
    with time_component("parameter_optimization"):
        param_time = random.uniform(0.2, 1.5)
        await asyncio.sleep(param_time)
        logger.debug(f"파라미터 최적화 완료: {param_time:.3f}초")
    
    # 5. 검증 시뮬레이션 (0.1-0.5초)
    with time_component("validation"):
        validation_time = random.uniform(0.1, 0.5)
        await asyncio.sleep(validation_time)
        logger.debug(f"검증 완료: {validation_time:.3f}초")
    
    # 랜덤 결과 생성
    opportunities_found = random.randint(0, 8)
    strategies_executed = min(opportunities_found, random.randint(0, 3))
    total_revenue = random.uniform(0, 15) if strategies_executed > 0 else 0
    gas_cost = random.uniform(0.01, 0.08) if strategies_executed > 0 else 0
    
    # 블록 처리 완료
    metrics = end_benchmarking(
        opportunities_found=opportunities_found,
        strategies_executed=strategies_executed,
        total_revenue=total_revenue,
        gas_cost=gas_cost
    )
    
    # 결과 출력
    if metrics.total_execution_time <= 6.43:
        status = "✅ 목표 달성"
        log_func = logger.info
    elif metrics.total_execution_time <= 6.43 * 1.2:  # 20% 여유
        status = "⚠️ 목표 근접"
        log_func = logger.warning
    else:
        status = "❌ 목표 초과"
        log_func = logger.error
    
    log_func(
        f"블록 {block_number}: {metrics.total_execution_time:.3f}초 "
        f"({status}) | 기회: {opportunities_found}개 | "
        f"실행: {strategies_executed}개 | 수익: {total_revenue:.3f} ETH"
    )

async def generate_final_report():
    """최종 성능 보고서 생성"""
    
    print("\n" + "="*80)
    print("📊 DEFIPOSER-ARB 성능 벤치마킹 최종 보고서")
    print("="*80)
    
    # 전체 성능 보고서 조회
    report = get_performance_report()
    
    if "error" in report:
        print(f"❌ 오류: {report['error']}")
        return
    
    summary = report["summary"]
    
    # 기본 통계
    print(f"🎯 목표 시간: {summary['target_time']:.3f}초")
    print(f"📈 분석된 블록: {summary['blocks_analyzed']}개")
    print(f"✅ 성공률: {summary['success_rate']:.1%}")
    print(f"⏱️ 평균 시간: {summary['average_time']:.3f}초")
    print(f"🏃 최고 기록: {summary['fastest_time']:.3f}초")
    print(f"🐌 최악 기록: {summary['slowest_time']:.3f}초")
    print(f"📊 표준편차: {summary['std_deviation']:.3f}초")
    
    # 성능 평가
    print(f"\n🏆 성능 평가:")
    if summary['success_rate'] >= 0.9:
        print("   우수함 - 논문 기준을 안정적으로 만족")
    elif summary['success_rate'] >= 0.7:
        print("   양호함 - 대부분 논문 기준을 만족")
    elif summary['success_rate'] >= 0.5:
        print("   개선 필요 - 논문 기준 달성률 부족")
    else:
        print("   심각함 - 논문 기준 대폭 미달")
    
    # 컴포넌트 분석
    if "component_analysis" in report and report["component_analysis"]:
        print(f"\n🔍 컴포넌트별 성능 분석:")
        for component, data in report["component_analysis"].items():
            print(f"   {component}: {data['average']:.3f}초 "
                  f"({data['percentage_of_total']:.1f}% of target)")
    
    # 리소스 사용량
    if "resource_usage" in report and report["resource_usage"]:
        print(f"\n💻 리소스 사용량:")
        resource = report["resource_usage"]
        if "memory" in resource:
            print(f"   메모리: 평균 {resource['memory']['average_mb']:.0f}MB, "
                  f"최대 {resource['memory']['max_mb']:.0f}MB")
        if "cpu" in resource:
            print(f"   CPU: 평균 {resource['cpu']['average_percent']:.1f}%, "
                  f"최대 {resource['cpu']['max_percent']:.1f}%")
    
    # 권장사항
    print(f"\n💡 권장사항:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"   {i}. {rec}")
    
    print("\n" + "="*80)

def test_benchmarker_api():
    """벤치마커 API 테스트"""
    print("🔧 PerformanceBenchmarker API 테스트")
    
    benchmarker = PerformanceBenchmarker(target_time=5.0)
    
    # 가상의 블록 처리 시뮬레이션
    for block_num in range(1, 6):
        benchmarker.start_block_processing(block_num)
        
        # 컴포넌트별 시간 측정
        with benchmarker.time_component("test_component"):
            time.sleep(random.uniform(0.5, 1.5))
        
        # 블록 처리 완료
        metrics = benchmarker.end_block_processing(
            opportunities_found=random.randint(0, 5),
            strategies_executed=random.randint(0, 2),
            total_revenue=random.uniform(0, 10),
            gas_cost=random.uniform(0.01, 0.05)
        )
        
        print(f"블록 {block_num}: {metrics.total_execution_time:.3f}초")
    
    # 보고서 생성 및 출력
    report = benchmarker.get_performance_report()
    success_rate = report["summary"]["success_rate"]
    avg_time = report["summary"]["average_time"]
    
    print(f"✅ API 테스트 완료: 성공률 {success_rate:.1%}, "
          f"평균 시간 {avg_time:.3f}초")

async def main():
    """메인 테스트 함수"""
    print("🚀 DEFIPOSER-ARB Performance Benchmarking Test Suite")
    print("📈 논문 목표: 평균 6.43초 이하 실행 시간")
    
    # 1. 기본 API 테스트
    test_benchmarker_api()
    
    print("\n" + "-"*60 + "\n")
    
    # 2. 실시간 블록 처리 시뮬레이션
    await simulate_block_processing()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 테스트 중단됨")
    except Exception as e:
        logger.error(f"테스트 오류: {e}")
        raise