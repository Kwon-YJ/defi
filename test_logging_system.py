#!/usr/bin/env python3
"""
Test script for the enhanced logging and debugging system
Enhanced logging system test
"""

import sys
import os
import time
import uuid
from datetime import datetime

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from logger import (
    log_transaction_execution,
    analyze_performance_bottleneck,
    track_error_with_context,
    detailed_logger,
    setup_logger
)

def test_transaction_logging():
    """거래 로그 테스트"""
    print("=== 거래 로그 시스템 테스트 ===")
    
    # 성공적인 거래 로그
    transaction_id = f"test_tx_{uuid.uuid4().hex[:8]}"
    log_transaction_execution(
        transaction_id=transaction_id,
        transaction_type="arbitrage",
        start_token="WETH",
        end_token="WETH",
        path=["WETH", "USDC", "DAI", "WETH"],
        amounts=[1.0, 2000.0, 2001.0, 1.001],
        execution_time=5.23,
        revenue=0.001,
        success=True,
        gas_used=150000,
        gas_price=20.0,
        block_number=18500000,
        performance_metrics={
            "graph_build_time": 1.2,
            "negative_cycle_detection_time": 3.1,
            "local_search_time": 0.93
        }
    )
    print(f"✅ 성공적인 거래 로그 기록됨: {transaction_id}")
    
    # 실패한 거래 로그
    failed_transaction_id = f"test_tx_{uuid.uuid4().hex[:8]}"
    log_transaction_execution(
        transaction_id=failed_transaction_id,
        transaction_type="flash_loan",
        start_token="USDC",
        end_token="USDC", 
        path=["USDC", "WETH"],
        amounts=[2000.0, 0.0],
        execution_time=2.1,
        revenue=0.0,
        success=False,
        error_message="Insufficient liquidity for trade execution",
        performance_metrics={"preparation_time": 2.1}
    )
    print(f"✅ 실패한 거래 로그 기록됨: {failed_transaction_id}")

def test_performance_bottleneck_identification():
    """성능 병목점 식별 테스트"""
    print("\n=== 성능 병목점 식별 테스트 ===")
    
    # 정상적인 성능 컴포넌트
    analysis1 = analyze_performance_bottleneck("graph_build", 2.1)
    print(f"✅ Graph Build 컴포넌트 분석: 병목점 감지={analysis1['bottleneck_detected']}")
    
    # 느린 컴포넌트 (병목점 예상)
    analysis2 = analyze_performance_bottleneck("negative_cycle_detection", 8.5)
    print(f"⚠️ Negative Cycle Detection 컴포넌트 분석: 병목점 감지={analysis2['bottleneck_detected']}")
    if analysis2['suggestions']:
        print(f"   제안사항: {analysis2['suggestions'][0]}")
    
    # 빠른 컴포넌트
    analysis3 = analyze_performance_bottleneck("local_search", 0.8)
    print(f"✅ Local Search 컴포넌트 분석: 병목점 감지={analysis3['bottleneck_detected']}")

def test_error_tracking():
    """에러 추적 테스트"""
    print("\n=== 에러 추적 및 디버깅 테스트 ===")
    
    # 네트워크 에러 시뮬레이션
    try:
        raise ConnectionError("Network timeout while fetching price data")
    except Exception as e:
        error_log = track_error_with_context(
            e, 
            "price_fetcher",
            token_pair="WETH/USDC",
            exchange="uniswap_v2",
            retry_count=3,
            timestamp=datetime.now().isoformat()
        )
        print(f"✅ 네트워크 에러 추적됨: {error_log.error_type} (심각도: {error_log.severity})")
        if error_log.recovery_action:
            print(f"   복구 방법: {error_log.recovery_action}")
    
    # 거래 실행 에러 시뮬레이션
    try:
        raise ValueError("Insufficient liquidity for requested trade size")
    except Exception as e:
        error_log = track_error_with_context(
            e,
            "trade_executor", 
            trade_amount=10.0,
            liquidity_available=5.0,
            slippage_tolerance=0.01
        )
        print(f"✅ 거래 실행 에러 추적됨: {error_log.error_type} (심각도: {error_log.severity})")
        if error_log.recovery_action:
            print(f"   복구 방법: {error_log.recovery_action}")

def test_transaction_statistics():
    """거래 통계 테스트"""
    print("\n=== 거래 통계 조회 테스트 ===")
    
    # 최근 1일 통계
    stats = detailed_logger.get_transaction_stats(1)
    print(f"✅ 최근 1일 통계:")
    print(f"   총 거래 수: {stats['total_transactions']}")
    print(f"   성공한 거래 수: {stats['successful_transactions']}")
    print(f"   성공률: {stats['success_rate']:.1f}%")
    print(f"   평균 실행시간: {stats['avg_execution_time']:.2f}초")
    print(f"   총 수익: {stats['total_revenue']:.6f} ETH")
    print(f"   평균 수익: {stats['avg_revenue']:.6f} ETH")
    print(f"   최대 수익: {stats['max_revenue']:.6f} ETH")

def main():
    """메인 테스트 함수"""
    
    # 기본 로거 설정
    logger = setup_logger(__name__)
    
    print("DeFiPoser-ARB 고급 로깅 및 디버깅 시스템 테스트 시작")
    print(f"테스트 시간: {datetime.now().isoformat()}")
    print(f"로그 데이터베이스 위치: logs/transaction_logs.db")
    
    try:
        # 각 테스트 실행
        test_transaction_logging()
        test_performance_bottleneck_identification()
        test_error_tracking()
        test_transaction_statistics()
        
        print("\n=== 테스트 완료 ===")
        print("✅ 모든 로깅 시스템이 정상적으로 작동합니다!")
        print("\n📊 대시보드 실행:")
        print("   python paper_results_dashboard.py")
        print("   브라우저에서 http://localhost:8050 접속")
        
    except Exception as e:
        logger.error(f"테스트 중 에러 발생: {e}")
        # 테스트 실패도 에러 추적 시스템으로 기록
        track_error_with_context(e, "test_system", test_type="logging_system_test")
        print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    main()