#!/usr/bin/env python3
"""
DeFi Arbitrage Validator - Complete System Runner
전체 검증 시스템을 실행하는 메인 스크립트
"""

import asyncio
import argparse
import sys
from src.testnet_validator import TestnetValidator
from src.arbitrage_detector import ArbitrageDetector
from src.performance_analyzer import PerformanceAnalyzer
from src.logger import setup_logger

logger = setup_logger(__name__)

async def run_testnet_validation():
    """테스트넷 검증 실행"""
    logger.info("=== 테스트넷 검증 시작 ===")
    validator = TestnetValidator()
    results = await validator.run_validation_suite()
    
    for test_name, result in results.items():
        status = "✅ 성공" if result.get('success', False) else "❌ 실패"
        logger.info(f"{test_name}: {status}")
        
        if not result.get('success', False) and 'error' in result:
            logger.error(f"  오류: {result['error']}")
    
    logger.info(f"전체 성공률: {results['overall_score']:.1%}")
    return results['overall_score'] > 0.8

async def run_arbitrage_detection():
    """차익거래 탐지 실행"""
    logger.info("=== 차익거래 탐지 시작 ===")
    detector = ArbitrageDetector()
    
    try:
        await detector.start_detection()
    except KeyboardInterrupt:
        detector.stop_detection()
        logger.info("차익거래 탐지 중지")

async def run_performance_analysis():
    """성과 분석 실행"""
    logger.info("=== 성과 분석 시작 ===")
    analyzer = PerformanceAnalyzer()
    
    # 일일 보고서 생성
    daily_report = await analyzer.generate_daily_report()
    logger.info(f"일일 보고서: {daily_report}")
    
    # 주간 요약 생성
    weekly_summary = await analyzer.generate_weekly_summary()
    logger.info(f"주간 요약: {weekly_summary}")
    
    # ROI 예측
    roi_projection = await analyzer.calculate_roi_projection(1550.0)  # $1550 투자
    logger.info(f"ROI 예측: {roi_projection}")

async def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='DeFi Arbitrage Validator')
    parser.add_argument('--mode', choices=['validate', 'detect', 'analyze', 'full'], 
                       default='full', help='실행 모드 선택')
    parser.add_argument('--skip-validation', action='store_true', 
                       help='테스트넷 검증 건너뛰기')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'validate':
            success = await run_testnet_validation()
            sys.exit(0 if success else 1)
            
        elif args.mode == 'detect':
            await run_arbitrage_detection()
            
        elif args.mode == 'analyze':
            await run_performance_analysis()
            
        elif args.mode == 'full':
            # 전체 시스템 실행
            if not args.skip_validation:
                logger.info("1단계: 테스트넷 검증")
                validation_success = await run_testnet_validation()
                
                if not validation_success:
                    logger.error("테스트넷 검증 실패. 시스템을 점검하세요.")
                    sys.exit(1)
            
            logger.info("2단계: 차익거래 탐지 시작")
            # 백그라운드에서 탐지 실행
            detection_task = asyncio.create_task(run_arbitrage_detection())
            
            # 주기적으로 성과 분석 실행
            while True:
                await asyncio.sleep(3600)  # 1시간마다
                logger.info("성과 분석 실행")
                await run_performance_analysis()
                
    except KeyboardInterrupt:
        logger.info("프로그램 종료")
    except Exception as e:
        logger.error(f"실행 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
