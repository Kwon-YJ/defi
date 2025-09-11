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
from config.config import config  # .env 로드를 위해 추가
from src.backtester import Backtester, BacktestConfig
from src.monitor import RealtimeMonitor
from src.roi_tracker import ROITracker, ROIConfig
from src.stability_tester import StabilityTester

logger = setup_logger(__name__)

async def run_testnet_validation():
    """테스트넷 검증 실행"""
    logger.info("=== 테스트넷 검증 시작 ===")
    validator = TestnetValidator()
    results = await validator.run_validation_suite()
    
    for test_name, result in results.items():
        # overall_score는 건너뛰기 (float 값)
        if test_name == 'overall_score':
            continue
            
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
    
    # 주간 수익 목표 평가 (논문 기준 191.48 ETH)
    weekly_target = await analyzer.evaluate_weekly_target()
    logger.info(f"주간 수익 목표 평가: {weekly_target}")
    
    # ROI 예측
    roi_projection = await analyzer.calculate_roi_projection(1550.0)  # $1550 투자
    logger.info(f"ROI 예측: {roi_projection}")

async def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='DeFi Arbitrage Validator')
    parser.add_argument('--mode', choices=['validate', 'detect', 'analyze', 'backtest', 'monitor', 'serve', 'roi', 'stability', 'flash', 'full'], 
                       default='full', help='실행 모드 선택')
    parser.add_argument('--skip-validation', action='store_true', 
                       help='테스트넷 검증 건너뛰기')
    # backtest 옵션
    parser.add_argument('--input', help='백테스트 입력 파일(JSONL/JSON/CSV). 미지정 시 합성', default=None)
    parser.add_argument('--days', type=int, default=150, help='백테스트 일수 (기본 150)')
    parser.add_argument('--synthesize', action='store_true', help='입력 대신 합성 데이터 사용')
    parser.add_argument('--daily-mean-eth', type=float, default=3.0, help='합성: 일평균 ETH 수익')
    parser.add_argument('--daily-std-eth', type=float, default=1.0, help='합성: 일표준편차 ETH')
    parser.add_argument('--out', default='reports', help='리포트 출력 디렉터리')
    parser.add_argument('--no-html', action='store_true', help='HTML 대시보드 생성 비활성화')
    # monitor 옵션
    parser.add_argument('--interval', type=int, default=None, help='모니터링 주기(초)')
    # serve 옵션
    parser.add_argument('--port', type=int, default=8000, help='대시보드 정적 서버 포트')
    # roi 옵션
    parser.add_argument('--roi-days', type=int, default=150, help='ROI 계산 기간(일)')
    parser.add_argument('--roi-capital-eth', type=float, default=None, help='ROI 초기 자본(ETH)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'validate':
            success = await run_testnet_validation()
            sys.exit(0 if success else 1)
            
        elif args.mode == 'detect':
            await run_arbitrage_detection()
            
        elif args.mode == 'analyze':
            await run_performance_analysis()
        
        elif args.mode == 'backtest':
            # 동기 백테스터 실행
            btcfg = BacktestConfig(
                input_path=args.input,
                days=args.days,
                synthesize=args.synthesize,
                synthetic_daily_mean_eth=args.daily_mean_eth,
                synthetic_daily_std_eth=args.daily_std_eth,
                out_dir=args.out,
                emit_html=not args.no_html,
            )
            bt = Backtester(btcfg)
            logger.info(f"=== 백테스트 시작 ({btcfg.days}일) ===")
            result = bt.run()
            logger.info(f"요약: total_profit={result['total_profit_eth']:.4f} ETH, avg_daily={result['avg_daily_profit_eth']:.4f} ETH, "
                        f"weeks_meeting_target={result['weeks_meeting_target']}/{len(result['weekly_buckets'])}")
            paths = bt.save_reports(result)
            logger.info(f"리포트 저장: {paths}")
        
        elif args.mode == 'monitor':
            monitor = RealtimeMonitor(interval_sec=args.interval)
            await monitor.run()

        elif args.mode == 'serve':
            import http.server
            import socketserver
            import os
            out_dir = getattr(config, 'dashboard_output_dir', 'reports')
            if not os.path.isdir(out_dir):
                logger.info(f"디렉터리 생성: {out_dir}")
                os.makedirs(out_dir, exist_ok=True)
            os.chdir(out_dir)
            handler = http.server.SimpleHTTPRequestHandler
            with socketserver.TCPServer(("0.0.0.0", args.port), handler) as httpd:
                logger.info(f"정적 서버 시작: http://0.0.0.0:{args.port} (dir={out_dir})")
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    logger.info("서버 종료")
            
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

        elif args.mode == 'roi':
            cfg = ROIConfig(lookback_days=args.roi_days,
                            initial_capital_eth=(args.roi_capital_eth if args.roi_capital_eth is not None else float(getattr(config, 'roi_initial_capital_eth', 1.0))))
            tracker = ROITracker(cfg)
            logger.info(f"=== ROI 리포트 생성 ({cfg.lookback_days}일) ===")
            rep = await tracker.generate_report()
            logger.info({k: rep[k] for k in ['total_profit_eth','avg_daily_profit_eth','sharpe_like','max_drawdown_eth','roi_percentage']})
            # 간단 저장
            import json, os
            os.makedirs('reports', exist_ok=True)
            from datetime import datetime
            p = f"reports/roi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(rep, f, ensure_ascii=False, indent=2)
            logger.info(f"ROI 리포트 저장: {p}")

        elif args.mode == 'stability':
            tester = StabilityTester()
            res = await tester.run()
            logger.info(f"안정성 테스트: total={res.total_actions}, executed={res.executed}, succeeded={res.succeeded}, failed={res.failed}, duration={res.duration_sec:.2f}s")
            if res.errors:
                logger.debug(f"에러 목록: {res.errors[:5]}{'...' if len(res.errors)>5 else ''}")

        elif args.mode == 'flash':
            # 한 번 탐지 후 최상위 기회에 대해 Flash 실행 경로 수행
            det = ArbitrageDetector()
            await det._run_detection(block_number=None, reason='manual-flash')
                
    except KeyboardInterrupt:
        logger.info("프로그램 종료")
    except Exception as e:
        logger.error(f"실행 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
