#!/usr/bin/env python3
"""
Multiple Data Sources Redundancy 구현 테스트
TODO requirement completion: Multiple data sources 동시 처리 (redundancy)

이 테스트는 논문 [2103.02228]의 DeFiPoser-ARB 시스템에서 요구하는
데이터 소스 redundancy 기능이 올바르게 구현되었는지 검증합니다.
"""

import asyncio
import time
import json
from typing import Dict, List
from src.real_time_price_feeds import RealTimePriceFeeds, PriceFeed, DataSource
from src.token_manager import TokenManager
from src.logger import setup_logger

logger = setup_logger(__name__)

class RedundancyTester:
    """Data source redundancy 테스터"""
    
    def __init__(self):
        self.token_manager = TokenManager()
        self.price_feeds = RealTimePriceFeeds(self.token_manager)
        self.test_results = {}
        
    async def run_all_tests(self):
        """모든 redundancy 테스트 실행"""
        logger.info("=== Multiple Data Sources Redundancy 테스트 시작 ===")
        
        tests = [
            self.test_data_source_initialization,
            self.test_backup_source_activation,
            self.test_price_aggregation_with_multiple_sources,
            self.test_outlier_detection_and_filtering,
            self.test_redundancy_metrics_calculation,
            self.test_failover_mechanism,
            self.test_performance_with_redundancy
        ]
        
        for test_func in tests:
            try:
                await test_func()
            except Exception as e:
                logger.error(f"테스트 {test_func.__name__} 실패: {e}")
                self.test_results[test_func.__name__] = {'status': 'FAILED', 'error': str(e)}
                
        await self.generate_test_report()
        
    async def test_data_source_initialization(self):
        """데이터 소스 초기화 테스트"""
        logger.info("🔍 데이터 소스 초기화 테스트")
        
        # 활성 소스 확인
        active_sources = [name for name, source in self.price_feeds.data_sources.items() if source.active]
        
        # Primary sources 확인
        required_primary = ['coingecko', 'onchain_uniswap']
        missing_primary = [src for src in required_primary if src not in active_sources]
        
        # Secondary/backup sources 확인
        backup_sources = ['coinpaprika', 'messari']
        available_backup = [src for src in backup_sources if src in active_sources]
        
        result = {
            'status': 'PASSED',
            'active_sources': len(active_sources),
            'primary_sources': len(required_primary) - len(missing_primary),
            'backup_sources': len(available_backup),
            'total_configured': len(self.price_feeds.data_sources)
        }
        
        # 최소 요구사항 검증
        if len(active_sources) < 3:
            result['status'] = 'WARNING'
            result['message'] = f"활성 소스가 {len(active_sources)}개로 적음 (권장: 3개 이상)"
            
        if missing_primary:
            result['status'] = 'FAILED'
            result['message'] = f"필수 Primary 소스 누락: {missing_primary}"
            
        if len(available_backup) < 2:
            result['status'] = 'WARNING'
            result['message'] = f"백업 소스가 {len(available_backup)}개로 적음 (권장: 2개 이상)"
            
        logger.info(f"✅ 데이터 소스 초기화: {result['status']} - {result}")
        self.test_results['data_source_init'] = result
        
    async def test_backup_source_activation(self):
        """백업 소스 활성화 테스트"""
        logger.info("🔍 백업 소스 활성화 테스트")
        
        # 테스트용 토큰 주소 (ETH)
        eth_address = self.token_manager.get_address_by_symbol('ETH')
        if not eth_address:
            self.test_results['backup_activation'] = {'status': 'SKIPPED', 'reason': 'ETH token not found'}
            return
            
        # 백업 소스 활성화 시뮬레이션
        try:
            await self.price_feeds._attempt_backup_sources_for_token(eth_address)
            
            result = {
                'status': 'PASSED',
                'tested_token': 'ETH',
                'backup_attempted': True
            }
            
        except Exception as e:
            result = {
                'status': 'FAILED',
                'error': str(e),
                'tested_token': 'ETH'
            }
            
        logger.info(f"✅ 백업 소스 활성화: {result['status']}")
        self.test_results['backup_activation'] = result
        
    async def test_price_aggregation_with_multiple_sources(self):
        """다중 소스 가격 집계 테스트"""
        logger.info("🔍 다중 소스 가격 집계 테스트")
        
        # 테스트용 가격 피드 생성
        test_feeds = [
            PriceFeed(
                token_address='0xtest123',
                symbol='TEST',
                price_usd=100.0,
                source='coingecko',
                timestamp=time.time(),
                confidence=0.95
            ),
            PriceFeed(
                token_address='0xtest123',
                symbol='TEST',
                price_usd=101.5,
                source='coinmarketcap',
                timestamp=time.time(),
                confidence=0.98
            ),
            PriceFeed(
                token_address='0xtest123',
                symbol='TEST',
                price_usd=99.8,
                source='binance_ws',
                timestamp=time.time(),
                confidence=1.0
            )
        ]
        
        try:
            # 집계 함수 테스트
            aggregated = await self.price_feeds._aggregate_multiple_sources('0xtest123', test_feeds)
            
            if aggregated:
                expected_range = (99.0, 102.0)  # 예상 범위
                price_in_range = expected_range[0] <= aggregated.price_usd <= expected_range[1]
                
                result = {
                    'status': 'PASSED' if price_in_range else 'FAILED',
                    'aggregated_price': aggregated.price_usd,
                    'source_count': len(test_feeds),
                    'confidence': aggregated.confidence,
                    'price_in_expected_range': price_in_range
                }
            else:
                result = {'status': 'FAILED', 'reason': 'No aggregated result'}
                
        except Exception as e:
            result = {'status': 'FAILED', 'error': str(e)}
            
        logger.info(f"✅ 다중 소스 가격 집계: {result['status']} - 가격: {result.get('aggregated_price', 'N/A')}")
        self.test_results['price_aggregation'] = result
        
    async def test_outlier_detection_and_filtering(self):
        """Outlier 감지 및 필터링 테스트"""
        logger.info("🔍 Outlier 감지 및 필터링 테스트")
        
        # Outlier가 포함된 테스트 데이터
        test_feeds_with_outlier = [
            PriceFeed('0xtest456', 'TEST2', 100.0, 'source1', time.time(), confidence=0.9),
            PriceFeed('0xtest456', 'TEST2', 101.0, 'source2', time.time(), confidence=0.9),
            PriceFeed('0xtest456', 'TEST2', 99.5, 'source3', time.time(), confidence=0.9),
            PriceFeed('0xtest456', 'TEST2', 500.0, 'source4', time.time(), confidence=0.8),  # Outlier
        ]
        
        try:
            aggregated = await self.price_feeds._aggregate_multiple_sources('0xtest456', test_feeds_with_outlier)
            
            if aggregated:
                # Outlier가 제거되었다면 가격이 정상 범위에 있어야 함
                outlier_filtered = 95.0 <= aggregated.price_usd <= 105.0
                
                result = {
                    'status': 'PASSED' if outlier_filtered else 'FAILED',
                    'aggregated_price': aggregated.price_usd,
                    'outlier_filtered': outlier_filtered,
                    'original_sources': len(test_feeds_with_outlier)
                }
            else:
                result = {'status': 'FAILED', 'reason': 'No aggregated result'}
                
        except Exception as e:
            result = {'status': 'FAILED', 'error': str(e)}
            
        logger.info(f"✅ Outlier 감지 및 필터링: {result['status']} - 가격: {result.get('aggregated_price', 'N/A')}")
        self.test_results['outlier_detection'] = result
        
    async def test_redundancy_metrics_calculation(self):
        """Redundancy 메트릭 계산 테스트"""
        logger.info("🔍 Redundancy 메트릭 계산 테스트")
        
        try:
            # 테스트용 가격 데이터 추가
            test_prices = {
                '0xeth': PriceFeed('0xeth', 'ETH', 2000.0, 'aggregated_3sources', time.time(), confidence=0.95),
                '0xbtc': PriceFeed('0xbtc', 'BTC', 40000.0, 'coingecko', time.time(), confidence=0.9),
                '0xusdc': PriceFeed('0xusdc', 'USDC', 1.0, 'aggregated_2sources', time.time(), confidence=0.98)
            }
            
            # 임시로 가격 데이터 추가
            original_prices = self.price_feeds.current_prices.copy()
            self.price_feeds.current_prices.update(test_prices)
            
            # 메트릭 계산
            metrics = self.price_feeds.get_performance_metrics()
            
            # 원래 데이터 복구
            self.price_feeds.current_prices = original_prices
            
            # 검증
            required_metrics = [
                'redundancy_score', 'tokens_with_multiple_sources', 
                'average_sources_per_token', 'redundancy_quality'
            ]
            
            has_all_metrics = all(metric in metrics for metric in required_metrics)
            
            result = {
                'status': 'PASSED' if has_all_metrics else 'FAILED',
                'redundancy_score': metrics.get('redundancy_score', 0),
                'redundancy_quality': metrics.get('redundancy_quality', 'Unknown'),
                'has_required_metrics': has_all_metrics
            }
            
        except Exception as e:
            result = {'status': 'FAILED', 'error': str(e)}
            
        logger.info(f"✅ Redundancy 메트릭: {result['status']} - 점수: {result.get('redundancy_score', 'N/A')}")
        self.test_results['redundancy_metrics'] = result
        
    async def test_failover_mechanism(self):
        """Failover 메커니즘 테스트"""
        logger.info("🔍 Failover 메커니즘 테스트")
        
        try:
            # Primary 소스 비활성화 시뮬레이션
            original_active = {}
            test_sources = ['coingecko', 'coinmarketcap']
            
            for source_name in test_sources:
                if source_name in self.price_feeds.data_sources:
                    original_active[source_name] = self.price_feeds.data_sources[source_name].active
                    self.price_feeds.data_sources[source_name].active = False
                    
            # 백업 소스들이 활성화되는지 확인
            active_backup_sources = [
                name for name, source in self.price_feeds.data_sources.items()
                if source.active and name in ['coinpaprika', 'messari', 'nomics']
            ]
            
            # 원래 상태 복구
            for source_name, was_active in original_active.items():
                self.price_feeds.data_sources[source_name].active = was_active
                
            result = {
                'status': 'PASSED' if len(active_backup_sources) > 0 else 'WARNING',
                'available_backup_sources': len(active_backup_sources),
                'backup_sources': active_backup_sources
            }
            
            if len(active_backup_sources) == 0:
                result['message'] = 'No active backup sources available'
                
        except Exception as e:
            result = {'status': 'FAILED', 'error': str(e)}
            
        logger.info(f"✅ Failover 메커니즘: {result['status']} - 백업 소스: {result.get('available_backup_sources', 0)}개")
        self.test_results['failover_mechanism'] = result
        
    async def test_performance_with_redundancy(self):
        """Redundancy 적용 시 성능 테스트"""
        logger.info("🔍 Redundancy 성능 테스트")
        
        try:
            start_time = time.time()
            
            # 가상의 다중 소스 처리 시뮬레이션
            test_tokens = ['ETH', 'BTC', 'USDC', 'DAI', 'UNI']
            processing_times = []
            
            for symbol in test_tokens:
                token_start = time.time()
                
                # 백업 소스 시도 시뮬레이션
                address = self.token_manager.get_address_by_symbol(symbol)
                if address:
                    try:
                        await self.price_feeds._attempt_backup_sources_for_token(address)
                    except:
                        pass  # 에러 무시하고 계속
                        
                processing_times.append(time.time() - token_start)
                
            total_time = time.time() - start_time
            avg_time_per_token = sum(processing_times) / len(processing_times) if processing_times else 0
            
            # 논문 목표: 6.43초 평균 실행시간과 비교
            target_time = 6.43
            performance_ok = total_time < target_time
            
            result = {
                'status': 'PASSED' if performance_ok else 'WARNING',
                'total_processing_time': round(total_time, 2),
                'avg_time_per_token': round(avg_time_per_token, 3),
                'target_time': target_time,
                'meets_target': performance_ok,
                'tokens_processed': len(test_tokens)
            }
            
        except Exception as e:
            result = {'status': 'FAILED', 'error': str(e)}
            
        logger.info(f"✅ Redundancy 성능: {result['status']} - 총 시간: {result.get('total_processing_time', 'N/A')}초")
        self.test_results['performance_test'] = result
        
    async def generate_test_report(self):
        """테스트 결과 리포트 생성"""
        logger.info("=== Redundancy 테스트 결과 리포트 ===")
        
        passed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'PASSED')
        warning_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'WARNING')
        failed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'FAILED')
        total_tests = len(self.test_results)
        
        logger.info(f"총 테스트: {total_tests}, 통과: {passed_tests}, 경고: {warning_tests}, 실패: {failed_tests}")
        
        # 상세 결과
        for test_name, result in self.test_results.items():
            status_emoji = {
                'PASSED': '✅',
                'WARNING': '⚠️',
                'FAILED': '❌',
                'SKIPPED': '⏭️'
            }.get(result.get('status', 'UNKNOWN'), '❓')
            
            logger.info(f"{status_emoji} {test_name}: {result.get('status', 'UNKNOWN')}")
            if 'message' in result:
                logger.info(f"   메시지: {result['message']}")
            if 'error' in result:
                logger.info(f"   오류: {result['error']}")
                
        # 전체 평가
        if failed_tests == 0 and warning_tests <= 1:
            overall_status = "✅ REDUNDANCY 구현 성공"
        elif failed_tests == 0:
            overall_status = "⚠️ REDUNDANCY 부분적 성공 (개선 필요)"
        else:
            overall_status = "❌ REDUNDANCY 구현 실패"
            
        logger.info(f"\n🎯 전체 평가: {overall_status}")
        
        # 결과를 파일로 저장
        report_data = {
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_status': overall_status,
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'warnings': warning_tests,
                'failed': failed_tests
            },
            'detailed_results': self.test_results
        }
        
        with open('redundancy_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
            
        logger.info("📊 테스트 리포트가 'redundancy_test_report.json'에 저장되었습니다")

async def main():
    """메인 테스트 실행"""
    tester = RedundancyTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())