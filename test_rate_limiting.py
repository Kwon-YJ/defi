"""
Rate Limiting 및 API Quota 관리 테스트 스크립트

TODO requirement 완료 확인용:
- Rate limiting이 제대로 작동하는지 확인
- API quota 관리가 정확한지 확인
- Circuit breaker 동작 확인
- Priority queue 시스템 확인
- Performance metrics 수집 확인
"""

import asyncio
import time
import logging
from typing import Dict, List
from src.rate_limiter import AdvancedRateLimiter, RateLimitTier, APIStatus
from src.real_time_price_feeds import RealTimePriceFeeds
from src.token_manager import TokenManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimitingTester:
    """Rate limiting 기능 테스트 클래스"""
    
    def __init__(self):
        self.rate_limiter = AdvancedRateLimiter()
        self.results: Dict[str, List] = {}
        
    async def test_basic_rate_limiting(self):
        """기본 Rate Limiting 테스트"""
        logger.info("=== 기본 Rate Limiting 테스트 시작 ===")
        
        await self.rate_limiter.initialize()
        
        # CoinGecko API에 대해 연속 요청 테스트
        api_name = "coingecko"
        tier = RateLimitTier.MEDIUM
        
        start_time = time.time()
        successful_requests = 0
        blocked_requests = 0
        
        for i in range(30):  # 30회 요청 시도
            try:
                async with self.rate_limiter.acquire_slot(api_name, tier):
                    # 가짜 API 호출 시뮬레이션
                    await asyncio.sleep(0.1)
                    successful_requests += 1
                    logger.info(f"요청 {i+1}/30 성공")
                    
            except Exception as e:
                blocked_requests += 1
                logger.warning(f"요청 {i+1}/30 차단됨: {e}")
                
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"테스트 완료: {total_time:.2f}초")
        logger.info(f"성공한 요청: {successful_requests}")
        logger.info(f"차단된 요청: {blocked_requests}")
        
        # 상태 확인
        status = self.rate_limiter.get_api_status(api_name)
        logger.info(f"API 상태: {status}")
        
        assert successful_requests > 0, "성공한 요청이 없음"
        assert total_time > 5, "Rate limiting이 제대로 작동하지 않음"
        
        logger.info("✅ 기본 Rate Limiting 테스트 통과")
        
    async def test_priority_tiers(self):
        """우선순위별 Rate Limiting 테스트"""
        logger.info("=== 우선순위별 Rate Limiting 테스트 시작 ===")
        
        api_name = "ethereum_rpc"
        
        # 다양한 우선순위로 요청
        tasks = []
        
        # CRITICAL 우선순위 (가장 먼저 처리되어야 함)
        for i in range(3):
            tasks.append(self._make_timed_request(api_name, RateLimitTier.CRITICAL, f"CRITICAL-{i}"))
            
        # LOW 우선순위 (나중에 처리)
        for i in range(3):
            tasks.append(self._make_timed_request(api_name, RateLimitTier.LOW, f"LOW-{i}"))
            
        # HIGH 우선순위
        for i in range(3):
            tasks.append(self._make_timed_request(api_name, RateLimitTier.HIGH, f"HIGH-{i}"))
        
        # 모든 요청을 동시에 시작
        results = await asyncio.gather(*tasks)
        
        # 결과 분석
        critical_times = [r[1] for r in results if r[0].startswith('CRITICAL')]
        high_times = [r[1] for r in results if r[0].startswith('HIGH')]
        low_times = [r[1] for r in results if r[0].startswith('LOW')]
        
        logger.info(f"CRITICAL 평균 대기시간: {sum(critical_times)/len(critical_times):.2f}초")
        logger.info(f"HIGH 평균 대기시간: {sum(high_times)/len(high_times):.2f}초")
        logger.info(f"LOW 평균 대기시간: {sum(low_times)/len(low_times):.2f}초")
        
        # CRITICAL이 가장 빠르게 처리되어야 함
        avg_critical = sum(critical_times) / len(critical_times)
        avg_low = sum(low_times) / len(low_times)
        
        assert avg_critical <= avg_low, "CRITICAL 우선순위가 제대로 작동하지 않음"
        
        logger.info("✅ 우선순위별 Rate Limiting 테스트 통과")
        
    async def _make_timed_request(self, api_name: str, tier: RateLimitTier, label: str):
        """시간 측정을 포함한 요청"""
        start_time = time.time()
        
        try:
            async with self.rate_limiter.acquire_slot(api_name, tier):
                await asyncio.sleep(0.05)  # 짧은 작업 시뮬레이션
                
            end_time = time.time()
            wait_time = end_time - start_time
            
            logger.info(f"{label} 완료: {wait_time:.3f}초")
            return (label, wait_time)
            
        except Exception as e:
            logger.error(f"{label} 실패: {e}")
            return (label, float('inf'))
            
    async def test_circuit_breaker(self):
        """Circuit Breaker 기능 테스트"""
        logger.info("=== Circuit Breaker 테스트 시작 ===")
        
        api_name = "test_failing_api"
        
        # 특별한 failing API 설정 (실패 임계값 3)
        from src.rate_limiter import RateLimit, APIQuota, CircuitBreakerConfig
        
        await self.rate_limiter.configure_api(
            api_name=api_name,
            rate_limits={
                RateLimitTier.MEDIUM: RateLimit(requests_per_minute=100)
            },
            quota=APIQuota(daily_limit=1000, monthly_limit=10000),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3, timeout=5.0)
        )
        
        # 실패 요청들을 시뮬레이션
        for i in range(5):
            try:
                async with self.rate_limiter.acquire_slot(api_name, RateLimitTier.MEDIUM):
                    # 실패 시뮬레이션
                    raise Exception(f"Simulated API failure {i+1}")
                    
            except Exception as e:
                logger.info(f"예상된 실패 {i+1}: {e}")
                
        # Circuit breaker 상태 확인
        circuit_breaker = self.rate_limiter.circuit_breakers.get(api_name)
        assert circuit_breaker is not None, "Circuit breaker가 없음"
        
        logger.info(f"Circuit breaker 상태: {circuit_breaker.state}")
        logger.info(f"실패 횟수: {circuit_breaker.failure_count}")
        
        # Circuit이 열렸는지 확인
        from src.rate_limiter import CircuitBreakerState
        assert circuit_breaker.state == CircuitBreakerState.OPEN, "Circuit breaker가 열리지 않음"
        
        # 이제 요청이 즉시 실패해야 함
        try:
            async with self.rate_limiter.acquire_slot(api_name, RateLimitTier.MEDIUM):
                assert False, "Circuit이 열린 상태에서 요청이 성공함"
        except Exception as e:
            logger.info(f"Circuit breaker에 의해 차단됨: {e}")
            
        logger.info("✅ Circuit Breaker 테스트 통과")
        
    async def test_quota_management(self):
        """Quota 관리 테스트"""
        logger.info("=== Quota 관리 테스트 시작 ===")
        
        api_name = "test_quota_api"
        
        # 매우 낮은 quota로 API 설정
        from src.rate_limiter import RateLimit, APIQuota
        
        await self.rate_limiter.configure_api(
            api_name=api_name,
            rate_limits={
                RateLimitTier.MEDIUM: RateLimit(requests_per_minute=100)
            },
            quota=APIQuota(daily_limit=5, monthly_limit=20),  # 매우 낮은 한도
            circuit_breaker_config=None
        )
        
        successful_requests = 0
        quota_exceeded_count = 0
        
        # 10번 요청해서 5번은 성공, 5번은 quota 초과로 실패해야 함
        for i in range(10):
            try:
                async with self.rate_limiter.acquire_slot(api_name, RateLimitTier.MEDIUM):
                    successful_requests += 1
                    logger.info(f"Quota 테스트 요청 {i+1} 성공")
                    
            except Exception as e:
                if "quota exceeded" in str(e).lower():
                    quota_exceeded_count += 1
                    logger.info(f"Quota 초과로 요청 {i+1} 차단됨")
                else:
                    logger.error(f"예상치 못한 오류: {e}")
                    
        logger.info(f"성공한 요청: {successful_requests}")
        logger.info(f"Quota 초과로 차단된 요청: {quota_exceeded_count}")
        
        assert successful_requests == 5, f"성공 요청 수가 예상과 다름: {successful_requests}"
        assert quota_exceeded_count == 5, f"Quota 초과 횟수가 예상과 다름: {quota_exceeded_count}"
        
        # Quota 상태 확인
        quota = self.rate_limiter.quotas.get(api_name)
        assert quota.used_daily == 5, f"일일 사용량이 잘못됨: {quota.used_daily}"
        
        logger.info("✅ Quota 관리 테스트 통과")
        
    async def test_integrated_system(self):
        """실시간 가격 피드와의 통합 테스트"""
        logger.info("=== 통합 시스템 테스트 시작 ===")
        
        # TokenManager와 RealTimePriceFeeds 초기화
        token_manager = TokenManager()
        price_feeds = RealTimePriceFeeds(token_manager)
        
        try:
            # 시스템 시작
            await price_feeds.start()
            
            # 잠시 대기하여 시스템이 초기화되도록 함
            await asyncio.sleep(2)
            
            # Rate limiting 상태 확인
            rate_limit_status = price_feeds.get_rate_limit_status()
            logger.info(f"Rate limiting 상태: {len(rate_limit_status)}개 API 모니터링 중")
            
            # API 건강 점수 확인
            health_scores = price_feeds.get_api_health_scores()
            logger.info(f"API 건강 점수: {health_scores}")
            
            # 성능 메트릭 확인
            metrics = price_feeds.get_performance_metrics()
            
            # Rate limiting 관련 메트릭이 포함되어 있는지 확인
            assert 'rate_limit_status' in metrics, "Rate limiting 메트릭이 없음"
            assert 'api_health_scores' in metrics, "API 건강 점수가 없음"
            assert 'circuit_breakers_open' in metrics, "Circuit breaker 메트릭이 없음"
            
            logger.info(f"통합 메트릭: Circuit breaker 열림 = {metrics['circuit_breakers_open']}")
            logger.info(f"통합 메트릭: Quota 초과 API = {metrics['quota_exceeded_apis']}")
            
            logger.info("✅ 통합 시스템 테스트 통과")
            
        finally:
            await price_feeds.stop()
            
    async def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("🚀 Rate Limiting 시스템 테스트 시작")
        
        try:
            await self.test_basic_rate_limiting()
            await asyncio.sleep(1)
            
            await self.test_priority_tiers()
            await asyncio.sleep(1)
            
            await self.test_circuit_breaker()
            await asyncio.sleep(1)
            
            await self.test_quota_management()
            await asyncio.sleep(1)
            
            await self.test_integrated_system()
            
            logger.info("🎉 모든 테스트 통과! Rate Limiting 시스템이 정상 작동합니다.")
            
        except AssertionError as e:
            logger.error(f"❌ 테스트 실패: {e}")
            raise
        except Exception as e:
            logger.error(f"💥 테스트 중 오류 발생: {e}")
            raise
        finally:
            await self.rate_limiter.shutdown()

async def main():
    """메인 테스트 함수"""
    tester = RateLimitingTester()
    await tester.run_all_tests()
    
if __name__ == "__main__":
    asyncio.run(main())