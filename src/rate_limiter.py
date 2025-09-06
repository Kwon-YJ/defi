"""
Advanced Rate Limiting and API Quota Management System
TODO requirement completion: Rate limiting 및 API quota 관리

DeFiPoser-ARB 시스템을 위한 고급 API 호출 제한 및 할당량 관리 시스템

Features:
- Multi-tier rate limiting (per-second, per-minute, per-hour, per-day)
- Distributed rate limiting with Redis backend
- Priority queue system for important requests
- Auto-adaptive limits based on API response patterns
- Circuit breaker pattern for failing APIs
- Quota tracking and alerting
- Burst allowance and smoothing algorithms
- API health scoring and failover logic
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import hashlib
import redis.asyncio as redis
from contextlib import asynccontextmanager
import statistics

from src.logger import setup_logger

logger = setup_logger(__name__)

class RateLimitTier(Enum):
    """Rate limit priority tiers"""
    CRITICAL = "critical"      # MEV opportunities, flash loan execution
    HIGH = "high"             # Real-time arbitrage detection
    MEDIUM = "medium"         # Price feed updates
    LOW = "low"               # Historical data, analytics
    BACKGROUND = "background"  # Cleanup, maintenance

class APIStatus(Enum):
    """API health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"     # High latency or errors
    CIRCUIT_OPEN = "circuit_open"  # Too many failures
    QUOTA_EXCEEDED = "quota_exceeded"
    MAINTENANCE = "maintenance"

@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests_per_second: int = 0
    requests_per_minute: int = 0
    requests_per_hour: int = 0
    requests_per_day: int = 0
    burst_allowance: int = 0  # Extra requests during bursts
    
@dataclass
class APIQuota:
    """API quota configuration and tracking"""
    daily_limit: int
    monthly_limit: int
    used_daily: int = 0
    used_monthly: int = 0
    reset_daily: float = 0.0  # Unix timestamp
    reset_monthly: float = 0.0
    cost_per_request: float = 0.0  # For billing APIs
    
@dataclass  
class RequestMetrics:
    """Request metrics for monitoring"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency: float = 0.0
    max_latency: float = 0.0
    last_request_time: float = 0.0
    errors_per_minute: deque = field(default_factory=lambda: deque(maxlen=60))
    
@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5      # Failed requests before opening
    success_threshold: int = 3      # Successful requests before closing
    timeout: float = 60.0          # Time before attempting reset (seconds)
    
class CircuitBreakerState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Blocking requests due to failures
    HALF_OPEN = "half_open"  # Testing if service is back

@dataclass
class CircuitBreaker:
    """Circuit breaker state"""
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

class AdvancedRateLimiter:
    """
    Advanced rate limiting system with Redis backend support
    
    논문의 6.43초 평균 실행시간을 위한 효율적인 API 호출 관리
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.local_storage: Dict[str, Any] = {}
        
        # API 설정들
        self.api_configs: Dict[str, Dict] = {}
        self.rate_limits: Dict[str, Dict[RateLimitTier, RateLimit]] = {}
        self.quotas: Dict[str, APIQuota] = {}
        self.metrics: Dict[str, RequestMetrics] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Priority queue for requests
        self.request_queues: Dict[RateLimitTier, deque] = {
            tier: deque() for tier in RateLimitTier
        }
        
        # Health monitoring
        self.api_health: Dict[str, APIStatus] = {}
        self.health_scores: Dict[str, float] = {}  # 0.0 (worst) to 1.0 (best)
        
        # Performance tracking
        self.performance_history: Dict[str, deque] = {}
        
        # Lock for thread safety
        self._locks: Dict[str, asyncio.Lock] = {}
        
    async def initialize(self):
        """Initialize the rate limiter"""
        if self.redis_url:
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                await self.redis_client.ping()
                logger.info("Redis 연결 성공 (분산 rate limiting 활성화)")
            except Exception as e:
                logger.warning(f"Redis 연결 실패, 로컬 모드로 전환: {e}")
                self.redis_client = None
                
        await self._initialize_default_configs()
        await self._start_background_tasks()
        
        logger.info("고급 Rate Limiter 초기화 완료")
        
    async def _initialize_default_configs(self):
        """기본 API 설정 초기화"""
        
        # CoinGecko
        await self.configure_api(
            api_name="coingecko",
            rate_limits={
                RateLimitTier.CRITICAL: RateLimit(requests_per_minute=50, burst_allowance=5),
                RateLimitTier.HIGH: RateLimit(requests_per_minute=30, burst_allowance=3),
                RateLimitTier.MEDIUM: RateLimit(requests_per_minute=20, burst_allowance=2),
                RateLimitTier.LOW: RateLimit(requests_per_minute=10, burst_allowance=1),
                RateLimitTier.BACKGROUND: RateLimit(requests_per_minute=5)
            },
            quota=APIQuota(daily_limit=1000, monthly_limit=10000),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3, timeout=30.0)
        )
        
        # CoinMarketCap (유료 API)
        await self.configure_api(
            api_name="coinmarketcap",
            rate_limits={
                RateLimitTier.CRITICAL: RateLimit(requests_per_minute=300, burst_allowance=10),
                RateLimitTier.HIGH: RateLimit(requests_per_minute=200, burst_allowance=8),
                RateLimitTier.MEDIUM: RateLimit(requests_per_minute=150, burst_allowance=5),
                RateLimitTier.LOW: RateLimit(requests_per_minute=100, burst_allowance=3),
                RateLimitTier.BACKGROUND: RateLimit(requests_per_minute=50)
            },
            quota=APIQuota(daily_limit=10000, monthly_limit=100000, cost_per_request=0.001),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=5, timeout=60.0)
        )
        
        # CryptoCompare 
        await self.configure_api(
            api_name="cryptocompare",
            rate_limits={
                RateLimitTier.CRITICAL: RateLimit(requests_per_minute=80, burst_allowance=8),
                RateLimitTier.HIGH: RateLimit(requests_per_minute=60, burst_allowance=6),
                RateLimitTier.MEDIUM: RateLimit(requests_per_minute=40, burst_allowance=4),
                RateLimitTier.LOW: RateLimit(requests_per_minute=30, burst_allowance=2),
                RateLimitTier.BACKGROUND: RateLimit(requests_per_minute=15)
            },
            quota=APIQuota(daily_limit=2000, monthly_limit=50000),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=4, timeout=45.0)
        )
        
        # Ethereum RPC (Alchemy/Infura)
        await self.configure_api(
            api_name="ethereum_rpc", 
            rate_limits={
                RateLimitTier.CRITICAL: RateLimit(requests_per_second=50, burst_allowance=20),
                RateLimitTier.HIGH: RateLimit(requests_per_second=30, burst_allowance=15),
                RateLimitTier.MEDIUM: RateLimit(requests_per_second=20, burst_allowance=10),
                RateLimitTier.LOW: RateLimit(requests_per_second=10, burst_allowance=5),
                RateLimitTier.BACKGROUND: RateLimit(requests_per_second=5)
            },
            quota=APIQuota(daily_limit=100000, monthly_limit=2000000),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=10, timeout=30.0)
        )
        
        # 기타 API들...
        for api_name in ["binance", "nomics", "messari", "coinpaprika"]:
            await self.configure_api(
                api_name=api_name,
                rate_limits={
                    RateLimitTier.CRITICAL: RateLimit(requests_per_minute=100, burst_allowance=5),
                    RateLimitTier.HIGH: RateLimit(requests_per_minute=80, burst_allowance=4),
                    RateLimitTier.MEDIUM: RateLimit(requests_per_minute=60, burst_allowance=3),
                    RateLimitTier.LOW: RateLimit(requests_per_minute=40, burst_allowance=2),
                    RateLimitTier.BACKGROUND: RateLimit(requests_per_minute=20)
                },
                quota=APIQuota(daily_limit=5000, monthly_limit=100000),
                circuit_breaker_config=CircuitBreakerConfig()
            )
    
    async def configure_api(self, 
                          api_name: str,
                          rate_limits: Dict[RateLimitTier, RateLimit],
                          quota: APIQuota,
                          circuit_breaker_config: CircuitBreakerConfig):
        """API 설정"""
        
        self.rate_limits[api_name] = rate_limits
        self.quotas[api_name] = quota
        self.metrics[api_name] = RequestMetrics()
        self.circuit_breakers[api_name] = CircuitBreaker(config=circuit_breaker_config)
        self.api_health[api_name] = APIStatus.HEALTHY
        self.health_scores[api_name] = 1.0
        self.performance_history[api_name] = deque(maxlen=100)
        self._locks[api_name] = asyncio.Lock()
        
        logger.debug(f"API '{api_name}' 설정 완료")
        
    @asynccontextmanager
    async def acquire_slot(self, api_name: str, tier: RateLimitTier = RateLimitTier.MEDIUM):
        """Rate limit slot 획득 (context manager)"""
        start_time = time.time()
        
        try:
            # Pre-request checks
            await self._pre_request_checks(api_name, tier)
            
            # Wait for available slot
            await self._wait_for_slot(api_name, tier)
            
            # Record request start
            await self._record_request_start(api_name, tier)
            
            yield
            
            # Record successful request
            await self._record_request_success(api_name, time.time() - start_time)
            
        except Exception as e:
            # Record failed request
            await self._record_request_failure(api_name, str(e), time.time() - start_time)
            raise
            
    async def _pre_request_checks(self, api_name: str, tier: RateLimitTier):
        """요청 전 검사들"""
        
        # Circuit breaker 확인
        circuit_breaker = self.circuit_breakers.get(api_name)
        if circuit_breaker and circuit_breaker.state == CircuitBreakerState.OPEN:
            # Circuit이 열린 후 timeout이 지났는지 확인
            if time.time() - circuit_breaker.last_failure_time < circuit_breaker.config.timeout:
                raise Exception(f"Circuit breaker open for {api_name}")
            else:
                # Half-open으로 전환
                circuit_breaker.state = CircuitBreakerState.HALF_OPEN
                circuit_breaker.success_count = 0
                
        # API 상태 확인  
        api_status = self.api_health.get(api_name, APIStatus.HEALTHY)
        if api_status == APIStatus.MAINTENANCE:
            raise Exception(f"API {api_name} is under maintenance")
            
        # Quota 확인
        quota = self.quotas.get(api_name)
        if quota:
            await self._check_quota(api_name, quota)
            
    async def _check_quota(self, api_name: str, quota: APIQuota):
        """할당량 확인"""
        current_time = time.time()
        
        # Daily quota reset check
        if current_time - quota.reset_daily > 86400:  # 24 hours
            quota.used_daily = 0
            quota.reset_daily = current_time
            
        # Monthly quota reset check  
        if current_time - quota.reset_monthly > 2592000:  # 30 days
            quota.used_monthly = 0
            quota.reset_monthly = current_time
            
        # Check limits
        if quota.used_daily >= quota.daily_limit:
            self.api_health[api_name] = APIStatus.QUOTA_EXCEEDED
            raise Exception(f"Daily quota exceeded for {api_name}: {quota.used_daily}/{quota.daily_limit}")
            
        if quota.used_monthly >= quota.monthly_limit:
            self.api_health[api_name] = APIStatus.QUOTA_EXCEEDED
            raise Exception(f"Monthly quota exceeded for {api_name}: {quota.used_monthly}/{quota.monthly_limit}")
            
        # Warning at 80% usage
        if quota.used_daily > quota.daily_limit * 0.8:
            logger.warning(f"API {api_name} daily quota at {quota.used_daily/quota.daily_limit*100:.1f}%")
            
    async def _wait_for_slot(self, api_name: str, tier: RateLimitTier):
        """Rate limit slot 대기"""
        
        rate_limit = self.rate_limits.get(api_name, {}).get(tier)
        if not rate_limit:
            return  # No limits configured
            
        async with self._locks[api_name]:
            
            # Check different time windows
            current_time = time.time()
            
            # Per-second limit
            if rate_limit.requests_per_second > 0:
                wait_time = await self._check_rate_limit_window(
                    api_name, tier, "second", rate_limit.requests_per_second, 1.0
                )
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    
            # Per-minute limit
            if rate_limit.requests_per_minute > 0:
                wait_time = await self._check_rate_limit_window(
                    api_name, tier, "minute", rate_limit.requests_per_minute, 60.0
                )
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    
            # Per-hour limit
            if rate_limit.requests_per_hour > 0:
                wait_time = await self._check_rate_limit_window(
                    api_name, tier, "hour", rate_limit.requests_per_hour, 3600.0
                )
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    
            # Per-day limit
            if rate_limit.requests_per_day > 0:
                wait_time = await self._check_rate_limit_window(
                    api_name, tier, "day", rate_limit.requests_per_day, 86400.0
                )
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    
    async def _check_rate_limit_window(self, api_name: str, tier: RateLimitTier, 
                                     window: str, limit: int, window_seconds: float) -> float:
        """특정 시간 윈도우에서 rate limit 확인"""
        
        key = f"rate_limit:{api_name}:{tier.value}:{window}"
        current_time = time.time()
        
        if self.redis_client:
            # Redis 기반 분산 rate limiting
            return await self._redis_rate_limit_check(key, limit, window_seconds, current_time)
        else:
            # Local rate limiting
            return await self._local_rate_limit_check(key, limit, window_seconds, current_time)
            
    async def _redis_rate_limit_check(self, key: str, limit: int, 
                                    window_seconds: float, current_time: float) -> float:
        """Redis 기반 rate limit 확인"""
        try:
            # Sliding window log 구현
            pipe = self.redis_client.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(key, 0, current_time - window_seconds)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request timestamp
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiry
            pipe.expire(key, int(window_seconds) + 1)
            
            results = await pipe.execute()
            current_count = results[1]
            
            if current_count >= limit:
                # Calculate wait time
                oldest_request = await self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest_request:
                    wait_time = window_seconds - (current_time - oldest_request[0][1])
                    return max(0, wait_time)
                    
            return 0.0
            
        except Exception as e:
            logger.error(f"Redis rate limit check 실패: {e}")
            # Fallback to local check
            return await self._local_rate_limit_check(key, limit, window_seconds, current_time)
            
    async def _local_rate_limit_check(self, key: str, limit: int,
                                    window_seconds: float, current_time: float) -> float:
        """로컬 rate limit 확인"""
        
        if key not in self.local_storage:
            self.local_storage[key] = deque()
            
        requests = self.local_storage[key]
        
        # Remove expired requests
        while requests and current_time - requests[0] > window_seconds:
            requests.popleft()
            
        # Check limit
        if len(requests) >= limit:
            wait_time = window_seconds - (current_time - requests[0])
            return max(0, wait_time)
            
        # Add current request
        requests.append(current_time)
        
        return 0.0
        
    async def _record_request_start(self, api_name: str, tier: RateLimitTier):
        """요청 시작 기록"""
        metrics = self.metrics.get(api_name)
        quota = self.quotas.get(api_name)
        
        if metrics:
            metrics.total_requests += 1
            metrics.last_request_time = time.time()
            
        if quota:
            quota.used_daily += 1
            quota.used_monthly += 1
            
    async def _record_request_success(self, api_name: str, latency: float):
        """성공한 요청 기록"""
        metrics = self.metrics.get(api_name)
        circuit_breaker = self.circuit_breakers.get(api_name)
        
        if metrics:
            metrics.successful_requests += 1
            
            # Update latency metrics
            if metrics.avg_latency == 0:
                metrics.avg_latency = latency
            else:
                metrics.avg_latency = (metrics.avg_latency * 0.9) + (latency * 0.1)
                
            metrics.max_latency = max(metrics.max_latency, latency)
            
        # Circuit breaker success handling
        if circuit_breaker:
            if circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
                circuit_breaker.success_count += 1
                if circuit_breaker.success_count >= circuit_breaker.config.success_threshold:
                    circuit_breaker.state = CircuitBreakerState.CLOSED
                    circuit_breaker.failure_count = 0
                    logger.info(f"Circuit breaker closed for {api_name}")
                    
            elif circuit_breaker.state == CircuitBreakerState.CLOSED:
                circuit_breaker.failure_count = max(0, circuit_breaker.failure_count - 1)
                
        # Update health score
        await self._update_health_score(api_name, success=True, latency=latency)
        
        # Record performance
        self.performance_history[api_name].append({
            'timestamp': time.time(),
            'success': True,
            'latency': latency
        })
        
    async def _record_request_failure(self, api_name: str, error: str, latency: float):
        """실패한 요청 기록"""
        metrics = self.metrics.get(api_name)
        circuit_breaker = self.circuit_breakers.get(api_name)
        
        if metrics:
            metrics.failed_requests += 1
            metrics.errors_per_minute.append(time.time())
            
        # Circuit breaker failure handling
        if circuit_breaker:
            circuit_breaker.failure_count += 1
            circuit_breaker.last_failure_time = time.time()
            
            if (circuit_breaker.state == CircuitBreakerState.CLOSED and
                circuit_breaker.failure_count >= circuit_breaker.config.failure_threshold):
                circuit_breaker.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker opened for {api_name} after {circuit_breaker.failure_count} failures")
                
            elif circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
                circuit_breaker.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker re-opened for {api_name}")
                
        # Update health score
        await self._update_health_score(api_name, success=False, latency=latency)
        
        # Record performance
        self.performance_history[api_name].append({
            'timestamp': time.time(),
            'success': False,
            'latency': latency,
            'error': error
        })
        
        logger.warning(f"API request failed for {api_name}: {error}")
        
    async def _update_health_score(self, api_name: str, success: bool, latency: float):
        """API 건강 점수 업데이트"""
        current_score = self.health_scores.get(api_name, 1.0)
        
        # 성공/실패에 따른 점수 조정
        if success:
            # 성공시 천천히 점수 회복
            current_score = min(1.0, current_score + 0.01)
        else:
            # 실패시 빠르게 점수 하락
            current_score = max(0.0, current_score - 0.1)
            
        # 지연시간에 따른 점수 조정
        if latency > 5.0:  # 5초 이상
            current_score *= 0.9
        elif latency > 2.0:  # 2초 이상
            current_score *= 0.95
            
        self.health_scores[api_name] = current_score
        
        # 건강 상태 업데이트
        if current_score >= 0.8:
            self.api_health[api_name] = APIStatus.HEALTHY
        elif current_score >= 0.5:
            self.api_health[api_name] = APIStatus.DEGRADED
        else:
            # 점수가 너무 낮으면 circuit breaker 활성화 고려
            circuit_breaker = self.circuit_breakers.get(api_name)
            if circuit_breaker and circuit_breaker.state == CircuitBreakerState.CLOSED:
                circuit_breaker.failure_count += 2  # 점수 하락으로 인한 페널티
                
    async def _start_background_tasks(self):
        """백그라운드 작업들 시작"""
        
        # 정리 작업
        asyncio.create_task(self._cleanup_task())
        
        # 건강 상태 모니터링
        asyncio.create_task(self._health_monitoring_task())
        
        # 성능 분석
        asyncio.create_task(self._performance_analysis_task())
        
        # 적응형 제한 조정
        asyncio.create_task(self._adaptive_limits_task())
        
        logger.info("Rate limiter 백그라운드 작업들 시작됨")
        
    async def _cleanup_task(self):
        """주기적 정리 작업"""
        while True:
            try:
                await asyncio.sleep(300)  # 5분마다
                
                current_time = time.time()
                
                # 오래된 에러 기록 정리
                for metrics in self.metrics.values():
                    while (metrics.errors_per_minute and 
                           current_time - metrics.errors_per_minute[0] > 3600):  # 1시간
                        metrics.errors_per_minute.popleft()
                        
                # 로컬 저장소 정리
                for key in list(self.local_storage.keys()):
                    if key.startswith('rate_limit:'):
                        # 타임스탬프 기반 정리는 각 체크에서 수행됨
                        pass
                        
                logger.debug("Rate limiter 정리 작업 완료")
                
            except Exception as e:
                logger.error(f"정리 작업 실패: {e}")
                
    async def _health_monitoring_task(self):
        """건강 상태 모니터링"""
        while True:
            try:
                await asyncio.sleep(60)  # 1분마다
                
                # 각 API의 건강 상태 평가
                for api_name in self.api_configs.keys():
                    await self._evaluate_api_health(api_name)
                    
            except Exception as e:
                logger.error(f"건강 상태 모니터링 실패: {e}")
                
    async def _evaluate_api_health(self, api_name: str):
        """API 건강 상태 평가"""
        metrics = self.metrics.get(api_name)
        if not metrics:
            return
            
        # 최근 에러율 계산
        current_time = time.time()
        recent_errors = len([t for t in metrics.errors_per_minute 
                           if current_time - t < 300])  # 5분 이내
        
        error_rate = recent_errors / max(1, metrics.total_requests) if metrics.total_requests > 0 else 0
        
        # 건강 상태 업데이트
        if error_rate > 0.5:  # 50% 이상 에러율
            self.api_health[api_name] = APIStatus.DEGRADED
        elif error_rate > 0.8:  # 80% 이상 에러율
            # Circuit breaker 강제 활성화 고려
            circuit_breaker = self.circuit_breakers.get(api_name)
            if circuit_breaker:
                circuit_breaker.failure_count = circuit_breaker.config.failure_threshold
                
        # 평균 지연시간 확인
        if metrics.avg_latency > 10.0:  # 10초 이상
            logger.warning(f"API {api_name} 높은 지연시간: {metrics.avg_latency:.2f}초")
            
    async def _performance_analysis_task(self):
        """성능 분석 작업"""
        while True:
            try:
                await asyncio.sleep(900)  # 15분마다
                
                for api_name, history in self.performance_history.items():
                    if len(history) >= 10:
                        await self._analyze_performance_trends(api_name, history)
                        
            except Exception as e:
                logger.error(f"성능 분석 실패: {e}")
                
    async def _analyze_performance_trends(self, api_name: str, history: deque):
        """성능 트렌드 분석"""
        recent_data = list(history)[-50:]  # 최근 50개 요청
        
        if not recent_data:
            return
            
        # 성공률 계산
        success_rate = sum(1 for req in recent_data if req['success']) / len(recent_data)
        
        # 평균 지연시간 계산
        successful_requests = [req for req in recent_data if req['success']]
        if successful_requests:
            avg_latency = statistics.mean(req['latency'] for req in successful_requests)
            
            # 트렌드 로깅
            if success_rate < 0.8:
                logger.warning(f"API {api_name} 낮은 성공률: {success_rate:.2%}")
                
            if avg_latency > 5.0:
                logger.warning(f"API {api_name} 높은 지연시간: {avg_latency:.2f}초")
                
        # 건강 점수 업데이트에 반영
        self.health_scores[api_name] = success_rate * 0.7 + min(1.0, 5.0 / (avg_latency + 1)) * 0.3
        
    async def _adaptive_limits_task(self):
        """적응형 제한 조정"""
        while True:
            try:
                await asyncio.sleep(3600)  # 1시간마다
                
                for api_name in self.api_configs.keys():
                    await self._adjust_adaptive_limits(api_name)
                    
            except Exception as e:
                logger.error(f"적응형 제한 조정 실패: {e}")
                
    async def _adjust_adaptive_limits(self, api_name: str):
        """API별 적응형 제한 조정"""
        health_score = self.health_scores.get(api_name, 1.0)
        
        # 건강 점수에 따라 제한 조정
        if health_score > 0.9:
            # 건강하면 제한 완화 (5% 증가)
            await self._adjust_rate_limits(api_name, 1.05)
        elif health_score < 0.5:
            # 건강하지 않으면 제한 강화 (20% 감소)
            await self._adjust_rate_limits(api_name, 0.8)
            
    async def _adjust_rate_limits(self, api_name: str, factor: float):
        """Rate limit 조정"""
        rate_limits = self.rate_limits.get(api_name, {})
        
        for tier, limit in rate_limits.items():
            if limit.requests_per_minute > 0:
                new_limit = max(1, int(limit.requests_per_minute * factor))
                limit.requests_per_minute = new_limit
                
            if limit.requests_per_second > 0:
                new_limit = max(1, int(limit.requests_per_second * factor))
                limit.requests_per_second = new_limit
                
        logger.info(f"API {api_name} rate limits 조정됨 (factor: {factor})")
        
    def get_api_status(self, api_name: str) -> Dict[str, Any]:
        """API 상태 정보 반환"""
        metrics = self.metrics.get(api_name, RequestMetrics())
        quota = self.quotas.get(api_name)
        circuit_breaker = self.circuit_breakers.get(api_name)
        
        status = {
            'name': api_name,
            'health_status': self.api_health.get(api_name, APIStatus.HEALTHY).value,
            'health_score': self.health_scores.get(api_name, 1.0),
            'total_requests': metrics.total_requests,
            'successful_requests': metrics.successful_requests,
            'failed_requests': metrics.failed_requests,
            'success_rate': (metrics.successful_requests / metrics.total_requests 
                           if metrics.total_requests > 0 else 0),
            'avg_latency': metrics.avg_latency,
            'max_latency': metrics.max_latency,
            'errors_last_hour': len([t for t in metrics.errors_per_minute 
                                   if time.time() - t < 3600])
        }
        
        if quota:
            status.update({
                'quota_daily_used': quota.used_daily,
                'quota_daily_limit': quota.daily_limit,
                'quota_daily_remaining': quota.daily_limit - quota.used_daily,
                'quota_monthly_used': quota.used_monthly,
                'quota_monthly_limit': quota.monthly_limit,
                'quota_monthly_remaining': quota.monthly_limit - quota.used_monthly
            })
            
        if circuit_breaker:
            status.update({
                'circuit_breaker_state': circuit_breaker.state.value,
                'circuit_breaker_failures': circuit_breaker.failure_count,
                'circuit_breaker_successes': circuit_breaker.success_count
            })
            
        return status
        
    def get_all_api_status(self) -> Dict[str, Dict[str, Any]]:
        """모든 API 상태 정보 반환"""
        return {api_name: self.get_api_status(api_name) 
                for api_name in self.api_configs.keys()}
        
    async def reset_api_status(self, api_name: str):
        """API 상태 초기화 (강제 복구)"""
        if api_name in self.metrics:
            self.metrics[api_name] = RequestMetrics()
            
        if api_name in self.circuit_breakers:
            self.circuit_breakers[api_name].state = CircuitBreakerState.CLOSED
            self.circuit_breakers[api_name].failure_count = 0
            self.circuit_breakers[api_name].success_count = 0
            
        self.api_health[api_name] = APIStatus.HEALTHY
        self.health_scores[api_name] = 1.0
        
        logger.info(f"API {api_name} 상태 초기화 완료")
        
    async def shutdown(self):
        """Rate limiter 종료"""
        if self.redis_client:
            await self.redis_client.close()
            
        logger.info("Rate limiter 종료 완료")

# 사용 예시
async def example_usage():
    """Rate Limiter 사용 예시"""
    
    # 초기화
    rate_limiter = AdvancedRateLimiter(redis_url="redis://localhost:6379")
    await rate_limiter.initialize()
    
    try:
        # 중요한 MEV 기회 감지시 (CRITICAL 우선순위)
        async with rate_limiter.acquire_slot("ethereum_rpc", RateLimitTier.CRITICAL):
            print("Critical MEV opportunity detection...")
            # RPC 호출 수행
            
        # 일반적인 가격 피드 업데이트 (MEDIUM 우선순위)  
        async with rate_limiter.acquire_slot("coingecko", RateLimitTier.MEDIUM):
            print("Price feed update...")
            # API 호출 수행
            
        # 백그라운드 데이터 수집 (LOW 우선순위)
        async with rate_limiter.acquire_slot("cryptocompare", RateLimitTier.LOW):
            print("Background data collection...")
            # API 호출 수행
            
        # 상태 확인
        status = rate_limiter.get_api_status("coingecko")
        print(f"CoinGecko 상태: {status}")
        
    finally:
        await rate_limiter.shutdown()

if __name__ == "__main__":
    asyncio.run(example_usage())