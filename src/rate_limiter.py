import asyncio
import time
from typing import Dict, Optional
from collections import defaultdict, deque
from src.logger import setup_logger

logger = setup_logger(__name__)

class RateLimiter:
    """API 요청 속도 제한기"""
    
    def __init__(self):
        # 각 API 엔드포인트별 속도 제한 설정
        self.limits = {
            'coingecko': {
                'requests_per_minute': 50,  # CoinGecko 무료 API 제한
                'requests_per_second': 10
            },
            'coinpaprika': {
                'requests_per_minute': 100,  # Coinpaprika API 제한
                'requests_per_second': 20
            },
            'cryptocompare': {
                'requests_per_minute': 300,  # CryptoCompare 무료 API 제한
                'requests_per_second': 50
            }
        }
        
        # 각 엔드포인트별 요청 기록
        self.request_history: Dict[str, deque] = defaultdict(deque)
        
        # API 키별 사용량 추적 (향후 확장을 위해)
        self.api_key_usage: Dict[str, Dict] = defaultdict(dict)
        
    def _clean_old_requests(self, endpoint: str, window: int = 300):
        """오래된 요청 기록 정리 (5분 윈도우)"""
        now = time.time()
        while self.request_history[endpoint] and self.request_history[endpoint][0] < now - window:
            self.request_history[endpoint].popleft()
    
    def is_allowed(self, endpoint: str) -> bool:
        """
        요청이 허용되는지 확인
        
        Args:
            endpoint: API 엔드포인트 이름
            
        Returns:
            허용 여부
        """
        if endpoint not in self.limits:
            return True  # 제한이 없는 엔드포인트는 허용
        
        now = time.time()
        
        # 오래된 요청 기록 정리
        self._clean_old_requests(endpoint)
        
        # 분당 제한 확인
        requests_last_minute = sum(1 for t in self.request_history[endpoint] if t > now - 60)
        if requests_last_minute >= self.limits[endpoint]['requests_per_minute']:
            return False
        
        # 초당 제한 확인
        requests_last_second = sum(1 for t in self.request_history[endpoint] if t > now - 1)
        if requests_last_second >= self.limits[endpoint]['requests_per_second']:
            return False
        
        return True
    
    def record_request(self, endpoint: str):
        """
        요청 기록
        
        Args:
            endpoint: API 엔드포인트 이름
        """
        now = time.time()
        self.request_history[endpoint].append(now)
        self._clean_old_requests(endpoint)
    
    async def wait_if_needed(self, endpoint: str) -> bool:
        """
        필요한 경우 속도 제한을 위해 대기
        
        Args:
            endpoint: API 엔드포인트 이름
            
        Returns:
            대기했는지 여부
        """
        if endpoint not in self.limits:
            return False
        
        wait_time = 0
        now = time.time()
        
        # 오래된 요청 기록 정리
        self._clean_old_requests(endpoint)
        
        # 분당 제한 확인
        requests_last_minute = sum(1 for t in self.request_history[endpoint] if t > now - 60)
        if requests_last_minute >= self.limits[endpoint]['requests_per_minute']:
            # 다음 분까지 대기
            wait_time = 60 - (now % 60)
            logger.warning(f"{endpoint} API 분당 제한 도달, {wait_time:.2f}초 대기")
        
        # 초당 제한 확인
        requests_last_second = sum(1 for t in self.request_history[endpoint] if t > now - 1)
        if requests_last_second >= self.limits[endpoint]['requests_per_second']:
            # 1초 대기
            wait_time = max(wait_time, 1.0)
            logger.warning(f"{endpoint} API 초당 제한 도달, {wait_time:.2f}초 대기")
        
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            return True
        
        return False
    
    def get_usage_stats(self, endpoint: str) -> Dict:
        """
        엔드포인트 사용 통계
        
        Args:
            endpoint: API 엔드포인트 이름
            
        Returns:
            사용 통계
        """
        if endpoint not in self.limits:
            return {}
        
        now = time.time()
        self._clean_old_requests(endpoint)
        
        requests_last_minute = sum(1 for t in self.request_history[endpoint] if t > now - 60)
        requests_last_second = sum(1 for t in self.request_history[endpoint] if t > now - 1)
        
        return {
            'requests_per_minute': requests_last_minute,
            'requests_per_second': requests_last_second,
            'minute_limit': self.limits[endpoint]['requests_per_minute'],
            'second_limit': self.limits[endpoint]['requests_per_second'],
            'minute_usage_percent': (requests_last_minute / self.limits[endpoint]['requests_per_minute']) * 100,
            'second_usage_percent': (requests_last_second / self.limits[endpoint]['requests_per_second']) * 100
        }
    
    def reset_usage(self, endpoint: Optional[str] = None):
        """
        사용량 초기화
        
        Args:
            endpoint: 특정 엔드포인트 (None이면 모두 초기화)
        """
        if endpoint:
            self.request_history[endpoint].clear()
        else:
            self.request_history.clear()

class APIQuotaManager:
    """API 할당량 관리자"""
    
    def __init__(self):
        # API 키별 할당량 설정
        self.quota_limits = {
            'coingecko': {
                'daily_limit': 10000,  # 일일 요청 제한
                'monthly_limit': 300000  # 월간 요청 제한
            },
            'coinpaprika': {
                'daily_limit': 20000,
                'monthly_limit': 600000
            },
            'cryptocompare': {
                'daily_limit': 100000,
                'monthly_limit': 3000000
            }
        }
        
        # API 키별 사용량 추적
        self.usage_tracking: Dict[str, Dict] = defaultdict(lambda: {
            'daily_count': 0,
            'monthly_count': 0,
            'last_reset_date': time.time(),
            'last_reset_month': time.time()
        })
        
        # 백오프 설정
        self.backoff_settings = {
            'initial_delay': 1.0,  # 초기 지연 시간 (초)
            'max_delay': 60.0,     # 최대 지연 시간 (초)
            'multiplier': 2.0,     # 지연 시간 배수
            'jitter': 0.1          # 지터 (랜덤성 추가)
        }
        
        self.backoff_delays: Dict[str, float] = defaultdict(lambda: self.backoff_settings['initial_delay'])
    
    def record_api_call(self, api_name: str, api_key: str = 'default'):
        """
        API 호출 기록
        
        Args:
            api_name: API 이름
            api_key: API 키 (기본값: 'default')
        """
        now = time.time()
        
        # 일일/월간 카운트 업데이트
        usage = self.usage_tracking[api_key]
        
        # 날짜 변경 확인 (일일 카운트 리셋)
        if self._is_new_day(usage['last_reset_date']):
            usage['daily_count'] = 0
            usage['last_reset_date'] = now
        
        # 월 변경 확인 (월간 카운트 리셋)
        if self._is_new_month(usage['last_reset_month']):
            usage['monthly_count'] = 0
            usage['last_reset_month'] = now
        
        # 카운트 증가
        usage['daily_count'] += 1
        usage['monthly_count'] += 1
    
    def is_quota_available(self, api_name: str, api_key: str = 'default') -> bool:
        """
        할당량이 남아있는지 확인
        
        Args:
            api_name: API 이름
            api_key: API 키 (기본값: 'default')
            
        Returns:
            할당량 사용 가능 여부
        """
        if api_name not in self.quota_limits:
            return True  # 제한이 없는 API는 항상 허용
        
        usage = self.usage_tracking[api_key]
        limits = self.quota_limits[api_name]
        
        return (usage['daily_count'] < limits['daily_limit'] and 
                usage['monthly_count'] < limits['monthly_limit'])
    
    def get_quota_usage(self, api_name: str, api_key: str = 'default') -> Dict:
        """
        할당량 사용률 조회
        
        Args:
            api_name: API 이름
            api_key: API 키 (기본값: 'default')
            
        Returns:
            할당량 사용 통계
        """
        if api_name not in self.quota_limits:
            return {}
        
        usage = self.usage_tracking[api_key]
        limits = self.quota_limits[api_name]
        
        return {
            'daily_used': usage['daily_count'],
            'daily_limit': limits['daily_limit'],
            'daily_percent': (usage['daily_count'] / limits['daily_limit']) * 100,
            'monthly_used': usage['monthly_count'],
            'monthly_limit': limits['monthly_limit'],
            'monthly_percent': (usage['monthly_count'] / limits['monthly_limit']) * 100
        }
    
    async def handle_rate_limit_error(self, api_name: str, api_key: str = 'default'):
        """
        속도 제한 오류 처리 (지수 백오프)
        
        Args:
            api_name: API 이름
            api_key: API 키 (기본값: 'default')
        """
        delay = self.backoff_delays[api_key]
        
        # 지터 추가 (랜덤성)
        import random
        jitter = random.uniform(-self.backoff_settings['jitter'], self.backoff_settings['jitter'])
        actual_delay = delay * (1 + jitter)
        
        logger.warning(f"{api_name} API 속도 제한 오류, {actual_delay:.2f}초 후 재시도")
        await asyncio.sleep(actual_delay)
        
        # 다음 요청을 위한 지연 시간 증가
        self.backoff_delays[api_key] = min(
            delay * self.backoff_settings['multiplier'],
            self.backoff_settings['max_delay']
        )
    
    def reset_backoff(self, api_key: str = 'default'):
        """
        백오프 지연 시간 리셋
        
        Args:
            api_key: API 키 (기본값: 'default')
        """
        self.backoff_delays[api_key] = self.backoff_settings['initial_delay']
    
    def _is_new_day(self, last_reset: float) -> bool:
        """새로운 날인지 확인"""
        now = time.time()
        return time.strftime('%Y-%m-%d', time.localtime(now)) != time.strftime('%Y-%m-%d', time.localtime(last_reset))
    
    def _is_new_month(self, last_reset: float) -> bool:
        """새로운 월인지 확인"""
        now = time.time()
        return time.strftime('%Y-%m', time.localtime(now)) != time.strftime('%Y-%m', time.localtime(last_reset))