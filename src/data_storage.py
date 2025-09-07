import redis
import json
import pickle
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from src.logger import setup_logger
from config.config import config

logger = setup_logger(__name__)

class DataStorage:
    def __init__(self):
        self.redis_client = redis.from_url(config.redis_url)
        self.pool_data_ttl = 300  # 5분
        self.price_data_ttl = 60   # 1분
        self.historical_data_ttl = 2592000  # 30일 (기본값)
        
        # 쿼리 최적화를 위한 인덱스 구조
        self._key_prefixes = {
            'pool': 'pool:',
            'pool_history': 'pool_history:',
            'pool_historical': 'pool_historical:',
            'arbitrage': 'arbitrage:',
            'arbitrage_historical': 'arbitrage_historical:',
            'price_historical': 'price_historical:',
            'performance': 'performance:'
        }
        
    async def store_pool_data(self, pool_address: str, pool_info: Dict):
        """풀 데이터 저장 (쿼리 최적화 버전)"""
        try:
            key = f"pool:{pool_address}"
            data = json.dumps(pool_info, default=str)
            self.redis_client.setex(key, self.pool_data_ttl, data)
            
            # 시계열 데이터도 저장
            timestamp = datetime.now().isoformat()
            ts_key = f"pool_history:{pool_address}:{timestamp}"
            self.redis_client.setex(ts_key, 3600, data)  # 1시간 보관
            
            # 인덱스 업데이트 (성능 최적화를 위한 패턴)
            self._update_pool_index(pool_address, timestamp)
            
        except Exception as e:
            logger.error(f"풀 데이터 저장 실패: {e}")
    
    def _update_pool_index(self, pool_address: str, timestamp: str):
        """풀 인덱스 업데이트 (성능 최적화를 위한 패턴)"""
        try:
            # 풀 주소 인덱스 업데이트
            index_key = "index:pools"
            self.redis_client.sadd(index_key, pool_address)
            
            # 타임스탬프 인덱스 업데이트
            time_index_key = f"index:pool_times:{pool_address}"
            self.redis_client.zadd(time_index_key, {timestamp: datetime.fromisoformat(timestamp).timestamp()})
            
            # 오래된 인덱스 항목 제거 (메모리 최적화)
            cutoff_time = (datetime.now() - timedelta(hours=24)).timestamp()
            self.redis_client.zremrangebyscore(time_index_key, 0, cutoff_time)
            
        except Exception as e:
            logger.debug(f"풀 인덱스 업데이트 실패 (비크리티컬): {e}")
    
    async def store_historical_pool_data(self, pool_address: str, pool_info: Dict, timestamp: datetime = None):
        """히스토리컬 풀 데이터 저장 (장기 보관)"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # 히스토리컬 데이터 저장 (더 긴 TTL)
            ts_key = f"pool_historical:{pool_address}:{timestamp.isoformat()}"
            data = json.dumps(pool_info, default=str)
            self.redis_client.setex(ts_key, self.historical_data_ttl, data)
            
            logger.debug(f"히스토리컬 풀 데이터 저장: {pool_address} at {timestamp}")
            
        except Exception as e:
            logger.error(f"히스토리컬 풀 데이터 저장 실패: {e}")
    
    async def get_pool_data(self, pool_address: str) -> Optional[Dict]:
        """풀 데이터 조회 (쿼리 최적화 버전)"""
        try:
            key = f"pool:{pool_address}"
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"풀 데이터 조회 실패: {e}")
            return None
    
    async def store_arbitrage_opportunity(self, opportunity: Dict):
        """차익거래 기회 저장 (쿼리 최적화 버전)"""
        try:
            timestamp = datetime.now().isoformat()
            key = f"arbitrage:{timestamp}"
            data = json.dumps(opportunity, default=str)
            self.redis_client.setex(key, 3600, data)  # 1시간 보관
            
            # 통계용 리스트에도 추가 (제한된 크기 유지)
            self.redis_client.lpush("arbitrage_opportunities", data)
            self.redis_client.ltrim("arbitrage_opportunities", 0, 999)  # 최근 1000개만 보관
            
            # 인덱스 업데이트
            self._update_arbitrage_index(timestamp)
            
        except Exception as e:
            logger.error(f"차익거래 기회 저장 실패: {e}")
    
    def _update_arbitrage_index(self, timestamp: str):
        """차익거래 인덱스 업데이트 (성능 최적화를 위한 패턴)"""
        try:
            # 타임스탬프 인덱스 업데이트
            index_key = "index:arbitrage_times"
            self.redis_client.zadd(index_key, {timestamp: datetime.fromisoformat(timestamp).timestamp()})
            
            # 오래된 인덱스 항목 제거 (메모리 최적화)
            cutoff_time = (datetime.now() - timedelta(hours=24)).timestamp()
            self.redis_client.zremrangebyscore(index_key, 0, cutoff_time)
            
        except Exception as e:
            logger.debug(f"차익거래 인덱스 업데이트 실패 (비크리티컬): {e}")
    
    async def store_historical_arbitrage_opportunity(self, opportunity: Dict, timestamp: datetime = None):
        """히스토리컬 차익거래 기회 저장 (장기 보관)"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # 히스토리컬 데이터 저장
            ts_key = f"arbitrage_historical:{timestamp.isoformat()}"
            data = json.dumps(opportunity, default=str)
            self.redis_client.setex(ts_key, self.historical_data_ttl, data)
            
            logger.debug(f"히스토리컬 차익거래 기회 저장: {ts_key}")
            
        except Exception as e:
            logger.error(f"히스토리컬 차익거래 기회 저장 실패: {e}")
    
    async def get_recent_opportunities(self, limit: int = 100) -> List[Dict]:
        """최근 차익거래 기회 조회 (쿼리 최적화 버전)"""
        try:
            data_list = self.redis_client.lrange("arbitrage_opportunities", 0, limit-1)
            opportunities = []
            for data in data_list:
                opportunities.append(json.loads(data))
            return opportunities
        except Exception as e:
            logger.error(f"차익거래 기회 조회 실패: {e}")
            return []
    
    async def get_pool_price_history(self, pool_address: str, 
                                   hours: int = 24) -> List[Dict]:
        """풀 가격 히스토리 조회 (쿼리 최적화 버전)"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Redis에서 시계열 데이터 조회 (인덱스 기반 최적화)
            # 시간 범위 기반으로 직접 키를 구성하여 조회
            history = []
            
            # 시간 범위 내의 키만 조회 (성능 최적화)
            current_time = start_time
            while current_time <= end_time:
                # 시간별 키 패턴 생성
                time_prefix = current_time.strftime("%Y-%m-%dT%H")
                pattern = f"pool_history:{pool_address}:{time_prefix}*"
                
                # KEYS 명령 대신 SCAN 사용 (메모리 최적화)
                cursor = 0
                while True:
                    cursor, keys = self.redis_client.scan(cursor=cursor, match=pattern, count=100)
                    for key in keys:
                        try:
                            # 키에서 타임스탬프 추출
                            timestamp_str = key.decode().split(':')[-1]
                            timestamp = datetime.fromisoformat(timestamp_str)
                            if start_time <= timestamp <= end_time:
                                data = self.redis_client.get(key)
                                if data:
                                    pool_data = json.loads(data)
                                    pool_data['timestamp'] = timestamp_str
                                    history.append(pool_data)
                        except ValueError:
                            continue
                    
                    # 모든 키를 스캔했으면 종료
                    if cursor == 0:
                        break
                
                # 다음 시간으로 이동
                current_time += timedelta(hours=1)
            
            # 시간순 정렬
            history.sort(key=lambda x: x['timestamp'])
            return history
            
        except Exception as e:
            logger.error(f"가격 히스토리 조회 실패: {e}")
            return []
    
    async def get_historical_pool_data(self, pool_address: str, 
                                     start_date: datetime, 
                                     end_date: datetime) -> List[Dict]:
        """특정 기간의 히스토리컬 풀 데이터 조회 (쿼리 최적화 버전)"""
        try:
            # 히스토리컬 데이터 조회 (인덱스 기반 최적화)
            # 시간 범위 기반으로 직접 키를 구성하여 조회
            history = []
            
            # 시간 범위 내의 키만 조회 (성능 최적화)
            current_date = start_date
            while current_date <= end_date:
                # 날짜별 키 패턴 생성
                date_prefix = current_date.strftime("%Y-%m-%d")
                pattern = f"pool_historical:{pool_address}:{date_prefix}*"
                
                # KEYS 명령 대신 SCAN 사용 (메모리 최적화)
                cursor = 0
                while True:
                    cursor, keys = self.redis_client.scan(cursor=cursor, match=pattern, count=100)
                    for key in keys:
                        try:
                            # 키에서 타임스탬프 추출
                            timestamp_str = key.decode().split(':')[-1]
                            timestamp = datetime.fromisoformat(timestamp_str)
                            if start_date <= timestamp <= end_date:
                                data = self.redis_client.get(key)
                                if data:
                                    pool_data = json.loads(data)
                                    pool_data['timestamp'] = timestamp_str
                                    history.append(pool_data)
                        except ValueError:
                            continue
                    
                    # 모든 키를 스캔했으면 종료
                    if cursor == 0:
                        break
                
                # 다음 날짜로 이동
                current_date += timedelta(days=1)
            
            # 시간순 정렬
            history.sort(key=lambda x: x['timestamp'])
            return history
            
        except Exception as e:
            logger.error(f"히스토리컬 풀 데이터 조회 실패: {e}")
            return []
    
    async def get_historical_arbitrage_opportunities(self, 
                                                   start_date: datetime, 
                                                   end_date: datetime) -> List[Dict]:
        """특정 기간의 히스토리컬 차익거래 기회 조회 (쿼리 최적화 버전)"""
        try:
            # 히스토리컬 데이터 조회 (인덱스 기반 최적화)
            # 시간 범위 기반으로 직접 키를 구성하여 조회
            opportunities = []
            
            # 시간 범위 내의 키만 조회 (성능 최적화)
            current_date = start_date
            while current_date <= end_date:
                # 날짜별 키 패턴 생성
                date_prefix = current_date.strftime("%Y-%m-%d")
                pattern = f"arbitrage_historical:{date_prefix}*"
                
                # KEYS 명령 대신 SCAN 사용 (메모리 최적화)
                cursor = 0
                while True:
                    cursor, keys = self.redis_client.scan(cursor=cursor, match=pattern, count=100)
                    for key in keys:
                        try:
                            # 키에서 타임스탬프 추출
                            timestamp_str = key.decode().split(':')[-1]
                            timestamp = datetime.fromisoformat(timestamp_str)
                            if start_date <= timestamp <= end_date:
                                data = self.redis_client.get(key)
                                if data:
                                    opportunity = json.loads(data)
                                    opportunity['timestamp'] = timestamp_str
                                    opportunities.append(opportunity)
                        except ValueError:
                            continue
                    
                    # 모든 키를 스캔했으면 종료
                    if cursor == 0:
                        break
                
                # 다음 날짜로 이동
                current_date += timedelta(days=1)
            
            # 시간순 정렬
            opportunities.sort(key=lambda x: x['timestamp'])
            return opportunities
            
        except Exception as e:
            logger.error(f"히스토리컬 차익거래 기회 조회 실패: {e}")
            return []
    
    def _scan_keys_optimized(self, pattern: str, count: int = 100) -> List[bytes]:
        """
        키 스캔 최적화 버전 (메모리 효율적)
        
        Args:
            pattern: 검색할 키 패턴
            count: 한 번에 스캔할 키 수
            
        Returns:
            매칭된 키 리스트
        """
        try:
            matched_keys = []
            cursor = 0
            
            # SCAN 명령 사용 (KEYS 대신 메모리 효율적)
            while True:
                cursor, keys = self.redis_client.scan(cursor=cursor, match=pattern, count=count)
                matched_keys.extend(keys)
                
                # 모든 키를 스캔했으면 종료
                if cursor == 0:
                    break
                    
                # 너무 많은 키를 스캔하지 않도록 제한
                if len(matched_keys) > 10000:  # 10,000개 제한
                    logger.warning(f"키 스캔 제한 도달: {len(matched_keys)}개 키")
                    break
            
            return matched_keys
            
        except Exception as e:
            logger.error(f"키 스캔 실패: {e}")
            return []
