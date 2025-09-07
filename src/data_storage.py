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
        
    async def store_pool_data(self, pool_address: str, pool_info: Dict):
        """풀 데이터 저장"""
        try:
            key = f"pool:{pool_address}"
            data = json.dumps(pool_info, default=str)
            self.redis_client.setex(key, self.pool_data_ttl, data)
            
            # 시계열 데이터도 저장
            timestamp = datetime.now().isoformat()
            ts_key = f"pool_history:{pool_address}:{timestamp}"
            self.redis_client.setex(ts_key, 3600, data)  # 1시간 보관
            
        except Exception as e:
            logger.error(f"풀 데이터 저장 실패: {e}")
    
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
        """풀 데이터 조회"""
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
        """차익거래 기회 저장"""
        try:
            timestamp = datetime.now().isoformat()
            key = f"arbitrage:{timestamp}"
            data = json.dumps(opportunity, default=str)
            self.redis_client.setex(key, 3600, data)  # 1시간 보관
            
            # 통계용 리스트에도 추가
            self.redis_client.lpush("arbitrage_opportunities", data)
            self.redis_client.ltrim("arbitrage_opportunities", 0, 999)  # 최근 1000개만 보관
            
        except Exception as e:
            logger.error(f"차익거래 기회 저장 실패: {e}")
    
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
        """최근 차익거래 기회 조회"""
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
        """풀 가격 히스토리 조회"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Redis에서 시계열 데이터 조회
            pattern = f"pool_history:{pool_address}:*"
            keys = self.redis_client.keys(pattern)
            
            history = []
            for key in keys:
                # 키에서 타임스탬프 추출
                timestamp_str = key.decode().split(':')[-1]
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if start_time <= timestamp <= end_time:
                        data = self.redis_client.get(key)
                        if data:
                            pool_data = json.loads(data)
                            pool_data['timestamp'] = timestamp_str
                            history.append(pool_data)
                except ValueError:
                    continue
            
            # 시간순 정렬
            history.sort(key=lambda x: x['timestamp'])
            return history
            
        except Exception as e:
            logger.error(f"가격 히스토리 조회 실패: {e}")
            return []
    
    async def get_historical_pool_data(self, pool_address: str, 
                                     start_date: datetime, 
                                     end_date: datetime) -> List[Dict]:
        """특정 기간의 히스토리컬 풀 데이터 조회"""
        try:
            # 히스토리컬 데이터 조회
            pattern = f"pool_historical:{pool_address}:*"
            keys = self.redis_client.keys(pattern)
            
            history = []
            for key in keys:
                # 키에서 타임스탬프 추출
                timestamp_str = key.decode().split(':')[-1]
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if start_date <= timestamp <= end_date:
                        data = self.redis_client.get(key)
                        if data:
                            pool_data = json.loads(data)
                            pool_data['timestamp'] = timestamp_str
                            history.append(pool_data)
                except ValueError:
                    continue
            
            # 시간순 정렬
            history.sort(key=lambda x: x['timestamp'])
            return history
            
        except Exception as e:
            logger.error(f"히스토리컬 풀 데이터 조회 실패: {e}")
            return []
    
    async def get_historical_arbitrage_opportunities(self, 
                                                   start_date: datetime, 
                                                   end_date: datetime) -> List[Dict]:
        """특정 기간의 히스토리컬 차익거래 기회 조회"""
        try:
            # 히스토리컬 데이터 조회
            pattern = f"arbitrage_historical:*"
            keys = self.redis_client.keys(pattern)
            
            opportunities = []
            for key in keys:
                # 키에서 타임스탬프 추출
                timestamp_str = key.decode().split(':')[-1]
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if start_date <= timestamp <= end_date:
                        data = self.redis_client.get(key)
                        if data:
                            opportunity = json.loads(data)
                            opportunity['timestamp'] = timestamp_str
                            opportunities.append(opportunity)
                except ValueError:
                    continue
            
            # 시간순 정렬
            opportunities.sort(key=lambda x: x['timestamp'])
            return opportunities
            
        except Exception as e:
            logger.error(f"히스토리컬 차익거래 기회 조회 실패: {e}")
            return []
