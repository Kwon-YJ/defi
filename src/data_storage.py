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

    # --- V3 fee tier stats ---
    async def upsert_v3_fee_stats(self, pool_address: str, fee_tier: int,
                                  fee0_per_L: float, fee1_per_L: float,
                                  alpha: float = 0.2) -> None:
        """feeGrowth 기반 per-L 수수료 샘플로 EMA 통계 업데이트."""
        try:
            key = f"v3stats:{pool_address}:{int(fee_tier)}"
            raw = self.redis_client.get(key)
            now = datetime.now().isoformat()
            if raw:
                obj = json.loads(raw)
            else:
                obj = {"ema_fee0": 0.0, "ema_fee1": 0.0, "count": 0}
            c = int(obj.get("count", 0))
            ema0 = float(obj.get("ema_fee0", 0.0))
            ema1 = float(obj.get("ema_fee1", 0.0))
            ema0 = (1 - alpha) * ema0 + alpha * float(fee0_per_L)
            ema1 = (1 - alpha) * ema1 + alpha * float(fee1_per_L)
            obj.update({
                "ema_fee0": ema0,
                "ema_fee1": ema1,
                "count": c + 1,
                "updated_at": now,
            })
            self.redis_client.setex(key, 24 * 3600, json.dumps(obj))
        except Exception as e:
            logger.debug(f"V3 fee stats upsert 실패: {e}")

    async def get_v3_fee_stats(self, pool_address: str, fee_tier: int) -> Optional[Dict]:
        try:
            key = f"v3stats:{pool_address}:{int(fee_tier)}"
            raw = self.redis_client.get(key)
            if not raw:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.debug(f"V3 fee stats 조회 실패: {e}")
            return None
