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

    # --- Price feeds ---
    async def store_token_price(self, token_address: str, price_usd: float) -> None:
        """토큰 USD 가격 저장 (TTL: price_data_ttl)."""
        try:
            key = f"price:{token_address.lower()}"
            obj = {"price_usd": float(price_usd), "updated_at": datetime.now().isoformat()}
            self.redis_client.setex(key, self.price_data_ttl, json.dumps(obj))
        except Exception as e:
            logger.debug(f"가격 저장 실패 {token_address[:6]}: {e}")

    async def get_token_price(self, token_address: str) -> Optional[Dict]:
        try:
            key = f"price:{token_address.lower()}"
            raw = self.redis_client.get(key)
            if not raw:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.debug(f"가격 조회 실패 {token_address[:6]}: {e}")
            return None
    
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

    # --- V3 band histogram (positions) ---
    async def upsert_v3_band_histogram(self, pool_address: str, tick_lower: int, tick_upper: int,
                                       delta_liquidity: Optional[int] = None) -> None:
        """밴드별 유동성 히스토그램을 업데이트. delta_liquidity가 있으면 유동성 총합에 가산.

        저장 구조 (JSON):
          {
            "bands": {"{lower}:{upper}": {"liq": float, "count": int}},
            "total_liq": float,
            "total_count": int,
            "updated_at": iso8601
          }
        """
        try:
            key = f"v3band:{pool_address}"
            raw = self.redis_client.get(key)
            if raw:
                obj = json.loads(raw)
            else:
                obj = {"bands": {}, "total_liq": 0.0, "total_count": 0}
            bkey = f"{int(tick_lower)}:{int(tick_upper)}"
            band = obj.setdefault("bands", {}).get(bkey, {"liq": 0.0, "count": 0})
            # count 증가
            band["count"] = int(band.get("count", 0)) + 1
            obj["total_count"] = int(obj.get("total_count", 0)) + 1
            # 유동성 합산
            if delta_liquidity is not None:
                inc = float(delta_liquidity)
                band["liq"] = float(band.get("liq", 0.0)) + inc
                obj["total_liq"] = float(obj.get("total_liq", 0.0)) + inc
            # 저장
            obj.setdefault("bands", {})[bkey] = band
            obj["updated_at"] = datetime.now().isoformat()
            self.redis_client.setex(key, 7 * 24 * 3600, json.dumps(obj))
        except Exception as e:
            logger.debug(f"V3 band histogram upsert 실패: {e}")

    async def get_v3_band_weight(self, pool_address: str, tick_lower: int, tick_upper: int) -> float:
        """해당 밴드의 상대 가중치(0..1)를 반환. 데이터 없으면 1.0."""
        try:
            key = f"v3band:{pool_address}"
            raw = self.redis_client.get(key)
            if not raw:
                return 1.0
            obj = json.loads(raw)
            bands = obj.get("bands", {})
            total_liq = float(obj.get("total_liq", 0.0))
            bkey = f"{int(tick_lower)}:{int(tick_upper)}"
            band = bands.get(bkey)
            if not band:
                return 1.0
            bliq = float(band.get("liq", 0.0))
            if total_liq <= 0.0:
                # total 부재 시 count 비율로 근사
                total_cnt = float(obj.get("total_count", 0.0))
                if total_cnt <= 0:
                    return 1.0
                return max(0.0, float(band.get("count", 0)) / total_cnt)
            return max(0.0, bliq / total_liq)
        except Exception as e:
            logger.debug(f"V3 band weight 조회 실패: {e}")
            return 1.0
