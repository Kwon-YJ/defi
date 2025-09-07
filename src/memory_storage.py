#!/usr/bin/env python3
"""
In-Memory Data Storage
Redis 대신 사용할 수 있는 메모리 기반 데이터 저장소
테스트 및 개발용
"""

import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
from src.logger import setup_logger

logger = setup_logger(__name__)

class MemoryStorage:
    """메모리 기반 데이터 저장소"""
    
    def __init__(self):
        self.data: Dict[str, Dict] = {}
        self.ttl_data: Dict[str, float] = {}  # key -> expiration_time
        self.pool_data_ttl = 300  # 5분
        self.price_data_ttl = 60   # 1분
        
        # 정기적으로 만료된 데이터 정리
        self._last_cleanup = time.time()
        
    def _cleanup_expired_data(self):
        """만료된 데이터 정리"""
        current_time = time.time()
        
        # 10초마다 한번씩만 정리
        if current_time - self._last_cleanup < 10:
            return
            
        expired_keys = []
        for key, expiry_time in self.ttl_data.items():
            if current_time > expiry_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self.data:
                del self.data[key]
            del self.ttl_data[key]
        
        if expired_keys:
            logger.debug(f"만료된 데이터 {len(expired_keys)}개 정리됨")
            
        self._last_cleanup = current_time
    
    def _set_with_ttl(self, key: str, value: Any, ttl: int):
        """TTL과 함께 데이터 저장"""
        self._cleanup_expired_data()
        
        self.data[key] = value
        self.ttl_data[key] = time.time() + ttl
    
    def _get(self, key: str) -> Optional[Any]:
        """키로 데이터 조회"""
        self._cleanup_expired_data()
        
        # TTL 체크
        if key in self.ttl_data:
            if time.time() > self.ttl_data[key]:
                # 만료됨
                if key in self.data:
                    del self.data[key]
                del self.ttl_data[key]
                return None
        
        return self.data.get(key)
        
    async def store_pool_data(self, pool_address: str, pool_info: Dict):
        """풀 데이터 저장"""
        try:
            key = f"pool:{pool_address}"
            self._set_with_ttl(key, pool_info, self.pool_data_ttl)
            
            # 시계열 데이터도 저장
            timestamp = datetime.now().isoformat()
            ts_key = f"pool_history:{pool_address}:{timestamp}"
            self._set_with_ttl(ts_key, pool_info, 3600)  # 1시간 보관
            
            logger.debug(f"풀 데이터 저장됨: {pool_address}")
            
        except Exception as e:
            logger.error(f"풀 데이터 저장 실패: {e}")
    
    async def get_pool_data(self, pool_address: str) -> Optional[Dict]:
        """풀 데이터 조회"""
        try:
            key = f"pool:{pool_address}"
            return self._get(key)
        except Exception as e:
            logger.error(f"풀 데이터 조회 실패: {e}")
            return None
    
    async def store_arbitrage_opportunity(self, opportunity: Dict):
        """차익거래 기회 저장"""
        try:
            timestamp = datetime.now().isoformat()
            key = f"arbitrage:{timestamp}"
            self._set_with_ttl(key, opportunity, 3600)  # 1시간 보관
            
            # 통계용 인덱스 업데이트
            stats_key = "arbitrage_stats"
            stats = self._get(stats_key) or {'count': 0, 'last_update': None}
            stats['count'] += 1
            stats['last_update'] = timestamp
            self._set_with_ttl(stats_key, stats, 86400)  # 24시간
            
            logger.debug(f"차익거래 기회 저장됨: {key}")
            
        except Exception as e:
            logger.error(f"차익거래 기회 저장 실패: {e}")
    
    async def get_recent_opportunities(self, limit: int = 10) -> List[Dict]:
        """최근 차익거래 기회들 조회"""
        try:
            opportunities = []
            
            for key, value in self.data.items():
                if key.startswith('arbitrage:'):
                    opportunities.append(value)
            
            # 타임스탬프로 정렬 (최신순)
            opportunities.sort(
                key=lambda x: x.get('timestamp', ''), 
                reverse=True
            )
            
            return opportunities[:limit]
            
        except Exception as e:
            logger.error(f"최근 기회 조회 실패: {e}")
            return []
    
    async def store_price_data(self, asset: str, price_info: Dict):
        """가격 데이터 저장"""
        try:
            key = f"price:{asset}"
            self._set_with_ttl(key, price_info, self.price_data_ttl)
            
            # 가격 히스토리도 저장
            timestamp = datetime.now().isoformat()
            history_key = f"price_history:{asset}:{timestamp}"
            self._set_with_ttl(history_key, price_info, 3600)  # 1시간
            
            logger.debug(f"가격 데이터 저장됨: {asset}")
            
        except Exception as e:
            logger.error(f"가격 데이터 저장 실패: {e}")
    
    async def get_price_data(self, asset: str) -> Optional[Dict]:
        """가격 데이터 조회"""
        try:
            key = f"price:{asset}"
            return self._get(key)
        except Exception as e:
            logger.error(f"가격 데이터 조회 실패: {e}")
            return None
    
    async def store_performance_metrics(self, metrics: Dict):
        """성능 메트릭 저장"""
        try:
            timestamp = datetime.now().isoformat()
            key = f"metrics:{timestamp}"
            self._set_with_ttl(key, metrics, 86400)  # 24시간
            
            # 최신 메트릭도 별도 저장
            latest_key = "latest_metrics"
            self._set_with_ttl(latest_key, metrics, 86400)
            
        except Exception as e:
            logger.error(f"성능 메트릭 저장 실패: {e}")
    
    async def get_latest_metrics(self) -> Optional[Dict]:
        """최신 성능 메트릭 조회"""
        try:
            return self._get("latest_metrics")
        except Exception as e:
            logger.error(f"최신 메트릭 조회 실패: {e}")
            return None
    
    async def store_transaction_data(self, tx_hash: str, tx_data: Dict):
        """트랜잭션 데이터 저장"""
        try:
            key = f"tx:{tx_hash}"
            self._set_with_ttl(key, tx_data, 3600)  # 1시간
            
        except Exception as e:
            logger.error(f"트랜잭션 데이터 저장 실패: {e}")
    
    async def get_transaction_data(self, tx_hash: str) -> Optional[Dict]:
        """트랜잭션 데이터 조회"""
        try:
            key = f"tx:{tx_hash}"
            return self._get(key)
        except Exception as e:
            logger.error(f"트랜잭션 데이터 조회 실패: {e}")
            return None
    
    def get_storage_stats(self) -> Dict:
        """저장소 통계"""
        self._cleanup_expired_data()
        
        stats = {
            'total_keys': len(self.data),
            'keys_with_ttl': len(self.ttl_data),
            'data_types': defaultdict(int)
        }
        
        # 키 타입별 통계
        for key in self.data.keys():
            key_type = key.split(':')[0]
            stats['data_types'][key_type] += 1
        
        return dict(stats)
    
    def clear_all_data(self):
        """모든 데이터 삭제 (테스트용)"""
        self.data.clear()
        self.ttl_data.clear()
        logger.info("모든 저장 데이터가 삭제되었습니다")

# 전역 인스턴스 (싱글톤 패턴)
_memory_storage_instance = None

def get_memory_storage() -> MemoryStorage:
    """메모리 저장소 싱글톤 인스턴스 반환"""
    global _memory_storage_instance
    if _memory_storage_instance is None:
        _memory_storage_instance = MemoryStorage()
    return _memory_storage_instance

# 사용 예시
async def main():
    storage = get_memory_storage()
    
    # 테스트 데이터 저장
    await storage.store_pool_data("0x1234", {
        "reserve0": 1000,
        "reserve1": 2000,
        "fee": 0.003
    })
    
    # 차익거래 기회 저장
    await storage.store_arbitrage_opportunity({
        "path": ["WETH", "USDC", "DAI", "WETH"],
        "profit": 1.5,
        "timestamp": datetime.now().isoformat()
    })
    
    # 통계 출력
    stats = storage.get_storage_stats()
    print(f"저장소 통계: {stats}")
    
    # 최근 기회들 조회
    opportunities = await storage.get_recent_opportunities()
    print(f"최근 기회: {len(opportunities)}개")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())