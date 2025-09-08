import redis
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# Handle imports properly
try:
    from src.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    # Fallback for direct execution
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
from config.config import config

class OptimizedDataStorage:
    def __init__(self):
        # Use connection pooling for better performance
        self.redis_client = redis.ConnectionPool(
            host='localhost',
            port=6379,
            db=0,
            max_connections=20,
            retry_on_timeout=True
        )
        self.redis_conn = redis.Redis(connection_pool=self.redis_client)
        
        self.pool_data_ttl = 300  # 5분
        self.price_data_ttl = 60   # 1분
        self.historical_data_ttl = 2592000  # 30일 (기본값)
        
        # Pre-defined key prefixes for consistency
        self._key_prefixes = {
            'pool': 'pool:',
            'pool_history': 'pool_history:',
            'pool_historical': 'pool_historical:',
            'arbitrage': 'arbitrage:',
            'arbitrage_historical': 'arbitrage_historical:',
            'price_historical': 'price_historical:',
            'performance': 'performance:'
        }
        
        # Pipeline for batch operations
        self._pipeline = None
    
    def get_connection(self):
        """Get Redis connection from pool"""
        return self.redis_conn
    
    def start_batch_operation(self):
        """Start a batch operation using pipeline"""
        if self._pipeline is None:
            self._pipeline = self.redis_conn.pipeline()
        return self._pipeline
    
    def execute_batch(self):
        """Execute all batched operations"""
        if self._pipeline:
            result = self._pipeline.execute()
            self._pipeline = None
            return result
        return []
    
    async def store_pool_data_batch(self, pool_data_list: List[Dict]):
        """Batch store pool data for better performance"""
        try:
            pipe = self.start_batch_operation()
            
            for pool_data in pool_data_list:
                pool_address = pool_data.get('address')
                if not pool_address:
                    continue
                    
                key = f"pool:{pool_address}"
                data = json.dumps(pool_data, default=str)
                pipe.setex(key, self.pool_data_ttl, data)
                
                # Store historical data
                timestamp = datetime.now().isoformat()
                ts_key = f"pool_history:{pool_address}:{timestamp}"
                pipe.setex(ts_key, 3600, data)
                
                # Update indexes
                self._update_pool_index_batch(pipe, pool_address, timestamp)
            
            self.execute_batch()
            logger.info(f"Batch stored {len(pool_data_list)} pool data entries")
            
        except Exception as e:
            logger.error(f"Batch pool data storage failed: {e}")
    
    def _update_pool_index_batch(self, pipe, pool_address: str, timestamp: str):
        """Update pool index using batch operations"""
        try:
            # Pool address index
            index_key = "index:pools"
            pipe.sadd(index_key, pool_address)
            
            # Timestamp index
            time_index_key = f"index:pool_times:{pool_address}"
            pipe.zadd(time_index_key, {timestamp: datetime.fromisoformat(timestamp).timestamp()})
            
            # Cleanup old entries
            cutoff_time = (datetime.now() - timedelta(hours=24)).timestamp()
            pipe.zremrangebyscore(time_index_key, 0, cutoff_time)
            
        except Exception as e:
            logger.debug(f"Pool index batch update failed (non-critical): {e}")
    
    async def store_pool_data(self, pool_address: str, pool_info: Dict):
        """Optimized 풀 데이터 저장"""
        try:
            key = f"pool:{pool_address}"
            data = json.dumps(pool_info, default=str)
            
            # Use pipeline for atomic operations
            pipe = self.redis_conn.pipeline()
            pipe.setex(key, self.pool_data_ttl, data)
            
            # Store time series data
            timestamp = datetime.now().isoformat()
            ts_key = f"pool_history:{pool_address}:{timestamp}"
            pipe.setex(ts_key, 3600, data)
            
            # Update indexes atomically
            self._update_pool_index_batch(pipe, pool_address, timestamp)
            
            pipe.execute()
            
        except Exception as e:
            logger.error(f"Pool data storage failed: {e}")
    
    async def get_pool_data(self, pool_address: str) -> Optional[Dict]:
        """Optimized 풀 데이터 조회 with connection pooling"""
        try:
            key = f"pool:{pool_address}"
            data = self.redis_conn.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Pool data retrieval failed: {e}")
            return None
    
    async def store_arbitrage_opportunity_batch(self, opportunities: List[Dict]):
        """Batch store arbitrage opportunities for better performance"""
        try:
            pipe = self.start_batch_operation()
            
            for opportunity in opportunities:
                timestamp = datetime.now().isoformat()
                key = f"arbitrage:{timestamp}"
                data = json.dumps(opportunity, default=str)
                pipe.setex(key, 3600, data)
                
                # Add to list for statistics
                pipe.lpush("arbitrage_opportunities", data)
                pipe.ltrim("arbitrage_opportunities", 0, 999)
                
                # Update index
                self._update_arbitrage_index_batch(pipe, timestamp)
            
            self.execute_batch()
            logger.info(f"Batch stored {len(opportunities)} arbitrage opportunities")
            
        except Exception as e:
            logger.error(f"Batch arbitrage storage failed: {e}")
    
    def _update_arbitrage_index_batch(self, pipe, timestamp: str):
        """Update arbitrage index using batch operations"""
        try:
            index_key = "index:arbitrage_times"
            pipe.zadd(index_key, {timestamp: datetime.fromisoformat(timestamp).timestamp()})
            
            # Cleanup old entries
            cutoff_time = (datetime.now() - timedelta(hours=24)).timestamp()
            pipe.zremrangebyscore(index_key, 0, cutoff_time)
            
        except Exception as e:
            logger.debug(f"Arbitrage index batch update failed (non-critical): {e}")
    
    async def store_arbitrage_opportunity(self, opportunity: Dict):
        """Optimized 차익거래 기회 저장"""
        try:
            timestamp = datetime.now().isoformat()
            key = f"arbitrage:{timestamp}"
            data = json.dumps(opportunity, default=str)
            
            # Use pipeline for atomic operations
            pipe = self.redis_conn.pipeline()
            pipe.setex(key, 3600, data)
            
            # Add to list for statistics with trim
            pipe.lpush("arbitrage_opportunities", data)
            pipe.ltrim("arbitrage_opportunities", 0, 999)
            
            # Update index atomically
            self._update_arbitrage_index_batch(pipe, timestamp)
            
            pipe.execute()
            
        except Exception as e:
            logger.error(f"Arbitrage opportunity storage failed: {e}")
    
    async def get_recent_opportunities(self, limit: int = 100) -> List[Dict]:
        """Optimized 최근 차익거래 기회 조회 with better connection handling"""
        try:
            # Use connection directly for better performance
            data_list = self.redis_conn.lrange("arbitrage_opportunities", 0, limit-1)
            opportunities = []
            for data in data_list:
                opportunities.append(json.loads(data))
            return opportunities
        except Exception as e:
            logger.error(f"Recent opportunities retrieval failed: {e}")
            return []
    
    async def get_performance_data_range(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Optimized 성능 데이터 범위 조회"""
        try:
            # Use sorted set for efficient range queries
            index_key = "index:performance_times"
            timestamp_strings = self.redis_conn.zrangebyscore(
                index_key,
                start_time.timestamp(),
                end_time.timestamp(),
                withscores=False
            )
            
            # Batch retrieve data using pipeline
            if not timestamp_strings:
                return []
            
            pipe = self.redis_conn.pipeline()
            keys_to_fetch = []
            
            for timestamp_str in timestamp_strings:
                try:
                    key = f"performance:execution_time:{timestamp_str.decode()}"
                    keys_to_fetch.append(key)
                    pipe.get(key)
                except (ValueError, UnicodeDecodeError):
                    continue
            
            # Execute batch get
            results = pipe.execute()
            
            # Process results
            history = []
            for i, data in enumerate(results):
                if data:
                    try:
                        execution_time = float(data.decode())
                        history.append({
                            'timestamp': keys_to_fetch[i].split(':')[-1],
                            'execution_time': execution_time
                        })
                    except (ValueError, UnicodeDecodeError):
                        continue
            
            # Sort by timestamp
            history.sort(key=lambda x: x['timestamp'])
            return history
            
        except Exception as e:
            logger.error(f"Performance data range retrieval failed: {e}")
            return []
    
    async def store_performance_data_batch(self, performance_data_list: List[Dict]):
        """Batch store performance data for better performance"""
        try:
            pipe = self.start_batch_operation()
            
            for perf_data in performance_data_list:
                timestamp = perf_data.get('timestamp', datetime.now().isoformat())
                execution_time = perf_data.get('execution_time', 0)
                
                key = f"performance:execution_time:{timestamp}"
                pipe.setex(key, 3600, str(execution_time))
                
                # Update index
                index_key = "index:performance_times"
                pipe.zadd(index_key, {timestamp: datetime.fromisoformat(timestamp).timestamp()})
            
            # Cleanup old entries in batch
            cutoff_time = (datetime.now() - timedelta(hours=24)).timestamp()
            index_key = "index:performance_times"
            pipe.zremrangebyscore(index_key, 0, cutoff_time)
            
            self.execute_batch()
            logger.info(f"Batch stored {len(performance_data_list)} performance data entries")
            
        except Exception as e:
            logger.error(f"Batch performance data storage failed: {e}")
    
    async def get_pool_historical_range(self, pool_address: str, 
                                      start_date: datetime, 
                                      end_date: datetime) -> List[Dict]:
        """Optimized 풀 히스토리컬 데이터 범위 조회"""
        try:
            # Use sorted set index for efficient range queries
            time_index_key = f"index:pool_times:{pool_address}"
            timestamp_strings = self.redis_conn.zrangebyscore(
                time_index_key,
                start_date.timestamp(),
                end_date.timestamp(),
                withscores=False
            )
            
            if not timestamp_strings:
                return []
            
            # Batch retrieve data using pipeline
            pipe = self.redis_conn.pipeline()
            keys_to_fetch = []
            
            for timestamp_str in timestamp_strings:
                try:
                    # Reconstruct key from timestamp
                    timestamp = timestamp_str.decode()
                    key = f"pool_history:{pool_address}:{timestamp}"
                    keys_to_fetch.append(key)
                    pipe.get(key)
                except (ValueError, UnicodeDecodeError):
                    continue
            
            # Execute batch get
            results = pipe.execute()
            
            # Process results
            history = []
            for i, data in enumerate(results):
                if data:
                    try:
                        pool_data = json.loads(data)
                        pool_data['timestamp'] = keys_to_fetch[i].split(':')[-1]
                        history.append(pool_data)
                    except (ValueError, json.JSONDecodeError):
                        continue
            
            # Sort by timestamp
            history.sort(key=lambda x: x['timestamp'])
            return history
            
        except Exception as e:
            logger.error(f"Pool historical range retrieval failed: {e}")
            return []
    
    def cleanup_old_data(self):
        """Clean up old data to optimize memory usage"""
        try:
            # Clean up old performance indexes
            cutoff_time = (datetime.now() - timedelta(hours=24)).timestamp()
            
            pipe = self.redis_conn.pipeline()
            
            # Clean performance indexes
            perf_index_key = "index:performance_times"
            pipe.zremrangebyscore(perf_index_key, 0, cutoff_time)
            
            # Clean pool time indexes
            # Get all pool addresses first
            pool_addresses = self.redis_conn.smembers("index:pools")
            for pool_addr in pool_addresses:
                try:
                    time_index_key = f"index:pool_times:{pool_addr.decode()}"
                    pipe.zremrangebyscore(time_index_key, 0, cutoff_time)
                except Exception:
                    continue
            
            # Clean arbitrage indexes
            arb_index_key = "index:arbitrage_times"
            pipe.zremrangebyscore(arb_index_key, 0, cutoff_time)
            
            pipe.execute()
            logger.info("Old data cleanup completed")
            
        except Exception as e:
            logger.error(f"Old data cleanup failed: {e}")