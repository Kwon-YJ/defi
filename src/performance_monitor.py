import asyncio
import time
from typing import Dict, List, Optional
from datetime import datetime
from src.logger import setup_logger
from src.data_storage import DataStorage

logger = setup_logger(__name__)

class ExecutionTimer:
    """실행 시간 측정기"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.checkpoints = {}
        
    def start(self):
        """타이머 시작"""
        self.start_time = time.time()
        self.checkpoints = {}
        logger.debug("Execution timer started")
        
    def checkpoint(self, name: str):
        """체크포인트 기록"""
        if self.start_time is None:
            logger.warning("Timer not started, checkpoint not recorded")
            return
            
        self.checkpoints[name] = time.time() - self.start_time
        logger.debug(f"Checkpoint '{name}' recorded at {self.checkpoints[name]:.4f}s")
        
    def stop(self) -> float:
        """타이머 종료 및 총 실행 시간 반환"""
        self.end_time = time.time()
        if self.start_time is None:
            logger.warning("Timer not started")
            return 0.0
            
        total_time = self.end_time - self.start_time
        logger.debug(f"Execution timer stopped. Total time: {total_time:.4f}s")
        return total_time
    
    def get_checkpoint_intervals(self) -> Dict[str, float]:
        """체크포인트 간 간격 계산"""
        intervals = {}
        prev_time = 0.0
        prev_name = "start"
        
        for name, timestamp in self.checkpoints.items():
            intervals[f"{prev_name}_to_{name}"] = timestamp - prev_time
            prev_time = timestamp
            prev_name = name
            
        # 마지막 체크포인트에서 종료까지의 시간
        if self.end_time and self.start_time:
            total_time = self.end_time - self.start_time
            intervals[f"{prev_name}_to_end"] = total_time - prev_time
            
        return intervals

class PerformanceBenchmark:
    """성능 벤치마킹"""
    
    def __init__(self, target_time: float = 6.43):
        self.target_execution_time = target_time  # 논문 기준 (초)
        self.execution_times: List[float] = []
        self.storage = DataStorage()
        
    def record_execution_time(self, execution_time: float):
        """
        실행 시간 기록
        
        Args:
            execution_time: 실행 시간 (초)
        """
        self.execution_times.append(execution_time)
        
        # 최근 100개만 유지
        if len(self.execution_times) > 100:
            self.execution_times = self.execution_times[-100:]
        
        # Redis에 저장
        try:
            key = f"performance:execution_time:{datetime.now().isoformat()}"
            self.storage.redis_client.setex(key, 3600, str(execution_time))  # 1시간 보관
        except Exception as e:
            logger.warning(f"Failed to store execution time in Redis: {e}")
        
        logger.info(f"Execution time recorded: {execution_time:.4f}s (target: {self.target_execution_time:.2f}s)")
        
        # 목표 시간을 초과한 경우 경고
        if execution_time > self.target_execution_time:
            logger.warning(f"Execution time exceeds target: {execution_time:.4f}s > {self.target_execution_time:.2f}s")
    
    def get_performance_stats(self) -> Dict:
        """
        성능 통계 계산
        
        Returns:
            성능 통계 딕셔너리
        """
        if not self.execution_times:
            return {}
            
        import statistics
        
        current_avg = statistics.mean(self.execution_times)
        current_min = min(self.execution_times)
        current_max = max(self.execution_times)
        
        # 표준 편차 계산
        if len(self.execution_times) > 1:
            std_dev = statistics.stdev(self.execution_times)
        else:
            std_dev = 0.0
            
        # 목표 시간 대비 성능
        meets_target = current_avg <= self.target_execution_time
        target_ratio = current_avg / self.target_execution_time if self.target_execution_time > 0 else 0
        
        # 최근 10회 평균
        recent_times = self.execution_times[-10:] if len(self.execution_times) >= 10 else self.execution_times
        recent_avg = statistics.mean(recent_times) if recent_times else 0
        
        return {
            'average_time': current_avg,
            'min_time': current_min,
            'max_time': current_max,
            'std_deviation': std_dev,
            'sample_count': len(self.execution_times),
            'target_time': self.target_execution_time,
            'meets_target': meets_target,
            'target_ratio': target_ratio,
            'recent_average': recent_avg,
            'recent_sample_count': len(recent_times)
        }
    
    def get_performance_rating(self) -> str:
        """
        성능 등급 계산
        
        Returns:
            성능 등급 (Excellent, Good, Fair, Poor, Critical)
        """
        stats = self.get_performance_stats()
        if not stats:
            return "Unknown"
            
        avg_time = stats['average_time']
        target_time = stats['target_time']
        
        if avg_time <= target_time * 0.8:
            return "Excellent"
        elif avg_time <= target_time:
            return "Good"
        elif avg_time <= target_time * 1.5:
            return "Fair"
        elif avg_time <= target_time * 2.0:
            return "Poor"
        else:
            return "Critical"
    
    async def get_historical_performance(self, hours: int = 24) -> List[Dict]:
        """
        과거 성능 데이터 조회
        
        Args:
            hours: 조회할 시간 범위 (시간)
            
        Returns:
            과거 성능 데이터 리스트
        """
        try:
            pattern = "performance:execution_time:*"
            keys = self.storage.redis_client.keys(pattern)
            
            history = []
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            for key in keys:
                try:
                    timestamp_str = key.decode().split(':')[-1]
                    timestamp = datetime.fromisoformat(timestamp_str)
                    
                    if timestamp >= cutoff_time:
                        data = self.storage.redis_client.get(key)
                        if data:
                            execution_time = float(data.decode())
                            history.append({
                                'timestamp': timestamp_str,
                                'execution_time': execution_time
                            })
                except (ValueError, UnicodeDecodeError):
                    continue
            
            # 시간순 정렬
            history.sort(key=lambda x: x['timestamp'])
            return history
            
        except Exception as e:
            logger.error(f"Failed to retrieve historical performance data: {e}")
            return []
    
    def reset_stats(self):
        """성능 통계 초기화"""
        self.execution_times = []
        logger.info("Performance statistics reset")

class PerformanceMonitor:
    """성능 모니터링"""
    
    def __init__(self):
        self.benchmark = PerformanceBenchmark()
        self.timer = ExecutionTimer()
        self.active_timers = {}
        
    def start_monitoring(self, task_id: str = "default") -> ExecutionTimer:
        """
        모니터링 시작
        
        Args:
            task_id: 작업 식별자
            
        Returns:
            ExecutionTimer 인스턴스
        """
        timer = ExecutionTimer()
        timer.start()
        self.active_timers[task_id] = timer
        return timer
    
    def stop_monitoring(self, task_id: str = "default") -> Optional[float]:
        """
        모니터링 종료 및 실행 시간 기록
        
        Args:
            task_id: 작업 식별자
            
        Returns:
            실행 시간 (초) 또는 None
        """
        if task_id not in self.active_timers:
            logger.warning(f"No active timer found for task: {task_id}")
            return None
            
        timer = self.active_timers.pop(task_id)
        execution_time = timer.stop()
        self.benchmark.record_execution_time(execution_time)
        return execution_time
    
    def get_current_stats(self) -> Dict:
        """
        현재 성능 통계 조회
        
        Returns:
            현재 성능 통계 딕셔너리
        """
        return self.benchmark.get_performance_stats()
    
    def get_performance_report(self) -> Dict:
        """
        성능 보고서 생성
        
        Returns:
            성능 보고서 딕셔너리
        """
        stats = self.benchmark.get_performance_stats()
        rating = self.benchmark.get_performance_rating()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'stats': stats,
            'rating': rating,
            'target_achieved': stats.get('meets_target', False) if stats else False,
            'recommendation': self._generate_recommendation(stats, rating)
        }
    
    def _generate_recommendation(self, stats: Dict, rating: str) -> str:
        """
        성능 개선 추천 생성
        
        Args:
            stats: 성능 통계
            rating: 성능 등급
            
        Returns:
            추천 내용
        """
        if not stats:
            return "Insufficient data for recommendation"
            
        if rating in ["Excellent", "Good"]:
            return "Performance meets or exceeds target. Continue monitoring."
        elif rating == "Fair":
            return "Performance is acceptable but close to target. Consider optimization."
        elif rating == "Poor":
            return "Performance needs improvement. Optimize critical components."
        else:  # Critical
            return "Performance is significantly below target. Immediate optimization required."