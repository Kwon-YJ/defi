"""
Performance Benchmarking Module for DEFIPOSER-ARB
실행 시간 측정 및 벤치마킹 구현 (논문 6.43초 목표)

이 모듈은 논문의 성능 기준을 달성하기 위한 실행 시간 측정 및 벤치마킹을 제공합니다.
- 목표: 평균 6.43초 이하 실행 시간
- 블록별 처리 시간 추적
- 성능 병목점 식별
- 실시간 모니터링
"""

import time
import asyncio
import statistics
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from contextlib import contextmanager
import psutil
import threading
from collections import deque, defaultdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    timestamp: datetime
    block_number: int
    total_execution_time: float  # seconds
    graph_building_time: float
    negative_cycle_detection_time: float
    local_search_time: float
    parameter_optimization_time: float
    validation_time: float
    memory_usage: float  # MB
    cpu_usage: float  # percentage
    opportunities_found: int
    strategies_executed: int
    total_revenue: float
    gas_cost: float

@dataclass
class ComponentTiming:
    """컴포넌트별 타이밍 정보"""
    name: str
    start_time: float
    end_time: float
    duration: float
    cpu_before: float
    cpu_after: float
    memory_before: float
    memory_after: float

class PerformanceBenchmarker:
    """
    DEFIPOSER-ARB 성능 벤치마킹 클래스
    논문의 6.43초 목표 달성을 위한 종합 성능 측정 도구
    """
    
    def __init__(self, target_time: float = 6.43, history_size: int = 1000):
        """
        Args:
            target_time: 목표 실행 시간 (초)
            history_size: 성능 기록 보관 개수
        """
        self.target_time = target_time
        self.history_size = history_size
        
        # 성능 기록 저장
        self.performance_history = deque(maxlen=history_size)
        self.component_timings = defaultdict(list)
        
        # 현재 측정 중인 메트릭
        self.current_metrics: Optional[PerformanceMetrics] = None
        self.timing_stack: List[ComponentTiming] = []
        
        # 통계 정보
        self.stats = {
            'total_blocks_processed': 0,
            'blocks_under_target': 0,
            'blocks_over_target': 0,
            'average_execution_time': 0.0,
            'fastest_execution': float('inf'),
            'slowest_execution': 0.0,
            'success_rate': 0.0
        }
        
        # 경고 임계값
        self.warning_threshold = target_time * 0.8  # 80% of target time
        self.critical_threshold = target_time  # 100% of target time
        
        # 로그 파일 설정
        self.log_file = Path("logs/performance_benchmark.json")
        self.log_file.parent.mkdir(exist_ok=True)
        
    def start_block_processing(self, block_number: int) -> None:
        """블록 처리 시작"""
        self.current_metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            block_number=block_number,
            total_execution_time=0.0,
            graph_building_time=0.0,
            negative_cycle_detection_time=0.0,
            local_search_time=0.0,
            parameter_optimization_time=0.0,
            validation_time=0.0,
            memory_usage=self._get_memory_usage(),
            cpu_usage=self._get_cpu_usage(),
            opportunities_found=0,
            strategies_executed=0,
            total_revenue=0.0,
            gas_cost=0.0
        )
        
        logger.info(f"🚀 블록 {block_number} 처리 시작 (목표: {self.target_time}초)")
        
    def end_block_processing(self, opportunities_found: int = 0, 
                           strategies_executed: int = 0, 
                           total_revenue: float = 0.0,
                           gas_cost: float = 0.0) -> PerformanceMetrics:
        """블록 처리 완료 및 메트릭 저장"""
        if not self.current_metrics:
            raise ValueError("start_block_processing()을 먼저 호출해야 합니다")
            
        # 최종 메트릭 업데이트
        end_time = datetime.now()
        self.current_metrics.total_execution_time = (
            end_time - self.current_metrics.timestamp
        ).total_seconds()
        
        self.current_metrics.opportunities_found = opportunities_found
        self.current_metrics.strategies_executed = strategies_executed
        self.current_metrics.total_revenue = total_revenue
        self.current_metrics.gas_cost = gas_cost
        
        # 성능 기록 저장
        self.performance_history.append(self.current_metrics)
        
        # 통계 업데이트
        self._update_stats()
        
        # 성능 평가 및 로깅
        self._evaluate_performance(self.current_metrics)
        
        # 로그 파일에 저장
        self._save_to_log()
        
        result = self.current_metrics
        self.current_metrics = None
        
        return result
    
    @contextmanager
    def time_component(self, component_name: str):
        """컴포넌트별 실행 시간 측정 컨텍스트 매니저"""
        timing = ComponentTiming(
            name=component_name,
            start_time=time.perf_counter(),
            end_time=0.0,
            duration=0.0,
            cpu_before=self._get_cpu_usage(),
            cpu_after=0.0,
            memory_before=self._get_memory_usage(),
            memory_after=0.0
        )
        
        self.timing_stack.append(timing)
        
        try:
            yield timing
        finally:
            timing.end_time = time.perf_counter()
            timing.duration = timing.end_time - timing.start_time
            timing.cpu_after = self._get_cpu_usage()
            timing.memory_after = self._get_memory_usage()
            
            # 현재 메트릭에 컴포넌트 시간 반영
            if self.current_metrics:
                self._update_component_timing(component_name, timing.duration)
                
            # 컴포넌트 타이밍 기록 저장
            self.component_timings[component_name].append(timing)
            
            # 스택에서 제거
            if self.timing_stack and self.timing_stack[-1] == timing:
                self.timing_stack.pop()
                
            logger.debug(f"⏱️ {component_name}: {timing.duration:.3f}초")
    
    def _update_component_timing(self, component_name: str, duration: float) -> None:
        """컴포넌트별 타이밍을 현재 메트릭에 반영"""
        if not self.current_metrics:
            return
            
        if component_name == "graph_building":
            self.current_metrics.graph_building_time += duration
        elif component_name == "negative_cycle_detection":
            self.current_metrics.negative_cycle_detection_time += duration
        elif component_name == "local_search":
            self.current_metrics.local_search_time += duration
        elif component_name == "parameter_optimization":
            self.current_metrics.parameter_optimization_time += duration
        elif component_name == "validation":
            self.current_metrics.validation_time += duration
    
    def _get_memory_usage(self) -> float:
        """현재 메모리 사용량 (MB)"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _get_cpu_usage(self) -> float:
        """현재 CPU 사용률 (%%)"""
        return psutil.cpu_percent()
    
    def _update_stats(self) -> None:
        """통계 정보 업데이트"""
        if not self.performance_history:
            return
            
        execution_times = [m.total_execution_time for m in self.performance_history]
        
        self.stats['total_blocks_processed'] = len(self.performance_history)
        self.stats['blocks_under_target'] = sum(
            1 for t in execution_times if t <= self.target_time
        )
        self.stats['blocks_over_target'] = (
            self.stats['total_blocks_processed'] - self.stats['blocks_under_target']
        )
        self.stats['average_execution_time'] = statistics.mean(execution_times)
        self.stats['fastest_execution'] = min(execution_times)
        self.stats['slowest_execution'] = max(execution_times)
        self.stats['success_rate'] = (
            self.stats['blocks_under_target'] / self.stats['total_blocks_processed']
        )
    
    def _evaluate_performance(self, metrics: PerformanceMetrics) -> None:
        """성능 평가 및 경고"""
        execution_time = metrics.total_execution_time
        
        if execution_time <= self.warning_threshold:
            logger.info(f"✅ 블록 {metrics.block_number}: {execution_time:.3f}초 "
                       f"(목표 {self.target_time}초 대비 우수)")
        elif execution_time <= self.critical_threshold:
            logger.warning(f"⚠️ 블록 {metrics.block_number}: {execution_time:.3f}초 "
                          f"(목표 {self.target_time}초 근접)")
        else:
            logger.error(f"❌ 블록 {metrics.block_number}: {execution_time:.3f}초 "
                        f"(목표 {self.target_time}초 초과)")
            
        # 컴포넌트별 병목점 분석
        self._analyze_bottlenecks(metrics)
    
    def _analyze_bottlenecks(self, metrics: PerformanceMetrics) -> None:
        """성능 병목점 분석"""
        components = {
            "그래프 구축": metrics.graph_building_time,
            "네거티브 사이클 탐지": metrics.negative_cycle_detection_time,
            "로컬 서치": metrics.local_search_time,
            "파라미터 최적화": metrics.parameter_optimization_time,
            "검증": metrics.validation_time
        }
        
        # 가장 시간이 오래 걸린 컴포넌트 찾기
        bottleneck = max(components, key=components.get)
        bottleneck_time = components[bottleneck]
        
        if bottleneck_time > self.target_time * 0.3:  # 30% 이상
            logger.warning(f"🔍 병목점 발견: {bottleneck} ({bottleneck_time:.3f}초)")
    
    def _save_to_log(self) -> None:
        """성능 기록을 JSON 파일에 저장"""
        if not self.current_metrics:
            return
            
        # datetime을 문자열로 변환
        metrics_dict = asdict(self.current_metrics)
        metrics_dict['timestamp'] = self.current_metrics.timestamp.isoformat()
        
        log_data = {
            "timestamp": self.current_metrics.timestamp.isoformat(),
            "metrics": metrics_dict,
            "stats": self.stats
        }
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False, default=str) + '\n')
        except Exception as e:
            logger.error(f"성능 로그 저장 실패: {e}")
    
    def get_performance_report(self, last_n_blocks: Optional[int] = None) -> Dict[str, Any]:
        """성능 보고서 생성"""
        if not self.performance_history:
            return {"error": "성능 기록이 없습니다"}
            
        # 최근 N개 블록만 분석 (지정된 경우)
        history = list(self.performance_history)
        if last_n_blocks:
            history = history[-last_n_blocks:]
            
        execution_times = [m.total_execution_time for m in history]
        
        report = {
            "summary": {
                "target_time": self.target_time,
                "blocks_analyzed": len(history),
                "success_rate": sum(1 for t in execution_times if t <= self.target_time) / len(history),
                "average_time": statistics.mean(execution_times),
                "median_time": statistics.median(execution_times),
                "fastest_time": min(execution_times),
                "slowest_time": max(execution_times),
                "std_deviation": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            },
            "component_analysis": self._get_component_analysis(history),
            "resource_usage": self._get_resource_analysis(history),
            "recommendations": self._get_performance_recommendations(history)
        }
        
        return report
    
    def _get_component_analysis(self, history: List[PerformanceMetrics]) -> Dict[str, Any]:
        """컴포넌트별 성능 분석"""
        components = {
            "graph_building": [m.graph_building_time for m in history],
            "negative_cycle_detection": [m.negative_cycle_detection_time for m in history],
            "local_search": [m.local_search_time for m in history],
            "parameter_optimization": [m.parameter_optimization_time for m in history],
            "validation": [m.validation_time for m in history]
        }
        
        analysis = {}
        for name, times in components.items():
            if times and any(t > 0 for t in times):
                analysis[name] = {
                    "average": statistics.mean(times),
                    "max": max(times),
                    "percentage_of_total": (statistics.mean(times) / self.target_time) * 100
                }
        
        return analysis
    
    def _get_resource_analysis(self, history: List[PerformanceMetrics]) -> Dict[str, Any]:
        """리소스 사용량 분석"""
        memory_usage = [m.memory_usage for m in history if m.memory_usage > 0]
        cpu_usage = [m.cpu_usage for m in history if m.cpu_usage > 0]
        
        analysis = {}
        
        if memory_usage:
            analysis["memory"] = {
                "average_mb": statistics.mean(memory_usage),
                "max_mb": max(memory_usage),
                "min_mb": min(memory_usage)
            }
            
        if cpu_usage:
            analysis["cpu"] = {
                "average_percent": statistics.mean(cpu_usage),
                "max_percent": max(cpu_usage),
                "min_percent": min(cpu_usage)
            }
            
        return analysis
    
    def _get_performance_recommendations(self, history: List[PerformanceMetrics]) -> List[str]:
        """성능 개선 권장사항"""
        recommendations = []
        
        execution_times = [m.total_execution_time for m in history]
        avg_time = statistics.mean(execution_times)
        
        if avg_time > self.target_time:
            recommendations.append(
                f"평균 실행 시간({avg_time:.3f}초)이 목표({self.target_time}초)를 초과합니다."
            )
            
        # 컴포넌트별 분석
        component_analysis = self._get_component_analysis(history)
        for name, data in component_analysis.items():
            if data["percentage_of_total"] > 40:  # 40% 이상
                recommendations.append(
                    f"{name} 컴포넌트가 전체 시간의 {data['percentage_of_total']:.1f}%를 차지합니다. 최적화가 필요합니다."
                )
                
        # 메모리 사용량 분석
        resource_analysis = self._get_resource_analysis(history)
        if "memory" in resource_analysis:
            avg_memory = resource_analysis["memory"]["average_mb"]
            if avg_memory > 1000:  # 1GB 이상
                recommendations.append(
                    f"평균 메모리 사용량({avg_memory:.0f}MB)이 높습니다. 메모리 최적화를 고려하세요."
                )
        
        if not recommendations:
            recommendations.append("성능이 목표 기준을 만족합니다. 현재 설정을 유지하세요.")
            
        return recommendations
    
    def real_time_monitor(self, check_interval: int = 60) -> None:
        """실시간 성능 모니터링 (별도 스레드에서 실행)"""
        def monitor():
            while True:
                try:
                    if len(self.performance_history) >= 10:  # 최소 10개 기록 필요
                        recent_report = self.get_performance_report(last_n_blocks=10)
                        
                        success_rate = recent_report["summary"]["success_rate"]
                        avg_time = recent_report["summary"]["average_time"]
                        
                        if success_rate < 0.8:  # 80% 미만 성공률
                            logger.warning(
                                f"🚨 성능 경고: 최근 성공률 {success_rate:.1%}, "
                                f"평균 실행시간 {avg_time:.3f}초"
                            )
                        else:
                            logger.info(
                                f"📊 성능 양호: 성공률 {success_rate:.1%}, "
                                f"평균 실행시간 {avg_time:.3f}초"
                            )
                            
                    time.sleep(check_interval)
                    
                except Exception as e:
                    logger.error(f"실시간 모니터링 오류: {e}")
                    time.sleep(check_interval)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        logger.info(f"실시간 성능 모니터링 시작 (체크 간격: {check_interval}초)")

# 전역 벤치마커 인스턴스
global_benchmarker = PerformanceBenchmarker()

# 편의 함수들
def start_benchmarking(block_number: int) -> None:
    """블록 처리 벤치마킹 시작"""
    global_benchmarker.start_block_processing(block_number)

def end_benchmarking(opportunities_found: int = 0, 
                    strategies_executed: int = 0,
                    total_revenue: float = 0.0,
                    gas_cost: float = 0.0) -> PerformanceMetrics:
    """블록 처리 벤치마킹 완료"""
    return global_benchmarker.end_block_processing(
        opportunities_found, strategies_executed, total_revenue, gas_cost
    )

def time_component(component_name: str):
    """컴포넌트 실행 시간 측정 데코레이터"""
    return global_benchmarker.time_component(component_name)

def get_performance_report(last_n_blocks: Optional[int] = None) -> Dict[str, Any]:
    """성능 보고서 조회"""
    return global_benchmarker.get_performance_report(last_n_blocks)

def start_monitoring(check_interval: int = 60) -> None:
    """실시간 성능 모니터링 시작"""
    global_benchmarker.real_time_monitor(check_interval)

if __name__ == "__main__":
    # 테스트 코드
    import random
    
    print("🧪 Performance Benchmarker 테스트 시작")
    
    # 모니터링 시작
    start_monitoring(check_interval=10)
    
    # 가상의 블록 처리 시뮬레이션
    for block_num in range(10000, 10010):
        start_benchmarking(block_num)
        
        # 각 컴포넌트 시뮬레이션
        with time_component("graph_building"):
            time.sleep(random.uniform(0.5, 2.0))
            
        with time_component("negative_cycle_detection"):
            time.sleep(random.uniform(1.0, 3.0))
            
        with time_component("local_search"):
            time.sleep(random.uniform(0.5, 2.5))
            
        with time_component("parameter_optimization"):
            time.sleep(random.uniform(0.3, 1.5))
            
        with time_component("validation"):
            time.sleep(random.uniform(0.1, 0.5))
        
        # 블록 처리 완료
        metrics = end_benchmarking(
            opportunities_found=random.randint(0, 5),
            strategies_executed=random.randint(0, 3),
            total_revenue=random.uniform(0, 10),
            gas_cost=random.uniform(0.01, 0.1)
        )
        
        print(f"블록 {block_num}: {metrics.total_execution_time:.3f}초")
    
    # 성능 보고서 출력
    report = get_performance_report()
    print("\n📊 성능 보고서:")
    print(f"성공률: {report['summary']['success_rate']:.1%}")
    print(f"평균 실행시간: {report['summary']['average_time']:.3f}초")
    print(f"최고 기록: {report['summary']['fastest_time']:.3f}초")
    
    print("\n💡 권장사항:")
    for rec in report['recommendations']:
        print(f"- {rec}")