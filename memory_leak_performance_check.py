#!/usr/bin/env python3
"""
Memory Leak and Performance Degradation Check
메모리 누수 및 성능 저하 점검 시스템

DeFiPoser-ARB 구현에서 메모리 누수와 성능 저하를 감지하고 분석
"""

import asyncio
import gc
import logging
import json
import time
import tracemalloc
import psutil
import resource
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sqlite3
import threading
import weakref
from collections import deque
import sys
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass 
class MemorySnapshot:
    timestamp: datetime
    memory_usage_mb: float
    peak_memory_mb: float
    object_count: int
    garbage_count: int
    active_threads: int
    open_files: int
    cpu_percent: float
    top_memory_objects: List[Tuple[str, int]]

@dataclass
class PerformanceMetric:
    timestamp: datetime  
    operation_name: str
    execution_time: float
    memory_before: float
    memory_after: float
    memory_delta: float
    cpu_usage: float
    success: bool
    details: Dict

class MemoryLeakDetector:
    """메모리 누수 감지기"""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.snapshots: deque = deque(maxlen=100)  # 최근 100개 스냅샷 보관
        self.performance_metrics: deque = deque(maxlen=1000)
        self.is_monitoring = False
        self.monitor_thread = None
        self.weak_references = set()
        
        # 트레이스 메모리 시작
        tracemalloc.start()
        
        # 초기 프로세스 정보
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def start_monitoring(self):
        """모니터링 시작"""
        if self.is_monitoring:
            logger.warning("이미 모니터링 중입니다.")
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("메모리 누수 모니터링을 시작했습니다.")
        
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("메모리 누수 모니터링을 중지했습니다.")
        
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                snapshot = self._take_memory_snapshot()
                self.snapshots.append(snapshot)
                
                # 메모리 누수 감지
                if len(self.snapshots) >= 3:
                    self._check_memory_leak()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"모니터링 루프 오류: {e}")
                time.sleep(self.check_interval)
                
    def _take_memory_snapshot(self) -> MemorySnapshot:
        """메모리 스냅샷 생성"""
        try:
            # 메모리 정보
            memory_info = self.process.memory_info()
            current_memory = memory_info.rss / 1024 / 1024  # MB
            
            # Peak memory from tracemalloc
            current_trace, peak_trace = tracemalloc.get_traced_memory()
            peak_memory = peak_trace / 1024 / 1024  # MB
            
            # 객체 수 및 가비지 수집
            gc.collect()  # 강제 가비지 수집
            object_count = len(gc.get_objects())
            garbage_count = len(gc.garbage)
            
            # 스레드 및 파일 핸들
            active_threads = threading.active_count()
            try:
                open_files = len(self.process.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = -1
                
            # CPU 사용률
            cpu_percent = self.process.cpu_percent()
            
            # 메모리를 많이 사용하는 객체 탑 10
            top_objects = self._get_top_memory_objects()
            
            return MemorySnapshot(
                timestamp=datetime.now(),
                memory_usage_mb=current_memory,
                peak_memory_mb=peak_memory,
                object_count=object_count,
                garbage_count=garbage_count,
                active_threads=active_threads,
                open_files=open_files,
                cpu_percent=cpu_percent,
                top_memory_objects=top_objects
            )
            
        except Exception as e:
            logger.error(f"스냅샷 생성 오류: {e}")
            # 기본 스냅샷 반환
            return MemorySnapshot(
                timestamp=datetime.now(),
                memory_usage_mb=0,
                peak_memory_mb=0,
                object_count=0,
                garbage_count=0,
                active_threads=0,
                open_files=0,
                cpu_percent=0,
                top_memory_objects=[]
            )
    
    def _get_top_memory_objects(self) -> List[Tuple[str, int]]:
        """메모리 사용량이 많은 객체들 반환"""
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            result = []
            for stat in top_stats[:10]:
                result.append((str(stat.traceback), stat.size))
                
            return result
            
        except Exception as e:
            logger.error(f"톱 메모리 객체 조회 오류: {e}")
            return []
    
    def _check_memory_leak(self):
        """메모리 누수 체크"""
        if len(self.snapshots) < 3:
            return
            
        # 최근 3개 스냅샷 분석
        recent_snapshots = list(self.snapshots)[-3:]
        
        # 메모리 사용량 증가 추세 분석
        memory_values = [s.memory_usage_mb for s in recent_snapshots]
        
        # 연속적인 증가인지 확인
        is_increasing = all(memory_values[i] <= memory_values[i+1] for i in range(len(memory_values)-1))
        
        if is_increasing:
            increase_rate = (memory_values[-1] - memory_values[0]) / len(memory_values)
            
            # 증가율이 임계값을 초과하면 경고
            if increase_rate > 5:  # 5MB/snapshot 이상 증가
                logger.warning(f"⚠️  잠재적 메모리 누수 감지: {increase_rate:.2f} MB/snapshot 증가")
                logger.warning(f"메모리 사용량: {memory_values[0]:.2f} -> {memory_values[-1]:.2f} MB")
                
                # 상세 분석
                self._analyze_memory_leak(recent_snapshots)
    
    def _analyze_memory_leak(self, snapshots: List[MemorySnapshot]):
        """메모리 누수 상세 분석"""
        logger.info("🔍 메모리 누수 상세 분석 중...")
        
        first_snapshot = snapshots[0]
        last_snapshot = snapshots[-1]
        
        # 객체 수 변화
        object_increase = last_snapshot.object_count - first_snapshot.object_count
        logger.info(f"객체 수 변화: {object_increase}")
        
        # 가비지 수집되지 않은 객체
        if last_snapshot.garbage_count > 0:
            logger.warning(f"가비지 수집되지 않은 객체: {last_snapshot.garbage_count}")
            
        # 스레드 및 파일 핸들 변화
        thread_increase = last_snapshot.active_threads - first_snapshot.active_threads
        if thread_increase > 0:
            logger.warning(f"활성 스레드 증가: +{thread_increase}")
            
        if last_snapshot.open_files > 0 and first_snapshot.open_files > 0:
            file_increase = last_snapshot.open_files - first_snapshot.open_files
            if file_increase > 0:
                logger.warning(f"열린 파일 핸들 증가: +{file_increase}")

    def measure_performance(self, operation_name: str):
        """성능 측정 데코레이터 팩토리"""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                return await self._measure_async_performance(operation_name, func, *args, **kwargs)
                
            def sync_wrapper(*args, **kwargs):
                return self._measure_sync_performance(operation_name, func, *args, **kwargs)
                
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
                
        return decorator
    
    async def _measure_async_performance(self, operation_name: str, func, *args, **kwargs):
        """비동기 함수 성능 측정"""
        # 시작 전 상태
        start_time = time.time()
        memory_before = self.process.memory_info().rss / 1024 / 1024
        cpu_before = self.process.cpu_percent()
        
        success = False
        result = None
        error = None
        
        try:
            result = await func(*args, **kwargs)
            success = True
        except Exception as e:
            error = str(e)
            logger.error(f"성능 측정 중 오류 ({operation_name}): {e}")
        
        # 종료 후 상태
        end_time = time.time()
        memory_after = self.process.memory_info().rss / 1024 / 1024
        execution_time = end_time - start_time
        memory_delta = memory_after - memory_before
        
        # 메트릭 기록
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            operation_name=operation_name,
            execution_time=execution_time,
            memory_before=memory_before,
            memory_after=memory_after,
            memory_delta=memory_delta,
            cpu_usage=self.process.cpu_percent() - cpu_before,
            success=success,
            details={
                'args_count': len(args),
                'kwargs_count': len(kwargs),
                'error': error
            }
        )
        
        self.performance_metrics.append(metric)
        
        # 성능 저하 체크
        if execution_time > 10:  # 10초 이상
            logger.warning(f"⚠️  성능 저하 감지 ({operation_name}): {execution_time:.2f}초")
            
        if memory_delta > 50:  # 50MB 이상 증가
            logger.warning(f"⚠️  메모리 사용량 급증 ({operation_name}): +{memory_delta:.2f}MB")
        
        if error:
            raise Exception(error)
            
        return result
    
    def _measure_sync_performance(self, operation_name: str, func, *args, **kwargs):
        """동기 함수 성능 측정"""
        # 시작 전 상태
        start_time = time.time()
        memory_before = self.process.memory_info().rss / 1024 / 1024
        cpu_before = self.process.cpu_percent()
        
        success = False
        result = None
        error = None
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            error = str(e)
            logger.error(f"성능 측정 중 오류 ({operation_name}): {e}")
        
        # 종료 후 상태
        end_time = time.time()
        memory_after = self.process.memory_info().rss / 1024 / 1024
        execution_time = end_time - start_time
        memory_delta = memory_after - memory_before
        
        # 메트릭 기록
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            operation_name=operation_name,
            execution_time=execution_time,
            memory_before=memory_before,
            memory_after=memory_after,
            memory_delta=memory_delta,
            cpu_usage=self.process.cpu_percent() - cpu_before,
            success=success,
            details={
                'args_count': len(args),
                'kwargs_count': len(kwargs),
                'error': error
            }
        )
        
        self.performance_metrics.append(metric)
        
        # 성능 저하 체크
        if execution_time > 10:  # 10초 이상
            logger.warning(f"⚠️  성능 저하 감지 ({operation_name}): {execution_time:.2f}초")
            
        if memory_delta > 50:  # 50MB 이상 증가
            logger.warning(f"⚠️  메모리 사용량 급증 ({operation_name}): +{memory_delta:.2f}MB")
        
        if error:
            raise Exception(error)
            
        return result
    
    def generate_report(self) -> Dict:
        """메모리 및 성능 보고서 생성"""
        now = datetime.now()
        
        # 메모리 스냅샷 분석
        memory_analysis = self._analyze_memory_snapshots()
        
        # 성능 메트릭 분석
        performance_analysis = self._analyze_performance_metrics()
        
        # 시스템 정보
        system_info = {
            'python_version': sys.version,
            'process_id': os.getpid(),
            'initial_memory_mb': self.initial_memory,
            'current_memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'memory_growth_mb': (self.process.memory_info().rss / 1024 / 1024) - self.initial_memory,
            'monitoring_duration_minutes': len(self.snapshots) * (self.check_interval / 60),
            'total_snapshots': len(self.snapshots),
            'total_performance_metrics': len(self.performance_metrics)
        }
        
        report = {
            'report_metadata': {
                'generated_at': now.isoformat(),
                'report_type': 'memory_leak_performance_check',
                'monitoring_active': self.is_monitoring
            },
            'system_info': system_info,
            'memory_analysis': memory_analysis,
            'performance_analysis': performance_analysis,
            'recommendations': self._generate_recommendations(memory_analysis, performance_analysis)
        }
        
        return report
    
    def _analyze_memory_snapshots(self) -> Dict:
        """메모리 스냅샷 분석"""
        if not self.snapshots:
            return {'error': 'No memory snapshots available'}
            
        snapshots_list = list(self.snapshots)
        
        # 기본 통계
        memory_values = [s.memory_usage_mb for s in snapshots_list]
        object_counts = [s.object_count for s in snapshots_list]
        thread_counts = [s.active_threads for s in snapshots_list]
        
        analysis = {
            'memory_usage': {
                'min_mb': min(memory_values),
                'max_mb': max(memory_values),
                'avg_mb': sum(memory_values) / len(memory_values),
                'current_mb': memory_values[-1],
                'trend': self._calculate_trend(memory_values)
            },
            'object_count': {
                'min': min(object_counts),
                'max': max(object_counts),
                'avg': sum(object_counts) / len(object_counts),
                'current': object_counts[-1],
                'trend': self._calculate_trend(object_counts)
            },
            'thread_count': {
                'min': min(thread_counts),
                'max': max(thread_counts),
                'avg': sum(thread_counts) / len(thread_counts),
                'current': thread_counts[-1]
            },
            'garbage_collection': {
                'total_garbage': sum(s.garbage_count for s in snapshots_list),
                'max_garbage': max(s.garbage_count for s in snapshots_list),
                'avg_garbage': sum(s.garbage_count for s in snapshots_list) / len(snapshots_list)
            }
        }
        
        # 메모리 누수 위험도 평가
        analysis['leak_risk'] = self._assess_leak_risk(analysis)
        
        return analysis
    
    def _analyze_performance_metrics(self) -> Dict:
        """성능 메트릭 분석"""
        if not self.performance_metrics:
            return {'error': 'No performance metrics available'}
            
        metrics_list = list(self.performance_metrics)
        
        # 연산별 그룹화
        operations = {}
        for metric in metrics_list:
            op_name = metric.operation_name
            if op_name not in operations:
                operations[op_name] = []
            operations[op_name].append(metric)
        
        # 각 연산별 분석
        operation_analysis = {}
        for op_name, op_metrics in operations.items():
            exec_times = [m.execution_time for m in op_metrics]
            memory_deltas = [m.memory_delta for m in op_metrics]
            success_rate = len([m for m in op_metrics if m.success]) / len(op_metrics)
            
            operation_analysis[op_name] = {
                'total_calls': len(op_metrics),
                'success_rate': success_rate,
                'execution_time': {
                    'min_seconds': min(exec_times),
                    'max_seconds': max(exec_times),
                    'avg_seconds': sum(exec_times) / len(exec_times),
                    'trend': self._calculate_trend(exec_times)
                },
                'memory_impact': {
                    'min_delta_mb': min(memory_deltas),
                    'max_delta_mb': max(memory_deltas),
                    'avg_delta_mb': sum(memory_deltas) / len(memory_deltas)
                }
            }
        
        # 전체 성능 요약
        all_exec_times = [m.execution_time for m in metrics_list]
        all_memory_deltas = [m.memory_delta for m in metrics_list]
        overall_success_rate = len([m for m in metrics_list if m.success]) / len(metrics_list)
        
        analysis = {
            'overall_metrics': {
                'total_operations': len(metrics_list),
                'unique_operations': len(operations),
                'overall_success_rate': overall_success_rate,
                'avg_execution_time': sum(all_exec_times) / len(all_exec_times),
                'avg_memory_impact': sum(all_memory_deltas) / len(all_memory_deltas)
            },
            'operation_breakdown': operation_analysis,
            'performance_issues': self._identify_performance_issues(operation_analysis)
        }
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """값들의 추세 계산"""
        if len(values) < 2:
            return 'insufficient_data'
            
        # 선형 회귀로 추세 계산
        n = len(values)
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if (n * sum_x2 - sum_x ** 2) == 0:
            return 'stable'
            
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _assess_leak_risk(self, memory_analysis: Dict) -> Dict:
        """메모리 누수 위험도 평가"""
        risk_score = 0
        risk_factors = []
        
        # 메모리 사용량 추세
        if memory_analysis['memory_usage']['trend'] == 'increasing':
            risk_score += 3
            risk_factors.append('Memory usage is consistently increasing')
        
        # 객체 수 추세
        if memory_analysis['object_count']['trend'] == 'increasing':
            risk_score += 2
            risk_factors.append('Object count is consistently increasing')
        
        # 가비지 수집 효율성
        if memory_analysis['garbage_collection']['avg_garbage'] > 100:
            risk_score += 2
            risk_factors.append('High number of uncollected garbage objects')
        
        # 메모리 성장률
        current_memory = memory_analysis['memory_usage']['current_mb']
        min_memory = memory_analysis['memory_usage']['min_mb']
        growth_ratio = (current_memory - min_memory) / min_memory if min_memory > 0 else 0
        
        if growth_ratio > 0.5:  # 50% 이상 증가
            risk_score += 3
            risk_factors.append(f'Memory usage increased by {growth_ratio:.1%}')
        
        # 위험도 레벨 결정
        if risk_score >= 7:
            risk_level = 'HIGH'
        elif risk_score >= 4:
            risk_level = 'MEDIUM'
        elif risk_score >= 1:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors
        }
    
    def _identify_performance_issues(self, operation_analysis: Dict) -> List[str]:
        """성능 문제 식별"""
        issues = []
        
        for op_name, analysis in operation_analysis.items():
            # 실행 시간 문제
            avg_time = analysis['execution_time']['avg_seconds']
            if avg_time > 6.43:  # 논문의 목표 시간
                issues.append(f'{op_name}: Average execution time ({avg_time:.2f}s) exceeds target (6.43s)')
            
            # 성공률 문제
            success_rate = analysis['success_rate']
            if success_rate < 0.9:  # 90% 미만
                issues.append(f'{op_name}: Low success rate ({success_rate:.1%})')
            
            # 메모리 증가 문제
            avg_memory_delta = analysis['memory_impact']['avg_delta_mb']
            if avg_memory_delta > 10:  # 10MB 이상 증가
                issues.append(f'{op_name}: High memory usage per operation (+{avg_memory_delta:.1f}MB)')
            
            # 실행 시간 증가 추세
            if analysis['execution_time']['trend'] == 'increasing':
                issues.append(f'{op_name}: Execution time is increasing over time')
        
        return issues
    
    def _generate_recommendations(self, memory_analysis: Dict, performance_analysis: Dict) -> List[str]:
        """개선 권고사항 생성"""
        recommendations = []
        
        # 메모리 관련 권고
        if 'memory_analysis' in memory_analysis:
            risk_level = memory_analysis.get('leak_risk', {}).get('risk_level', 'UNKNOWN')
            
            if risk_level in ['HIGH', 'MEDIUM']:
                recommendations.append('Consider implementing more aggressive garbage collection')
                recommendations.append('Review object lifecycle management and ensure proper cleanup')
                recommendations.append('Use weak references for cached data that can be recreated')
        
        # 성능 관련 권고
        if 'performance_issues' in performance_analysis:
            issues = performance_analysis['performance_issues']
            
            if any('execution time' in issue.lower() for issue in issues):
                recommendations.append('Optimize slow operations with caching or parallel processing')
                recommendations.append('Consider breaking down complex operations into smaller chunks')
            
            if any('memory usage' in issue.lower() for issue in issues):
                recommendations.append('Implement memory pooling for frequently allocated objects')
                recommendations.append('Use streaming processing for large datasets')
            
            if any('success rate' in issue.lower() for issue in issues):
                recommendations.append('Improve error handling and retry mechanisms')
                recommendations.append('Add input validation to prevent operation failures')
        
        # 일반적인 권고사항
        if not recommendations:
            recommendations.append('System is performing well, continue monitoring')
        else:
            recommendations.append('Consider running this check more frequently during high-load periods')
        
        return recommendations

class DeFiSystemMemoryCheck:
    """DeFi 시스템 전용 메모리 점검"""
    
    def __init__(self):
        self.detector = MemoryLeakDetector(check_interval=30)  # 30초마다 체크
        
    async def run_comprehensive_check(self) -> str:
        """종합적인 메모리 및 성능 점검"""
        logger.info("🔍 DeFi 시스템 메모리 누수 및 성능 점검 시작")
        
        # 모니터링 시작
        self.detector.start_monitoring()
        
        try:
            # 주요 DeFi 시스템 컴포넌트들을 시뮬레이션하여 테스트
            await self._simulate_defi_operations()
            
            # 잠시 대기하여 메모리 패턴 관찰
            logger.info("⏳ 메모리 패턴 관찰을 위해 대기 중...")
            await asyncio.sleep(180)  # 3분 대기
            
            # 보고서 생성
            report = self.detector.generate_report()
            
            # 보고서 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"memory_leak_check_report_{timestamp}.json"
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # 요약 출력
            self._print_summary(report)
            
            return report_filename
            
        finally:
            # 모니터링 중지
            self.detector.stop_monitoring()
    
    async def _simulate_defi_operations(self):
        """DeFi 작업들을 시뮬레이션하여 메모리 사용 패턴 확인"""
        
        # 1. 데이터 수집 시뮬레이션
        await self._simulate_data_collection()
        
        # 2. 아비트래지 탐지 시뮬레이션
        await self._simulate_arbitrage_detection()
        
        # 3. 그래프 빌딩 시뮬레이션
        await self._simulate_graph_building()
        
        # 4. 메모리 집약적 작업 시뮬레이션
        await self._simulate_memory_intensive_operations()
    
    @property 
    def _measured_data_collection(self):
        return self.detector.measure_performance("data_collection")
    
    @property
    def _measured_arbitrage_detection(self):
        return self.detector.measure_performance("arbitrage_detection")
    
    @property
    def _measured_graph_building(self):
        return self.detector.measure_performance("graph_building")
    
    @property
    def _measured_memory_intensive_op(self):
        return self.detector.measure_performance("memory_intensive_operation")
    
    async def _simulate_data_collection(self):
        """데이터 수집 시뮬레이션"""
        @self._measured_data_collection
        async def collect_data():
            # 대량의 데이터 수집을 시뮬레이션
            data = []
            for i in range(10000):
                data.append({
                    'block_number': i,
                    'price_data': [random.uniform(1, 1000) for _ in range(25)],  # 25개 자산
                    'pool_data': {f'pool_{j}': {'reserve0': random.uniform(1000, 100000), 
                                              'reserve1': random.uniform(1000, 100000)} 
                                for j in range(96)}  # 96개 프로토콜
                })
            
            # 메모리 사용을 시뮬레이션하기 위해 일부 데이터 보관
            return data[:1000]  # 일부만 반환하여 메모리 해제 확인
        
        import random
        result = await collect_data()
        logger.info(f"데이터 수집 완료: {len(result)}개 항목")
    
    async def _simulate_arbitrage_detection(self):
        """아비트래지 탐지 시뮬레이션"""
        @self._measured_arbitrage_detection
        async def detect_arbitrage():
            opportunities = []
            
            # Bellman-Ford 알고리즘 시뮬레이션
            for cycle in range(100):  # 100개의 negative cycle 체크
                path = []
                for step in range(5):  # 평균 5단계 경로
                    path.append({
                        'dex': f'dex_{step}',
                        'asset_in': f'asset_{step}',
                        'asset_out': f'asset_{(step+1)%25}',
                        'exchange_rate': random.uniform(0.95, 1.05)
                    })
                
                # 수익성 계산
                profit = random.uniform(-0.1, 0.5)
                if profit > 0:
                    opportunities.append({
                        'path': path,
                        'profit': profit,
                        'confidence': random.uniform(0.7, 0.95)
                    })
            
            return opportunities
        
        import random
        result = await detect_arbitrage()
        logger.info(f"아비트래지 탐지 완료: {len(result)}개 기회 발견")
    
    async def _simulate_graph_building(self):
        """그래프 빌딩 시뮬레이션"""
        @self._measured_graph_building
        async def build_graph():
            # 25개 노드, 2400개 엣지 (96 protocols * 25 assets) 그래프 구축
            nodes = [f'asset_{i}' for i in range(25)]
            edges = {}
            
            for i in range(25):
                for j in range(25):
                    if i != j:
                        edge_key = f"{nodes[i]}_{nodes[j]}"
                        edges[edge_key] = {
                            'weight': random.uniform(-0.1, 0.1),  # log price
                            'protocols': [f'protocol_{k}' for k in range(random.randint(1, 5))]
                        }
            
            # 메모리 사용량이 많은 그래프 연산 시뮬레이션
            adjacency_matrix = [[0.0 for _ in range(25)] for _ in range(25)]
            for i in range(25):
                for j in range(25):
                    adjacency_matrix[i][j] = random.uniform(-0.5, 0.5)
            
            return {'nodes': nodes, 'edges': edges, 'matrix': adjacency_matrix}
        
        import random
        result = await build_graph()
        logger.info(f"그래프 빌딩 완료: {len(result['nodes'])}개 노드, {len(result['edges'])}개 엣지")
    
    async def _simulate_memory_intensive_operations(self):
        """메모리 집약적 작업 시뮬레이션"""
        @self._measured_memory_intensive_op
        async def intensive_operation():
            # 큰 데이터 구조 생성 및 조작
            large_data = []
            
            # 96개 프로토콜 * 25개 자산 * 150일 = 360,000 데이터 포인트
            for protocol in range(96):
                for asset in range(25):
                    for day in range(150):
                        large_data.append({
                            'protocol': f'protocol_{protocol}',
                            'asset': f'asset_{asset}',
                            'day': day,
                            'transactions': [
                                {
                                    'timestamp': datetime.now() - timedelta(days=day, hours=hour),
                                    'amount': random.uniform(0.1, 100),
                                    'price': random.uniform(1, 1000)
                                }
                                for hour in range(24)  # 시간당 데이터
                            ]
                        })
            
            # 데이터 처리 (메모리 사용량 증가)
            processed_data = {}
            for item in large_data:
                key = f"{item['protocol']}_{item['asset']}"
                if key not in processed_data:
                    processed_data[key] = []
                processed_data[key].append(item)
            
            # 일부 데이터만 반환하여 메모리 해제 테스트
            return {k: v for k, v in list(processed_data.items())[:100]}
        
        import random
        result = await intensive_operation()
        logger.info(f"집약적 연산 완료: {len(result)}개 처리된 키")
    
    def _print_summary(self, report: Dict):
        """보고서 요약 출력"""
        print("\n" + "="*70)
        print("🔍 DeFi 시스템 메모리 누수 및 성능 점검 결과")
        print("="*70)
        
        # 시스템 정보
        system_info = report.get('system_info', {})
        print(f"\n📊 시스템 정보:")
        print(f"  초기 메모리: {system_info.get('initial_memory_mb', 0):.2f} MB")
        print(f"  현재 메모리: {system_info.get('current_memory_mb', 0):.2f} MB")
        print(f"  메모리 증가: {system_info.get('memory_growth_mb', 0):.2f} MB")
        print(f"  모니터링 시간: {system_info.get('monitoring_duration_minutes', 0):.1f} 분")
        
        # 메모리 분석
        memory_analysis = report.get('memory_analysis', {})
        if 'leak_risk' in memory_analysis:
            leak_risk = memory_analysis['leak_risk']
            print(f"\n⚠️  메모리 누수 위험도: {leak_risk.get('risk_level', 'UNKNOWN')}")
            print(f"  위험 점수: {leak_risk.get('risk_score', 0)}/10")
            
            risk_factors = leak_risk.get('risk_factors', [])
            if risk_factors:
                print(f"  위험 요소:")
                for factor in risk_factors:
                    print(f"    • {factor}")
        
        # 성능 분석
        performance_analysis = report.get('performance_analysis', {})
        if 'overall_metrics' in performance_analysis:
            overall = performance_analysis['overall_metrics']
            print(f"\n📈 성능 지표:")
            print(f"  전체 연산: {overall.get('total_operations', 0)}개")
            print(f"  성공률: {overall.get('overall_success_rate', 0):.1%}")
            print(f"  평균 실행시간: {overall.get('avg_execution_time', 0):.2f}초")
            print(f"  평균 메모리 영향: {overall.get('avg_memory_impact', 0):.2f}MB")
        
        # 성능 문제
        performance_issues = performance_analysis.get('performance_issues', [])
        if performance_issues:
            print(f"\n❌ 성능 문제:")
            for issue in performance_issues:
                print(f"    • {issue}")
        
        # 권고사항
        recommendations = report.get('recommendations', [])
        if recommendations:
            print(f"\n💡 권고사항:")
            for rec in recommendations:
                print(f"    • {rec}")
        
        print("\n" + "="*70)

async def main():
    """메인 함수"""
    print("🔍 DeFi 시스템 메모리 누수 및 성능 저하 점검")
    print("=" * 70)
    
    checker = DeFiSystemMemoryCheck()
    
    try:
        report_file = await checker.run_comprehensive_check()
        print(f"\n✅ 점검 완료! 상세 보고서: {report_file}")
        
    except Exception as e:
        logger.error(f"점검 중 오류 발생: {e}")
        print(f"❌ 점검 실패: {e}")

if __name__ == "__main__":
    asyncio.run(main())