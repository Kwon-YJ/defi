#!/usr/bin/env python3
"""
Simple Memory Leak and Performance Check
psutil 없이도 작동하는 간단한 메모리 누수 및 성능 점검

내장 모듈만 사용하여 DeFi 시스템의 메모리 및 성능 상태를 점검
"""

import asyncio
import gc
import json
import logging
import os
import resource
import sys
import time
import tracemalloc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading
from collections import deque
import random

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimpleMemorySnapshot:
    timestamp: datetime
    memory_usage_mb: float
    peak_memory_mb: float
    object_count: int
    garbage_count: int
    thread_count: int

@dataclass
class SimplePerformanceMetric:
    timestamp: datetime
    operation_name: str
    execution_time: float
    memory_before: float
    memory_after: float
    memory_delta: float
    success: bool

class SimpleMemoryChecker:
    """간단한 메모리 점검기 (psutil 불필요)"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.snapshots = deque(maxlen=50)
        self.performance_metrics = deque(maxlen=500)
        self.is_monitoring = False
        self.monitor_thread = None
        
        # 트레이스 메모리 시작
        tracemalloc.start()
        
        # 초기 메모리 정보
        self.initial_memory = self._get_memory_usage()
        
    def _get_memory_usage(self) -> float:
        """현재 메모리 사용량 (MB)"""
        try:
            # resource 모듈 사용 (Unix 계열)
            if hasattr(resource, 'RUSAGE_SELF'):
                usage = resource.getrusage(resource.RUSAGE_SELF)
                # Linux에서는 KB 단위, macOS에서는 bytes 단위
                if sys.platform == 'darwin':
                    return usage.ru_maxrss / 1024 / 1024  # bytes to MB
                else:
                    return usage.ru_maxrss / 1024  # KB to MB
        except:
            pass
            
        # tracemalloc 사용 (fallback)
        try:
            current, peak = tracemalloc.get_traced_memory()
            return current / 1024 / 1024  # bytes to MB
        except:
            return 0.0
    
    def start_monitoring(self):
        """모니터링 시작"""
        if self.is_monitoring:
            logger.warning("이미 모니터링 중입니다.")
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("메모리 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=3)
        logger.info("메모리 모니터링 중지")
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                
                # 메모리 누수 체크
                if len(self.snapshots) >= 3:
                    self._check_memory_trend()
                
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"모니터링 오류: {e}")
                time.sleep(self.check_interval)
    
    def _take_snapshot(self) -> SimpleMemorySnapshot:
        """메모리 스냅샷 생성"""
        try:
            # 메모리 사용량
            memory_usage = self._get_memory_usage()
            
            # tracemalloc에서 peak memory
            try:
                current_trace, peak_trace = tracemalloc.get_traced_memory()
                peak_memory = peak_trace / 1024 / 1024  # MB
            except:
                peak_memory = memory_usage
            
            # 가비지 수집
            gc.collect()
            object_count = len(gc.get_objects())
            garbage_count = len(gc.garbage)
            
            # 스레드 수
            thread_count = threading.active_count()
            
            return SimpleMemorySnapshot(
                timestamp=datetime.now(),
                memory_usage_mb=memory_usage,
                peak_memory_mb=peak_memory,
                object_count=object_count,
                garbage_count=garbage_count,
                thread_count=thread_count
            )
            
        except Exception as e:
            logger.error(f"스냅샷 생성 오류: {e}")
            return SimpleMemorySnapshot(
                timestamp=datetime.now(),
                memory_usage_mb=0,
                peak_memory_mb=0,
                object_count=0,
                garbage_count=0,
                thread_count=0
            )
    
    def _check_memory_trend(self):
        """메모리 증가 추세 체크"""
        if len(self.snapshots) < 3:
            return
            
        recent_snapshots = list(self.snapshots)[-3:]
        memory_values = [s.memory_usage_mb for s in recent_snapshots]
        
        # 연속적인 증가인지 확인
        is_increasing = all(memory_values[i] <= memory_values[i+1] for i in range(len(memory_values)-1))
        
        if is_increasing and len(memory_values) > 1:
            increase = memory_values[-1] - memory_values[0]
            if increase > 5:  # 5MB 이상 증가
                logger.warning(f"⚠️  메모리 증가 감지: +{increase:.2f}MB")
                logger.warning(f"메모리: {memory_values[0]:.1f} -> {memory_values[-1]:.1f} MB")
    
    def measure_performance(self, operation_name: str):
        """성능 측정 데코레이터"""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    return await self._measure_async_performance(operation_name, func, *args, **kwargs)
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    return self._measure_sync_performance(operation_name, func, *args, **kwargs)
                return sync_wrapper
        return decorator
    
    async def _measure_async_performance(self, operation_name: str, func, *args, **kwargs):
        """비동기 함수 성능 측정"""
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        success = False
        result = None
        
        try:
            result = await func(*args, **kwargs)
            success = True
        except Exception as e:
            logger.error(f"성능 측정 중 오류 ({operation_name}): {e}")
            raise
        
        end_time = time.time()
        memory_after = self._get_memory_usage()
        execution_time = end_time - start_time
        memory_delta = memory_after - memory_before
        
        # 메트릭 기록
        metric = SimplePerformanceMetric(
            timestamp=datetime.now(),
            operation_name=operation_name,
            execution_time=execution_time,
            memory_before=memory_before,
            memory_after=memory_after,
            memory_delta=memory_delta,
            success=success
        )
        
        self.performance_metrics.append(metric)
        
        # 경고 체크
        if execution_time > 6.43:  # 논문의 목표 시간
            logger.warning(f"⚠️  성능 저하 ({operation_name}): {execution_time:.2f}초")
        
        if memory_delta > 20:  # 20MB 이상 증가
            logger.warning(f"⚠️  메모리 사용량 급증 ({operation_name}): +{memory_delta:.1f}MB")
        
        return result
    
    def _measure_sync_performance(self, operation_name: str, func, *args, **kwargs):
        """동기 함수 성능 측정"""
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        success = False
        result = None
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            logger.error(f"성능 측정 중 오류 ({operation_name}): {e}")
            raise
        
        end_time = time.time()
        memory_after = self._get_memory_usage()
        execution_time = end_time - start_time
        memory_delta = memory_after - memory_before
        
        # 메트릭 기록
        metric = SimplePerformanceMetric(
            timestamp=datetime.now(),
            operation_name=operation_name,
            execution_time=execution_time,
            memory_before=memory_before,
            memory_after=memory_after,
            memory_delta=memory_delta,
            success=success
        )
        
        self.performance_metrics.append(metric)
        
        # 경고 체크
        if execution_time > 6.43:  # 논문의 목표 시간
            logger.warning(f"⚠️  성능 저하 ({operation_name}): {execution_time:.2f}초")
        
        if memory_delta > 20:  # 20MB 이상 증가
            logger.warning(f"⚠️  메모리 사용량 급증 ({operation_name}): +{memory_delta:.1f}MB")
        
        return result
    
    def generate_report(self) -> Dict:
        """간단한 보고서 생성"""
        now = datetime.now()
        
        # 메모리 분석
        memory_analysis = self._analyze_memory()
        
        # 성능 분석
        performance_analysis = self._analyze_performance()
        
        # 시스템 정보
        system_info = {
            'python_version': sys.version.split()[0],
            'platform': sys.platform,
            'process_id': os.getpid(),
            'initial_memory_mb': self.initial_memory,
            'current_memory_mb': self._get_memory_usage(),
            'memory_growth_mb': self._get_memory_usage() - self.initial_memory,
            'total_snapshots': len(self.snapshots),
            'total_metrics': len(self.performance_metrics)
        }
        
        return {
            'report_metadata': {
                'generated_at': now.isoformat(),
                'report_type': 'simple_memory_performance_check',
                'monitoring_active': self.is_monitoring
            },
            'system_info': system_info,
            'memory_analysis': memory_analysis,
            'performance_analysis': performance_analysis,
            'recommendations': self._generate_recommendations(memory_analysis, performance_analysis)
        }
    
    def _analyze_memory(self) -> Dict:
        """메모리 분석"""
        if not self.snapshots:
            return {'error': 'No memory snapshots available'}
        
        snapshots_list = list(self.snapshots)
        memory_values = [s.memory_usage_mb for s in snapshots_list]
        object_counts = [s.object_count for s in snapshots_list]
        
        # 기본 통계
        analysis = {
            'memory_usage': {
                'min_mb': min(memory_values) if memory_values else 0,
                'max_mb': max(memory_values) if memory_values else 0,
                'avg_mb': sum(memory_values) / len(memory_values) if memory_values else 0,
                'current_mb': memory_values[-1] if memory_values else 0,
                'trend': self._calculate_trend(memory_values)
            },
            'object_count': {
                'min': min(object_counts) if object_counts else 0,
                'max': max(object_counts) if object_counts else 0,
                'avg': sum(object_counts) / len(object_counts) if object_counts else 0,
                'current': object_counts[-1] if object_counts else 0,
                'trend': self._calculate_trend(object_counts)
            }
        }
        
        # 위험도 평가
        analysis['risk_assessment'] = self._assess_risk(analysis)
        
        return analysis
    
    def _analyze_performance(self) -> Dict:
        """성능 분석"""
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
        
        # 연산별 분석
        operation_analysis = {}
        for op_name, op_metrics in operations.items():
            exec_times = [m.execution_time for m in op_metrics]
            memory_deltas = [m.memory_delta for m in op_metrics]
            success_count = len([m for m in op_metrics if m.success])
            
            operation_analysis[op_name] = {
                'total_calls': len(op_metrics),
                'success_rate': success_count / len(op_metrics) if op_metrics else 0,
                'avg_execution_time': sum(exec_times) / len(exec_times) if exec_times else 0,
                'max_execution_time': max(exec_times) if exec_times else 0,
                'avg_memory_impact': sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0,
                'max_memory_impact': max(memory_deltas) if memory_deltas else 0
            }
        
        # 전체 요약
        all_exec_times = [m.execution_time for m in metrics_list]
        all_memory_deltas = [m.memory_delta for m in metrics_list]
        total_success = len([m for m in metrics_list if m.success])
        
        analysis = {
            'overall': {
                'total_operations': len(metrics_list),
                'success_rate': total_success / len(metrics_list) if metrics_list else 0,
                'avg_execution_time': sum(all_exec_times) / len(all_exec_times) if all_exec_times else 0,
                'avg_memory_impact': sum(all_memory_deltas) / len(all_memory_deltas) if all_memory_deltas else 0
            },
            'by_operation': operation_analysis,
            'performance_issues': self._identify_issues(operation_analysis)
        }
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """추세 계산"""
        if len(values) < 2:
            return 'insufficient_data'
        
        # 간단한 추세 계산
        increases = 0
        decreases = 0
        
        for i in range(1, len(values)):
            if values[i] > values[i-1]:
                increases += 1
            elif values[i] < values[i-1]:
                decreases += 1
        
        if increases > decreases * 1.5:
            return 'increasing'
        elif decreases > increases * 1.5:
            return 'decreasing'
        else:
            return 'stable'
    
    def _assess_risk(self, memory_analysis: Dict) -> Dict:
        """위험도 평가"""
        risk_score = 0
        risk_factors = []
        
        memory_usage = memory_analysis.get('memory_usage', {})
        
        # 메모리 증가 추세
        if memory_usage.get('trend') == 'increasing':
            risk_score += 3
            risk_factors.append('Memory usage is increasing')
        
        # 현재 메모리 사용량
        current_memory = memory_usage.get('current_mb', 0)
        if current_memory > 500:  # 500MB 이상
            risk_score += 2
            risk_factors.append(f'High memory usage: {current_memory:.1f}MB')
        
        # 메모리 성장률
        max_memory = memory_usage.get('max_mb', 0)
        min_memory = memory_usage.get('min_mb', 0)
        if min_memory > 0:
            growth_ratio = (max_memory - min_memory) / min_memory
            if growth_ratio > 0.3:  # 30% 이상 증가
                risk_score += 2
                risk_factors.append(f'Memory grew by {growth_ratio:.1%}')
        
        # 위험도 레벨
        if risk_score >= 6:
            risk_level = 'HIGH'
        elif risk_score >= 3:
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
    
    def _identify_issues(self, operation_analysis: Dict) -> List[str]:
        """성능 문제 식별"""
        issues = []
        
        for op_name, analysis in operation_analysis.items():
            # 실행 시간 문제
            avg_time = analysis.get('avg_execution_time', 0)
            if avg_time > 6.43:  # 논문 목표
                issues.append(f'{op_name}: Slow execution ({avg_time:.2f}s > 6.43s target)')
            
            # 성공률 문제
            success_rate = analysis.get('success_rate', 0)
            if success_rate < 0.9:
                issues.append(f'{op_name}: Low success rate ({success_rate:.1%})')
            
            # 메모리 문제
            avg_memory = analysis.get('avg_memory_impact', 0)
            if avg_memory > 10:  # 10MB 이상
                issues.append(f'{op_name}: High memory impact (+{avg_memory:.1f}MB)')
        
        return issues
    
    def _generate_recommendations(self, memory_analysis: Dict, performance_analysis: Dict) -> List[str]:
        """권고사항 생성"""
        recommendations = []
        
        # 메모리 관련
        risk_level = memory_analysis.get('risk_assessment', {}).get('risk_level', 'UNKNOWN')
        if risk_level in ['HIGH', 'MEDIUM']:
            recommendations.append('Consider implementing regular garbage collection')
            recommendations.append('Review memory-intensive operations')
            
        # 성능 관련
        issues = performance_analysis.get('performance_issues', [])
        if issues:
            recommendations.append('Optimize slow operations identified in performance issues')
            recommendations.append('Consider caching for repeated operations')
        
        # 일반적인 권고
        if not recommendations:
            recommendations.append('System performance appears normal')
        else:
            recommendations.append('Monitor system regularly during high-load periods')
        
        return recommendations

class DeFiSimpleChecker:
    """DeFi 시스템 간단 점검기"""
    
    def __init__(self):
        self.checker = SimpleMemoryChecker(check_interval=20)  # 20초마다 체크
    
    async def run_check(self) -> str:
        """점검 실행"""
        logger.info("🔍 DeFi 시스템 메모리 및 성능 점검 시작")
        
        # 모니터링 시작
        self.checker.start_monitoring()
        
        try:
            # DeFi 작업 시뮬레이션
            await self._simulate_defi_work()
            
            # 대기 시간 (패턴 관찰)
            logger.info("⏳ 메모리 패턴 관찰을 위해 90초 대기...")
            await asyncio.sleep(90)
            
            # 보고서 생성
            report = self.checker.generate_report()
            
            # 보고서 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"simple_memory_check_{timestamp}.json"
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # 결과 출력
            self._print_results(report)
            
            return report_filename
            
        finally:
            self.checker.stop_monitoring()
    
    async def _simulate_defi_work(self):
        """DeFi 작업 시뮬레이션"""
        
        # 1. 데이터 수집
        await self._simulate_data_collection()
        
        # 2. 그래프 빌딩
        await self._simulate_graph_building()
        
        # 3. 아비트래지 탐지
        await self._simulate_arbitrage_detection()
        
        # 4. 메모리 집약적 작업
        await self._simulate_intensive_work()
    
    async def _simulate_data_collection(self):
        """데이터 수집 시뮬레이션"""
        @self.checker.measure_performance("data_collection")
        async def collect_data():
            logger.info("📊 데이터 수집 시뮬레이션...")
            data = []
            
            # 96개 프로토콜 * 25개 자산 데이터
            for protocol in range(96):
                for asset in range(25):
                    data.append({
                        'protocol': f'protocol_{protocol}',
                        'asset': f'asset_{asset}',
                        'price': random.uniform(1, 1000),
                        'liquidity': random.uniform(10000, 1000000),
                        'volume': random.uniform(1000, 100000)
                    })
            
            # 데이터 처리
            processed = {}
            for item in data:
                key = item['protocol']
                if key not in processed:
                    processed[key] = []
                processed[key].append(item)
            
            return len(processed)
        
        result = await collect_data()
        logger.info(f"  완료: {result}개 프로토콜 데이터 수집")
    
    async def _simulate_graph_building(self):
        """그래프 빌딩 시뮬레이션"""
        @self.checker.measure_performance("graph_building")
        async def build_graph():
            logger.info("🕸️  그래프 빌딩 시뮬레이션...")
            
            # 25개 자산 노드
            nodes = [f'asset_{i}' for i in range(25)]
            
            # 모든 쌍에 대해 edge 생성 (25 * 24 = 600개)
            edges = {}
            for i in range(25):
                for j in range(25):
                    if i != j:
                        key = f"{nodes[i]}_{nodes[j]}"
                        edges[key] = {
                            'weight': random.uniform(-0.1, 0.1),
                            'protocols': [f'protocol_{k}' for k in range(random.randint(1, 5))]
                        }
            
            # 인접 행렬 계산
            matrix = [[0.0 for _ in range(25)] for _ in range(25)]
            for i in range(25):
                for j in range(25):
                    if i != j:
                        matrix[i][j] = random.uniform(-0.5, 0.5)
            
            return len(edges)
        
        result = await build_graph()
        logger.info(f"  완료: {result}개 엣지 생성")
    
    async def _simulate_arbitrage_detection(self):
        """아비트래지 탐지 시뮬레이션"""
        @self.checker.measure_performance("arbitrage_detection")
        async def detect_arbitrage():
            logger.info("🔍 아비트래지 탐지 시뮬레이션...")
            
            opportunities = []
            
            # Bellman-Ford 알고리즘 시뮬레이션
            for cycle in range(50):
                path_length = random.randint(3, 6)
                path = []
                
                for step in range(path_length):
                    path.append({
                        'from': f'asset_{step % 25}',
                        'to': f'asset_{(step + 1) % 25}',
                        'rate': random.uniform(0.95, 1.05),
                        'protocol': f'protocol_{step % 96}'
                    })
                
                # 수익성 계산
                total_rate = 1.0
                for step in path:
                    total_rate *= step['rate']
                
                profit = total_rate - 1.0
                if profit > 0.001:  # 0.1% 이상 수익
                    opportunities.append({
                        'path': path,
                        'profit': profit,
                        'confidence': random.uniform(0.7, 0.95)
                    })
            
            return len(opportunities)
        
        result = await detect_arbitrage()
        logger.info(f"  완료: {result}개 기회 발견")
    
    async def _simulate_intensive_work(self):
        """메모리 집약적 작업"""
        @self.checker.measure_performance("intensive_operation")
        async def intensive_work():
            logger.info("💾 메모리 집약적 작업 시뮬레이션...")
            
            # 큰 데이터 구조 생성
            large_data = []
            
            # 96 프로토콜 * 25 자산 * 100 일 = 240,000 데이터 포인트
            for protocol in range(96):
                for asset in range(25):
                    for day in range(100):
                        large_data.append({
                            'protocol': protocol,
                            'asset': asset,
                            'day': day,
                            'price_history': [random.uniform(1, 100) for _ in range(24)],  # 시간당 가격
                            'volume_history': [random.uniform(100, 10000) for _ in range(24)]
                        })
            
            # 데이터 집계
            aggregated = {}
            for item in large_data:
                key = f"{item['protocol']}_{item['asset']}"
                if key not in aggregated:
                    aggregated[key] = {
                        'total_volume': 0,
                        'avg_price': 0,
                        'days': 0
                    }
                
                aggregated[key]['total_volume'] += sum(item['volume_history'])
                aggregated[key]['avg_price'] += sum(item['price_history']) / len(item['price_history'])
                aggregated[key]['days'] += 1
            
            # 평균 계산
            for key, data in aggregated.items():
                if data['days'] > 0:
                    data['avg_price'] /= data['days']
            
            return len(aggregated)
        
        result = await intensive_work()
        logger.info(f"  완료: {result}개 집계 데이터 생성")
    
    def _print_results(self, report: Dict):
        """결과 출력"""
        print("\n" + "="*60)
        print("🔍 DeFi 시스템 메모리 및 성능 점검 결과")
        print("="*60)
        
        # 시스템 정보
        system_info = report.get('system_info', {})
        print(f"\n📊 시스템 정보:")
        print(f"  Python 버전: {system_info.get('python_version', 'Unknown')}")
        print(f"  초기 메모리: {system_info.get('initial_memory_mb', 0):.1f} MB")
        print(f"  현재 메모리: {system_info.get('current_memory_mb', 0):.1f} MB")
        print(f"  메모리 증가: {system_info.get('memory_growth_mb', 0):.1f} MB")
        
        # 메모리 분석
        memory_analysis = report.get('memory_analysis', {})
        if 'risk_assessment' in memory_analysis:
            risk = memory_analysis['risk_assessment']
            print(f"\n⚠️  메모리 위험도: {risk.get('risk_level', 'UNKNOWN')}")
            print(f"  위험 점수: {risk.get('risk_score', 0)}/7")
            
            factors = risk.get('risk_factors', [])
            if factors:
                print("  위험 요소:")
                for factor in factors:
                    print(f"    • {factor}")
        
        # 성능 분석
        performance_analysis = report.get('performance_analysis', {})
        if 'overall' in performance_analysis:
            overall = performance_analysis['overall']
            print(f"\n📈 성능 요약:")
            print(f"  전체 연산: {overall.get('total_operations', 0)}개")
            print(f"  성공률: {overall.get('success_rate', 0):.1%}")
            print(f"  평균 실행시간: {overall.get('avg_execution_time', 0):.2f}초")
            print(f"  평균 메모리 영향: {overall.get('avg_memory_impact', 0):.1f}MB")
        
        # 성능 문제
        issues = performance_analysis.get('performance_issues', [])
        if issues:
            print(f"\n❌ 성능 문제:")
            for issue in issues:
                print(f"    • {issue}")
        
        # 권고사항
        recommendations = report.get('recommendations', [])
        print(f"\n💡 권고사항:")
        for rec in recommendations:
            print(f"    • {rec}")
        
        print("\n" + "="*60)

async def main():
    """메인 함수"""
    print("🔍 DeFi 시스템 간단 메모리 및 성능 점검")
    print("=" * 60)
    
    checker = DeFiSimpleChecker()
    
    try:
        report_file = await checker.run_check()
        print(f"\n✅ 점검 완료! 보고서: {report_file}")
        
    except Exception as e:
        logger.error(f"점검 중 오류: {e}")
        print(f"❌ 점검 실패: {e}")

if __name__ == "__main__":
    asyncio.run(main())