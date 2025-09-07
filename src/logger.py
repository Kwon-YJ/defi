import logging
import sys
import os
import json
import sqlite3
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque, defaultdict
from config.config import config

@dataclass
class TransactionLog:
    """거래 로그 구조체"""
    timestamp: str
    transaction_id: str
    transaction_type: str  # 'arbitrage', 'flash_loan', 'multi_hop', etc.
    start_token: str
    end_token: str
    path: List[str]  # 거래 경로
    amounts: List[float]  # 각 단계별 거래량
    gas_used: Optional[int]
    gas_price: Optional[float]
    execution_time: float  # 실행 시간 (초)
    revenue: float  # ETH 단위 수익
    revenue_usd: Optional[float]  # USD 수익
    block_number: Optional[int]
    success: bool
    error_message: Optional[str]
    performance_metrics: Dict[str, Any]

@dataclass
class PerformanceLog:
    """성능 로그 구조체"""
    timestamp: str
    component: str  # 'graph_build', 'negative_cycle_detection', 'local_search', etc.
    execution_time: float
    memory_usage: float
    cpu_usage: float
    bottleneck_detected: bool
    optimization_suggestions: List[str]

@dataclass
class ErrorLog:
    """에러 로그 구조체"""
    timestamp: str
    error_type: str
    error_message: str
    stack_trace: str
    component: str
    context: Dict[str, Any]
    recovery_action: Optional[str]
    severity: str  # 'low', 'medium', 'high', 'critical'

class DetailedTransactionLogger:
    """상세한 거래 로그 시스템"""
    
    def __init__(self, db_path: str = "logs/transaction_logs.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
        
    def _init_database(self):
        """데이터베이스 초기화"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 거래 로그 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transaction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    transaction_id TEXT UNIQUE NOT NULL,
                    transaction_type TEXT NOT NULL,
                    start_token TEXT NOT NULL,
                    end_token TEXT NOT NULL,
                    path TEXT NOT NULL,  -- JSON 형태
                    amounts TEXT NOT NULL,  -- JSON 형태
                    gas_used INTEGER,
                    gas_price REAL,
                    execution_time REAL NOT NULL,
                    revenue REAL NOT NULL,
                    revenue_usd REAL,
                    block_number INTEGER,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    performance_metrics TEXT  -- JSON 형태
                )
            """)
            
            # 성능 로그 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    component TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    memory_usage REAL NOT NULL,
                    cpu_usage REAL NOT NULL,
                    bottleneck_detected BOOLEAN NOT NULL,
                    optimization_suggestions TEXT  -- JSON 형태
                )
            """)
            
            # 에러 로그 테이블  
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    stack_trace TEXT NOT NULL,
                    component TEXT NOT NULL,
                    context TEXT NOT NULL,  -- JSON 형태
                    recovery_action TEXT,
                    severity TEXT NOT NULL
                )
            """)
            
            conn.commit()
    
    def log_transaction(self, transaction_log: TransactionLog):
        """거래 로그 기록"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO transaction_logs 
                    (timestamp, transaction_id, transaction_type, start_token, end_token,
                     path, amounts, gas_used, gas_price, execution_time, revenue, 
                     revenue_usd, block_number, success, error_message, performance_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    transaction_log.timestamp,
                    transaction_log.transaction_id,
                    transaction_log.transaction_type,
                    transaction_log.start_token,
                    transaction_log.end_token,
                    json.dumps(transaction_log.path),
                    json.dumps(transaction_log.amounts),
                    transaction_log.gas_used,
                    transaction_log.gas_price,
                    transaction_log.execution_time,
                    transaction_log.revenue,
                    transaction_log.revenue_usd,
                    transaction_log.block_number,
                    transaction_log.success,
                    transaction_log.error_message,
                    json.dumps(transaction_log.performance_metrics)
                ))
                conn.commit()

    def log_performance(self, performance_log: PerformanceLog):
        """성능 로그 기록"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO performance_logs 
                    (timestamp, component, execution_time, memory_usage, cpu_usage, 
                     bottleneck_detected, optimization_suggestions)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    performance_log.timestamp,
                    performance_log.component,
                    performance_log.execution_time,
                    performance_log.memory_usage,
                    performance_log.cpu_usage,
                    performance_log.bottleneck_detected,
                    json.dumps(performance_log.optimization_suggestions)
                ))
                conn.commit()

    def log_error(self, error_log: ErrorLog):
        """에러 로그 기록"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO error_logs 
                    (timestamp, error_type, error_message, stack_trace, component,
                     context, recovery_action, severity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    error_log.timestamp,
                    error_log.error_type,
                    error_log.error_message,
                    error_log.stack_trace,
                    error_log.component,
                    json.dumps(error_log.context),
                    error_log.recovery_action,
                    error_log.severity
                ))
                conn.commit()

    def get_transaction_stats(self, days: int = 1) -> Dict[str, Any]:
        """거래 통계 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_transactions,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_transactions,
                    AVG(execution_time) as avg_execution_time,
                    MAX(execution_time) as max_execution_time,
                    SUM(revenue) as total_revenue,
                    AVG(revenue) as avg_revenue,
                    MAX(revenue) as max_revenue
                FROM transaction_logs
                WHERE timestamp >= ?
            """, (since_date,))
            
            result = cursor.fetchone()
            return {
                'total_transactions': result[0] or 0,
                'successful_transactions': result[1] or 0,
                'success_rate': (result[1] / result[0] * 100) if result[0] else 0,
                'avg_execution_time': result[2] or 0,
                'max_execution_time': result[3] or 0,
                'total_revenue': result[4] or 0,
                'avg_revenue': result[5] or 0,
                'max_revenue': result[6] or 0
            }

class PerformanceBottleneckIdentifier:
    """성능 병목점 식별 시스템"""
    
    def __init__(self):
        self.performance_history = defaultdict(deque)
        self.bottleneck_thresholds = {
            'execution_time': 6.43,  # 논문 목표 시간 (초)
            'memory_usage': 80.0,    # 80% 이상시 경고
            'cpu_usage': 90.0        # 90% 이상시 경고
        }
        
    def analyze_component_performance(self, component: str, 
                                    execution_time: float,
                                    memory_usage: float,
                                    cpu_usage: float) -> Dict[str, Any]:
        """컴포넌트 성능 분석"""
        
        # 성능 기록 업데이트
        self.performance_history[component].append({
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'cpu_usage': cpu_usage
        })
        
        # 최근 100개 기록만 유지
        if len(self.performance_history[component]) > 100:
            self.performance_history[component].popleft()
            
        recent_data = list(self.performance_history[component])
        
        if len(recent_data) < 5:
            return {'bottleneck_detected': False, 'suggestions': []}
            
        # 병목점 분석
        avg_execution_time = statistics.mean([d['execution_time'] for d in recent_data])
        avg_memory_usage = statistics.mean([d['memory_usage'] for d in recent_data])
        avg_cpu_usage = statistics.mean([d['cpu_usage'] for d in recent_data])
        
        bottlenecks = []
        suggestions = []
        
        # 실행 시간 병목
        if avg_execution_time > self.bottleneck_thresholds['execution_time']:
            bottlenecks.append('execution_time')
            suggestions.append(f"{component} 실행 시간이 목표치 {self.bottleneck_thresholds['execution_time']}초를 초과 (평균: {avg_execution_time:.2f}초)")
            
        # 메모리 병목
        if avg_memory_usage > self.bottleneck_thresholds['memory_usage']:
            bottlenecks.append('memory_usage')
            suggestions.append(f"{component} 메모리 사용량이 {avg_memory_usage:.1f}%로 높음. 메모리 최적화 필요")
            
        # CPU 병목
        if avg_cpu_usage > self.bottleneck_thresholds['cpu_usage']:
            bottlenecks.append('cpu_usage')
            suggestions.append(f"{component} CPU 사용량이 {avg_cpu_usage:.1f}%로 높음. 병렬 처리 고려")
            
        return {
            'bottleneck_detected': len(bottlenecks) > 0,
            'bottlenecks': bottlenecks,
            'suggestions': suggestions,
            'avg_execution_time': avg_execution_time,
            'avg_memory_usage': avg_memory_usage,
            'avg_cpu_usage': avg_cpu_usage
        }

class ErrorTrackingDebugger:
    """에러 추적 및 디버깅 시스템"""
    
    def __init__(self, transaction_logger: DetailedTransactionLogger):
        self.transaction_logger = transaction_logger
        self.error_patterns = {}
        self.recovery_strategies = {
            'network_error': 'retry_with_backoff',
            'insufficient_liquidity': 'skip_and_continue',
            'gas_estimation_error': 'use_fixed_gas_limit',
            'price_impact_too_high': 'reduce_trade_size'
        }
    
    def track_error(self, error: Exception, component: str, context: Dict[str, Any]):
        """에러 추적 및 패턴 분석"""
        import traceback
        
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # 심각도 판정
        severity = self._determine_severity(error_type, error_message)
        
        # 복구 방법 제안
        recovery_action = self._suggest_recovery(error_type, error_message)
        
        error_log = ErrorLog(
            timestamp=datetime.now().isoformat(),
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            component=component,
            context=context,
            recovery_action=recovery_action,
            severity=severity
        )
        
        self.transaction_logger.log_error(error_log)
        
        # 에러 패턴 업데이트
        pattern_key = f"{error_type}:{component}"
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = {'count': 0, 'recent_occurrences': deque(maxlen=50)}
        
        self.error_patterns[pattern_key]['count'] += 1
        self.error_patterns[pattern_key]['recent_occurrences'].append({
            'timestamp': datetime.now().isoformat(),
            'context': context
        })
        
        return error_log
    
    def _determine_severity(self, error_type: str, error_message: str) -> str:
        """에러 심각도 판정"""
        critical_patterns = ['out of memory', 'disk full', 'database corruption']
        high_patterns = ['network timeout', 'api rate limit', 'insufficient balance']
        medium_patterns = ['price impact', 'slippage', 'gas estimation']
        
        error_lower = error_message.lower()
        
        if any(pattern in error_lower for pattern in critical_patterns):
            return 'critical'
        elif any(pattern in error_lower for pattern in high_patterns):
            return 'high'
        elif any(pattern in error_lower for pattern in medium_patterns):
            return 'medium'
        else:
            return 'low'
    
    def _suggest_recovery(self, error_type: str, error_message: str) -> Optional[str]:
        """복구 방법 제안"""
        for pattern, strategy in self.recovery_strategies.items():
            if pattern in error_message.lower() or pattern in error_type.lower():
                return strategy
        return None

# 글로벌 인스턴스들
detailed_logger = DetailedTransactionLogger()
bottleneck_identifier = PerformanceBottleneckIdentifier()
error_debugger = ErrorTrackingDebugger(detailed_logger)

def setup_logger(name: str) -> logging.Logger:
    """로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.log_level or 'INFO'))
    
    # 로그 디렉토리 생성
    os.makedirs('logs', exist_ok=True)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(
        f'logs/arbitrage_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # 포매터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 핸들러가 이미 추가되지 않았다면 추가
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

def log_transaction_execution(transaction_id: str, transaction_type: str, 
                            start_token: str, end_token: str, path: List[str],
                            amounts: List[float], execution_time: float,
                            revenue: float, success: bool = True,
                            error_message: Optional[str] = None,
                            gas_used: Optional[int] = None,
                            gas_price: Optional[float] = None,
                            block_number: Optional[int] = None,
                            performance_metrics: Optional[Dict[str, Any]] = None):
    """거래 실행 로그 기록 (편의 함수)"""
    
    transaction_log = TransactionLog(
        timestamp=datetime.now().isoformat(),
        transaction_id=transaction_id,
        transaction_type=transaction_type,
        start_token=start_token,
        end_token=end_token,
        path=path,
        amounts=amounts,
        gas_used=gas_used,
        gas_price=gas_price,
        execution_time=execution_time,
        revenue=revenue,
        revenue_usd=None,  # USD 환율 계산 추가 가능
        block_number=block_number,
        success=success,
        error_message=error_message,
        performance_metrics=performance_metrics or {}
    )
    
    detailed_logger.log_transaction(transaction_log)

def analyze_performance_bottleneck(component: str, execution_time: float):
    """성능 병목점 분석 (편의 함수)"""
    import psutil
    
    memory_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    
    analysis = bottleneck_identifier.analyze_component_performance(
        component, execution_time, memory_usage, cpu_usage
    )
    
    performance_log = PerformanceLog(
        timestamp=datetime.now().isoformat(),
        component=component,
        execution_time=execution_time,
        memory_usage=memory_usage,
        cpu_usage=cpu_usage,
        bottleneck_detected=analysis['bottleneck_detected'],
        optimization_suggestions=analysis['suggestions']
    )
    
    detailed_logger.log_performance(performance_log)
    
    return analysis

def track_error_with_context(error: Exception, component: str, **context):
    """컨텍스트와 함께 에러 추적 (편의 함수)"""
    return error_debugger.track_error(error, component, context)
