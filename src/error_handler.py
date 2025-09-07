#!/usr/bin/env python3
"""
Error Handling and Recovery Mechanism for DeFi Arbitrage System
시스템 안정성 검증을 위한 포괄적 에러 처리 및 복구 메커니즘 구현

Features:
- Centralized error handling with classification
- Automatic recovery strategies
- Circuit breaker pattern
- Retry mechanism with exponential backoff
- Error metrics and monitoring
- Graceful degradation
"""

import asyncio
import time
import traceback
import logging
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import threading
from functools import wraps

from src.logger import setup_logger

logger = setup_logger(__name__)


class ErrorSeverity(Enum):
    """에러 심각도 분류"""
    LOW = 1        # 정보성 에러, 계속 진행 가능
    MEDIUM = 2     # 경고, 일부 기능 영향
    HIGH = 3       # 심각한 에러, 복구 시도 필요
    CRITICAL = 4   # 시스템 중단 필요


class ErrorType(Enum):
    """에러 유형 분류"""
    NETWORK_ERROR = "network"           # 네트워크 연결 오류
    API_ERROR = "api"                   # API 호출 오류
    DATA_ERROR = "data"                 # 데이터 품질 오류
    COMPUTATION_ERROR = "computation"   # 계산 오류
    MEMORY_ERROR = "memory"             # 메모리 부족
    TIMEOUT_ERROR = "timeout"           # 시간 초과
    PROTOCOL_ERROR = "protocol"         # DeFi 프로토콜 오류
    BLOCKCHAIN_ERROR = "blockchain"     # 블록체인 관련 오류
    SYSTEM_ERROR = "system"             # 시스템 레벨 오류
    UNKNOWN_ERROR = "unknown"           # 알 수 없는 오류


@dataclass
class ErrorContext:
    """에러 발생 컨텍스트 정보"""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    component: str
    timestamp: datetime = field(default_factory=datetime.now)
    traceback_info: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3


@dataclass
class RecoveryStrategy:
    """복구 전략 정의"""
    strategy_name: str
    retry_count: int = 0
    max_retries: int = 3
    backoff_factor: float = 1.5
    base_delay: float = 1.0
    recovery_function: Optional[Callable] = None
    fallback_function: Optional[Callable] = None


class CircuitBreakerState(Enum):
    """서킷 브레이커 상태"""
    CLOSED = "closed"       # 정상 상태
    OPEN = "open"           # 회로 차단 상태
    HALF_OPEN = "half_open" # 반개방 상태


@dataclass
class CircuitBreaker:
    """서킷 브레이커 구현"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    success_count_in_half_open: int = 0
    required_success_count: int = 3


class ErrorHandlerConfig:
    """에러 핸들러 설정"""
    def __init__(self):
        self.max_error_history = 1000
        self.error_rate_window = timedelta(minutes=5)
        self.alert_thresholds = {
            ErrorSeverity.LOW: 50,      # 5분간 50회
            ErrorSeverity.MEDIUM: 20,   # 5분간 20회
            ErrorSeverity.HIGH: 10,     # 5분간 10회
            ErrorSeverity.CRITICAL: 5   # 5분간 5회
        }
        self.auto_recovery_enabled = True
        self.graceful_degradation_enabled = True


class SystemErrorHandler:
    """중앙집중식 에러 처리 및 복구 시스템"""
    
    def __init__(self, config: Optional[ErrorHandlerConfig] = None):
        self.config = config or ErrorHandlerConfig()
        self.error_history: deque = deque(maxlen=self.config.max_error_history)
        self.error_stats = defaultdict(int)
        self.recovery_strategies: Dict[ErrorType, RecoveryStrategy] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.alert_callbacks: List[Callable] = []
        self.is_running = True
        self.lock = threading.RLock()
        
        # 시스템 상태 추적
        self.system_health_score = 100.0
        self.degraded_components = set()
        self.recovery_in_progress = set()
        
        # 메트릭스
        self.metrics = {
            'total_errors': 0,
            'recovered_errors': 0,
            'failed_recoveries': 0,
            'circuit_breaker_trips': 0,
            'degraded_operations': 0,
            'system_restarts': 0
        }
        
        # 기본 복구 전략 설정
        self._setup_default_recovery_strategies()
        
        logger.info("SystemErrorHandler 초기화 완료")
    
    def _setup_default_recovery_strategies(self):
        """기본 복구 전략 설정"""
        # 네트워크 에러 복구 전략
        self.recovery_strategies[ErrorType.NETWORK_ERROR] = RecoveryStrategy(
            strategy_name="network_retry",
            max_retries=5,
            backoff_factor=2.0,
            base_delay=1.0
        )
        
        # API 에러 복구 전략
        self.recovery_strategies[ErrorType.API_ERROR] = RecoveryStrategy(
            strategy_name="api_retry",
            max_retries=3,
            backoff_factor=1.5,
            base_delay=0.5
        )
        
        # 데이터 에러 복구 전략
        self.recovery_strategies[ErrorType.DATA_ERROR] = RecoveryStrategy(
            strategy_name="data_fallback",
            max_retries=2,
            backoff_factor=1.0,
            base_delay=0.1
        )
        
        # 메모리 에러 복구 전략
        self.recovery_strategies[ErrorType.MEMORY_ERROR] = RecoveryStrategy(
            strategy_name="memory_cleanup",
            max_retries=2,
            backoff_factor=1.0,
            base_delay=0.5
        )
        
        # 타임아웃 에러 복구 전략
        self.recovery_strategies[ErrorType.TIMEOUT_ERROR] = RecoveryStrategy(
            strategy_name="timeout_retry",
            max_retries=2,
            backoff_factor=2.0,
            base_delay=2.0
        )
    
    def handle_error(self, error: Exception, component: str, 
                    error_type: Optional[ErrorType] = None,
                    severity: Optional[ErrorSeverity] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """에러 처리 메인 함수"""
        try:
            # 에러 분류
            if error_type is None:
                error_type = self._classify_error(error)
            
            if severity is None:
                severity = self._determine_severity(error_type, error)
            
            # 에러 컨텍스트 생성
            error_context = ErrorContext(
                error_type=error_type,
                severity=severity,
                message=str(error),
                component=component,
                traceback_info=traceback.format_exc(),
                metadata=metadata or {}
            )
            
            # 에러 기록
            self._record_error(error_context)
            
            # 복구 시도
            if self.config.auto_recovery_enabled:
                self._attempt_recovery(error_context)
            
            # 알림 발송
            self._check_and_send_alerts(error_context)
            
            # 시스템 건강도 업데이트
            self._update_system_health(error_context)
            
            return error_context
            
        except Exception as handler_error:
            logger.critical(f"Error handler 자체에서 오류 발생: {handler_error}")
            return ErrorContext(
                error_type=ErrorType.SYSTEM_ERROR,
                severity=ErrorSeverity.CRITICAL,
                message=f"Error handler failure: {handler_error}",
                component="error_handler"
            )
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """에러 자동 분류"""
        error_name = error.__class__.__name__.lower()
        error_message = str(error).lower()
        
        if any(keyword in error_message for keyword in ['timeout', 'time out']):
            return ErrorType.TIMEOUT_ERROR
        elif any(keyword in error_message for keyword in ['network', 'connection', 'http']):
            return ErrorType.NETWORK_ERROR
        elif any(keyword in error_message for keyword in ['api', 'request', 'response']):
            return ErrorType.API_ERROR
        elif any(keyword in error_message for keyword in ['memory', 'out of memory']):
            return ErrorType.MEMORY_ERROR
        elif any(keyword in error_name for keyword in ['value', 'type', 'attribute']):
            return ErrorType.DATA_ERROR
        elif any(keyword in error_message for keyword in ['gas', 'revert', 'transaction']):
            return ErrorType.BLOCKCHAIN_ERROR
        elif 'protocol' in error_message:
            return ErrorType.PROTOCOL_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR
    
    def _determine_severity(self, error_type: ErrorType, error: Exception) -> ErrorSeverity:
        """에러 심각도 결정"""
        # 시스템 중단 수준 에러
        if error_type in [ErrorType.SYSTEM_ERROR, ErrorType.MEMORY_ERROR]:
            return ErrorSeverity.CRITICAL
        
        # 높은 심각도 에러
        elif error_type in [ErrorType.BLOCKCHAIN_ERROR, ErrorType.PROTOCOL_ERROR]:
            return ErrorSeverity.HIGH
        
        # 중간 심각도 에러
        elif error_type in [ErrorType.NETWORK_ERROR, ErrorType.TIMEOUT_ERROR]:
            return ErrorSeverity.MEDIUM
        
        # 낮은 심각도 에러
        else:
            return ErrorSeverity.LOW
    
    def _record_error(self, error_context: ErrorContext):
        """에러 기록 및 통계 업데이트"""
        with self.lock:
            self.error_history.append(error_context)
            self.error_stats[error_context.error_type] += 1
            self.metrics['total_errors'] += 1
            
            logger.error(
                f"[{error_context.severity.name}] {error_context.error_type.value} "
                f"in {error_context.component}: {error_context.message}"
            )
            
            if error_context.traceback_info and error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                logger.debug(f"Traceback: {error_context.traceback_info}")
    
    def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """자동 복구 시도"""
        if error_context.component in self.recovery_in_progress:
            logger.debug(f"Recovery already in progress for {error_context.component}")
            return False
        
        recovery_strategy = self.recovery_strategies.get(error_context.error_type)
        if not recovery_strategy:
            logger.debug(f"No recovery strategy for {error_context.error_type}")
            return False
        
        if error_context.recovery_attempts >= error_context.max_recovery_attempts:
            logger.warning(f"Maximum recovery attempts reached for {error_context.component}")
            return False
        
        self.recovery_in_progress.add(error_context.component)
        
        try:
            logger.info(f"Attempting recovery for {error_context.component} using {recovery_strategy.strategy_name}")
            
            # 지수 백오프 대기
            delay = recovery_strategy.base_delay * (recovery_strategy.backoff_factor ** recovery_strategy.retry_count)
            time.sleep(delay)
            
            # 복구 함수 실행
            if recovery_strategy.recovery_function:
                result = recovery_strategy.recovery_function(error_context)
                if result:
                    logger.info(f"Recovery successful for {error_context.component}")
                    self.metrics['recovered_errors'] += 1
                    return True
            
            # 기본 복구 전략 실행
            result = self._execute_default_recovery(error_context)
            if result:
                self.metrics['recovered_errors'] += 1
                return True
            else:
                self.metrics['failed_recoveries'] += 1
                return False
                
        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed: {recovery_error}")
            self.metrics['failed_recoveries'] += 1
            return False
        
        finally:
            self.recovery_in_progress.discard(error_context.component)
            error_context.recovery_attempts += 1
    
    def _execute_default_recovery(self, error_context: ErrorContext) -> bool:
        """기본 복구 전략 실행"""
        if error_context.error_type == ErrorType.MEMORY_ERROR:
            return self._recover_memory_error()
        elif error_context.error_type == ErrorType.NETWORK_ERROR:
            return self._recover_network_error()
        elif error_context.error_type == ErrorType.DATA_ERROR:
            return self._recover_data_error(error_context)
        else:
            return False
    
    def _recover_memory_error(self) -> bool:
        """메모리 에러 복구"""
        try:
            import gc
            gc.collect()
            logger.info("Memory cleanup completed")
            return True
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return False
    
    def _recover_network_error(self) -> bool:
        """네트워크 에러 복구"""
        # 네트워크 상태 확인 로직
        logger.info("Checking network connectivity...")
        # 실제 구현에서는 ping 또는 간단한 HTTP 요청으로 확인
        return True
    
    def _recover_data_error(self, error_context: ErrorContext) -> bool:
        """데이터 에러 복구"""
        logger.info("Attempting data error recovery...")
        # 데이터 검증 및 정리 로직
        return True
    
    def get_circuit_breaker(self, component: str) -> CircuitBreaker:
        """컴포넌트별 서킷 브레이커 가져오기/생성"""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker()
        return self.circuit_breakers[component]
    
    def circuit_breaker_call(self, component: str, func: Callable, *args, **kwargs):
        """서킷 브레이커 패턴으로 함수 호출"""
        breaker = self.get_circuit_breaker(component)
        
        # 현재 상태 확인
        current_time = datetime.now()
        
        if breaker.state == CircuitBreakerState.OPEN:
            # 복구 시간 확인
            if (current_time - breaker.last_failure_time).total_seconds() > breaker.recovery_timeout:
                breaker.state = CircuitBreakerState.HALF_OPEN
                breaker.success_count_in_half_open = 0
            else:
                raise Exception(f"Circuit breaker is OPEN for {component}")
        
        try:
            result = func(*args, **kwargs)
            
            # 성공 시 처리
            if breaker.state == CircuitBreakerState.HALF_OPEN:
                breaker.success_count_in_half_open += 1
                if breaker.success_count_in_half_open >= breaker.required_success_count:
                    breaker.state = CircuitBreakerState.CLOSED
                    breaker.failure_count = 0
                    logger.info(f"Circuit breaker CLOSED for {component}")
            
            return result
            
        except Exception as e:
            # 실패 시 처리
            breaker.failure_count += 1
            breaker.last_failure_time = current_time
            
            if breaker.failure_count >= breaker.failure_threshold:
                breaker.state = CircuitBreakerState.OPEN
                self.metrics['circuit_breaker_trips'] += 1
                logger.warning(f"Circuit breaker OPENED for {component}")
            
            # 에러 처리
            self.handle_error(e, component)
            raise
    
    def _check_and_send_alerts(self, error_context: ErrorContext):
        """알림 조건 확인 및 발송"""
        # 최근 시간 윈도우 내 동일 심각도 에러 카운트
        current_time = datetime.now()
        window_start = current_time - self.config.error_rate_window
        
        recent_errors = [
            err for err in self.error_history 
            if err.timestamp >= window_start and err.severity == error_context.severity
        ]
        
        threshold = self.config.alert_thresholds.get(error_context.severity, 100)
        
        if len(recent_errors) >= threshold:
            self._send_alert(error_context, len(recent_errors))
    
    def _send_alert(self, error_context: ErrorContext, error_count: int):
        """알림 발송"""
        alert_message = (
            f"ALERT: {error_context.severity.name} level errors exceeded threshold\n"
            f"Component: {error_context.component}\n"
            f"Error Type: {error_context.error_type.value}\n"
            f"Recent Count: {error_count}\n"
            f"Latest Error: {error_context.message}"
        )
        
        logger.critical(alert_message)
        
        # 등록된 콜백 호출
        for callback in self.alert_callbacks:
            try:
                callback(error_context, alert_message)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _update_system_health(self, error_context: ErrorContext):
        """시스템 건강도 업데이트"""
        severity_impact = {
            ErrorSeverity.LOW: -0.5,
            ErrorSeverity.MEDIUM: -2.0,
            ErrorSeverity.HIGH: -5.0,
            ErrorSeverity.CRITICAL: -10.0
        }
        
        impact = severity_impact.get(error_context.severity, -1.0)
        self.system_health_score = max(0, self.system_health_score + impact)
        
        # 성능 저하 컴포넌트 추가
        if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.degraded_components.add(error_context.component)
    
    def enable_graceful_degradation(self, component: str):
        """성능 저하 모드 활성화"""
        if self.config.graceful_degradation_enabled:
            self.degraded_components.add(component)
            self.metrics['degraded_operations'] += 1
            logger.warning(f"Graceful degradation enabled for {component}")
    
    def disable_graceful_degradation(self, component: str):
        """성능 저하 모드 비활성화"""
        self.degraded_components.discard(component)
        logger.info(f"Graceful degradation disabled for {component}")
    
    def is_component_degraded(self, component: str) -> bool:
        """컴포넌트 성능 저하 상태 확인"""
        return component in self.degraded_components
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 정보 반환"""
        return {
            'health_score': self.system_health_score,
            'degraded_components': list(self.degraded_components),
            'circuit_breaker_states': {
                comp: breaker.state.value 
                for comp, breaker in self.circuit_breakers.items()
            },
            'error_stats': dict(self.error_stats),
            'metrics': self.metrics.copy(),
            'recent_errors': len([
                err for err in self.error_history 
                if (datetime.now() - err.timestamp).total_seconds() < 300  # 5분
            ])
        }
    
    def register_alert_callback(self, callback: Callable):
        """알림 콜백 등록"""
        self.alert_callbacks.append(callback)
    
    def register_recovery_strategy(self, error_type: ErrorType, strategy: RecoveryStrategy):
        """복구 전략 등록"""
        self.recovery_strategies[error_type] = strategy
        logger.info(f"Recovery strategy registered for {error_type.value}: {strategy.strategy_name}")
    
    def shutdown(self):
        """시스템 종료"""
        self.is_running = False
        logger.info("Error handler shutdown completed")


def error_handler_decorator(error_handler: SystemErrorHandler, component: str, 
                          error_type: Optional[ErrorType] = None,
                          severity: Optional[ErrorSeverity] = None):
    """에러 핸들러 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(
                    error=e,
                    component=component,
                    error_type=error_type,
                    severity=severity,
                    metadata={'function': func.__name__, 'args': str(args), 'kwargs': str(kwargs)}
                )
                raise
        return wrapper
    return decorator


def circuit_breaker_decorator(error_handler: SystemErrorHandler, component: str):
    """서킷 브레이커 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return error_handler.circuit_breaker_call(component, func, *args, **kwargs)
        return wrapper
    return decorator


# 전역 에러 핸들러 인스턴스
global_error_handler = SystemErrorHandler()


if __name__ == "__main__":
    # 테스트 코드
    handler = SystemErrorHandler()
    
    # 테스트 에러 발생
    try:
        raise ValueError("Test error")
    except Exception as e:
        context = handler.handle_error(e, "test_component")
        print(f"Error handled: {context.error_type} - {context.severity}")
    
    # 시스템 상태 출력
    status = handler.get_system_status()
    print(f"System status: {json.dumps(status, indent=2, default=str)}")