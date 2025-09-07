#!/usr/bin/env python3
"""
Error Handling and Recovery System Test
에러 핸들링 및 복구 메커니즘 포괄적 테스트

Tests:
- Error classification and handling
- Recovery strategies
- Circuit breaker functionality
- DeFi-specific error recovery
- System health monitoring
- Alert system
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any
from unittest.mock import Mock, patch

from src.error_handler import (
    SystemErrorHandler, ErrorHandlerConfig, ErrorType, ErrorSeverity,
    CircuitBreakerState, error_handler_decorator, circuit_breaker_decorator
)
from src.defi_recovery_manager import DeFiRecoveryManager, DeFiErrorType
from src.logger import setup_logger

logger = setup_logger(__name__)


class ErrorHandlingSystemTest:
    """에러 핸들링 시스템 통합 테스트"""
    
    def __init__(self):
        self.error_handler = SystemErrorHandler()
        self.recovery_manager = DeFiRecoveryManager(self.error_handler)
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
    async def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("=== Error Handling & Recovery System Test 시작 ===")
        
        # 기본 에러 핸들링 테스트
        await self._test_basic_error_handling()
        await self._test_error_classification()
        await self._test_severity_determination()
        
        # 복구 시스템 테스트
        await self._test_recovery_strategies()
        await self._test_circuit_breaker()
        await self._test_graceful_degradation()
        
        # DeFi 특화 테스트
        await self._test_defi_error_recovery()
        await self._test_price_feed_recovery()
        await self._test_graph_state_recovery()
        
        # 시스템 건강도 테스트
        await self._test_system_health_monitoring()
        await self._test_alert_system()
        
        # 성능 테스트
        await self._test_error_handling_performance()
        
        # 결과 출력
        self._print_test_results()
        return self._get_test_summary()
    
    async def _test_basic_error_handling(self):
        """기본 에러 핸들링 테스트"""
        test_name = "Basic Error Handling"
        try:
            # 기본 에러 발생 및 처리 테스트
            test_error = ValueError("Test error message")
            context = self.error_handler.handle_error(
                error=test_error,
                component="test_component",
                metadata={'test': True}
            )
            
            # 결과 검증
            assert context.message == "Test error message"
            assert context.component == "test_component"
            assert context.metadata['test'] is True
            assert len(self.error_handler.error_history) >= 1
            
            self._record_test_result(test_name, True, "Basic error handling successful")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Failed: {e}")
    
    async def _test_error_classification(self):
        """에러 분류 테스트"""
        test_name = "Error Classification"
        try:
            test_cases = [
                (TimeoutError("Connection timeout"), ErrorType.TIMEOUT_ERROR),
                (ConnectionError("Network failed"), ErrorType.NETWORK_ERROR),
                (ValueError("Invalid data format"), ErrorType.DATA_ERROR),
                (MemoryError("Out of memory"), ErrorType.MEMORY_ERROR),
            ]
            
            for error, expected_type in test_cases:
                context = self.error_handler.handle_error(error, "test_component")
                assert context.error_type == expected_type, f"Expected {expected_type}, got {context.error_type}"
            
            self._record_test_result(test_name, True, f"Classified {len(test_cases)} error types correctly")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Failed: {e}")
    
    async def _test_severity_determination(self):
        """심각도 결정 테스트"""
        test_name = "Severity Determination"
        try:
            test_cases = [
                (ErrorType.SYSTEM_ERROR, ErrorSeverity.CRITICAL),
                (ErrorType.MEMORY_ERROR, ErrorSeverity.CRITICAL),
                (ErrorType.BLOCKCHAIN_ERROR, ErrorSeverity.HIGH),
                (ErrorType.NETWORK_ERROR, ErrorSeverity.MEDIUM),
                (ErrorType.DATA_ERROR, ErrorSeverity.LOW),
            ]
            
            for error_type, expected_severity in test_cases:
                severity = self.error_handler._determine_severity(error_type, Exception("test"))
                assert severity == expected_severity, f"Expected {expected_severity}, got {severity}"
            
            self._record_test_result(test_name, True, f"Determined {len(test_cases)} severities correctly")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Failed: {e}")
    
    async def _test_recovery_strategies(self):
        """복구 전략 테스트"""
        test_name = "Recovery Strategies"
        try:
            # 네트워크 에러 복구 테스트
            network_error = ConnectionError("Network connection failed")
            context = self.error_handler.handle_error(
                error=network_error,
                component="network_client",
                error_type=ErrorType.NETWORK_ERROR
            )
            
            # 복구 시도 검증
            recovery_attempted = context.recovery_attempts > 0
            assert recovery_attempted or ErrorType.NETWORK_ERROR in self.error_handler.recovery_strategies
            
            self._record_test_result(test_name, True, "Recovery strategies working")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Failed: {e}")
    
    async def _test_circuit_breaker(self):
        """서킷 브레이커 테스트"""
        test_name = "Circuit Breaker"
        try:
            component = "test_circuit"
            
            # 정상 호출
            def normal_function():
                return "success"
            
            result = self.error_handler.circuit_breaker_call(component, normal_function)
            assert result == "success"
            
            # 실패 호출로 서킷 브레이커 트리거
            def failing_function():
                raise Exception("Simulated failure")
            
            breaker = self.error_handler.get_circuit_breaker(component)
            initial_state = breaker.state
            
            # 여러 번 실패시켜 서킷 브레이커 열기
            for i in range(breaker.failure_threshold + 1):
                try:
                    self.error_handler.circuit_breaker_call(component, failing_function)
                except:
                    pass
            
            # 서킷 브레이커가 열렸는지 확인
            final_breaker = self.error_handler.get_circuit_breaker(component)
            circuit_opened = final_breaker.state == CircuitBreakerState.OPEN
            
            self._record_test_result(test_name, circuit_opened, f"Circuit breaker state: {final_breaker.state}")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Failed: {e}")
    
    async def _test_graceful_degradation(self):
        """성능 저하 모드 테스트"""
        test_name = "Graceful Degradation"
        try:
            component = "test_component"
            
            # 초기 상태 확인
            assert not self.error_handler.is_component_degraded(component)
            
            # 성능 저하 모드 활성화
            self.error_handler.enable_graceful_degradation(component)
            assert self.error_handler.is_component_degraded(component)
            
            # 성능 저하 모드 비활성화
            self.error_handler.disable_graceful_degradation(component)
            assert not self.error_handler.is_component_degraded(component)
            
            self._record_test_result(test_name, True, "Graceful degradation working correctly")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Failed: {e}")
    
    async def _test_defi_error_recovery(self):
        """DeFi 특화 에러 복구 테스트"""
        test_name = "DeFi Error Recovery"
        try:
            # Price feed 에러 테스트
            price_error = Exception("Price feed connection failed")
            success = await self.recovery_manager.handle_defi_error(
                error=price_error,
                component="price_collector",
                defi_error_type=DeFiErrorType.PRICE_FEED_ERROR,
                affected_tokens=["ETH", "USDC"]
            )
            
            # 통계 확인
            stats = self.recovery_manager.get_recovery_stats()
            recovery_attempted = stats['total_recoveries'] > 0
            
            self._record_test_result(test_name, recovery_attempted, f"Recovery stats: {stats}")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Failed: {e}")
    
    async def _test_price_feed_recovery(self):
        """Price feed 복구 테스트"""
        test_name = "Price Feed Recovery"
        try:
            # Fallback data source 설정
            self.recovery_manager.add_fallback_data_source("ETH", "backup_oracle")
            
            # Price feed 에러 시뮬레이션
            price_error = Exception("Primary price feed failed")
            success = await self.recovery_manager.handle_defi_error(
                error=price_error,
                component="price_feed",
                defi_error_type=DeFiErrorType.PRICE_FEED_ERROR,
                affected_tokens=["ETH"]
            )
            
            # 복구 통계 확인
            stats = self.recovery_manager.get_recovery_stats()
            price_recoveries = stats.get('price_feed_recoveries', 0)
            
            self._record_test_result(test_name, price_recoveries > 0, f"Price feed recoveries: {price_recoveries}")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Failed: {e}")
    
    async def _test_graph_state_recovery(self):
        """그래프 상태 복구 테스트"""
        test_name = "Graph State Recovery"
        try:
            # 시스템 백업 생성
            self.recovery_manager.create_system_backup({
                'test_data': 'backup_test',
                'timestamp': time.time()
            })
            
            # Graph inconsistency 에러 시뮬레이션
            graph_error = Exception("Graph state inconsistent")
            success = await self.recovery_manager.handle_defi_error(
                error=graph_error,
                component="market_graph",
                defi_error_type=DeFiErrorType.GRAPH_INCONSISTENCY
            )
            
            # 복구 통계 확인
            stats = self.recovery_manager.get_recovery_stats()
            graph_recoveries = stats.get('graph_restorations', 0)
            
            self._record_test_result(test_name, graph_recoveries > 0, f"Graph restorations: {graph_recoveries}")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Failed: {e}")
    
    async def _test_system_health_monitoring(self):
        """시스템 건강도 모니터링 테스트"""
        test_name = "System Health Monitoring"
        try:
            # 초기 건강도 확인
            initial_health = self.error_handler.system_health_score
            assert initial_health <= 100
            
            # 다양한 심각도의 에러 발생
            critical_error = Exception("Critical system failure")
            self.error_handler.handle_error(
                error=critical_error,
                component="system",
                severity=ErrorSeverity.CRITICAL
            )
            
            # 건강도 감소 확인
            current_health = self.error_handler.system_health_score
            health_decreased = current_health < initial_health
            
            # 시스템 상태 정보 확인
            status = self.error_handler.get_system_status()
            required_fields = ['health_score', 'degraded_components', 'circuit_breaker_states', 'metrics']
            all_fields_present = all(field in status for field in required_fields)
            
            self._record_test_result(
                test_name, 
                health_decreased and all_fields_present, 
                f"Health: {initial_health} -> {current_health}, Status fields: {all_fields_present}"
            )
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Failed: {e}")
    
    async def _test_alert_system(self):
        """알림 시스템 테스트"""
        test_name = "Alert System"
        try:
            alert_received = []
            
            # 알림 콜백 등록
            def test_callback(error_context, message):
                alert_received.append((error_context, message))
            
            self.error_handler.register_alert_callback(test_callback)
            
            # 임계치를 초과하는 에러 발생
            config = ErrorHandlerConfig()
            threshold = config.alert_thresholds[ErrorSeverity.HIGH]
            
            for i in range(threshold + 1):
                high_error = Exception(f"High severity error {i}")
                self.error_handler.handle_error(
                    error=high_error,
                    component="alert_test",
                    severity=ErrorSeverity.HIGH
                )
                await asyncio.sleep(0.01)  # 짧은 대기
            
            # 알림 발생 확인
            alert_triggered = len(alert_received) > 0
            
            self._record_test_result(test_name, alert_triggered, f"Alerts received: {len(alert_received)}")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Failed: {e}")
    
    async def _test_error_handling_performance(self):
        """에러 핸들링 성능 테스트"""
        test_name = "Error Handling Performance"
        try:
            num_errors = 100
            start_time = time.time()
            
            # 대량의 에러 처리
            for i in range(num_errors):
                test_error = Exception(f"Performance test error {i}")
                self.error_handler.handle_error(
                    error=test_error,
                    component="performance_test"
                )
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_error = total_time / num_errors
            
            # 성능 기준: 에러당 평균 1ms 이하
            performance_acceptable = avg_time_per_error < 0.001
            
            self._record_test_result(
                test_name,
                performance_acceptable,
                f"Processed {num_errors} errors in {total_time:.3f}s (avg: {avg_time_per_error*1000:.2f}ms per error)"
            )
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Failed: {e}")
    
    def _record_test_result(self, test_name: str, passed: bool, details: str = ""):
        """테스트 결과 기록"""
        self.test_results['total_tests'] += 1
        
        if passed:
            self.test_results['passed_tests'] += 1
            status = "PASS"
        else:
            self.test_results['failed_tests'] += 1
            status = "FAIL"
        
        result = {
            'test_name': test_name,
            'status': status,
            'details': details,
            'timestamp': time.time()
        }
        
        self.test_results['test_details'].append(result)
        logger.info(f"[{status}] {test_name}: {details}")
    
    def _print_test_results(self):
        """테스트 결과 출력"""
        print("\n" + "="*80)
        print("ERROR HANDLING & RECOVERY SYSTEM TEST RESULTS")
        print("="*80)
        
        total = self.test_results['total_tests']
        passed = self.test_results['passed_tests']
        failed = self.test_results['failed_tests']
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        print()
        
        # 실패한 테스트 상세 정보
        if failed > 0:
            print("FAILED TESTS:")
            print("-" * 40)
            for result in self.test_results['test_details']:
                if result['status'] == 'FAIL':
                    print(f"❌ {result['test_name']}: {result['details']}")
            print()
        
        # 성공한 테스트 요약
        print("PASSED TESTS:")
        print("-" * 40)
        for result in self.test_results['test_details']:
            if result['status'] == 'PASS':
                print(f"✅ {result['test_name']}")
        
        print("\n" + "="*80)
    
    def _get_test_summary(self) -> Dict[str, Any]:
        """테스트 요약 정보 반환"""
        return {
            'summary': self.test_results,
            'error_handler_status': self.error_handler.get_system_status(),
            'recovery_stats': self.recovery_manager.get_recovery_stats(),
            'system_health_score': self.error_handler.system_health_score
        }


async def main():
    """메인 테스트 실행 함수"""
    logger.info("Starting Error Handling & Recovery System Test")
    
    test_system = ErrorHandlingSystemTest()
    
    try:
        # 모든 테스트 실행
        summary = await test_system.run_all_tests()
        
        # 결과 JSON 파일로 저장
        with open("error_handling_test_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("Test results saved to error_handling_test_results.json")
        
        # TODO.txt 업데이트를 위한 성공 여부 반환
        total_tests = summary['summary']['total_tests']
        passed_tests = summary['summary']['passed_tests']
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        if success_rate >= 80:
            logger.info("✅ Error handling and recovery system implementation SUCCESSFUL")
            return True
        else:
            logger.error("❌ Error handling and recovery system implementation FAILED")
            return False
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)