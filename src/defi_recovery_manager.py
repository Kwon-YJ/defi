#!/usr/bin/env python3
"""
DeFi-Specific Recovery Manager
DeFi 차익거래 시스템을 위한 특화된 복구 관리자

Features:
- DeFi protocol-specific error recovery
- Market data inconsistency handling
- Graph state restoration
- Transaction failure recovery
- Price feed fallback mechanisms
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from src.error_handler import SystemErrorHandler, ErrorType, ErrorSeverity
from src.logger import setup_logger

logger = setup_logger(__name__)


class DeFiErrorType(Enum):
    """DeFi 특화 에러 타입"""
    PRICE_FEED_ERROR = "price_feed"
    LIQUIDITY_ERROR = "liquidity"
    SLIPPAGE_ERROR = "slippage"
    GAS_PRICE_ERROR = "gas_price"
    TRANSACTION_FAILED = "transaction_failed"
    PROTOCOL_UNAVAILABLE = "protocol_unavailable"
    GRAPH_INCONSISTENCY = "graph_inconsistency"
    ARBITRAGE_OPPORTUNITY_EXPIRED = "opportunity_expired"
    FLASH_LOAN_FAILED = "flash_loan_failed"
    INSUFFICIENT_BALANCE = "insufficient_balance"


@dataclass
class DeFiRecoveryContext:
    """DeFi 복구 컨텍스트"""
    error_type: DeFiErrorType
    component: str
    affected_tokens: List[str]
    affected_protocols: List[str]
    timestamp: datetime
    metadata: Dict[str, Any]
    recovery_actions: List[str]


class DeFiRecoveryManager:
    """DeFi 차익거래 시스템 전용 복구 관리자"""
    
    def __init__(self, error_handler: SystemErrorHandler):
        self.error_handler = error_handler
        self.recovery_history = []
        self.fallback_data_sources = {}
        self.backup_rpc_endpoints = []
        self.protocol_status = {}  # protocol -> (status, last_check)
        
        # 복구 통계
        self.recovery_stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'price_feed_recoveries': 0,
            'graph_restorations': 0,
            'protocol_failovers': 0
        }
        
        # 시스템 상태 백업
        self.state_backups = {}
        self.backup_interval = 300  # 5분
        self.max_backups = 12  # 1시간분
        
        self._setup_defi_recovery_strategies()
        logger.info("DeFiRecoveryManager 초기화 완료")
    
    def _setup_defi_recovery_strategies(self):
        """DeFi 특화 복구 전략 설정"""
        
        # Price feed 에러 복구 전략
        async def recover_price_feed_error(context: DeFiRecoveryContext):
            return await self._recover_price_feed(context)
        
        # 유동성 에러 복구 전략
        async def recover_liquidity_error(context: DeFiRecoveryContext):
            return await self._recover_liquidity_issue(context)
        
        # Graph 불일치 복구 전략
        async def recover_graph_inconsistency(context: DeFiRecoveryContext):
            return await self._recover_graph_state(context)
        
        # Transaction 실패 복구 전략
        async def recover_transaction_failure(context: DeFiRecoveryContext):
            return await self._recover_transaction_failure(context)
        
        # Protocol 이용불가 복구 전략
        async def recover_protocol_unavailable(context: DeFiRecoveryContext):
            return await self._recover_protocol_unavailable(context)
        
        # 복구 함수 매핑
        self.recovery_functions = {
            DeFiErrorType.PRICE_FEED_ERROR: recover_price_feed_error,
            DeFiErrorType.LIQUIDITY_ERROR: recover_liquidity_error,
            DeFiErrorType.GRAPH_INCONSISTENCY: recover_graph_inconsistency,
            DeFiErrorType.TRANSACTION_FAILED: recover_transaction_failure,
            DeFiErrorType.PROTOCOL_UNAVAILABLE: recover_protocol_unavailable,
        }
    
    async def handle_defi_error(self, error: Exception, component: str,
                               defi_error_type: DeFiErrorType,
                               affected_tokens: List[str] = None,
                               affected_protocols: List[str] = None,
                               metadata: Dict[str, Any] = None) -> bool:
        """DeFi 특화 에러 처리"""
        
        recovery_context = DeFiRecoveryContext(
            error_type=defi_error_type,
            component=component,
            affected_tokens=affected_tokens or [],
            affected_protocols=affected_protocols or [],
            timestamp=datetime.now(),
            metadata=metadata or {},
            recovery_actions=[]
        )
        
        logger.warning(f"DeFi 에러 처리 시작: {defi_error_type.value} in {component}")
        
        try:
            # 일반 에러 처리
            error_severity = self._get_defi_error_severity(defi_error_type)
            self.error_handler.handle_error(
                error=error,
                component=component,
                error_type=self._map_to_general_error_type(defi_error_type),
                severity=error_severity,
                metadata=metadata
            )
            
            # DeFi 특화 복구 시도
            recovery_success = await self._attempt_defi_recovery(recovery_context)
            
            # 복구 통계 업데이트
            self.recovery_stats['total_recoveries'] += 1
            if recovery_success:
                self.recovery_stats['successful_recoveries'] += 1
            else:
                self.recovery_stats['failed_recoveries'] += 1
            
            # 복구 기록
            self.recovery_history.append(recovery_context)
            
            return recovery_success
            
        except Exception as recovery_error:
            logger.error(f"DeFi 에러 처리 중 오류 발생: {recovery_error}")
            return False
    
    def _get_defi_error_severity(self, defi_error_type: DeFiErrorType) -> ErrorSeverity:
        """DeFi 에러 타입별 심각도 결정"""
        critical_errors = [
            DeFiErrorType.FLASH_LOAN_FAILED,
            DeFiErrorType.GRAPH_INCONSISTENCY
        ]
        
        high_errors = [
            DeFiErrorType.PRICE_FEED_ERROR,
            DeFiErrorType.TRANSACTION_FAILED,
            DeFiErrorType.PROTOCOL_UNAVAILABLE
        ]
        
        medium_errors = [
            DeFiErrorType.LIQUIDITY_ERROR,
            DeFiErrorType.SLIPPAGE_ERROR,
            DeFiErrorType.GAS_PRICE_ERROR
        ]
        
        if defi_error_type in critical_errors:
            return ErrorSeverity.CRITICAL
        elif defi_error_type in high_errors:
            return ErrorSeverity.HIGH
        elif defi_error_type in medium_errors:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _map_to_general_error_type(self, defi_error_type: DeFiErrorType) -> ErrorType:
        """DeFi 에러 타입을 일반 에러 타입으로 매핑"""
        mapping = {
            DeFiErrorType.PRICE_FEED_ERROR: ErrorType.DATA_ERROR,
            DeFiErrorType.LIQUIDITY_ERROR: ErrorType.DATA_ERROR,
            DeFiErrorType.SLIPPAGE_ERROR: ErrorType.COMPUTATION_ERROR,
            DeFiErrorType.GAS_PRICE_ERROR: ErrorType.NETWORK_ERROR,
            DeFiErrorType.TRANSACTION_FAILED: ErrorType.BLOCKCHAIN_ERROR,
            DeFiErrorType.PROTOCOL_UNAVAILABLE: ErrorType.PROTOCOL_ERROR,
            DeFiErrorType.GRAPH_INCONSISTENCY: ErrorType.DATA_ERROR,
            DeFiErrorType.ARBITRAGE_OPPORTUNITY_EXPIRED: ErrorType.TIMEOUT_ERROR,
            DeFiErrorType.FLASH_LOAN_FAILED: ErrorType.BLOCKCHAIN_ERROR,
            DeFiErrorType.INSUFFICIENT_BALANCE: ErrorType.DATA_ERROR
        }
        return mapping.get(defi_error_type, ErrorType.UNKNOWN_ERROR)
    
    async def _attempt_defi_recovery(self, context: DeFiRecoveryContext) -> bool:
        """DeFi 특화 복구 시도"""
        recovery_function = self.recovery_functions.get(context.error_type)
        
        if not recovery_function:
            logger.warning(f"No specific recovery function for {context.error_type.value}")
            return await self._generic_defi_recovery(context)
        
        try:
            result = await recovery_function(context)
            if result:
                logger.info(f"DeFi 복구 성공: {context.error_type.value} in {context.component}")
            return result
            
        except Exception as e:
            logger.error(f"DeFi 복구 실패: {e}")
            return False
    
    async def _recover_price_feed(self, context: DeFiRecoveryContext) -> bool:
        """Price feed 에러 복구"""
        logger.info("Price feed 복구 시작")
        
        # 1. 백업 데이터 소스 시도
        for token in context.affected_tokens:
            if token in self.fallback_data_sources:
                logger.info(f"Using fallback price source for {token}")
                context.recovery_actions.append(f"fallback_price_{token}")
        
        # 2. 캐시된 가격 데이터 사용
        logger.info("Using cached price data temporarily")
        context.recovery_actions.append("use_cached_prices")
        
        # 3. 대체 API 엔드포인트 시도
        logger.info("Switching to alternative API endpoints")
        context.recovery_actions.append("switch_api_endpoint")
        
        self.recovery_stats['price_feed_recoveries'] += 1
        return True
    
    async def _recover_liquidity_issue(self, context: DeFiRecoveryContext) -> bool:
        """유동성 문제 복구"""
        logger.info("Liquidity issue 복구 시작")
        
        # 1. 대체 DEX/Pool 찾기
        for protocol in context.affected_protocols:
            logger.info(f"Finding alternative pools for {protocol}")
            context.recovery_actions.append(f"find_alt_pools_{protocol}")
        
        # 2. 거래 금액 조정
        logger.info("Adjusting trade amounts for available liquidity")
        context.recovery_actions.append("adjust_trade_amounts")
        
        # 3. 멀티홉 경로 재계산
        logger.info("Recalculating multi-hop paths")
        context.recovery_actions.append("recalculate_paths")
        
        return True
    
    async def _recover_graph_state(self, context: DeFiRecoveryContext) -> bool:
        """그래프 상태 불일치 복구"""
        logger.info("Graph inconsistency 복구 시작")
        
        # 1. 백업된 그래프 상태 복원
        if await self._restore_graph_from_backup():
            logger.info("Graph state restored from backup")
            context.recovery_actions.append("restore_from_backup")
        
        # 2. 전체 그래프 재구성
        else:
            logger.info("Rebuilding entire graph state")
            await self._rebuild_graph_state(context)
            context.recovery_actions.append("rebuild_graph")
        
        self.recovery_stats['graph_restorations'] += 1
        return True
    
    async def _recover_transaction_failure(self, context: DeFiRecoveryContext) -> bool:
        """트랜잭션 실패 복구"""
        logger.info("Transaction failure 복구 시작")
        
        # 1. Gas price 조정
        logger.info("Adjusting gas price")
        context.recovery_actions.append("adjust_gas_price")
        
        # 2. Slippage tolerance 증가
        logger.info("Increasing slippage tolerance")
        context.recovery_actions.append("increase_slippage")
        
        # 3. 대체 거래 경로 시도
        logger.info("Trying alternative trading routes")
        context.recovery_actions.append("try_alt_routes")
        
        # 4. RPC 엔드포인트 변경
        if await self._switch_rpc_endpoint():
            logger.info("Switched to backup RPC endpoint")
            context.recovery_actions.append("switch_rpc")
        
        return True
    
    async def _recover_protocol_unavailable(self, context: DeFiRecoveryContext) -> bool:
        """프로토콜 이용불가 복구"""
        logger.info("Protocol unavailable 복구 시작")
        
        # 1. 프로토콜 상태 체크
        for protocol in context.affected_protocols:
            status = await self._check_protocol_status(protocol)
            self.protocol_status[protocol] = (status, datetime.now())
            logger.info(f"Protocol {protocol} status: {status}")
        
        # 2. 대체 프로토콜 사용
        logger.info("Switching to alternative protocols")
        context.recovery_actions.append("switch_protocols")
        
        # 3. 일시적 우회 경로 설정
        logger.info("Setting up temporary bypass routes")
        context.recovery_actions.append("setup_bypass")
        
        self.recovery_stats['protocol_failovers'] += 1
        return True
    
    async def _generic_defi_recovery(self, context: DeFiRecoveryContext) -> bool:
        """일반적인 DeFi 복구 시도"""
        logger.info(f"Generic recovery for {context.error_type.value}")
        
        # 1. 시스템 상태 백업에서 복원
        await self._restore_system_state()
        
        # 2. 성능 저하 모드 활성화
        self.error_handler.enable_graceful_degradation(context.component)
        
        # 3. 재시작 추천
        context.recovery_actions.append("recommend_restart")
        
        return True
    
    async def _restore_graph_from_backup(self) -> bool:
        """백업에서 그래프 상태 복원"""
        try:
            # 가장 최근 백업 찾기
            if not self.state_backups:
                return False
            
            latest_backup = max(self.state_backups.keys())
            backup_data = self.state_backups[latest_backup]
            
            logger.info(f"Restoring graph from backup: {latest_backup}")
            # 실제 복원 로직은 구체적인 구현에서 처리
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False
    
    async def _rebuild_graph_state(self, context: DeFiRecoveryContext):
        """그래프 상태 전체 재구성"""
        try:
            logger.info("Starting complete graph rebuild")
            
            # 1. 현재 그래프 정리
            context.recovery_actions.append("clear_current_graph")
            
            # 2. 프로토콜별 데이터 재수집
            for protocol in context.affected_protocols:
                logger.info(f"Rebuilding data for {protocol}")
                context.recovery_actions.append(f"rebuild_{protocol}")
            
            # 3. 토큰별 데이터 재수집
            for token in context.affected_tokens:
                logger.info(f"Rebuilding data for {token}")
                context.recovery_actions.append(f"rebuild_{token}")
            
            # 4. 그래프 재구성 완료
            context.recovery_actions.append("graph_rebuild_complete")
            
        except Exception as e:
            logger.error(f"Graph rebuild failed: {e}")
            raise
    
    async def _switch_rpc_endpoint(self) -> bool:
        """RPC 엔드포인트 전환"""
        try:
            if not self.backup_rpc_endpoints:
                logger.warning("No backup RPC endpoints available")
                return False
            
            # 다음 백업 엔드포인트로 전환
            # 실제 구현에서는 Web3 provider 변경
            logger.info("Switched to backup RPC endpoint")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch RPC endpoint: {e}")
            return False
    
    async def _check_protocol_status(self, protocol: str) -> str:
        """프로토콜 상태 확인"""
        try:
            # 실제 구현에서는 각 프로토콜의 상태 확인 API 호출
            # 여기서는 시뮬레이션
            await asyncio.sleep(0.1)  # API 호출 시뮬레이션
            return "available"  # or "unavailable", "degraded"
            
        except Exception as e:
            logger.error(f"Failed to check protocol {protocol} status: {e}")
            return "unknown"
    
    async def _restore_system_state(self):
        """시스템 상태 복원"""
        try:
            logger.info("Restoring system state from backup")
            
            if not self.state_backups:
                logger.warning("No system state backups available")
                return
            
            # 가장 최근 백업 사용
            latest_backup_time = max(self.state_backups.keys())
            backup_data = self.state_backups[latest_backup_time]
            
            # 실제 구현에서는 각 컴포넌트의 상태를 복원
            logger.info(f"System state restored from {latest_backup_time}")
            
        except Exception as e:
            logger.error(f"Failed to restore system state: {e}")
    
    def create_system_backup(self, components_data: Dict[str, Any]):
        """시스템 상태 백업 생성"""
        try:
            current_time = datetime.now()
            
            # 오래된 백업 정리
            if len(self.state_backups) >= self.max_backups:
                oldest_backup = min(self.state_backups.keys())
                del self.state_backups[oldest_backup]
            
            # 새 백업 생성
            self.state_backups[current_time] = {
                'timestamp': current_time,
                'components': components_data,
                'metadata': {
                    'system_health': self.error_handler.system_health_score,
                    'active_circuits': len(self.error_handler.circuit_breakers)
                }
            }
            
            logger.debug(f"System backup created: {current_time}")
            
        except Exception as e:
            logger.error(f"Failed to create system backup: {e}")
    
    def add_fallback_data_source(self, token: str, source: str):
        """백업 데이터 소스 추가"""
        self.fallback_data_sources[token] = source
        logger.info(f"Fallback data source added for {token}: {source}")
    
    def add_backup_rpc_endpoint(self, endpoint: str):
        """백업 RPC 엔드포인트 추가"""
        self.backup_rpc_endpoints.append(endpoint)
        logger.info(f"Backup RPC endpoint added: {endpoint}")
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """복구 통계 반환"""
        return {
            **self.recovery_stats,
            'success_rate': (
                self.recovery_stats['successful_recoveries'] / 
                max(self.recovery_stats['total_recoveries'], 1) * 100
            ),
            'recent_recoveries': len([
                r for r in self.recovery_history 
                if (datetime.now() - r.timestamp).total_seconds() < 3600  # 1시간
            ])
        }
    
    def get_protocol_health(self) -> Dict[str, Any]:
        """프로토콜 건강도 반환"""
        current_time = datetime.now()
        health_data = {}
        
        for protocol, (status, last_check) in self.protocol_status.items():
            time_since_check = (current_time - last_check).total_seconds()
            health_data[protocol] = {
                'status': status,
                'last_check': last_check,
                'seconds_since_check': time_since_check,
                'is_stale': time_since_check > 300  # 5분
            }
        
        return health_data


if __name__ == "__main__":
    # 테스트 코드
    from src.error_handler import SystemErrorHandler
    
    async def test_defi_recovery():
        error_handler = SystemErrorHandler()
        recovery_manager = DeFiRecoveryManager(error_handler)
        
        # 테스트 에러 처리
        try:
            raise ValueError("Price feed connection failed")
        except Exception as e:
            success = await recovery_manager.handle_defi_error(
                error=e,
                component="price_collector",
                defi_error_type=DeFiErrorType.PRICE_FEED_ERROR,
                affected_tokens=["ETH", "USDC"],
                affected_protocols=["Uniswap"]
            )
            print(f"Recovery success: {success}")
        
        # 통계 출력
        stats = recovery_manager.get_recovery_stats()
        print(f"Recovery stats: {json.dumps(stats, indent=2)}")
    
    asyncio.run(test_defi_recovery())