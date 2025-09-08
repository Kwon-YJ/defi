#!/usr/bin/env python3
"""
Enhanced State Change Detection and Response System
향상된 상태 변경 감지 및 즉시 대응 시스템
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Callable, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict

from src.logger import setup_logger
from src.data_storage import DataStorage
from config.config import config

logger = setup_logger(__name__)

class StateChangeType(Enum):
    """상태 변경 유형"""
    POOL_RESERVE_CHANGE = "pool_reserve_change"
    TOKEN_PRICE_CHANGE = "token_price_change"
    PROTOCOL_STATE_CHANGE = "protocol_state_change"
    LIQUIDITY_CHANGE = "liquidity_change"
    FEE_CHANGE = "fee_change"

@dataclass
class StateChange:
    """상태 변경 데이터"""
    change_type: StateChangeType
    entity_id: str  # Pool address, token symbol, etc.
    timestamp: datetime
    old_value: Any
    new_value: Any
    change_percentage: float
    source: str  # Which protocol/pool this change came from
    metadata: Dict[str, Any] = None

class StateChangeDetector:
    """상태 변경 감지기"""
    
    def __init__(self):
        self.storage = DataStorage()
        self.subscribers: Dict[StateChangeType, List[Callable]] = defaultdict(list)
        self.running = False
        
        # State tracking
        self.previous_states = {}  # entity_id -> {attribute: value}
        self.change_thresholds = {
            StateChangeType.POOL_RESERVE_CHANGE: 0.05,  # 5% change threshold
            StateChangeType.TOKEN_PRICE_CHANGE: 0.02,   # 2% change threshold
            StateChangeType.LIQUIDITY_CHANGE: 0.10,     # 10% change threshold
            StateChangeType.FEE_CHANGE: 0.05,           # 5% change threshold
        }
        
        # Monitoring intervals
        self.pool_monitoring_interval = 1  # seconds
        self.price_monitoring_interval = 5  # seconds
        self.protocol_monitoring_interval = 10  # seconds
        
        # Performance tracking
        self.detection_stats = {
            'total_changes_detected': 0,
            'changes_by_type': defaultdict(int),
            'response_times': [],
            'errors': 0
        }
        
        # HTTP session for API calls
        self.http_session = None
        
        # Monitored entities
        self.monitored_pools: Set[str] = set()
        self.monitored_tokens: Set[str] = set()
        self.monitored_protocols: Set[str] = set()
    
    async def initialize(self):
        """시스템 초기화"""
        logger.info("상태 변경 감지 시스템 초기화 시작")
        
        # Initialize HTTP session
        self.http_session = aiohttp.ClientSession()
        
        logger.info("상태 변경 감지 시스템 초기화 완료")
    
    async def start_detection_system(self):
        """감지 시스템 시작"""
        self.running = True
        logger.info("상태 변경 감지 시스템 시작")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitor_pool_reserves()),
            asyncio.create_task(self._monitor_token_prices()),
            asyncio.create_task(self._monitor_protocol_states())
        ]
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"상태 변경 감지 시스템 오류: {e}")
        finally:
            logger.info("상태 변경 감지 시스템 종료")
    
    def subscribe(self, change_type: StateChangeType, callback: Callable):
        """상태 변경 구독"""
        self.subscribers[change_type].append(callback)
        logger.debug(f"새 구독자 추가: {change_type}")
    
    def subscribe_all(self, callback: Callable):
        """모든 상태 변경 구독"""
        for change_type in StateChangeType:
            self.subscribe(change_type, callback)
    
    async def add_monitored_pool(self, pool_address: str):
        """감지할 풀 추가"""
        self.monitored_pools.add(pool_address)
        logger.debug(f"감지 풀 추가: {pool_address}")
    
    async def add_monitored_token(self, token_symbol: str):
        """감지할 토큰 추가"""
        self.monitored_tokens.add(token_symbol)
        logger.debug(f"감지 토큰 추가: {token_symbol}")
    
    async def add_monitored_protocol(self, protocol_name: str):
        """감지할 프로토콜 추가"""
        self.monitored_protocols.add(protocol_name)
        logger.debug(f"감지 프로토콜 추가: {protocol_name}")
    
    async def _monitor_pool_reserves(self):
        """풀 리저브 모니터링"""
        while self.running:
            try:
                for pool_address in self.monitored_pools:
                    await self._check_pool_reserve_change(pool_address)
                
                await asyncio.sleep(self.pool_monitoring_interval)
                
            except asyncio.CancelledError:
                logger.info("풀 리저브 모니터링 작업 취소됨")
                break
            except Exception as e:
                logger.error(f"풀 리저브 모니터링 오류: {e}")
                self.detection_stats['errors'] += 1
                await asyncio.sleep(self.pool_monitoring_interval)
    
    async def _check_pool_reserve_change(self, pool_address: str):
        """풀 리저브 변경 확인"""
        try:
            # Get current pool data from storage
            current_data = await self.storage.get_pool_data(pool_address)
            if not current_data:
                return
            
            # Get previous state
            previous_state = self.previous_states.get(pool_address, {})
            current_reserves = {
                'reserve0': current_data.get('reserve0', 0),
                'reserve1': current_data.get('reserve1', 0)
            }
            
            # Check for changes
            significant_change = False
            changes = []
            
            for reserve_key in ['reserve0', 'reserve1']:
                current_value = current_reserves[reserve_key]
                previous_value = previous_state.get(reserve_key, current_value)
                
                if previous_value != 0:
                    change_percentage = abs((current_value - previous_value) / previous_value)
                    threshold = self.change_thresholds[StateChangeType.POOL_RESERVE_CHANGE]
                    
                    if change_percentage >= threshold:
                        significant_change = True
                        changes.append({
                            'reserve': reserve_key,
                            'old_value': previous_value,
                            'new_value': current_value,
                            'change_percentage': change_percentage
                        })
            
            # If significant change detected
            if significant_change:
                state_change = StateChange(
                    change_type=StateChangeType.POOL_RESERVE_CHANGE,
                    entity_id=pool_address,
                    timestamp=datetime.now(),
                    old_value=previous_state,
                    new_value=current_reserves,
                    change_percentage=max(change['change_percentage'] for change in changes),
                    source="pool_monitoring",
                    metadata={'changes': changes}
                )
                
                await self._notify_subscribers(state_change)
                await self._update_previous_state(pool_address, current_reserves)
                await self._update_statistics(state_change)
                
                logger.info(f"풀 리저브 변경 감지: {pool_address}, 변경율: {state_change.change_percentage:.2%}")
            
            # Update previous state
            await self._update_previous_state(pool_address, current_reserves)
            
        except Exception as e:
            logger.error(f"풀 리저브 변경 확인 오류 ({pool_address}): {e}")
            self.detection_stats['errors'] += 1
    
    async def _monitor_token_prices(self):
        """토큰 가격 모니터링"""
        while self.running:
            try:
                for token_symbol in self.monitored_tokens:
                    await self._check_token_price_change(token_symbol)
                
                await asyncio.sleep(self.price_monitoring_interval)
                
            except asyncio.CancelledError:
                logger.info("토큰 가격 모니터링 작업 취소됨")
                break
            except Exception as e:
                logger.error(f"토큰 가격 모니터링 오류: {e}")
                self.detection_stats['errors'] += 1
                await asyncio.sleep(self.price_monitoring_interval)
    
    async def _check_token_price_change(self, token_symbol: str):
        """토큰 가격 변경 확인"""
        try:
            # This would typically involve checking price feeds or DEX data
            # For now, we'll simulate by checking stored price data
            # In a real implementation, this would connect to price oracles
            
            # Simulate price check (this is a placeholder)
            current_price = await self._get_current_token_price(token_symbol)
            if current_price is None:
                return
            
            # Get previous state
            previous_state = self.previous_states.get(f"price_{token_symbol}", {})
            previous_price = previous_state.get('price', current_price)
            
            # Check for significant change
            if previous_price != 0:
                change_percentage = abs((current_price - previous_price) / previous_price)
                threshold = self.change_thresholds[StateChangeType.TOKEN_PRICE_CHANGE]
                
                if change_percentage >= threshold:
                    state_change = StateChange(
                        change_type=StateChangeType.TOKEN_PRICE_CHANGE,
                        entity_id=token_symbol,
                        timestamp=datetime.now(),
                        old_value={'price': previous_price},
                        new_value={'price': current_price},
                        change_percentage=change_percentage,
                        source="price_monitoring"
                    )
                    
                    await self._notify_subscribers(state_change)
                    await self._update_previous_state(f"price_{token_symbol}", {'price': current_price})
                    await self._update_statistics(state_change)
                    
                    logger.info(f"토큰 가격 변경 감지: {token_symbol}, 변경율: {change_percentage:.2%}")
            
            # Update previous state
            await self._update_previous_state(f"price_{token_symbol}", {'price': current_price})
            
        except Exception as e:
            logger.error(f"토큰 가격 변경 확인 오류 ({token_symbol}): {e}")
            self.detection_stats['errors'] += 1
    
    async def _get_current_token_price(self, token_symbol: str) -> Optional[float]:
        """현재 토큰 가격 조회 (시뮬레이션)"""
        # This is a placeholder - in a real implementation, this would connect to price oracles
        # For now, we'll return a simulated price based on stored data or a fixed value
        
        try:
            # Try to get from storage first
            price_key = f"price:{token_symbol}"
            stored_price = self.storage.redis_client.get(price_key)
            if stored_price:
                return float(stored_price.decode())
            
            # If not found, return a default value (this is just for testing)
            default_prices = {
                'ETH': 2000.0,
                'WETH': 2000.0,
                'USDC': 1.0,
                'USDT': 1.0,
                'DAI': 1.0
            }
            
            return default_prices.get(token_symbol, 100.0)
            
        except Exception as e:
            logger.debug(f"토큰 가격 조회 실패 ({token_symbol}): {e}")
            return None
    
    async def _monitor_protocol_states(self):
        """프로토콜 상태 모니터링"""
        while self.running:
            try:
                for protocol_name in self.monitored_protocols:
                    await self._check_protocol_state_change(protocol_name)
                
                await asyncio.sleep(self.protocol_monitoring_interval)
                
            except asyncio.CancelledError:
                logger.info("프로토콜 상태 모니터링 작업 취소됨")
                break
            except Exception as e:
                logger.error(f"프로토콜 상태 모니터링 오류: {e}")
                self.detection_stats['errors'] += 1
                await asyncio.sleep(self.protocol_monitoring_interval)
    
    async def _check_protocol_state_change(self, protocol_name: str):
        """프로토콜 상태 변경 확인"""
        try:
            # This would check protocol-specific state changes
            # For example, fee changes, parameter updates, etc.
            
            # Simulate protocol state check
            current_state = await self._get_protocol_state(protocol_name)
            if current_state is None:
                return
            
            # Get previous state
            previous_state = self.previous_states.get(f"protocol_{protocol_name}", {})
            
            # Check for any changes in state
            changes_detected = False
            changed_fields = []
            
            for key, current_value in current_state.items():
                previous_value = previous_state.get(key, current_value)
                if previous_value != current_value:
                    changes_detected = True
                    changed_fields.append({
                        'field': key,
                        'old_value': previous_value,
                        'new_value': current_value
                    })
            
            # If changes detected
            if changes_detected:
                # Calculate overall change significance (simplified)
                change_percentage = len(changed_fields) / max(1, len(current_state))
                
                state_change = StateChange(
                    change_type=StateChangeType.PROTOCOL_STATE_CHANGE,
                    entity_id=protocol_name,
                    timestamp=datetime.now(),
                    old_value=previous_state,
                    new_value=current_state,
                    change_percentage=change_percentage,
                    source="protocol_monitoring",
                    metadata={'changed_fields': changed_fields}
                )
                
                await self._notify_subscribers(state_change)
                await self._update_previous_state(f"protocol_{protocol_name}", current_state)
                await self._update_statistics(state_change)
                
                logger.info(f"프로토콜 상태 변경 감지: {protocol_name}, 변경 필드: {len(changed_fields)}개")
            
            # Update previous state
            await self._update_previous_state(f"protocol_{protocol_name}", current_state)
            
        except Exception as e:
            logger.error(f"프로토콜 상태 변경 확인 오류 ({protocol_name}): {e}")
            self.detection_stats['errors'] += 1
    
    async def _get_protocol_state(self, protocol_name: str) -> Optional[Dict]:
        """프로토콜 상태 조회 (시뮬레이션)"""
        # This is a placeholder - in a real implementation, this would query protocol contracts
        # For now, we'll return simulated states
        
        try:
            # Simulate different protocol states
            protocol_states = {
                'uniswap_v2': {
                    'fee': 0.003,
                    'total_liquidity': 1000000000,  # 1B USD
                    'active_pools': 15000
                },
                'uniswap_v3': {
                    'fee_tiers': [0.0005, 0.003, 0.01],
                    'total_liquidity': 2500000000,  # 2.5B USD
                    'active_pools': 80000
                },
                'sushiswap': {
                    'fee': 0.003,
                    'total_liquidity': 500000000,  # 500M USD
                    'active_pools': 8000
                }
            }
            
            return protocol_states.get(protocol_name)
            
        except Exception as e:
            logger.debug(f"프로토콜 상태 조회 실패 ({protocol_name}): {e}")
            return None
    
    async def _notify_subscribers(self, state_change: StateChange):
        """구독자들에게 상태 변경 알림"""
        start_time = datetime.now()
        
        # Notify specific type subscribers
        if state_change.change_type in self.subscribers:
            for callback in self.subscribers[state_change.change_type]:
                try:
                    await callback(state_change)
                except Exception as e:
                    logger.error(f"구독자 콜백 오류 ({state_change.change_type}): {e}")
        
        # Notify all subscribers
        for callback in self.subscribers.get(StateChangeType.PROTOCOL_STATE_CHANGE, []):  # For backward compatibility
            try:
                await callback(state_change)
            except Exception as e:
                logger.error(f"구독자 콜백 오류 (모든 상태 변경): {e}")
        
        # Track response time
        response_time = (datetime.now() - start_time).total_seconds()
        self.detection_stats['response_times'].append(response_time)
    
    async def _update_previous_state(self, entity_id: str, state: Dict):
        """이전 상태 업데이트"""
        self.previous_states[entity_id] = state.copy()
    
    async def _update_statistics(self, state_change: StateChange):
        """통계 업데이트"""
        self.detection_stats['total_changes_detected'] += 1
        self.detection_stats['changes_by_type'][state_change.change_type] += 1
    
    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        avg_response_time = (
            sum(self.detection_stats['response_times']) / len(self.detection_stats['response_times'])
            if self.detection_stats['response_times'] else 0
        )
        
        return {
            'total_changes_detected': self.detection_stats['total_changes_detected'],
            'changes_by_type': dict(self.detection_stats['changes_by_type']),
            'average_response_time': avg_response_time,
            'total_response_times': len(self.detection_stats['response_times']),
            'errors': self.detection_stats['errors'],
            'monitored_pools': len(self.monitored_pools),
            'monitored_tokens': len(self.monitored_tokens),
            'monitored_protocols': len(self.monitored_protocols)
        }
    
    async def cleanup(self):
        """정리"""
        self.running = False
        
        if self.http_session:
            await self.http_session.close()
        
        logger.info("상태 변경 감지 시스템 정리 완료")
    
    def stop(self):
        """시스템 중지"""
        self.running = False
        logger.info("상태 변경 감지 시스템 중지")

class ImmediateResponseSystem:
    """즉시 대응 시스템"""
    
    def __init__(self, state_detector: StateChangeDetector):
        self.state_detector = state_detector
        self.response_handlers = {}
        self.running = False
        
        # Response configuration
        self.response_priorities = {
            StateChangeType.POOL_RESERVE_CHANGE: 1,  # Highest priority
            StateChangeType.TOKEN_PRICE_CHANGE: 2,
            StateChangeType.LIQUIDITY_CHANGE: 2,
            StateChangeType.FEE_CHANGE: 3,
            StateChangeType.PROTOCOL_STATE_CHANGE: 3
        }
        
        # Response thresholds
        self.response_thresholds = {
            StateChangeType.POOL_RESERVE_CHANGE: 0.02,  # 2% change triggers response
            StateChangeType.TOKEN_PRICE_CHANGE: 0.01,   # 1% change triggers response
            StateChangeType.LIQUIDITY_CHANGE: 0.05,     # 5% change triggers response
            StateChangeType.FEE_CHANGE: 0.03,           # 3% change triggers response
            StateChangeType.PROTOCOL_STATE_CHANGE: 0.01 # 1% change triggers response
        }
        
        # Performance tracking
        self.response_stats = {
            'total_responses': 0,
            'responses_by_type': defaultdict(int),
            'response_times': [],
            'errors': 0
        }
    
    async def initialize(self):
        """시스템 초기화"""
        # Subscribe to all state changes
        self.state_detector.subscribe_all(self._handle_state_change)
        logger.info("즉시 대응 시스템 초기화 완료")
    
    async def _handle_state_change(self, state_change: StateChange):
        """상태 변경 처리"""
        try:
            # Check if response is needed based on thresholds
            response_threshold = self.response_thresholds.get(state_change.change_type, 0.01)
            
            if state_change.change_percentage >= response_threshold:
                # Handle the state change with appropriate response
                await self._respond_to_state_change(state_change)
                
        except Exception as e:
            logger.error(f"상태 변경 처리 오류: {e}")
            self.response_stats['errors'] += 1
    
    async def _respond_to_state_change(self, state_change: StateChange):
        """상태 변경에 대한 대응"""
        start_time = datetime.now()
        
        try:
            logger.info(f"상태 변경에 대한 즉시 대응: {state_change.change_type.value} for {state_change.entity_id}")
            
            # Determine response priority
            priority = self.response_priorities.get(state_change.change_type, 3)
            
            # Execute appropriate response based on change type
            if state_change.change_type == StateChangeType.POOL_RESERVE_CHANGE:
                await self._respond_to_pool_reserve_change(state_change)
            elif state_change.change_type == StateChangeType.TOKEN_PRICE_CHANGE:
                await self._respond_to_token_price_change(state_change)
            elif state_change.change_type == StateChangeType.LIQUIDITY_CHANGE:
                await self._respond_to_liquidity_change(state_change)
            elif state_change.change_type == StateChangeType.FEE_CHANGE:
                await self._respond_to_fee_change(state_change)
            elif state_change.change_type == StateChangeType.PROTOCOL_STATE_CHANGE:
                await self._respond_to_protocol_state_change(state_change)
            
            # Track response
            response_time = (datetime.now() - start_time).total_seconds()
            self.response_stats['response_times'].append(response_time)
            self.response_stats['total_responses'] += 1
            self.response_stats['responses_by_type'][state_change.change_type] += 1
            
            logger.info(f"즉시 대응 완료: {state_change.change_type.value} for {state_change.entity_id} "
                       f"(응답 시간: {response_time:.4f}초)")
            
        except Exception as e:
            logger.error(f"상태 변경 대응 오류: {e}")
            self.response_stats['errors'] += 1
    
    async def _respond_to_pool_reserve_change(self, state_change: StateChange):
        """풀 리저브 변경에 대한 대응"""
        # In a real implementation, this would:
        # 1. Update market graph with new reserves
        # 2. Recalculate arbitrage opportunities
        # 3. Trigger immediate processing if profitable
        
        logger.debug(f"풀 리저브 변경 대응: {state_change.entity_id}")
        
        # Simulate response actions
        await asyncio.sleep(0.01)  # Simulate processing time
        
        # This would typically trigger reprocessing of arbitrage opportunities
        # for the affected pool
    
    async def _respond_to_token_price_change(self, state_change: StateChange):
        """토큰 가격 변경에 대한 대응"""
        # In a real implementation, this would:
        # 1. Update price feeds
        # 2. Recalculate exchange rates in market graph
        # 3. Trigger immediate processing if profitable
        
        logger.debug(f"토큰 가격 변경 대응: {state_change.entity_id}")
        
        # Simulate response actions
        await asyncio.sleep(0.005)  # Simulate processing time
    
    async def _respond_to_liquidity_change(self, state_change: StateChange):
        """유동성 변경에 대한 대응"""
        # In a real implementation, this would:
        # 1. Update liquidity metrics
        # 2. Adjust arbitrage filtering criteria
        # 3. Trigger immediate processing if needed
        
        logger.debug(f"유동성 변경 대응: {state_change.entity_id}")
        
        # Simulate response actions
        await asyncio.sleep(0.005)  # Simulate processing time
    
    async def _respond_to_fee_change(self, state_change: StateChange):
        """수수료 변경에 대한 대응"""
        # In a real implementation, this would:
        # 1. Update fee structures in market graph
        # 2. Recalculate profitability thresholds
        # 3. Trigger immediate processing if profitable
        
        logger.debug(f"수수료 변경 대응: {state_change.entity_id}")
        
        # Simulate response actions
        await asyncio.sleep(0.002)  # Simulate processing time
    
    async def _respond_to_protocol_state_change(self, state_change: StateChange):
        """프로토콜 상태 변경에 대한 대응"""
        # In a real implementation, this would:
        # 1. Update protocol parameters
        # 2. Adjust processing strategies
        # 3. Trigger immediate processing if needed
        
        logger.debug(f"프로토콜 상태 변경 대응: {state_change.entity_id}")
        
        # Simulate response actions
        await asyncio.sleep(0.01)  # Simulate processing time
    
    def get_response_statistics(self) -> Dict:
        """응답 통계 반환"""
        avg_response_time = (
            sum(self.response_stats['response_times']) / len(self.response_stats['response_times'])
            if self.response_stats['response_times'] else 0
        )
        
        return {
            'total_responses': self.response_stats['total_responses'],
            'responses_by_type': dict(self.response_stats['responses_by_type']),
            'average_response_time': avg_response_time,
            'total_response_times': len(self.response_stats['response_times']),
            'errors': self.response_stats['errors']
        }
    
    async def cleanup(self):
        """정리"""
        self.running = False
        logger.info("즉시 대응 시스템 정리 완료")

# Enhanced state change detection with arbitrage integration
class EnhancedStateChangeDetectionSystem:
    """향상된 상태 변경 감지 및 대응 시스템"""
    
    def __init__(self, arbitrage_detector=None):
        self.state_detector = StateChangeDetector()
        self.response_system = ImmediateResponseSystem(self.state_detector)
        self.arbitrage_detector = arbitrage_detector
        self.running = False
    
    async def initialize(self):
        """시스템 초기화"""
        await self.state_detector.initialize()
        await self.response_system.initialize()
        logger.info("향상된 상태 변경 감지 시스템 초기화 완료")
    
    async def start_system(self):
        """시스템 시작"""
        self.running = True
        
        # Start state detection
        detection_task = asyncio.create_task(self.state_detector.start_detection_system())
        
        logger.info("향상된 상태 변경 감지 시스템 시작")
        
        # Wait for tasks
        await detection_task
    
    async def add_monitored_entities(self, pools: List[str] = None, tokens: List[str] = None, protocols: List[str] = None):
        """감지할 엔티티 추가"""
        if pools:
            for pool in pools:
                await self.state_detector.add_monitored_pool(pool)
        
        if tokens:
            for token in tokens:
                await self.state_detector.add_monitored_token(token)
        
        if protocols:
            for protocol in protocols:
                await self.state_detector.add_monitored_protocol(protocol)
    
    def get_system_statistics(self) -> Dict:
        """시스템 통계 반환"""
        detection_stats = self.state_detector.get_statistics()
        response_stats = self.response_system.get_response_statistics()
        
        return {
            'detection': detection_stats,
            'response': response_stats
        }
    
    async def cleanup(self):
        """정리"""
        await self.state_detector.cleanup()
        await self.response_system.cleanup()
        logger.info("향상된 상태 변경 감지 시스템 정리 완료")
    
    def stop(self):
        """시스템 중지"""
        self.running = False
        self.state_detector.stop()
        logger.info("향상된 상태 변경 감지 시스템 중지")

async def main():
    """메인 실행 함수"""
    # This would be integrated with the main arbitrage detection system
    pass

if __name__ == "__main__":
    asyncio.run(main())