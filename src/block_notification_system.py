#!/usr/bin/env python3
"""
Enhanced Block Notification System
향상된 블록 알림 시스템
"""

import asyncio
import aiohttp
import websockets
import json
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

from src.logger import setup_logger
from config.config import config

logger = setup_logger(__name__)

class BlockNotificationType(Enum):
    """블록 알림 유형"""
    NEW_BLOCK = "new_block"
    BLOCK_REORG = "block_reorg"
    PENDING_TRANSACTION = "pending_transaction"
    TRANSACTION_CONFIRMED = "transaction_confirmed"

@dataclass
class BlockNotification:
    """블록 알림 데이터"""
    notification_type: BlockNotificationType
    block_number: Optional[int] = None
    block_hash: Optional[str] = None
    timestamp: datetime = None
    data: Dict[str, Any] = None
    source: str = None

class BlockNotificationSystem:
    """향상된 블록 알림 시스템"""
    
    def __init__(self):
        # Multiple Ethereum node sources for redundancy
        self.node_configs = [
            {"url": config.ethereum_mainnet_ws, "type": "websocket", "name": "Alchemy_WS"},
            {"url": config.ethereum_mainnet_rpc, "type": "http", "name": "Alchemy_HTTP"},
            {"url": f"wss://mainnet.infura.io/ws/v3/{config.infura_project_id}", "type": "websocket", "name": "Infura_WS"} if config.infura_project_id else None,
            {"url": f"https://mainnet.infura.io/v3/{config.infura_project_id}", "type": "http", "name": "Infura_HTTP"} if config.infura_project_id else None,
        ]
        # Filter out None values
        self.node_configs = [config for config in self.node_configs if config]
        
        self.subscribers: Dict[BlockNotificationType, List[Callable]] = {}
        self.running = False
        self.websocket_connections = {}
        self.http_sessions = {}
        self.notification_stats = {
            'total_notifications': 0,
            'by_type': {},
            'by_source': {},
            'errors': 0
        }
        
        # Connection management
        self.connection_attempts = {}
        self.max_connection_attempts = 3
        self.reconnect_delay = 5  # seconds
        
        # Performance monitoring
        self.notification_latency = []  # Track notification latency
        self.max_latency_samples = 1000
        
    async def initialize(self):
        """시스템 초기화"""
        logger.info("블록 알림 시스템 초기화 시작")
        
        # Initialize HTTP sessions
        for node_config in self.node_configs:
            if node_config["type"] == "http":
                session = aiohttp.ClientSession()
                self.http_sessions[node_config["name"]] = session
        
        logger.info(f"블록 알림 시스템 초기화 완료 - {len(self.node_configs)}개 노드 구성")
    
    async def start_notification_system(self):
        """알림 시스템 시작"""
        self.running = True
        logger.info("블록 알림 시스템 시작")
        
        # Start WebSocket connections
        websocket_tasks = []
        for node_config in self.node_configs:
            if node_config["type"] == "websocket":
                task = asyncio.create_task(self._start_websocket_connection(node_config))
                websocket_tasks.append(task)
        
        # Start HTTP polling for nodes that don't support WebSocket
        http_tasks = []
        for node_config in self.node_configs:
            if node_config["type"] == "http":
                task = asyncio.create_task(self._start_http_polling(node_config))
                http_tasks.append(task)
        
        # Wait for all tasks
        all_tasks = websocket_tasks + http_tasks
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        
        logger.info("블록 알림 시스템 종료")
    
    async def _start_websocket_connection(self, node_config: Dict):
        """WebSocket 연결 시작"""
        node_name = node_config["name"]
        node_url = node_config["url"]
        
        if not node_url:
            logger.warning(f"{node_name}에 대한 URL이 없습니다")
            return
        
        self.connection_attempts[node_name] = 0
        
        while self.running:
            try:
                logger.info(f"{node_name}에 WebSocket 연결 시도 중: {node_url}")
                async with websockets.connect(node_url) as websocket:
                    self.websocket_connections[node_name] = websocket
                    self.connection_attempts[node_name] = 0  # Reset on successful connection
                    
                    # Subscribe to events
                    await self._setup_websocket_subscriptions(websocket, node_name)
                    
                    # Listen for messages
                    async for message in websocket:
                        await self._handle_websocket_message(message, node_name)
                        
            except websockets.exceptions.ConnectionClosed:
                self.connection_attempts[node_name] += 1
                logger.warning(f"{node_name} WebSocket 연결 끊김, 재연결 시도... "
                             f"(시도 {self.connection_attempts[node_name]}/{self.max_connection_attempts})")
                
                if self.connection_attempts[node_name] >= self.max_connection_attempts:
                    logger.error(f"{node_name} WebSocket 최대 재연결 시도 도달")
                    break
                    
                await asyncio.sleep(self.reconnect_delay)
                
            except Exception as e:
                self.connection_attempts[node_name] += 1
                logger.error(f"{node_name} WebSocket 연결 오류: {e}")
                
                if self.connection_attempts[node_name] >= self.max_connection_attempts:
                    logger.error(f"{node_name} WebSocket 최대 재연결 시도 도달")
                    break
                    
                await asyncio.sleep(self.reconnect_delay)
    
    async def _setup_websocket_subscriptions(self, websocket, node_name: str):
        """WebSocket 구독 설정"""
        try:
            # Subscribe to new heads
            new_heads_subscription = {
                "id": 1,
                "method": "eth_subscribe",
                "params": ["newHeads"]
            }
            await websocket.send(json.dumps(new_heads_subscription))
            
            # Subscribe to new pending transactions
            pending_tx_subscription = {
                "id": 2,
                "method": "eth_subscribe",
                "params": ["newPendingTransactions"]
            }
            await websocket.send(json.dumps(pending_tx_subscription))
            
            logger.info(f"{node_name} WebSocket 구독 설정 완료")
            
        except Exception as e:
            logger.error(f"{node_name} WebSocket 구독 설정 실패: {e}")
    
    async def _handle_websocket_message(self, message: str, node_name: str):
        """WebSocket 메시지 처리"""
        try:
            data = json.loads(message)
            
            # Handle subscription confirmations
            if 'id' in data and 'result' in data:
                logger.debug(f"{node_name} 구독 확인: ID {data['id']}, 결과 {data['result']}")
                return
            
            # Handle actual notifications
            if 'params' in data and 'subscription' in data['params']:
                await self._process_notification(data, node_name)
                
        except json.JSONDecodeError:
            logger.error(f"{node_name} JSON 파싱 실패: {message}")
        except Exception as e:
            logger.error(f"{node_name} 메시지 처리 오류: {e}")
    
    async def _start_http_polling(self, node_config: Dict):
        """HTTP 폴링 시작"""
        node_name = node_config["name"]
        node_url = node_config["url"]
        
        if not node_url:
            logger.warning(f"{node_name}에 대한 URL이 없습니다")
            return
        
        session = self.http_sessions.get(node_name)
        if not session:
            logger.error(f"{node_name}에 대한 HTTP 세션이 없습니다")
            return
        
        self.connection_attempts[node_name] = 0
        last_block_number = None
        
        while self.running:
            try:
                # Get latest block number
                payload = {
                    "jsonrpc": "2.0",
                    "method": "eth_blockNumber",
                    "params": [],
                    "id": 1
                }
                
                async with session.post(node_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if 'result' in result:
                            current_block_number = int(result['result'], 16)
                            
                            # If new block detected
                            if last_block_number is None or current_block_number > last_block_number:
                                # Get block details
                                block_payload = {
                                    "jsonrpc": "2.0",
                                    "method": "eth_getBlockByNumber",
                                    "params": [result['result'], False],  # False = don't include full tx objects
                                    "id": 2
                                }
                                
                                async with session.post(node_url, json=block_payload) as block_response:
                                    if block_response.status == 200:
                                        block_result = await block_response.json()
                                        if 'result' in block_result and block_result['result']:
                                            await self._process_notification({
                                                "params": {
                                                    "result": block_result['result']
                                                }
                                            }, node_name)
                            
                            last_block_number = current_block_number
                            self.connection_attempts[node_name] = 0  # Reset on successful request
                    
                # Wait before next poll
                await asyncio.sleep(1)  # Poll every second
                
            except Exception as e:
                self.connection_attempts[node_name] += 1
                logger.error(f"{node_name} HTTP 폴링 오류: {e}")
                
                if self.connection_attempts[node_name] >= self.max_connection_attempts:
                    logger.error(f"{node_name} HTTP 최대 재시도 도달")
                    break
                    
                await asyncio.sleep(self.reconnect_delay)
    
    async def _process_notification(self, data: Dict, source: str):
        """알림 처리"""
        try:
            start_time = datetime.now()
            
            if 'params' in data and 'result' in data['params']:
                result = data['params']['result']
                
                # Determine notification type
                notification = None
                
                # New block notification
                if 'number' in result and 'hash' in result:
                    block_number = int(result['number'], 16)
                    block_hash = result['hash']
                    
                    notification = BlockNotification(
                        notification_type=BlockNotificationType.NEW_BLOCK,
                        block_number=block_number,
                        block_hash=block_hash,
                        timestamp=datetime.now(),
                        data=result,
                        source=source
                    )
                    
                    logger.info(f"새 블록 알림: {block_number} ({block_hash}) from {source}")
                
                # Pending transaction notification
                elif isinstance(result, str) and result.startswith('0x'):
                    notification = BlockNotification(
                        notification_type=BlockNotificationType.PENDING_TRANSACTION,
                        timestamp=datetime.now(),
                        data={"transaction_hash": result},
                        source=source
                    )
                    
                    logger.debug(f"펜딩 트랜잭션 알림: {result} from {source}")
                
                # Process notification if created
                if notification:
                    await self._notify_subscribers(notification)
                    await self._update_statistics(notification, start_time)
                    
        except Exception as e:
            logger.error(f"알림 처리 오류: {e}")
            self.notification_stats['errors'] += 1
    
    async def _notify_subscribers(self, notification: BlockNotification):
        """구독자들에게 알림"""
        notification_type = notification.notification_type
        
        # Notify specific type subscribers
        if notification_type in self.subscribers:
            for callback in self.subscribers[notification_type]:
                try:
                    await callback(notification)
                except Exception as e:
                    logger.error(f"구독자 콜백 오류 ({notification_type}): {e}")
        
        # Notify all subscribers
        if BlockNotificationType.NEW_BLOCK in self.subscribers:  # For backward compatibility
            for callback in self.subscribers.get(BlockNotificationType.NEW_BLOCK, []):
                try:
                    await callback(notification)
                except Exception as e:
                    logger.error(f"구독자 콜백 오류 (모든 알림): {e}")
    
    async def _update_statistics(self, notification: BlockNotification, start_time: datetime):
        """통계 업데이트"""
        # Total count
        self.notification_stats['total_notifications'] += 1
        
        # Count by type
        notification_type = notification.notification_type.value
        if notification_type not in self.notification_stats['by_type']:
            self.notification_stats['by_type'][notification_type] = 0
        self.notification_stats['by_type'][notification_type] += 1
        
        # Count by source
        source = notification.source
        if source not in self.notification_stats['by_source']:
            self.notification_stats['by_source'][source] = 0
        self.notification_stats['by_source'][source] += 1
        
        # Track latency
        latency = (datetime.now() - start_time).total_seconds()
        self.notification_latency.append(latency)
        if len(self.notification_latency) > self.max_latency_samples:
            self.notification_latency = self.notification_latency[-self.max_latency_samples:]
    
    def subscribe(self, notification_type: BlockNotificationType, callback: Callable):
        """알림 구독"""
        if notification_type not in self.subscribers:
            self.subscribers[notification_type] = []
        self.subscribers[notification_type].append(callback)
        logger.debug(f"새 구독자 추가: {notification_type}")
    
    def subscribe_all(self, callback: Callable):
        """모든 알림 구독"""
        for notification_type in BlockNotificationType:
            self.subscribe(notification_type, callback)
    
    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        avg_latency = sum(self.notification_latency) / len(self.notification_latency) if self.notification_latency else 0
        
        return {
            'total_notifications': self.notification_stats['total_notifications'],
            'notifications_by_type': self.notification_stats['by_type'],
            'notifications_by_source': self.notification_stats['by_source'],
            'errors': self.notification_stats['errors'],
            'average_latency': avg_latency,
            'total_latency_samples': len(self.notification_latency)
        }
    
    async def cleanup(self):
        """정리"""
        self.running = False
        
        # Close WebSocket connections
        for node_name, websocket in self.websocket_connections.items():
            try:
                await websocket.close()
            except Exception as e:
                logger.error(f"{node_name} WebSocket 연결 종료 오류: {e}")
        
        # Close HTTP sessions
        for node_name, session in self.http_sessions.items():
            try:
                await session.close()
            except Exception as e:
                logger.error(f"{node_name} HTTP 세션 종료 오류: {e}")
        
        logger.info("블록 알림 시스템 정리 완료")
    
    def stop(self):
        """시스템 중지"""
        self.running = False
        logger.info("블록 알림 시스템 중지")

# Enhanced notification handler for arbitrage detector
class EnhancedNotificationHandler:
    """향상된 알림 처리기"""
    
    def __init__(self, arbitrage_detector):
        self.arbitrage_detector = arbitrage_detector
        self.notification_system = BlockNotificationSystem()
        self.processing_queue = asyncio.Queue()
        self.processing_task = None
    
    async def initialize(self):
        """초기화"""
        await self.notification_system.initialize()
        
        # Subscribe to new block notifications
        self.notification_system.subscribe(
            BlockNotificationType.NEW_BLOCK, 
            self._handle_new_block_notification
        )
        
        # Subscribe to pending transactions
        self.notification_system.subscribe(
            BlockNotificationType.PENDING_TRANSACTION,
            self._handle_pending_transaction_notification
        )
        
        logger.info("향상된 알림 처리기 초기화 완료")
    
    async def start_notification_handling(self):
        """알림 처리 시작"""
        # Start notification system
        notification_task = asyncio.create_task(self.notification_system.start_notification_system())
        
        # Start processing queue
        self.processing_task = asyncio.create_task(self._process_notification_queue())
        
        logger.info("향상된 알림 처리 시작")
        
        # Wait for tasks
        await asyncio.gather(notification_task, self.processing_task, return_exceptions=True)
    
    async def _handle_new_block_notification(self, notification: BlockNotification):
        """새 블록 알림 처리"""
        logger.debug(f"새 블록 알림 수신: {notification.block_number} from {notification.source}")
        
        # Add to processing queue
        await self.processing_queue.put(notification)
    
    async def _handle_pending_transaction_notification(self, notification: BlockNotification):
        """펜딩 트랜잭션 알림 처리"""
        logger.debug(f"펜딩 트랜잭션 알림 수신: {notification.data.get('transaction_hash')} from {notification.source}")
        
        # For now, we're focusing on block notifications for arbitrage detection
        # Pending transactions could be used for more advanced MEV strategies
    
    async def _process_notification_queue(self):
        """알림 큐 처리"""
        while True:
            try:
                notification = await self.processing_queue.get()
                
                if notification.notification_type == BlockNotificationType.NEW_BLOCK:
                    await self._process_new_block(notification)
                
                self.processing_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("알림 큐 처리 작업 취소됨")
                break
            except Exception as e:
                logger.error(f"알림 큐 처리 오류: {e}")
    
    async def _process_new_block(self, notification: BlockNotification):
        """새 블록 처리"""
        try:
            logger.info(f"새 블록 처리 시작: {notification.block_number}")
            
            # Process with the arbitrage detector
            if hasattr(self.arbitrage_detector, '_process_new_block'):
                # Use existing block processing method
                await self.arbitrage_detector._process_new_block({
                    'number': hex(notification.block_number),
                    'hash': notification.block_hash,
                    **notification.data
                })
            else:
                logger.warning("Arbitrage detector에 _process_new_block 메서드가 없습니다")
                
        except Exception as e:
            logger.error(f"새 블록 처리 오류: {e}")
    
    def get_notification_statistics(self) -> Dict:
        """알림 통계 반환"""
        return self.notification_system.get_statistics()
    
    async def cleanup(self):
        """정리"""
        self.notification_system.stop()
        await self.notification_system.cleanup()
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("향상된 알림 처리기 정리 완료")

# Integration with existing arbitrage detector
class EnhancedArbitrageDetectorWithNotifications:
    """알림 시스템이 통합된 향상된 차익거래 탐지기"""
    
    def __init__(self):
        # This would integrate with the existing EnhancedArbitrageDetector
        # For now, we'll focus on the notification system implementation
        pass

async def main():
    """메인 실행 함수"""
    # This would be integrated with the main arbitrage detection system
    pass

if __name__ == "__main__":
    asyncio.run(main())