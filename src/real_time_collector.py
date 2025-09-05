import asyncio
import websockets
import json
from typing import Dict, List, Callable
from web3 import Web3
from src.logger import setup_logger
from config.config import config

logger = setup_logger(__name__)

class RealTimeDataCollector:
    def __init__(self):
        self.ws_url = config.ethereum_mainnet_ws
        self.w3 = Web3(Web3.HTTPProvider(config.ethereum_mainnet_rpc))
        self.subscribers: Dict[str, List[Callable]] = {}
        self.running = False
        
    async def subscribe_to_blocks(self, callback: Callable):
        """새 블록 구독"""
        if 'new_blocks' not in self.subscribers:
            self.subscribers['new_blocks'] = []
        self.subscribers['new_blocks'].append(callback)
    
    async def subscribe_to_pending_transactions(self, callback: Callable):
        """펜딩 트랜잭션 구독"""
        if 'pending_txs' not in self.subscribers:
            self.subscribers['pending_txs'] = []
        self.subscribers['pending_txs'].append(callback)
    
    async def start_websocket_listener(self):
        """WebSocket 리스너 시작"""
        self.running = True
        
        while self.running:
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    # 새 블록 구독
                    await websocket.send(json.dumps({
                        "id": 1,
                        "method": "eth_subscribe",
                        "params": ["newHeads"]
                    }))
                    
                    # 펜딩 트랜잭션 구독
                    await websocket.send(json.dumps({
                        "id": 2,
                        "method": "eth_subscribe", 
                        "params": ["newPendingTransactions"]
                    }))
                    
                    logger.info("WebSocket 연결 성공, 구독 시작")
                    
                    async for message in websocket:
                        await self._handle_websocket_message(message)
                        
            except Exception as e:
                logger.error(f"WebSocket 연결 오류: {e}")
                await asyncio.sleep(5)  # 5초 후 재연결 시도
    
    async def _handle_websocket_message(self, message: str):
        """WebSocket 메시지 처리"""
        try:
            data = json.loads(message)
            
            if 'params' in data:
                subscription_id = data['params']['subscription']
                result = data['params']['result']
                
                # 새 블록 처리
                if 'number' in result:
                    await self._notify_subscribers('new_blocks', result)
                
                # 펜딩 트랜잭션 처리
                elif isinstance(result, str) and result.startswith('0x'):
                    await self._notify_subscribers('pending_txs', result)
                    
        except Exception as e:
            logger.error(f"메시지 처리 오류: {e}")
    
    async def _notify_subscribers(self, event_type: str, data):
        """구독자들에게 알림"""
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"콜백 실행 오류: {e}")
    
    def stop(self):
        """데이터 수집 중지"""
        self.running = False

# 사용 예시
async def on_new_block(block_data):
    """새 블록 처리"""
    block_number = int(block_data['number'], 16)
    logger.info(f"새 블록 수신: {block_number}")

async def on_pending_transaction(tx_hash):
    """펜딩 트랜잭션 처리"""
    logger.debug(f"펜딩 트랜잭션: {tx_hash}")
