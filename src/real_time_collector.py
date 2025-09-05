import asyncio
import websockets
import json
from typing import Dict, List, Callable, Optional
from web3 import Web3
from src.logger import setup_logger
from src.data_storage import DataStorage
from config.config import config

logger = setup_logger(__name__)

class RealTimeDataCollector:
    def __init__(self):
        self.ws_url = config.ethereum_mainnet_ws
        self.w3 = Web3(Web3.HTTPProvider(config.ethereum_mainnet_rpc))
        self.storage = DataStorage()
        self.subscribers: Dict[str, List[Callable]] = {}
        self.running = False
        self.websocket = None
        
        # 모니터링할 이벤트들
        self.monitored_events = {
            'Swap': '0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822',
            'Sync': '0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1'
        }
        
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
    
    async def subscribe_to_logs(self, callback: Callable):
        """로그 이벤트 구독"""
        if 'logs' not in self.subscribers:
            self.subscribers['logs'] = []
        self.subscribers['logs'].append(callback)
    
    async def start_websocket_listener(self):
        """WebSocket 리스너 시작"""
        self.running = True
        logger.info("WebSocket 연결 시작")
        
        while self.running:
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    self.websocket = websocket
                    
                    # 구독 설정
                    await self._setup_subscriptions()
                    
                    # 메시지 수신 루프
                    async for message in websocket:
                        await self._handle_message(message)
                        
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket 연결 끊김, 재연결 시도...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"WebSocket 연결 오류: {e}")
                await asyncio.sleep(10)
    
    async def _setup_subscriptions(self):
        """구독 설정"""
        try:
            # 새 블록 구독
            if 'new_blocks' in self.subscribers:
                await self.websocket.send(json.dumps({
                    "id": 1,
                    "method": "eth_subscribe",
                    "params": ["newHeads"]
                }))
            
            # 펜딩 트랜잭션 구독
            if 'pending_txs' in self.subscribers:
                await self.websocket.send(json.dumps({
                    "id": 2,
                    "method": "eth_subscribe",
                    "params": ["newPendingTransactions"]
                }))
            
            # 로그 구독 (Swap 이벤트)
            if 'logs' in self.subscribers:
                await self.websocket.send(json.dumps({
                    "id": 3,
                    "method": "eth_subscribe",
                    "params": [
                        "logs",
                        {
                            "topics": [list(self.monitored_events.values())]
                        }
                    ]
                }))
            
            logger.info("WebSocket 구독 설정 완료")
            
        except Exception as e:
            logger.error(f"구독 설정 실패: {e}")
    
    async def _handle_message(self, message: str):
        """WebSocket 메시지 처리"""
        try:
            data = json.loads(message)
            
            # 구독 확인 메시지
            if 'id' in data and 'result' in data:
                logger.info(f"구독 ID {data['id']} 설정됨: {data['result']}")
                return
            
            # 실제 이벤트 데이터
            if 'params' in data:
                subscription_id = data['params']['subscription']
                result = data['params']['result']
                
                # 블록 데이터
                if 'number' in result and 'hash' in result:
                    await self._handle_new_block(result)
                
                # 트랜잭션 해시
                elif isinstance(result, str) and result.startswith('0x'):
                    await self._handle_pending_transaction(result)
                
                # 로그 데이터
                elif 'topics' in result:
                    await self._handle_log_event(result)
                    
        except json.JSONDecodeError:
            logger.error(f"JSON 파싱 실패: {message}")
        except Exception as e:
            logger.error(f"메시지 처리 실패: {e}")
    
    async def _handle_new_block(self, block_data: Dict):
        """새 블록 처리"""
        try:
            block_number = int(block_data['number'], 16)
            block_hash = block_data['hash']
            
            logger.debug(f"새 블록: {block_number} ({block_hash})")
            
            # 구독자들에게 알림
            for callback in self.subscribers.get('new_blocks', []):
                try:
                    await callback(block_data)
                except Exception as e:
                    logger.error(f"블록 콜백 실행 실패: {e}")
                    
        except Exception as e:
            logger.error(f"블록 처리 실패: {e}")
    
    async def _handle_pending_transaction(self, tx_hash: str):
        """펜딩 트랜잭션 처리"""
        try:
            # 구독자들에게 알림
            for callback in self.subscribers.get('pending_txs', []):
                try:
                    await callback(tx_hash)
                except Exception as e:
                    logger.error(f"펜딩 트랜잭션 콜백 실행 실패: {e}")
                    
        except Exception as e:
            logger.error(f"펜딩 트랜잭션 처리 실패: {e}")
    
    async def _handle_log_event(self, log_data: Dict):
        """로그 이벤트 처리"""
        try:
            # Swap 이벤트인지 확인
            if log_data['topics'][0] == self.monitored_events['Swap']:
                await self._handle_swap_event(log_data)
            
            # Sync 이벤트인지 확인
            elif log_data['topics'][0] == self.monitored_events['Sync']:
                await self._handle_sync_event(log_data)
            
            # 구독자들에게 알림
            for callback in self.subscribers.get('logs', []):
                try:
                    await callback(log_data)
                except Exception as e:
                    logger.error(f"로그 콜백 실행 실패: {e}")
                    
        except Exception as e:
            logger.error(f"로그 이벤트 처리 실패: {e}")
    
    async def _handle_swap_event(self, log_data: Dict):
        """Swap 이벤트 처리"""
        try:
            pool_address = log_data['address']
            tx_hash = log_data['transactionHash']
            
            logger.debug(f"Swap 이벤트 감지: {pool_address} in {tx_hash}")
            
            # 풀 데이터 업데이트 트리거
            # TODO: 실제 풀 데이터 업데이트 로직 구현
            
        except Exception as e:
            logger.error(f"Swap 이벤트 처리 실패: {e}")
    
    async def _handle_sync_event(self, log_data: Dict):
        """Sync 이벤트 처리 (리저브 업데이트)"""
        try:
            pool_address = log_data['address']
            
            # Sync 이벤트 데이터 파싱
            # topics[1]: reserve0, topics[2]: reserve1
            if len(log_data['topics']) >= 3:
                reserve0 = int(log_data['topics'][1], 16)
                reserve1 = int(log_data['topics'][2], 16)
                
                # 풀 데이터 저장
                pool_data = {
                    'address': pool_address,
                    'reserve0': reserve0,
                    'reserve1': reserve1,
                    'timestamp': log_data.get('blockNumber', 0)
                }
                
                await self.storage.store_pool_data(pool_address, pool_data)
                
                logger.debug(f"풀 리저브 업데이트: {pool_address}")
                
        except Exception as e:
            logger.error(f"Sync 이벤트 처리 실패: {e}")
    
    def stop(self):
        """데이터 수집 중지"""
        self.running = False
        logger.info("실시간 데이터 수집 중지")

# 사용 예시
async def main():
    collector = RealTimeDataCollector()
    
    # 콜백 함수 정의
    async def on_new_block(block_data):
        print(f"새 블록: {int(block_data['number'], 16)}")
    
    async def on_swap_event(log_data):
        print(f"Swap 이벤트: {log_data['address']}")
    
    # 구독 설정
    await collector.subscribe_to_blocks(on_new_block)
    await collector.subscribe_to_logs(on_swap_event)
    
    # 시작
    await collector.start_websocket_listener()

if __name__ == "__main__":
    asyncio.run(main())