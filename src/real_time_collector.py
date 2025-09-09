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
        # WS 다중 소스 구성 (쉼표 구분 목록 지원)
        raw_eps = []
        try:
            env_eps = os.getenv('WS_ENDPOINTS', '')
            if env_eps:
                raw_eps = [e.strip() for e in env_eps.split(',') if e.strip()]
        except Exception:
            raw_eps = []
        self.ws_endpoints: List[str] = [e for e in raw_eps if e]
        if self.ws_url and self.ws_url not in self.ws_endpoints:
            self.ws_endpoints.insert(0, self.ws_url)
        if not self.ws_endpoints:
            # fallback to HTTP rpc via websocket proxy not available; keep placeholder
            self.ws_endpoints = [self.ws_url] if self.ws_url else []
        self.ws_index = 0
        self.ws_idle_timeout = int(os.getenv('WS_IDLE_TIMEOUT', '20'))  # seconds without messages → rotate
        self.ws_backoff_min = float(os.getenv('WS_BACKOFF_MIN', '1.0'))
        self.ws_backoff_max = float(os.getenv('WS_BACKOFF_MAX', '10.0'))
        self.w3 = Web3(Web3.HTTPProvider(config.ethereum_mainnet_rpc))
        self.storage = DataStorage()
        self.subscribers: Dict[str, List[Callable]] = {}
        self.running = False
        self.websocket = None
        self.log_addresses: Optional[List[str]] = None
        
        # 모니터링할 이벤트들
        self.monitored_events = {
            'Swap': '0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822',
            'Sync': '0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1',
            # Uniswap V2 Pair events (Mint/Burn)
            # keccak256("Mint(address,uint256,uint256)")
            'Mint': '0x4c209b5fc8ad50758f13e2e1088ba56a560dff690a1c6fef26394f4c03821c4f',
            # keccak256("Burn(address,uint256,uint256,address)")
            'Burn': '0xdccd412f0b1252819cb1fd330b93224ca42612892bb3f4f789976e6d81936496',
            # Uniswap V3 Pool Collect event
            # keccak256("Collect(address,address,int24,int24,uint128,uint256,uint256)")
            'V3Collect': '0xf4ffc589144636c0d56f9ddc7c2399571e6401783c0aafedc9d34ab7cadad2ba',
            # Uniswap V3 NonfungiblePositionManager events (position-level)
            # keccak256("IncreaseLiquidity(uint256,uint128,uint256,uint256)")
            'V3PMIncrease': '0x3067048beee31b25b2f1681f88dac838c8bba36af25bfb2b7cf7473a5847e35f',
            # keccak256("DecreaseLiquidity(uint256,uint128,uint256,uint256)")
            'V3PMDecrease': '0x26f6a048ee9138f2c0ce266f322cb99228e8d619ae2bff30c67f8dcf9d2377b4',
            # keccak256("Collect(uint256,address,uint128,uint128)") (PositionManager)
            'V3PMCollect': '0x4d8babf9b22e68d8f3c8653392a91073d3f3d246ad70593d8c8ed3fe381b3c96'
            ,
            # ERC20 Transfer (for LP mint/burn detection)
            # keccak256("Transfer(address,address,uint256)")
            'ERC20Transfer': '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
            ,
            # Balancer V2 Vault PoolBalanceChanged
            'BalancerPoolBalanceChanged': '0xe5ce249087ce04f05a957192435400fd97868dba0e6a4b4c049abf8af80dae78',
            # Compound cToken AccrueInterest
            'CompoundAccrueInterest': '0x4dec04e750ca11537cabcd8a9eab06494de08da3735bc8871cd41250e190bc04',
            # Aave v2 LendingPool ReserveDataUpdated
            'AaveReserveDataUpdated': '0x804c9b842b2748a22bb64b345453a3de7ca54a6ca45ce00d415894979e22897a'
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
        logger.info("WebSocket 연결 시작 (다중 소스 지원)")

        last_message = 0.0
        backoff = self.ws_backoff_min
        while self.running:
            ep = None
            try:
                if not self.ws_endpoints:
                    logger.error("WS 엔드포인트가 구성되지 않았습니다.")
                    await asyncio.sleep(5)
                    continue
                ep = self.ws_endpoints[self.ws_index % len(self.ws_endpoints)]
                logger.info(f"WS 연결 시도: {ep}")
                async with websockets.connect(ep, ping_interval=15, ping_timeout=10) as websocket:
                    self.websocket = websocket
                    last_message = asyncio.get_event_loop().time()
                    backoff = self.ws_backoff_min
                    # 구독 설정
                    await self._setup_subscriptions()
                    # 메시지 수신 루프
                    async for message in websocket:
                        await self._handle_message(message)
                        last_message = asyncio.get_event_loop().time()
                        # 유휴 타임아웃 검사 → 다른 엔드포인트로 회전
                        if self.ws_idle_timeout > 0 and (asyncio.get_event_loop().time() - last_message) > self.ws_idle_timeout:
                            logger.warning("WS 유휴 타임아웃, 엔드포인트 회전")
                            self.ws_index = (self.ws_index + 1) % len(self.ws_endpoints)
                            break
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket 연결 끊김, 재연결/회전 시도...")
                self.ws_index = (self.ws_index + 1) % len(self.ws_endpoints)
                await asyncio.sleep(backoff)
                backoff = min(self.ws_backoff_max, backoff * 1.5)
            except Exception as e:
                logger.error(f"WebSocket 연결 오류: {e}")
                self.ws_index = (self.ws_index + 1) % len(self.ws_endpoints)
                await asyncio.sleep(backoff)
                backoff = min(self.ws_backoff_max, backoff * 1.5)
    
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
                            "topics": [list(self.monitored_events.values())],
                            **({"address": self.log_addresses} if self.log_addresses else {})
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
            
            # Mint/Burn 이벤트 처리 (유동성 변화 → 리저브 변화 가능)
            elif log_data['topics'][0] == self.monitored_events['Mint']:
                await self._handle_mint_event(log_data)
            elif log_data['topics'][0] == self.monitored_events['Burn']:
                await self._handle_burn_event(log_data)
            
            # Uniswap V3 Collect 이벤트 처리 (수수료 수취)
            elif log_data['topics'][0] == self.monitored_events['V3Collect']:
                await self._handle_v3_collect_event(log_data)
            # Uniswap V3 PositionManager 이벤트 처리
            elif log_data['topics'][0] in (
                self.monitored_events['V3PMIncrease'],
                self.monitored_events['V3PMDecrease'],
                self.monitored_events['V3PMCollect'],
            ):
                await self._handle_v3_pm_event(log_data)
            # ERC20 Transfer: LP mint/burn (from/to zero address)
            elif log_data['topics'][0] == self.monitored_events['ERC20Transfer']:
                await self._handle_erc20_transfer(log_data)
            
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
            # Sync 이벤트 데이터 파싱 (데이터 필드에 ABI 인코딩된 reserve0/reserve1 포함)
            data_hex = log_data.get('data')
            if isinstance(data_hex, str) and data_hex.startswith('0x') and len(data_hex) >= 2 + 64 * 2:
                try:
                    r0_hex = data_hex[2:2+64]
                    r1_hex = data_hex[2+64:2+128]
                    reserve0 = int(r0_hex, 16)
                    reserve1 = int(r1_hex, 16)

                    pool_data = {
                        'address': pool_address,
                        'reserve0': reserve0,
                        'reserve1': reserve1,
                        'timestamp': log_data.get('blockNumber', 0)
                    }
                    await self.storage.store_pool_data(pool_address, pool_data)
                    logger.debug(f"풀 리저브 업데이트: {pool_address}")
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Sync 이벤트 처리 실패: {e}")
    
    async def _handle_mint_event(self, log_data: Dict):
        """Mint 이벤트 처리 (LP 발행 → 유동성 증가)"""
        try:
            pool_address = log_data['address']
            logger.debug(f"Mint 이벤트 감지: {pool_address}")
            # 실제 그래프 갱신은 상위 구독자(on_log_event)에서 처리
        except Exception as e:
            logger.error(f"Mint 이벤트 처리 실패: {e}")

    async def _handle_burn_event(self, log_data: Dict):
        """Burn 이벤트 처리 (LP 소각 → 유동성 감소)"""
        try:
            pool_address = log_data['address']
            logger.debug(f"Burn 이벤트 감지: {pool_address}")
            # 실제 그래프 갱신은 상위 구독자(on_log_event)에서 처리
        except Exception as e:
            logger.error(f"Burn 이벤트 처리 실패: {e}")

    async def _handle_v3_collect_event(self, log_data: Dict):
        """Uniswap V3 Collect 이벤트 처리 (수수료 수취)"""
        try:
            pool_address = log_data['address']
            logger.debug(f"V3 Collect 이벤트 감지: {pool_address}")
            # 실제 그래프 갱신은 상위 구독자(on_log_event)에서 처리
        except Exception as e:
            logger.error(f"V3 Collect 이벤트 처리 실패: {e}")

    async def _handle_v3_pm_event(self, log_data: Dict):
        """Uniswap V3 PositionManager 이벤트 처리 (증가/감소/수취)"""
        try:
            addr = log_data.get('address')
            logger.debug(f"V3 PM 이벤트 감지: {addr} topic={log_data['topics'][0]}")
            # 심화 처리는 BlockGraphUpdater에서 필요 시 구현
        except Exception as e:
            logger.error(f"V3 PM 이벤트 처리 실패: {e}")

    async def _handle_erc20_transfer(self, log_data: Dict):
        """ERC20 Transfer 이벤트 처리: LP mint/burn 감지용."""
        try:
            addr = log_data.get('address')
            topics = log_data.get('topics', [])
            if len(topics) < 3:
                return
            from_addr = '0x' + topics[1][-40:]
            to_addr = '0x' + topics[2][-40:]
            if from_addr.lower() == '0x0000000000000000000000000000000000000000' or \
               to_addr.lower() == '0x0000000000000000000000000000000000000000':
                logger.debug(f"LP Mint/Burn 감지: token={addr} from={from_addr} to={to_addr}")
                # 실제 그래프 반영은 BlockGraphUpdater에서 수행
        except Exception as e:
            logger.error(f"ERC20 Transfer 이벤트 처리 실패: {e}")
    
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
