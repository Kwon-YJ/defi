#!/usr/bin/env python3
"""
Enhanced Block-Based DeFi Arbitrage Detector with Ethereum Block Time Guarantee
실시간 블록 기반 차익거래 기회 탐지 및 실행 (이더리움 블록 시간 13.5초 보장)
"""

import asyncio
from typing import List, Dict
from datetime import datetime
from web3 import Web3
import websockets
import json

from src.market_graph import DeFiMarketGraph
from src.bellman_ford_arbitrage import BellmanFordArbitrage
from src.protocol_actions import ProtocolActionsManager
from src.token_manager import TokenManager
from src.data_storage import DataStorage
from src.performance_monitor import PerformanceMonitor
from src.logger import setup_logger
from config.config import config

logger = setup_logger(__name__)

class EnhancedBlockBasedArbitrageDetector:
    def __init__(self):
        self.market_graph = DeFiMarketGraph()
        self.bellman_ford = BellmanFordArbitrage(self.market_graph)
        self.protocol_manager = ProtocolActionsManager(self.market_graph)
        self.token_manager = TokenManager()
        self.storage = DataStorage()
        self.performance_monitor = PerformanceMonitor()
        self.running = False
        
        # Ethereum block time constants
        self.ETHEREUM_BLOCK_TIME = 13.5  # seconds
        self.TARGET_PROCESSING_TIME = 6.43  # seconds (from paper)
        self.WARNING_THRESHOLD = 10.0  # seconds (warning when approaching block time)
        self.CRITICAL_THRESHOLD = 12.0  # seconds (critical when very close to block time)
        
        # WebSocket connection
        self.websocket = None
        self.ws_url = config.ethereum_mainnet_ws or "wss://ethereum-rpc.publicnode.com"
        
        # Timeout settings for processing
        self.processing_timeout = 12.0  # seconds - timeout for processing a single block
        
        # Initialize supported tokens (expand from 4 to 25 assets as per paper)
        self._initialize_supported_tokens()
        
        # Get base tokens for arbitrage detection (from the 25 supported assets)
        self.base_tokens = self.token_manager.get_supported_tokens()
        
        # Log the number of protocol actions and supported tokens
        logger.info(f"Initialized with {self.protocol_manager.get_total_action_count()} protocol actions")
        logger.info(f"Initialized with {self.token_manager.get_token_count()} supported tokens")
        
        # Performance tracking
        self.block_processing_times = []  # Track recent block processing times
        self.max_recent_blocks = 100  # Keep track of last 100 blocks
        
    def _initialize_supported_tokens(self):
        """Initialize the supported tokens to match the paper's 25 assets."""
        # Add all 25 assets as supported tokens
        for token_symbol in self.token_manager.get_all_asset_symbols():
            self.token_manager.add_supported_token(token_symbol)
    
    async def start_block_based_detection(self):
        """블록 기반 차익거래 탐지 시작 (이더리움 블록 시간 보장)"""
        self.running = True
        logger.info(f"블록 기반 차익거래 탐지 시작 (이더리움 블록 시간 {self.ETHEREUM_BLOCK_TIME}초 보장)")
        
        # 연결 시도
        connection_attempts = 0
        max_attempts = 5
        
        while self.running and connection_attempts < max_attempts:
            try:
                logger.info(f"WebSocket에 연결 시도 중: {self.ws_url}")
                async with websockets.connect(self.ws_url) as websocket:
                    self.websocket = websocket
                    connection_attempts = 0  # 성공 시 카운트 리셋
                    
                    # 새 블록 구독
                    await self._subscribe_to_new_blocks()
                    
                    # 메시지 수신 루프
                    async for message in websocket:
                        await self._handle_websocket_message(message)
                        
            except websockets.exceptions.ConnectionClosed:
                connection_attempts += 1
                logger.warning(f"WebSocket 연결 끊김, 재연결 시도... (시도 {connection_attempts}/{max_attempts})")
                await asyncio.sleep(5)
            except Exception as e:
                connection_attempts += 1
                logger.error(f"WebSocket 연결 오류: {e}")
                await asyncio.sleep(10)
        
        logger.info("블록 기반 차익거래 탐지 종료")
    
    async def _subscribe_to_new_blocks(self):
        """새 블록 구독 설정"""
        try:
            subscription_request = {
                "id": 1,
                "method": "eth_subscribe",
                "params": ["newHeads"]
            }
            await self.websocket.send(json.dumps(subscription_request))
            logger.info("새 블록 구독 설정 완료")
        except Exception as e:
            logger.error(f"새 블록 구독 설정 실패: {e}")
    
    async def _handle_websocket_message(self, message: str):
        """WebSocket 메시지 처리"""
        try:
            data = json.loads(message)
            
            # 구독 확인 메시지
            if 'id' in data and 'result' in data:
                logger.info(f"구독 ID {data['id']} 설정됨: {data['result']}")
                return
            
            # 실제 블록 데이터
            if 'params' in data and 'result' in data['params']:
                block_data = data['params']['result']
                if 'number' in block_data and 'hash' in block_data:
                    await self._process_new_block_with_timeout(block_data)
                    
        except json.JSONDecodeError:
            logger.error(f"JSON 파싱 실패: {message}")
        except Exception as e:
            logger.error(f"메시지 처리 실패: {e}")
    
    async def _process_new_block_with_timeout(self, block_data: Dict):
        """타임아웃이 있는 새 블록 처리"""
        block_number = int(block_data['number'], 16)
        block_hash = block_data['hash']
        
        logger.info(f"새 블록 처리 시작: {block_number} ({block_hash})")
        
        try:
            # Create a task for block processing with timeout
            processing_task = asyncio.create_task(self._process_new_block(block_data))
            
            # Wait for processing to complete or timeout
            await asyncio.wait_for(processing_task, timeout=self.processing_timeout)
            
        except asyncio.TimeoutError:
            logger.error(f"블록 {block_number} 처리 시간 초과 ({self.processing_timeout}초)")
            # Log performance metrics even on timeout
            self._log_performance_metrics(block_number, self.processing_timeout, timeout=True)
        except Exception as e:
            logger.error(f"블록 {block_number} 처리 중 오류 발생: {e}")
    
    async def _process_new_block(self, block_data: Dict):
        """새 블록 처리 및 차익거래 탐지"""
        start_time = datetime.now()
        block_number = int(block_data['number'], 16)
        block_hash = block_data['hash']
        
        try:
            # 성능 모니터링 시작
            timer = self.performance_monitor.start_monitoring(f"block_{block_number}")
            
            # 1. 시장 데이터 업데이트
            timer.checkpoint("market_data_update_start")
            await self._update_market_data_with_timeout()
            timer.checkpoint("market_data_update_end")
            
            # 2. 각 기준 토큰에서 차익거래 기회 탐색
            all_opportunities = []
            
            # Process each base token with timeout awareness
            for base_token in self.base_tokens:
                opportunities = self.bellman_ford.find_negative_cycles(base_token)
                
                # 3. 기회 최적화 및 필터링
                for opp in opportunities:
                    if opp.net_profit > 0.001:  # 최소 수익 임계값
                        all_opportunities.append(opp)
            
            timer.checkpoint("opportunity_search_end")
            
            # 4. 기회 저장 및 알림
            if all_opportunities:
                await self._process_opportunities_with_timeout(all_opportunities)
            
            timer.checkpoint("opportunity_processing_end")
            
            # 5. 성능 모니터링 종료
            execution_time = self.performance_monitor.stop_monitoring(f"block_{block_number}")
            
            # Calculate actual processing time
            end_time = datetime.now()
            actual_processing_time = (end_time - start_time).total_seconds()
            
            # 성능 통계 로깅
            if execution_time:
                stats = self.performance_monitor.get_current_stats()
                rating = self.performance_monitor.benchmark.get_performance_rating()
                logger.info(f"블록 {block_number} 처리 완료 in {execution_time:.4f}s (Rating: {rating})")
                
                # 목표 시간을 초과한 경우 경고
                if execution_time > self.TARGET_PROCESSING_TIME:
                    logger.warning(f"실행 시간 ({execution_time:.4f}s)이 목표 시간 ({self.TARGET_PROCESSING_TIME}s)을 초과했습니다")
            
            # 블록 처리 완료 로그
            logger.info(f"블록 {block_number} 처리 완료 - {len(all_opportunities)}개의 차익거래 기회 발견")
            
            # Track processing time for performance analysis
            self._track_processing_time(actual_processing_time)
            
            # Check if we're within Ethereum block time limits
            self._check_ethereum_block_time_compliance(block_number, actual_processing_time)
            
        except Exception as e:
            logger.error(f"블록 {block_number} 처리 중 오류 발생: {e}")
            raise
    
    async def _update_market_data_with_timeout(self):
        """타임아웃이 있는 시장 데이터 업데이트"""
        try:
            # Create a task for market data update with timeout
            update_task = asyncio.create_task(self._update_market_data())
            
            # Wait for update to complete or timeout (half of processing time)
            await asyncio.wait_for(update_task, timeout=self.processing_timeout / 2)
            
        except asyncio.TimeoutError:
            logger.warning("시장 데이터 업데이트 시간 초과")
            raise
        except Exception as e:
            logger.error(f"시장 데이터 업데이트 중 오류 발생: {e}")
            raise
    
    async def _update_market_data(self):
        """시장 데이터 업데이트"""
        # Use the protocol actions manager to update all 96 protocol actions
        await self.protocol_manager.update_all_protocol_pools()
    
    async def _process_opportunities_with_timeout(self, opportunities: List):
        """타임아웃이 있는 기회들 처리"""
        try:
            # Create a task for opportunity processing with timeout
            process_task = asyncio.create_task(self._process_opportunities(opportunities))
            
            # Wait for processing to complete or timeout (quarter of processing time)
            await asyncio.wait_for(process_task, timeout=self.processing_timeout / 4)
            
        except asyncio.TimeoutError:
            logger.warning("기회 처리 시간 초과")
            raise
        except Exception as e:
            logger.error(f"기회 처리 중 오류 발생: {e}")
            raise
    
    async def _process_opportunities(self, opportunities: List):
        """발견된 기회들 처리"""
        logger.info(f"{len(opportunities)}개의 차익거래 기회 발견")
        
        for opp in opportunities:
            # 기회 정보 로깅
            logger.info(
                f"차익거래 기회: {' -> '.join(opp.path)} "
                f"수익률: {opp.profit_ratio:.4f} "
                f"순수익: {opp.net_profit:.6f} ETH "
                f"신뢰도: {opp.confidence:.2f}"
            )
            
            # 데이터베이스에 저장
            await self.storage.store_arbitrage_opportunity({
                'timestamp': datetime.now().isoformat(),
                'path': opp.path,
                'profit_ratio': opp.profit_ratio,
                'net_profit': opp.net_profit,
                'required_capital': opp.required_capital,
                'confidence': opp.confidence,
                'dexes': [edge.dex for edge in opp.edges]
            })
    
    def _track_processing_time(self, processing_time: float):
        """처리 시간 추적"""
        self.block_processing_times.append(processing_time)
        
        # 최근 100개 블록만 유지
        if len(self.block_processing_times) > self.max_recent_blocks:
            self.block_processing_times = self.block_processing_times[-self.max_recent_blocks:]
    
    def _check_ethereum_block_time_compliance(self, block_number: int, processing_time: float):
        """이더리움 블록 시간 준수 확인"""
        # 로그 처리 시간
        self._log_performance_metrics(block_number, processing_time)
        
        # 경고 및 알림
        if processing_time > self.CRITICAL_THRESHOLD:
            logger.critical(f"블록 {block_number} 처리 시간이 매우 위험 수준: {processing_time:.4f}초 "
                          f"(이더리움 블록 시간 {self.ETHEREUM_BLOCK_TIME}초에 근접)")
        elif processing_time > self.WARNING_THRESHOLD:
            logger.warning(f"블록 {block_number} 처리 시간이 경고 수준: {processing_time:.4f}초 "
                         f"(이더리움 블록 시간 {self.ETHEREUM_BLOCK_TIME}초에 근접)")
        elif processing_time > self.TARGET_PROCESSING_TIME:
            logger.info(f"블록 {block_number} 처리 시간이 목표 시간을 초과: {processing_time:.4f}초 "
                       f"(목표: {self.TARGET_PROCESSING_TIME}초)")
        else:
            logger.debug(f"블록 {block_number} 처리 시간이 목표 시간 내: {processing_time:.4f}초")
    
    def _log_performance_metrics(self, block_number: int, processing_time: float, timeout: bool = False):
        """성능 메트릭 로깅"""
        # Calculate statistics
        if self.block_processing_times:
            avg_time = sum(self.block_processing_times) / len(self.block_processing_times)
            min_time = min(self.block_processing_times)
            max_time = max(self.block_processing_times)
            
            # Performance rating based on Ethereum block time
            if avg_time <= self.TARGET_PROCESSING_TIME:
                rating = "Excellent"
            elif avg_time <= self.WARNING_THRESHOLD:
                rating = "Good"
            elif avg_time <= self.CRITICAL_THRESHOLD:
                rating = "Fair"
            elif avg_time <= self.ETHEREUM_BLOCK_TIME:
                rating = "Poor"
            else:
                rating = "Critical"
            
            logger.info(f"블록 {block_number} 처리 시간: {processing_time:.4f}초 "
                       f"(평균: {avg_time:.4f}초, 최소: {min_time:.4f}초, 최대: {max_time:.4f}초) "
                       f"등급: {rating}")
        else:
            logger.info(f"블록 {block_number} 처리 시간: {processing_time:.4f}초")
        
        if timeout:
            logger.error(f"블록 {block_number} 처리 시간 초과로 인해 중단됨")
    
    def get_performance_statistics(self) -> Dict:
        """성능 통계 반환"""
        if not self.block_processing_times:
            return {}
        
        avg_time = sum(self.block_processing_times) / len(self.block_processing_times)
        min_time = min(self.block_processing_times)
        max_time = max(self.block_processing_times)
        
        # Calculate percentage within thresholds
        within_target = sum(1 for t in self.block_processing_times if t <= self.TARGET_PROCESSING_TIME)
        within_warning = sum(1 for t in self.block_processing_times if t <= self.WARNING_THRESHOLD)
        within_critical = sum(1 for t in self.block_processing_times if t <= self.CRITICAL_THRESHOLD)
        within_block_time = sum(1 for t in self.block_processing_times if t <= self.ETHEREUM_BLOCK_TIME)
        
        total_blocks = len(self.block_processing_times)
        
        return {
            'average_processing_time': avg_time,
            'min_processing_time': min_time,
            'max_processing_time': max_time,
            'total_blocks_processed': total_blocks,
            'within_target_percentage': (within_target / total_blocks) * 100,
            'within_warning_percentage': (within_warning / total_blocks) * 100,
            'within_critical_percentage': (within_critical / total_blocks) * 100,
            'within_block_time_percentage': (within_block_time / total_blocks) * 100,
            'ethereum_block_time': self.ETHEREUM_BLOCK_TIME,
            'target_processing_time': self.TARGET_PROCESSING_TIME
        }
    
    def stop_detection(self):
        """탐지 중지"""
        self.running = False
        logger.info("블록 기반 차익거래 탐지 중지 (이더리움 블록 시간 보장)")

# Enhanced ArbitrageDetector with Ethereum block time guarantee
class EnhancedArbitrageDetector:
    def __init__(self):
        self.market_graph = DeFiMarketGraph()
        self.bellman_ford = BellmanFordArbitrage(self.market_graph)
        self.protocol_manager = ProtocolActionsManager(self.market_graph)
        self.token_manager = TokenManager()
        self.storage = DataStorage()
        self.performance_monitor = PerformanceMonitor()
        self.running = False
        
        # Initialize supported tokens (expand from 4 to 25 assets as per paper)
        self._initialize_supported_tokens()
        
        # Get base tokens for arbitrage detection (from the 25 supported assets)
        self.base_tokens = self.token_manager.get_supported_tokens()
        
        # Log the number of protocol actions and supported tokens
        logger.info(f"Initialized with {self.protocol_manager.get_total_action_count()} protocol actions")
        logger.info(f"Initialized with {self.token_manager.get_token_count()} supported tokens")
        
        # Enhanced block 기반 처리기 추가
        self.enhanced_block_detector = EnhancedBlockBasedArbitrageDetector()
        
    def _initialize_supported_tokens(self):
        """Initialize the supported tokens to match the paper's 25 assets."""
        # Add all 25 assets as supported tokens
        for token_symbol in self.token_manager.get_all_asset_symbols():
            self.token_manager.add_supported_token(token_symbol)
    
    async def start_detection(self):
        """차익거래 탐지 시작 (기존 5초 딜레이 방식)"""
        self.running = True
        logger.info("차익거래 탐지 시작 (5초 딜레이 방식)")
        
        while self.running:
            try:
                # 성능 모니터링 시작
                timer = self.performance_monitor.start_monitoring("arbitrage_detection_cycle")
                
                # 1. 시장 데이터 업데이트
                timer.checkpoint("market_data_update_start")
                await self._update_market_data()
                timer.checkpoint("market_data_update_end")
                
                # 2. 각 기준 토큰에서 차익거래 기회 탐색
                all_opportunities = []
                
                for base_token in self.base_tokens:
                    opportunities = self.bellman_ford.find_negative_cycles(base_token)
                    
                    # 3. 기회 최적화 및 필터링
                    for opp in opportunities:
                        if opp.net_profit > 0.001:  # 최소 수익 임계값
                            all_opportunities.append(opp)
                
                timer.checkpoint("opportunity_search_end")
                
                # 4. 기회 저장 및 알림
                if all_opportunities:
                    await self._process_opportunities(all_opportunities)
                
                timer.checkpoint("opportunity_processing_end")
                
                # 5. 성능 모니터링 종료
                execution_time = self.performance_monitor.stop_monitoring("arbitrage_detection_cycle")
                
                # 성능 통계 로깅
                if execution_time:
                    stats = self.performance_monitor.get_current_stats()
                    rating = self.performance_monitor.benchmark.get_performance_rating()
                    logger.info(f"Detection cycle completed in {execution_time:.4f}s (Rating: {rating})")
                    
                    # 목표 시간을 초과한 경우 경고
                    if execution_time > 6.43:
                        logger.warning(f"Execution time ({execution_time:.4f}s) exceeds target (6.43s)")
                
                # 6. 잠시 대기 (너무 자주 실행하지 않도록)
                await asyncio.sleep(5)  # 5초 간격
                
            except Exception as e:
                logger.error(f"탐지 루프 오류: {e}")
                await asyncio.sleep(10)
    
    async def start_enhanced_block_based_detection(self):
        """향상된 블록 기반 차익거래 탐지 시작 (이더리움 블록 시간 13.5초 보장)"""
        logger.info("향상된 블록 기반 차익거래 탐지 시작 (이더리움 블록 시간 13.5초 보장)")
        await self.enhanced_block_detector.start_block_based_detection()
    
    async def _update_market_data(self):
        """시장 데이터 업데이트"""
        # Use the protocol actions manager to update all 96 protocol actions
        await self.protocol_manager.update_all_protocol_pools()
    
    async def _process_opportunities(self, opportunities: List):
        """발견된 기회들 처리"""
        logger.info(f"{len(opportunities)}개의 차익거래 기회 발견")
        
        for opp in opportunities:
            # 기회 정보 로깅
            logger.info(
                f"차익거래 기회: {' -> '.join(opp.path)} "
                f"수익률: {opp.profit_ratio:.4f} "
                f"순수익: {opp.net_profit:.6f} ETH "
                f"신뢰도: {opp.confidence:.2f}"
            )
            
            # 데이터베이스에 저장
            await self.storage.store_arbitrage_opportunity({
                'timestamp': datetime.now().isoformat(),
                'path': opp.path,
                'profit_ratio': opp.profit_ratio,
                'net_profit': opp.net_profit,
                'required_capital': opp.required_capital,
                'confidence': opp.confidence,
                'dexes': [edge.dex for edge in opp.edges]
            })
    
    def stop_detection(self):
        """탐지 중지"""
        self.running = False
        self.enhanced_block_detector.stop_detection()
        logger.info("차익거래 탐지 중지 (이더리움 블록 시간 보장)")

async def main():
    """메인 실행 함수"""
    detector = EnhancedArbitrageDetector()
    
    try:
        # 향상된 블록 기반 탐지 모드로 실행
        await detector.start_enhanced_block_based_detection()
    except KeyboardInterrupt:
        detector.stop_detection()
        logger.info("프로그램 종료")

if __name__ == "__main__":
    asyncio.run(main())