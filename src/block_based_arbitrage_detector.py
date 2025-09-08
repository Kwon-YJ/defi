#!/usr/bin/env python3
"""
Block-Based DeFi Arbitrage Detector
실시간 블록 기반 차익거래 기회 탐지 및 실행
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

class BlockBasedArbitrageDetector:
    def __init__(self):
        self.market_graph = DeFiMarketGraph()
        self.bellman_ford = BellmanFordArbitrage(self.market_graph)
        self.protocol_manager = ProtocolActionsManager(self.market_graph)
        self.token_manager = TokenManager()
        self.storage = DataStorage()
        self.performance_monitor = PerformanceMonitor()
        self.running = False
        
        # WebSocket connection
        self.websocket = None
        self.ws_url = config.ethereum_mainnet_ws or "wss://ethereum-rpc.publicnode.com"
        
        # Initialize supported tokens (expand from 4 to 25 assets as per paper)
        self._initialize_supported_tokens()
        
        # Get base tokens for arbitrage detection (from the 25 supported assets)
        self.base_tokens = self.token_manager.get_supported_tokens()
        
        # Log the number of protocol actions and supported tokens
        logger.info(f"Initialized with {self.protocol_manager.get_total_action_count()} protocol actions")
        logger.info(f"Initialized with {self.token_manager.get_token_count()} supported tokens")
        
    def _initialize_supported_tokens(self):
        """Initialize the supported tokens to match the paper's 25 assets."""
        # Add all 25 assets as supported tokens
        for token_symbol in self.token_manager.get_all_asset_symbols():
            self.token_manager.add_supported_token(token_symbol)
    
    async def start_block_based_detection(self):
        """블록 기반 차익거래 탐지 시작"""
        self.running = True
        logger.info("블록 기반 차익거래 탐지 시작")
        
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
                    await self._process_new_block(block_data)
                    
        except json.JSONDecodeError:
            logger.error(f"JSON 파싱 실패: {message}")
        except Exception as e:
            logger.error(f"메시지 처리 실패: {e}")
    
    async def _process_new_block(self, block_data: Dict):
        """새 블록 처리 및 차익거래 탐지"""
        try:
            block_number = int(block_data['number'], 16)
            block_hash = block_data['hash']
            
            logger.info(f"새 블록 처리 시작: {block_number} ({block_hash})")
            
            # 성능 모니터링 시작
            timer = self.performance_monitor.start_monitoring(f"block_{block_number}")
            
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
            execution_time = self.performance_monitor.stop_monitoring(f"block_{block_number}")
            
            # 성능 통계 로깅
            if execution_time:
                stats = self.performance_monitor.get_current_stats()
                rating = self.performance_monitor.benchmark.get_performance_rating()
                logger.info(f"블록 {block_number} 처리 완료 in {execution_time:.4f}s (Rating: {rating})")
                
                # 목표 시간을 초과한 경우 경고
                if execution_time > 6.43:
                    logger.warning(f"실행 시간 ({execution_time:.4f}s)이 목표 시간 (6.43s)을 초과했습니다")
            
            # 블록 처리 완료 로그
            logger.info(f"블록 {block_number} 처리 완료 - {len(all_opportunities)}개의 차익거래 기회 발견")
            
        except Exception as e:
            logger.error(f"블록 처리 중 오류 발생: {e}")
    
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
        logger.info("블록 기반 차익거래 탐지 중지")

# 기존 ArbitrageDetector 클래스를 확장하여 블록 기반 처리를 지원하도록 수정
class ArbitrageDetector:
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
        
        # 블록 기반 처리기 추가
        self.block_detector = BlockBasedArbitrageDetector()
        
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
    
    async def start_block_based_detection(self):
        """블록 기반 차익거래 탐지 시작 (논문 기준 실시간 처리)"""
        logger.info("블록 기반 차익거래 탐지 시작 (실시간 블록 처리)")
        await self.block_detector.start_block_based_detection()
    
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
        self.block_detector.stop_detection()
        logger.info("차익거래 탐지 중지")

async def main():
    """메인 실행 함수"""
    detector = ArbitrageDetector()
    
    try:
        # 블록 기반 탐지 모드로 실행
        await detector.start_block_based_detection()
    except KeyboardInterrupt:
        detector.stop_detection()
        logger.info("프로그램 종료")

if __name__ == "__main__":
    asyncio.run(main())