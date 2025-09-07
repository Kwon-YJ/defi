#!/usr/bin/env python3
"""
DeFi Arbitrage Detector - Main execution script
실시간 차익거래 기회 탐지 및 실행
"""

import asyncio
from typing import List, Dict
from datetime import datetime
from src.market_graph import DeFiMarketGraph
from src.bellman_ford_arbitrage import BellmanFordArbitrage
from src.protocol_actions import ProtocolActionsManager
from src.token_manager import TokenManager
from src.data_storage import DataStorage
from src.logger import setup_logger

logger = setup_logger(__name__)

class ArbitrageDetector:
    def __init__(self):
        self.market_graph = DeFiMarketGraph()
        self.bellman_ford = BellmanFordArbitrage(self.market_graph)
        self.protocol_manager = ProtocolActionsManager(self.market_graph)
        self.token_manager = TokenManager()
        self.storage = DataStorage()
        self.running = False
        
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
    
    async def start_detection(self):
        """차익거래 탐지 시작"""
        self.running = True
        logger.info("차익거래 탐지 시작")
        
        while self.running:
            try:
                # 1. 시장 데이터 업데이트
                await self._update_market_data()
                
                # 2. 각 기준 토큰에서 차익거래 기회 탐색
                all_opportunities = []
                
                for base_token in self.base_tokens:
                    opportunities = self.bellman_ford.find_negative_cycles(base_token)
                    
                    # 3. 기회 최적화 및 필터링
                    for opp in opportunities:
                        if opp.net_profit > 0.001:  # 최소 수익 임계값
                            all_opportunities.append(opp)
                
                # 4. 기회 저장 및 알림
                if all_opportunities:
                    await self._process_opportunities(all_opportunities)
                
                # 5. 잠시 대기 (너무 자주 실행하지 않도록)
                await asyncio.sleep(5)  # 5초 간격
                
            except Exception as e:
                logger.error(f"탐지 루프 오류: {e}")
                await asyncio.sleep(10)
    
    async def _update_market_data(self):
        """시장 데이터 업데이트"""
        # Use the protocol actions manager to update all 96 protocol actions
        await self.protocol_manager.update_all_protocol_pools()
    
    async def _update_dex_pools(self, dex_config: Dict):
        """특정 DEX의 풀 데이터 업데이트"""
        # This method is now handled by the protocol actions manager
        pass
    
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
        logger.info("차익거래 탐지 중지")

async def main():
    """메인 실행 함수"""
    detector = ArbitrageDetector()
    
    try:
        await detector.start_detection()
    except KeyboardInterrupt:
        detector.stop_detection()
        logger.info("프로그램 종료")

if __name__ == "__main__":
    asyncio.run(main())