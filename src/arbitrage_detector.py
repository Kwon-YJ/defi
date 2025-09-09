#!/usr/bin/env python3
"""
DeFi Arbitrage Detector - Main execution script
실시간 차익거래 기회 탐지 및 실행
"""

import asyncio
from typing import List, Dict
from datetime import datetime
from src.market_graph import DeFiMarketGraph
from src.block_graph_updater import BlockGraphUpdater
from src.bellman_ford_arbitrage import BellmanFordArbitrage
from src.data_storage import DataStorage
from src.logger import setup_logger

logger = setup_logger(__name__)

class ArbitrageDetector:
    def __init__(self):
        self.market_graph = DeFiMarketGraph()
        self.bellman_ford = BellmanFordArbitrage(self.market_graph)
        self.storage = DataStorage()
        self.running = False
        self.updater: BlockGraphUpdater = BlockGraphUpdater(self.market_graph)
        
        # 주요 토큰들 (차익거래 시작점)
        self.base_tokens = [
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC (수정됨)
            "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
        ]
        
    async def start_detection(self):
        """차익거래 탐지 시작 (블록 기반 루프)"""
        self.running = True
        logger.info("차익거래 탐지 시작 - 블록 기반")

        async def on_new_block(block_data: Dict):
            try:
                bn = int(block_data['number'], 16)
            except Exception:
                bn = None
            await self._run_detection(block_number=bn)

        # BlockGraphUpdater 시작 및 블록 구독
        await self.updater.rt.subscribe_to_blocks(on_new_block)
        await self.updater.start()
        # 초기 한번 실행
        await self._run_detection(block_number=None)
        # 루프 유지
        while self.running:
            await asyncio.sleep(3600)
    
    async def _update_market_data(self):
        """시장 데이터 업데이트"""
        # 주요 DEX들의 풀 데이터 업데이트
        dex_configs = [
            {
                'name': 'uniswap_v2',
                'factory': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',
                'fee': 0.003
            },
            {
                'name': 'sushiswap', 
                'factory': '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac',
                'fee': 0.003
            }
        ]
        
        for dex_config in dex_configs:
            await self._update_dex_pools(dex_config)
    
    async def _update_dex_pools(self, dex_config: Dict):
        """특정 DEX의 풀 데이터 업데이트"""
        # 주요 토큰 쌍들의 풀 데이터 조회 및 업데이트
        major_pairs = [
            ("WETH", "USDC"),
            ("WETH", "DAI"), 
            ("WETH", "USDT"),
            ("USDC", "DAI"),
            ("USDC", "USDT"),
            ("DAI", "USDT")
        ]
        
        for token0_symbol, token1_symbol in major_pairs:
            # 실제 구현에서는 토큰 주소 매핑 필요
            # pool_data = await get_pool_data(token0, token1, dex_config)
            # self.market_graph.add_trading_pair(...)
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
