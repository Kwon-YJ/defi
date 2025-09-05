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
from src.data_storage import DataStorage
from src.logger import setup_logger

logger = setup_logger(__name__)

class ArbitrageDetector:
    def __init__(self):
        self.market_graph = DeFiMarketGraph()
        self.bellman_ford = BellmanFordArbitrage(self.market_graph)
        self.storage = DataStorage()
        self.running = False
        
        # 주요 토큰들 (차익거래 시작점)
        self.base_tokens = [
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "0xA0b86a33E6441b8e5c7F5c8b5e8b5e8b5e8b5e8b",  # USDC
            "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
        ]
        
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
