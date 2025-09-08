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
            "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC (수정됨)
            "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
        ]
        
    async def start_detection(self):
        """
        차익거래 탐지를 시작합니다.
        Local Search 로직을 포함하여, 가장 수익성 높은 기회를 찾고,
        그래프 상태를 업데이트한 후, 다시 탐색을 반복합니다.
        """
        self.running = True
        logger.info("차익거래 탐지 시작 (Local Search 활성화)")
        
        while self.running:
            try:
                # 1. 시장 데이터 업데이트 (매 탐색 주기 시작 시)
                await self._update_market_data()
                
                # Local Search 루프
                while True:
                    # 2. 모든 기준 토큰에서 병렬로 차익거래 기회 탐색
                    logger.debug(f"{len(self.base_tokens)}개의 시작점에서 병렬 탐색 시작")
                    tasks = [asyncio.to_thread(self.bellman_ford.find_negative_cycles, base_token) for base_token in self.base_tokens]
                    results = await asyncio.gather(*tasks)

                    all_opportunities = [opp for sublist in results for opp in sublist]
                    
                    best_opportunity = None
                    if all_opportunities:
                        best_opportunity = max(all_opportunities, key=lambda op: op.net_profit)

                    # 3. 수익성 있는 기회가 없으면 Local Search 종료
                    if best_opportunity is None or best_opportunity.net_profit <= 0.001: # 최소 수익 임계값
                        logger.info("수익성 있는 차익거래 기회를 더 이상 찾을 수 없음. 다음 탐색 주기로 넘어갑니다.")
                        break
                        
                    # 4. 최고의 기회 처리 및 그래프 업데이트
                    logger.info(f"최고의 차익거래 기회 발견: 순수익 {best_opportunity.net_profit:.6f} ETH")
                    await self._process_opportunities([best_opportunity])
                    
                    # 5. 그래프 상태 업데이트 (시뮬레이션된 거래 실행)
                    logger.info("그래프 상태를 업데이트하고 Local Search를 계속합니다.")
                    self.market_graph.update_graph_with_trade(best_opportunity.edges, best_opportunity.required_capital)

                # 6. 다음 탐색 주기까지 대기
                await asyncio.sleep(5)
                
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
