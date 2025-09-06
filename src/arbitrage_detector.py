#!/usr/bin/env python3
"""
DeFi Arbitrage Detector - Main execution script
실시간 차익거래 기회 탐지 및 실행

Updated: 블록 기반 실시간 그래프 상태 업데이트 구현
논문 요구사항: 매 블록마다 그래프 상태 업데이트 (13.5초 블록 시간 내 6.43초 처리)
"""

import asyncio
from typing import List, Dict
from datetime import datetime
from src.market_graph import DeFiMarketGraph
from src.bellman_ford_arbitrage import BellmanFordArbitrage
from src.block_based_detector import BlockBasedArbitrageDetector
from src.data_storage import DataStorage
from src.logger import setup_logger

logger = setup_logger(__name__)

class ArbitrageDetector:
    def __init__(self, use_block_based: bool = True):
        """
        Args:
            use_block_based: True면 블록 기반 탐지 사용, False면 기존 5초 주기 탐지 사용
        """
        self.use_block_based = use_block_based
        
        if use_block_based:
            # 블록 기반 탐지기 사용 (논문 요구사항)
            self.block_detector = BlockBasedArbitrageDetector()
            logger.info("블록 기반 탐지 모드 활성화 (논문 요구사항)")
        else:
            # 기존 방식 (하위 호환성)
            self.market_graph = DeFiMarketGraph()
            self.bellman_ford = BellmanFordArbitrage(self.market_graph)
            self.storage = DataStorage()
            logger.info("기존 5초 주기 탐지 모드 (하위 호환성)")
        
        self.running = False
        
        # 주요 토큰들 (차익거래 시작점)
        self.base_tokens = [
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "0xA0b86a91c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC (수정됨)
            "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
        ]
        
    async def start_detection(self):
        """차익거래 탐지 시작"""
        self.running = True
        
        if self.use_block_based:
            # 블록 기반 탐지 시작 (논문 요구사항)
            logger.info("블록 기반 차익거래 탐지 시작")
            logger.info("목표: 매 블록마다 그래프 상태 실시간 업데이트, 평균 6.43초 처리")
            await self.block_detector.start_detection()
        else:
            # 기존 5초 주기 탐지 (하위 호환성)
            logger.info("기존 5초 주기 차익거래 탐지 시작")
            await self._legacy_detection_loop()
    
    async def _legacy_detection_loop(self):
        """기존 5초 주기 탐지 루프 (하위 호환성)"""
        while self.running:
            try:
                # 1. 시장 데이터 업데이트
                await self._update_market_data()
                
                # 2. 각 기준 토큰에서 차익거래 기회 탐색
                all_opportunities = []
                
                for base_token in self.base_tokens:
                    opportunities = await self._find_negative_cycles_async(base_token)
                    
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
    
    async def _find_negative_cycles_async(self, source_token: str, 
                                        max_path_length: int = 4) -> List:
        """
        비동기 음의 사이클 탐지 - 병렬 local search 포함
        """
        loop = asyncio.get_event_loop()
        
        # Bellman-Ford 알고리즘을 별도 스레드에서 실행 (CPU 집약적)
        opportunities = await loop.run_in_executor(
            None, 
            self.bellman_ford.find_negative_cycles, 
            source_token, 
            max_path_length
        )
        
        return opportunities
    
    def stop_detection(self):
        """탐지 중지"""
        self.running = False
        
        if self.use_block_based:
            self.block_detector.stop_detection()
        
        logger.info("차익거래 탐지 중지")
    
    def get_metrics(self) -> Dict:
        """성능 메트릭 조회"""
        if self.use_block_based:
            return self.block_detector.get_metrics()
        else:
            return {
                'mode': 'legacy',
                'message': '기존 모드에서는 상세 메트릭을 제공하지 않습니다.'
            }

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
