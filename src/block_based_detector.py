#!/usr/bin/env python3
"""
Block-Based DeFi Arbitrage Detector
블록 기반 실시간 그래프 상태 업데이트 및 차익거래 탐지

논문 요구사항 구현:
- 매 블록마다 그래프 상태 실시간 업데이트
- 13.5초 블록 시간 내 6.43초 평균 실행 시간 달성
- 96개 protocol actions 처리 가능한 확장성
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable
from datetime import datetime
from web3 import Web3
from concurrent.futures import ThreadPoolExecutor
import json

from src.market_graph import DeFiMarketGraph
from src.bellman_ford_arbitrage import BellmanFordArbitrage  
from src.real_time_collector import RealTimeDataCollector
from src.data_storage import DataStorage
from src.logger import setup_logger
from config.config import config

logger = setup_logger(__name__)

class BlockBasedArbitrageDetector:
    """
    블록 기반 차익거래 탐지기
    논문의 실시간 처리 요구사항을 만족하는 구현
    """
    
    def __init__(self):
        self.market_graph = DeFiMarketGraph()
        self.bellman_ford = BellmanFordArbitrage(self.market_graph)
        self.real_time_collector = RealTimeDataCollector()
        self.storage = DataStorage()
        self.w3 = Web3(Web3.HTTPProvider(config.ethereum_mainnet_rpc))
        
        # 성능 모니터링
        self.execution_times = []
        self.blocks_processed = 0
        self.total_opportunities_found = 0
        
        # 실행 상태
        self.running = False
        self.current_block = None
        self.processing_block = False
        
        # 스레드 풀 (병렬 처리용)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 주요 토큰들 (차익거래 시작점)
        self.base_tokens = [
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "0xA0b86a91c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
            "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI  
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
        ]
        
        # DEX 설정 (96개 protocol actions로 확장 예정)
        self.dex_configs = self._initialize_dex_configs()
        
        # 메트릭스
        self.metrics = {
            'total_blocks_processed': 0,
            'average_execution_time': 0.0,
            'opportunities_per_block': 0.0,
            'graph_update_time': 0.0,
            'negative_cycle_detection_time': 0.0,
            'local_search_time': 0.0
        }
    
    def _initialize_dex_configs(self) -> List[Dict]:
        """
        DEX 설정 초기화
        TODO: 96개 protocol actions로 확장
        """
        return [
            {
                'name': 'uniswap_v2',
                'factory': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',
                'fee': 0.003,
                'enabled': True
            },
            {
                'name': 'uniswap_v3', 
                'factory': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
                'fee': 0.003,  # 가변 수수료
                'enabled': True
            },
            {
                'name': 'sushiswap',
                'factory': '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac', 
                'fee': 0.003,
                'enabled': True
            },
            {
                'name': 'curve',
                'factory': None,  # Registry 기반
                'fee': 0.0004,  # 0.04%
                'enabled': False  # TODO: 구현 예정
            },
            {
                'name': 'balancer',
                'factory': '0x9424B1412450D0f8Fc2255FAf6046b98213B76Bd',
                'fee': 0.005,  # 가변 수수료
                'enabled': False  # TODO: 구현 예정
            }
        ]
    
    async def start_detection(self):
        """블록 기반 차익거래 탐지 시작"""
        self.running = True
        logger.info("=== 블록 기반 차익거래 탐지 시작 ===")
        logger.info(f"목표: 평균 실행 시간 6.43초 이내 (블록 시간 13.5초)")
        
        # WebSocket 구독 설정
        await self._setup_block_subscriptions()
        
        # 실시간 데이터 수집기 시작
        collection_task = asyncio.create_task(
            self.real_time_collector.start_websocket_listener()
        )
        
        # 메인 탐지 루프
        try:
            while self.running:
                await asyncio.sleep(0.1)  # 짧은 대기로 응답성 유지
                
        except Exception as e:
            logger.error(f"탐지 시스템 오류: {e}")
        finally:
            collection_task.cancel()
            self.executor.shutdown(wait=False)
    
    async def _setup_block_subscriptions(self):
        """블록 구독 설정"""
        # 새 블록 구독
        await self.real_time_collector.subscribe_to_blocks(self._on_new_block)
        
        # Swap 이벤트 구독 (그래프 업데이트용)
        await self.real_time_collector.subscribe_to_logs(self._on_swap_event)
        
        logger.info("블록 및 이벤트 구독 설정 완료")
    
    async def _on_new_block(self, block_data: Dict):
        """
        새 블록 처리 - 논문의 핵심 요구사항
        매 블록마다 실시간 그래프 상태 업데이트 및 차익거래 탐지
        """
        if self.processing_block:
            logger.warning("이전 블록 처리 중... 스킵")
            return
        
        self.processing_block = True
        block_number = int(block_data['number'], 16)
        block_hash = block_data['hash']
        
        start_time = time.time()
        
        try:
            logger.info(f"=== 블록 {block_number} 처리 시작 ===")
            
            # 1. 그래프 상태 실시간 업데이트 (논문 요구사항)
            graph_update_start = time.time()
            await self._update_graph_state_for_block(block_number)
            graph_update_time = time.time() - graph_update_start
            
            # 2. 병렬 차익거래 탐지 (각 base token별)
            detection_start = time.time()
            all_opportunities = await self._parallel_arbitrage_detection()
            detection_time = time.time() - detection_start
            
            # 3. 기회 처리 및 저장
            if all_opportunities:
                await self._process_block_opportunities(
                    block_number, block_hash, all_opportunities
                )
            
            # 4. 성능 메트릭 업데이트
            total_time = time.time() - start_time
            self._update_performance_metrics(
                total_time, graph_update_time, detection_time, len(all_opportunities)
            )
            
            # 5. 논문 기준 성능 체크
            if total_time > 6.43:
                logger.warning(
                    f"실행 시간 초과: {total_time:.3f}s > 6.43s 목표"
                )
            else:
                logger.info(
                    f"실행 시간 목표 달성: {total_time:.3f}s < 6.43s"
                )
            
        except Exception as e:
            logger.error(f"블록 {block_number} 처리 오류: {e}")
        finally:
            self.processing_block = False
            self.current_block = block_number
    
    async def _update_graph_state_for_block(self, block_number: int):
        """
        블록별 그래프 상태 실시간 업데이트
        논문 요구사항: "Graph state building을 매 블록마다 실시간 업데이트"
        """
        logger.debug(f"블록 {block_number}: 그래프 상태 업데이트 시작")
        
        # 병렬로 각 DEX의 상태 업데이트
        update_tasks = []
        
        for dex_config in self.dex_configs:
            if dex_config['enabled']:
                task = asyncio.create_task(
                    self._update_dex_state(dex_config, block_number)
                )
                update_tasks.append(task)
        
        # 모든 DEX 상태 업데이트 완료 대기
        if update_tasks:
            await asyncio.gather(*update_tasks, return_exceptions=True)
        
        # 그래프 통계 로깅
        stats = self.market_graph.get_graph_stats()
        logger.debug(
            f"그래프 업데이트 완료: {stats['nodes']}개 노드, "
            f"{stats['edges']}개 엣지"
        )
    
    async def _update_dex_state(self, dex_config: Dict, block_number: int):
        """특정 DEX의 상태 업데이트"""
        try:
            dex_name = dex_config['name']
            
            # 주요 거래 쌍들에 대한 리저브 정보 조회
            major_pairs = [
                ("WETH", "USDC"),
                ("WETH", "DAI"),
                ("WETH", "USDT"), 
                ("USDC", "DAI"),
                ("USDC", "USDT"),
                ("DAI", "USDT")
            ]
            
            for token0_symbol, token1_symbol in major_pairs:
                # TODO: 실제 온체인 데이터 조회 구현
                # pool_data = await self._get_pool_reserves(token0, token1, dex_config)
                # if pool_data:
                #     self.market_graph.add_trading_pair(
                #         token0, token1, dex_name,
                #         pool_data['address'], 
                #         pool_data['reserve0'], 
                #         pool_data['reserve1'],
                #         dex_config['fee']
                #     )
                
                # 임시 모의 데이터 (실제 구현시 제거)
                await self._add_mock_trading_pair(
                    token0_symbol, token1_symbol, dex_name, dex_config['fee']
                )
                
        except Exception as e:
            logger.error(f"DEX {dex_config['name']} 상태 업데이트 오류: {e}")
    
    async def _add_mock_trading_pair(self, token0_symbol: str, token1_symbol: str, 
                                   dex_name: str, fee: float):
        """모의 거래 쌍 추가 (개발/테스트용)"""
        # 토큰 주소 매핑
        token_addresses = {
            "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "USDC": "0xA0b86a91c6218b36c1d19D4a2e9Eb0cE3606eB48", 
            "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
            "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7"
        }
        
        token0_addr = token_addresses.get(token0_symbol)
        token1_addr = token_addresses.get(token1_symbol)
        
        if token0_addr and token1_addr:
            # 모의 리저브 데이터 (실제로는 온체인에서 조회)
            import random
            reserve0 = random.uniform(100, 10000)  # 100-10000 ETH
            reserve1 = random.uniform(100000, 50000000)  # 100K-50M USDC/DAI/USDT
            
            # 실제 환율 반영 (ETH = $2000 기준)
            if token0_symbol == "WETH":
                reserve1 = reserve0 * 2000 * random.uniform(0.95, 1.05)
            
            pool_address = f"0x{hash((token0_addr, token1_addr, dex_name)) % (16**40):040x}"
            
            self.market_graph.add_trading_pair(
                token0_addr, token1_addr, dex_name,
                pool_address, reserve0, reserve1, fee
            )
    
    async def _parallel_arbitrage_detection(self) -> List:
        """병렬 차익거래 탐지"""
        detection_tasks = []
        
        for base_token in self.base_tokens:
            task = asyncio.create_task(
                self._detect_opportunities_for_token(base_token)
            )
            detection_tasks.append(task)
        
        # 모든 토큰에 대한 탐지 완료 대기
        results = await asyncio.gather(*detection_tasks, return_exceptions=True)
        
        # 결과 통합
        all_opportunities = []
        for result in results:
            if isinstance(result, list):
                all_opportunities.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"병렬 탐지 오류: {result}")
        
        return all_opportunities
    
    async def _detect_opportunities_for_token(self, base_token: str) -> List:
        """특정 토큰에 대한 차익거래 기회 탐지"""
        loop = asyncio.get_event_loop()
        
        # CPU 집약적 작업을 별도 스레드에서 실행
        opportunities = await loop.run_in_executor(
            self.executor,
            self.bellman_ford.find_negative_cycles,
            base_token,
            4  # max_path_length
        )
        
        return opportunities
    
    async def _process_block_opportunities(self, block_number: int, 
                                        block_hash: str, opportunities: List):
        """블록에서 발견된 기회들 처리"""
        logger.info(
            f"블록 {block_number}: {len(opportunities)}개 차익거래 기회 발견"
        )
        
        total_revenue = 0
        processed_count = 0
        
        for opp in opportunities:
            if opp.net_profit > 0.001:  # 최소 수익 임계값
                # 기회 정보 로깅
                logger.info(
                    f"  기회: {' -> '.join(opp.path)} "
                    f"수익: {opp.net_profit:.6f} ETH "
                    f"신뢰도: {opp.confidence:.2f}"
                )
                
                # 데이터베이스 저장
                await self.storage.store_arbitrage_opportunity({
                    'block_number': block_number,
                    'block_hash': block_hash,
                    'timestamp': datetime.now().isoformat(),
                    'path': opp.path,
                    'profit_ratio': opp.profit_ratio,
                    'net_profit': opp.net_profit,
                    'required_capital': opp.required_capital,
                    'confidence': opp.confidence,
                    'dexes': [edge.dex for edge in opp.edges]
                })
                
                total_revenue += opp.net_profit
                processed_count += 1
        
        if processed_count > 0:
            logger.info(
                f"블록 {block_number} 처리 완료: {processed_count}개 기회, "
                f"총 예상 수익: {total_revenue:.6f} ETH"
            )
        
        self.total_opportunities_found += processed_count
    
    async def _on_swap_event(self, log_data: Dict):
        """Swap 이벤트 처리 - 그래프 상태 즉시 업데이트"""
        try:
            pool_address = log_data['address']
            
            # TODO: 실제 Swap 이벤트 파싱 및 그래프 업데이트
            # swap_data = parse_swap_event(log_data)
            # self.market_graph.update_pool_data(
            #     pool_address, 
            #     swap_data['new_reserve0'], 
            #     swap_data['new_reserve1']
            # )
            
            logger.debug(f"Swap 이벤트 처리: {pool_address}")
            
        except Exception as e:
            logger.error(f"Swap 이벤트 처리 오류: {e}")
    
    def _update_performance_metrics(self, total_time: float, 
                                  graph_update_time: float,
                                  detection_time: float, 
                                  opportunities_count: int):
        """성능 메트릭 업데이트"""
        self.execution_times.append(total_time)
        self.blocks_processed += 1
        
        # 최근 100블록의 평균 성능 계산
        recent_times = self.execution_times[-100:]
        avg_execution_time = sum(recent_times) / len(recent_times)
        
        # 메트릭 업데이트
        self.metrics.update({
            'total_blocks_processed': self.blocks_processed,
            'average_execution_time': avg_execution_time,
            'opportunities_per_block': self.total_opportunities_found / max(self.blocks_processed, 1),
            'graph_update_time': graph_update_time,
            'negative_cycle_detection_time': detection_time,
            'current_execution_time': total_time
        })
        
        # 100블록마다 성능 보고
        if self.blocks_processed % 100 == 0:
            self._log_performance_report()
    
    def _log_performance_report(self):
        """성능 보고서 출력"""
        logger.info("=== 성능 보고서 (최근 100블록) ===")
        logger.info(f"평균 실행 시간: {self.metrics['average_execution_time']:.3f}s")
        logger.info(f"목표 대비: {self.metrics['average_execution_time']:.3f}s / 6.43s")
        logger.info(f"블록당 평균 기회: {self.metrics['opportunities_per_block']:.2f}개")
        logger.info(f"총 처리 블록: {self.metrics['total_blocks_processed']}개")
        logger.info(f"총 발견 기회: {self.total_opportunities_found}개")
        
        # 성능 목표 달성 여부
        target_achieved = self.metrics['average_execution_time'] <= 6.43
        status = "✅ 달성" if target_achieved else "❌ 미달성"
        logger.info(f"논문 성능 기준: {status}")
    
    def get_metrics(self) -> Dict:
        """현재 성능 메트릭 반환"""
        return self.metrics.copy()
    
    def stop_detection(self):
        """탐지 중지"""
        self.running = False
        self.real_time_collector.stop()
        self.executor.shutdown(wait=True)
        logger.info("블록 기반 차익거래 탐지 중지")

# 사용 예시
async def main():
    detector = BlockBasedArbitrageDetector()
    
    try:
        await detector.start_detection()
    except KeyboardInterrupt:
        detector.stop_detection()
        logger.info("프로그램 종료")

if __name__ == "__main__":
    asyncio.run(main())