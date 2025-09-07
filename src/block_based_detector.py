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
from src.transaction_pool_monitor import TransactionPoolMonitor
from src.data_storage import DataStorage
from src.logger import setup_logger
from config.config import config
from src.performance_benchmarking import (
    start_benchmarking, end_benchmarking, time_component, 
    get_performance_report
)
from src.error_handler import SystemErrorHandler, ErrorType, ErrorSeverity, error_handler_decorator, circuit_breaker_decorator
from src.defi_recovery_manager import DeFiRecoveryManager, DeFiErrorType

logger = setup_logger(__name__)

class BlockBasedArbitrageDetector:
    """
    블록 기반 차익거래 탐지기
    논문의 실시간 처리 요구사항을 만족하는 구현
    """
    
    def __init__(self):
        # Error handling 및 recovery 시스템 초기화
        self.error_handler = SystemErrorHandler()
        self.recovery_manager = DeFiRecoveryManager(self.error_handler)
        
        self.market_graph = DeFiMarketGraph()
        self.bellman_ford = BellmanFordArbitrage(self.market_graph)
        self.real_time_collector = RealTimeDataCollector()
        self.transaction_pool_monitor = TransactionPoolMonitor()  # 새로 추가
        self.storage = DataStorage()
        self.w3 = Web3(Web3.HTTPProvider(config.ethereum_mainnet_rpc))
        
        # **DYNAMIC GRAPH UPDATE**: 실시간 상태 변화 리스너 설정
        self.market_graph.register_state_change_listener(self._on_graph_state_change)
        
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
            'local_search_time': 0.0,
            'blocks_within_target_time': 0,     # 6.43초 이내 처리 블록 수
            'blocks_exceeding_target_time': 0,  # 6.43초 초과 처리 블록 수
            'ethereum_block_time_violations': 0  # 13.5초 초과 처리 횟수
        }
        
        # 블록 처리 시간 보장 (논문 요구사항)
        self.target_processing_time = 6.43    # 논문 목표
        self.ethereum_block_time = 13.5       # Ethereum 평균 블록 시간
        self.processing_timeout = 12.0        # 처리 타임아웃 (여유 1.5초)
        
        # Error handling 설정
        self._setup_error_recovery()
        
        logger.info("BlockBasedArbitrageDetector 초기화 완료 (with error handling)")
    
    def _setup_error_recovery(self):
        """에러 핸들링 및 복구 시스템 설정"""
        # Backup RPC endpoints 추가
        backup_rpcs = [
            "https://mainnet.infura.io/v3/backup",
            "https://eth-mainnet.alchemyapi.io/v2/backup",
            "https://cloudflare-eth.com"
        ]
        for rpc in backup_rpcs:
            self.recovery_manager.add_backup_rpc_endpoint(rpc)
        
        # Fallback data sources for price feeds
        fallback_sources = {
            "ETH": "coinbase_api",
            "USDC": "compound_oracle", 
            "DAI": "maker_oracle",
            "USDT": "chainlink_oracle"
        }
        for token, source in fallback_sources.items():
            self.recovery_manager.add_fallback_data_source(token, source)
        
        # Alert callback 등록
        self.error_handler.register_alert_callback(self._handle_system_alert)
    
    async def _handle_system_alert(self, error_context, alert_message: str):
        """시스템 알림 처리"""
        try:
            logger.critical(f"SYSTEM ALERT: {alert_message}")
            
            # Critical 에러의 경우 graceful degradation 활성화
            if error_context.severity == ErrorSeverity.CRITICAL:
                self.error_handler.enable_graceful_degradation(error_context.component)
                logger.warning(f"Graceful degradation enabled for {error_context.component}")
            
            # 시스템 건강도가 너무 낮으면 재시작 권고
            if self.error_handler.system_health_score < 20:
                logger.critical("System health critically low - recommending restart")
                
                # Telegram으로 알림 전송 (사용자 요구사항)
                import subprocess
                try:
                    subprocess.run([
                        "python3", "telegram_send.py", 
                        "--msg", f"DeFi 시스템 위험 상태: Health Score {self.error_handler.system_health_score:.1f}. 재시작 필요."
                    ], timeout=5)
                except Exception as telegram_error:
                    logger.error(f"Telegram 알림 전송 실패: {telegram_error}")
            
        except Exception as e:
            logger.error(f"Alert handler 에러: {e}")
    
    def _classify_block_error(self, error: Exception) -> DeFiErrorType:
        """블록 처리 에러 분류"""
        error_message = str(error).lower()
        
        if any(keyword in error_message for keyword in ['timeout', 'time out']):
            return DeFiErrorType.ARBITRAGE_OPPORTUNITY_EXPIRED
        elif any(keyword in error_message for keyword in ['price', 'rate', 'feed']):
            return DeFiErrorType.PRICE_FEED_ERROR
        elif any(keyword in error_message for keyword in ['liquidity', 'insufficient']):
            return DeFiErrorType.LIQUIDITY_ERROR
        elif any(keyword in error_message for keyword in ['gas', 'transaction']):
            return DeFiErrorType.TRANSACTION_FAILED
        elif any(keyword in error_message for keyword in ['graph', 'inconsist']):
            return DeFiErrorType.GRAPH_INCONSISTENCY
        elif any(keyword in error_message for keyword in ['protocol', 'unavailable']):
            return DeFiErrorType.PROTOCOL_UNAVAILABLE
        else:
            return DeFiErrorType.GRAPH_INCONSISTENCY  # Default fallback
    
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
        
        # 트랜잭션 풀 모니터링 시작 (논문 요구사항)
        txpool_task = asyncio.create_task(
            self.transaction_pool_monitor.start_monitoring()
        )
        
        # 트랜잭션 풀 모니터의 상태 변화 리스너 연결
        self.transaction_pool_monitor.register_state_change_listener(
            self._on_mempool_state_change
        )
        
        # 메인 탐지 루프
        try:
            while self.running:
                await asyncio.sleep(0.1)  # 짧은 대기로 응답성 유지
                
        except Exception as e:
            # 전체 시스템 오류 처리
            try:
                await self.recovery_manager.handle_defi_error(
                    error=e,
                    component="main_detection_loop",
                    defi_error_type=DeFiErrorType.GRAPH_INCONSISTENCY,
                    metadata={'system_state': 'main_loop_failure'}
                )
                
                # Critical system failure의 경우 재시작 권고
                if self.error_handler.system_health_score < 10:
                    import subprocess
                    try:
                        subprocess.run([
                            "python3", "telegram_send.py", 
                            "--msg", "DeFi 시스템 Critical Failure 발생. 즉시 재시작이 필요합니다."
                        ], timeout=5)
                        
                        # pkill로 무한루프 중단 (사용자 요구사항)
                        logger.critical("Critical failure detected - initiating system shutdown")
                        subprocess.run(["pkill", "-f", "claude"], timeout=5)
                        
                    except Exception as emergency_error:
                        logger.error(f"Emergency shutdown failed: {emergency_error}")
                        
            except Exception as recovery_error:
                logger.error(f"Main loop recovery handling failed: {recovery_error}")
            
            logger.error(f"탐지 시스템 오류: {e}")
            
        finally:
            # 리소스 정리
            try:
                collection_task.cancel()
                txpool_task.cancel()
                self.executor.shutdown(wait=False)
                
                # Error handler 종료
                self.error_handler.shutdown()
                
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {cleanup_error}")
    
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
        Enhanced with comprehensive error handling and recovery
        """
        if self.processing_block:
            logger.warning("이전 블록 처리 중... 스킵")
            return
        
        self.processing_block = True
        block_number = None
        block_hash = None
        
        try:
            # Block data validation with error handling
            try:
                block_number = int(block_data['number'], 16)
                block_hash = block_data['hash']
            except (KeyError, ValueError, TypeError) as e:
                await self.recovery_manager.handle_defi_error(
                    error=e,
                    component="block_processor",
                    defi_error_type=DeFiErrorType.GRAPH_INCONSISTENCY,
                    metadata={'block_data': str(block_data)}
                )
                return
            
            # Create system backup before processing
            try:
                self.recovery_manager.create_system_backup({
                    'block_number': block_number,
                    'processing_state': 'starting',
                    'graph_state': 'current'
                })
            except Exception as e:
                logger.warning(f"Failed to create system backup: {e}")
        
        # 성능 벤치마킹 시작
        start_benchmarking(block_number)
        
        try:
            logger.info(f"=== 블록 {block_number} 처리 시작 (목표: {self.target_processing_time}초) ===")
            
            # **논문 요구사항**: 처리 시간 제한 (Ethereum 블록 시간 내 처리 보장)
            processing_start_time = time.time()
            
            # 타임아웃을 적용한 처리
            try:
                opportunities_found = 0
                strategies_executed = 0
                total_revenue = 0.0
                
                # 1. 그래프 상태 실시간 업데이트 (논문 요구사항)
                with time_component("graph_building"):
                    await asyncio.wait_for(
                        self._update_graph_state_for_block(block_number),
                        timeout=4.0  # 그래프 업데이트에 최대 4초
                    )
                    
                    # **DYNAMIC GRAPH UPDATE**: 대기 중인 업데이트들 즉시 처리
                    queued_updates = self.market_graph.process_update_queue(max_items=100)
                    if queued_updates > 0:
                        logger.debug(f"블록 {block_number}: {queued_updates}개 동적 업데이트 처리")
                
                # 2. 병렬 차익거래 탐지 (각 base token별)
                with time_component("negative_cycle_detection"):
                    remaining_time = self.processing_timeout - (time.time() - processing_start_time)
                    if remaining_time > 2.0:  # 최소 2초는 남아야 함
                        all_opportunities = await asyncio.wait_for(
                            self._parallel_arbitrage_detection(),
                            timeout=remaining_time - 1.0
                        )
                        opportunities_found = len(all_opportunities)
                    else:
                        logger.warning(f"블록 {block_number}: 시간 부족으로 탐지 스킵")
                        all_opportunities = []
                
                # 3. 기회 처리 및 저장
                if all_opportunities:
                    remaining_time = self.processing_timeout - (time.time() - processing_start_time)
                    if remaining_time > 1.0:
                        with time_component("validation"):
                            strategies_executed, total_revenue = await asyncio.wait_for(
                                self._process_block_opportunities(
                                    block_number, block_hash, all_opportunities
                                ),
                                timeout=remaining_time - 0.5
                            )
                    else:
                        logger.warning(f"블록 {block_number}: 시간 부족으로 처리 스킵")
                        
            except asyncio.TimeoutError:
                processing_time = time.time() - processing_start_time
                logger.error(
                    f"🚨 블록 {block_number} 처리 타임아웃 ({processing_time:.2f}s > {self.processing_timeout}s)"
                )
                self.metrics['ethereum_block_time_violations'] += 1
                opportunities_found = 0
                strategies_executed = 0
                total_revenue = 0.0
            
            # 성능 벤치마킹 완료
            metrics = end_benchmarking(
                opportunities_found=opportunities_found,
                strategies_executed=strategies_executed,
                total_revenue=total_revenue,
                gas_cost=0.02  # 예상 가스 비용
            )
            
            # 논문 기준 성능 체크
            if metrics.total_execution_time > self.target_processing_time:
                logger.warning(
                    f"⚠️ 실행 시간 초과: {metrics.total_execution_time:.3f}s > {self.target_processing_time}s 목표"
                )
                self.metrics['blocks_exceeding_target_time'] += 1
            else:
                logger.info(
                    f"✅ 실행 시간 목표 달성: {metrics.total_execution_time:.3f}s < {self.target_processing_time}s"
                )
                self.metrics['blocks_within_target_time'] += 1
            
            # Ethereum 블록 시간 체크
            if metrics.total_execution_time > self.ethereum_block_time:
                logger.error(
                    f"🚨 Ethereum 블록 시간 초과: {metrics.total_execution_time:.3f}s > {self.ethereum_block_time}s"
                )
                self.metrics['ethereum_block_time_violations'] += 1
            
        except Exception as e:
            # 블록 처리 에러를 DeFi Recovery Manager로 전달
            try:
                await self.recovery_manager.handle_defi_error(
                    error=e,
                    component="block_processor",
                    defi_error_type=self._classify_block_error(e),
                    affected_tokens=self.base_tokens,
                    affected_protocols=[dex['name'] for dex in self.dex_configs],
                    metadata={
                        'block_number': block_number,
                        'block_hash': block_hash,
                        'processing_stage': 'main_processing'
                    }
                )
            except Exception as recovery_error:
                logger.error(f"Recovery handling failed: {recovery_error}")
            
            logger.error(f"블록 {block_number} 처리 오류: {e}")
            
            # 오류 시에도 벤치마킹 완료
            try:
                end_benchmarking(opportunities_found=0, strategies_executed=0)
            except Exception as bench_error:
                logger.error(f"Benchmarking cleanup failed: {bench_error}")
                
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
            # DEX 상태 업데이트 에러를 복구 시스템으로 전달
            try:
                await self.recovery_manager.handle_defi_error(
                    error=e,
                    component=f"dex_updater_{dex_name}",
                    defi_error_type=DeFiErrorType.PROTOCOL_UNAVAILABLE,
                    affected_protocols=[dex_name],
                    metadata={'block_number': block_number, 'dex_config': dex_config}
                )
            except Exception as recovery_error:
                logger.error(f"DEX recovery handling failed: {recovery_error}")
            
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
        # Local search는 Bellman-Ford 내부에서 수행됨
        with time_component("local_search"):
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
        """Swap 이벤트 처리 - 동적 그래프 상태 즉시 업데이트"""
        try:
            pool_address = log_data['address']
            
            # **DYNAMIC GRAPH UPDATE**: 큐를 통한 즉시 업데이트
            # 실제 구현에서는 log_data에서 새로운 리저브 정보를 파싱
            # 여기서는 임시로 모의 데이터 사용
            import random
            mock_reserve0 = random.uniform(100, 10000)
            mock_reserve1 = random.uniform(100000, 50000000)
            
            # 높은 우선순위로 업데이트 큐에 추가
            self.market_graph.queue_update('pool_update', {
                'pool_address': pool_address,
                'reserve0': mock_reserve0,
                'reserve1': mock_reserve1,
                'source': 'swap_event',
                'tx_hash': log_data.get('transactionHash')
            }, priority=1)  # 최고 우선순위
            
            logger.debug(f"동적 그래프 업데이트 - Swap 이벤트: {pool_address}")
            
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
    
    async def _on_graph_state_change(self, notification: Dict):
        """
        그래프 상태 변화 리스너 콜백
        동적 그래프 업데이트 시 호출됨
        """
        change_type = notification['type']
        change_data = notification['data']
        graph_hash = notification['graph_hash']
        
        logger.debug(f"그래프 상태 변화 감지: {change_type} (hash: {graph_hash[:8]}...)")
        
        # 상태 변화 통계 업데이트
        if change_type == 'pool_update':
            updated_pairs = change_data.get('updated_pairs', 0)
            update_time = change_data.get('update_time', 0)
            logger.debug(f"풀 업데이트: {updated_pairs}개 쌍, {update_time:.3f}초")
            
        elif change_type == 'queue_processed':
            processed_count = change_data.get('processed_count', 0)
            logger.debug(f"업데이트 큐 처리: {processed_count}개 완료")
            
        elif change_type == 'auto_detection':
            if change_data.get('changed'):
                logger.info(f"자동 상태 변화 감지: {change_data['previous_hash'][:8]} -> {change_data['current_hash'][:8]}")
    
    async def _on_mempool_state_change(self, change_data: Dict):
        """
        Mempool 상태 변화 리스너 콜백
        트랜잭션 풀 모니터링에서 감지한 상태 변화 처리
        """
        change_type = change_data['type']
        
        if change_type == 'new_block':
            block_number = change_data['block_number']
            logger.debug(f"Mempool에서 새 블록 감지: {block_number}")
            
        elif change_type == 'arbitrage_detected':
            tx_hash = change_data.get('tx_hash', '')
            logger.info(f"Mempool에서 차익거래 트랜잭션 감지: {tx_hash[:10]}...")
            
            # 즉시 그래프 상태 업데이트 트리거
            await self._trigger_immediate_graph_update(f"arbitrage_tx_{tx_hash}")
            
        elif change_type == 'mev_opportunity':
            mev_score = change_data.get('mev_score', 0)
            tx_hash = change_data.get('tx_hash', '')
            logger.info(f"MEV 기회 감지 (점수: {mev_score:.2f}): {tx_hash[:10]}...")
            
            # 높은 MEV 점수면 우선순위 처리
            if mev_score > 0.8:
                await self._trigger_priority_processing(change_data)
    
    async def _trigger_immediate_graph_update(self, trigger_reason: str):
        """즉시 그래프 상태 업데이트 트리거"""
        try:
            logger.debug(f"즉시 그래프 업데이트 트리거: {trigger_reason}")
            
            # 우선순위가 높은 업데이트들 즉시 처리
            processed_updates = self.market_graph.process_update_queue(
                max_items=50, 
                priority_only=True
            )
            
            if processed_updates > 0:
                logger.info(f"즉시 처리된 그래프 업데이트: {processed_updates}개")
                
        except Exception as e:
            logger.error(f"즉시 그래프 업데이트 실패: {e}")
    
    async def _trigger_priority_processing(self, mev_data: Dict):
        """우선순위 처리 트리거"""
        try:
            tx_hash = mev_data.get('tx_hash', '')
            mev_score = mev_data.get('mev_score', 0)
            
            logger.info(f"우선순위 MEV 처리: {tx_hash[:10]}... (점수: {mev_score:.2f})")
            
            # 여기서 즉시 차익거래 탐지 실행 가능
            # 하지만 현재 블록 처리 중이 아닐 때만
            if not self.processing_block:
                logger.info("블록 처리 중이 아니므로 즉시 MEV 분석 시작")
                # 즉시 분석 로직 구현 가능
                
        except Exception as e:
            logger.error(f"우선순위 처리 실패: {e}")
    
    def stop_detection(self):
        """탐지 중지"""
        self.running = False
    
    def get_system_status_with_error_info(self) -> Dict[str, Any]:
        """에러 핸들링 정보를 포함한 전체 시스템 상태"""
        base_metrics = {
            'blocks_processed': self.blocks_processed,
            'average_execution_time': (
                sum(self.execution_times) / len(self.execution_times) 
                if self.execution_times else 0
            ),
            'opportunities_found': self.total_opportunities_found,
            'processing_status': 'running' if self.running else 'stopped',
            'current_block': self.current_block
        }
        
        # Error handling 상태 정보 추가
        error_status = self.error_handler.get_system_status()
        recovery_stats = self.recovery_manager.get_recovery_stats()
        protocol_health = self.recovery_manager.get_protocol_health()
        
        return {
            'basic_metrics': base_metrics,
            'error_handling': error_status,
            'recovery_performance': recovery_stats,
            'protocol_health': protocol_health,
            'system_recommendations': self._generate_system_recommendations(error_status)
        }
    
    def _generate_system_recommendations(self, error_status: Dict[str, Any]) -> List[str]:
        """시스템 상태 기반 권고사항 생성"""
        recommendations = []
        
        health_score = error_status.get('health_score', 100)
        recent_errors = error_status.get('recent_errors', 0)
        degraded_components = error_status.get('degraded_components', [])
        
        if health_score < 20:
            recommendations.append("CRITICAL: System restart required immediately")
        elif health_score < 50:
            recommendations.append("WARNING: System performance degraded - consider restart")
        
        if recent_errors > 20:
            recommendations.append("HIGH: Too many recent errors - check system resources")
        
        if len(degraded_components) > 3:
            recommendations.append("MEDIUM: Multiple components degraded - investigate")
        
        if self.metrics['ethereum_block_time_violations'] > 5:
            recommendations.append("PERFORMANCE: Block processing too slow - optimize")
            
        return recommendations

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