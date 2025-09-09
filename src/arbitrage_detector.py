#!/usr/bin/env python3
"""
DeFi Arbitrage Detector - Main execution script
실시간 차익거래 기회 탐지 및 실행
"""

import asyncio
import time
import hashlib
import os
from typing import List, Dict, Optional
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
        # 상태 변화 감지/즉시 대응 구성
        self._last_state_fingerprint: Optional[str] = None
        self._detect_lock = asyncio.Lock()
        self._event_debounce_sec = float(os.getenv('EVENT_DETECT_DEBOUNCE_SEC', '0.5')) if 'EVENT_DETECT_DEBOUNCE_SEC' in os.environ else 0.5
        self._block_min_interval_sec = float(os.getenv('BLOCK_DETECT_MIN_INTERVAL_SEC', '0')) if 'BLOCK_DETECT_MIN_INTERVAL_SEC' in os.environ else 0.0
        self._last_block_detect_ts: float = 0.0
        self._event_task: Optional[asyncio.Task] = None
        self._event_run_scheduled = False
        
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

        # 블록/로그/펜딩 이벤트 콜백을 먼저 구독한 뒤 WS를 시작하여,
        # 초기 WS 구독 단계에서 pending 포함 모든 구독이 설정되도록 함.
        # 블록 이벤트: 그래프 갱신 이후 탐지 실행
        async def on_new_block(block_data: Dict):
            try:
                bn = int(block_data['number'], 16)
            except Exception:
                bn = None
            # 블록 최소 간격 스로틀링(옵션)
            now = time.time()
            if self._block_min_interval_sec > 0 and (now - self._last_block_detect_ts) < self._block_min_interval_sec:
                return
            self._last_block_detect_ts = now
            await self._run_detection(block_number=bn, reason='block')

        await self.updater.rt.subscribe_to_blocks(on_new_block)

        # 로그 이벤트: 즉시 대응(디바운스) - 그래프는 BlockGraphUpdater가 실시간 갱신
        async def on_log_event(_log: Dict):
            # 너무 자주 호출되는 것을 방지하고, 짧은 시간 창에서 이벤트를 모아서 1회 실행
            if self._event_run_scheduled:
                return
            self._event_run_scheduled = True
            async def _delayed_run():
                try:
                    await asyncio.sleep(max(0.05, self._event_debounce_sec))
                    await self._run_detection(block_number=None, reason='event')
                finally:
                    self._event_run_scheduled = False
            # 백그라운드 태스크로 스케줄
            self._event_task = asyncio.create_task(_delayed_run())

        await self.updater.rt.subscribe_to_logs(on_log_event)

        # 펜딩 트랜잭션 모니터링: 관련 주소(풀/라우터)로 향하는 트랜잭션 감지 시 즉시 탐지
        self._mempool_run_scheduled = False
        self._mempool_task: Optional[asyncio.Task] = None
        self._mempool_debounce_sec = float(os.getenv('MEMPOOL_DEBOUNCE_SEC', '0.4'))

        async def on_pending_tx(tx_hash: str):
            try:
                # 트랜잭션 상세 조회 (HTTP RPC)
                w3 = getattr(self.updater, 'w3', None)
                if w3 is None:
                    return
                tx = None
                try:
                    tx = w3.eth.get_transaction(tx_hash)
                except Exception:
                    return
                if not tx:
                    return
                to_addr = tx.get('to') if isinstance(tx, dict) else getattr(tx, 'to', None)
                if not to_addr:
                    return  # 컨트랙트 생성 등은 스킵
                to_addr = to_addr.lower()

                # 관심 주소 집합: (1) 그래프 기반 풀/LP, (2) 주요 라우터/볼트
                try:
                    pool_addrs = set([a.lower() for a in self.updater._compute_log_addresses()])
                except Exception:
                    pool_addrs = set()
                router_like = {
                    # Uniswap V2 Router02
                    '0x7a250d5630b4cf539739df2c5dacb4c659f2488d',
                    # SushiSwap Router
                    '0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f',
                    # Uniswap V3 SwapRouter
                    '0xe592427a0aece92de3edee1f18e0157c05861564',
                    # Uniswap V3 NonfungiblePositionManager
                    '0xc36442b4a4522e871399cd717abdd847ab11fe88',
                    # Balancer V2 Vault
                    '0xba12222222228d8ba445958a75a0704d566bf2c8',
                }
                interesting = (to_addr in pool_addrs) or (to_addr in router_like)
                if not interesting:
                    return

                # 디바운스 스케줄링으로 탐지 실행
                if self._mempool_run_scheduled:
                    return
                self._mempool_run_scheduled = True

                async def _delayed_mempool_run():
                    try:
                        await asyncio.sleep(max(0.05, self._mempool_debounce_sec))
                        await self._run_detection(block_number=None, reason='mempool')
                    finally:
                        self._mempool_run_scheduled = False

                self._mempool_task = asyncio.create_task(_delayed_mempool_run())
            except Exception as e:
                logger.debug(f"mempool 모니터링 처리 실패: {e}")

        await self.updater.rt.subscribe_to_pending_transactions(on_pending_tx)

        # BlockGraphUpdater 시작 (내부에서 블록/로그 구독 및 그래프 빌드/갱신 수행)
        await self.updater.start()

        # 초기 한번 실행 (상태 지문 저장 포함)
        await self._run_detection(block_number=None, reason='init')
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

    # -------------------- 내부 유틸/핵심 로직 --------------------
    def _compute_graph_fingerprint(self) -> str:
        """현재 그래프 상태에 대한 지문(해시) 계산.
        - 엣지의 (pool_address, from, to, exchange_rate, liquidity)를 정렬된 순서로 반영
        - exchange_rate/liquidity는 과도한 민감도를 줄이기 위해 1e-6 정밀도로 반올림
        """
        items: List[bytes] = []
        try:
            if hasattr(self.market_graph.graph, 'edges'):
                if self.market_graph.graph.is_multigraph():
                    iterator = self.market_graph.graph.edges(keys=True, data=True)
                    for u, v, k, data in iterator:
                        if not isinstance(data, dict):
                            continue
                        pa = str(data.get('pool_address') or '')
                        ex = float(data.get('exchange_rate') or 0.0)
                        liq = float(data.get('liquidity') or 0.0)
                        exr = round(ex, 6)
                        liqr = round(liq, 6)
                        items.append(f"{pa}|{u}|{v}|{exr}|{liqr}".encode())
                else:
                    for u, v, data in self.market_graph.graph.edges(data=True):
                        if not isinstance(data, dict):
                            continue
                        pa = str(data.get('pool_address') or '')
                        ex = float(data.get('exchange_rate') or 0.0)
                        liq = float(data.get('liquidity') or 0.0)
                        exr = round(ex, 6)
                        liqr = round(liq, 6)
                        items.append(f"{pa}|{u}|{v}|{exr}|{liqr}".encode())
        except Exception:
            pass
        items.sort()
        m = hashlib.sha256()
        for b in items:
            m.update(b)
            m.update(b"\x00")
        return m.hexdigest()

    async def _run_detection(self, block_number: Optional[int] = None, reason: str = 'manual'):
        """상태 변화 감지 후 차익거래 탐지 실행.
        - reason: 'init' | 'block' | 'event' | 'manual'
        - 동일 지문이면 블록 기반 탐지는 스킵 (중복 연산 방지)
        - 이벤트 기반은 지문 동일하더라도 짧은 디바운스 이후 1회 실행하여 즉시성 보장
        """
        async with self._detect_lock:
            try:
                fp = self._compute_graph_fingerprint()
            except Exception as e:
                logger.debug(f"상태 지문 계산 실패: {e}")
                fp = None

            # 블록 기반: 상태 변화 없으면 스킵
            if reason == 'block' and fp is not None and self._last_state_fingerprint is not None and fp == self._last_state_fingerprint:
                logger.debug("블록 상태 변화 없음 → 탐지 스킵")
                return

            # 탐지 실행
            try:
                all_opps = []
                for base in self.base_tokens:
                    try:
                        opps = self.bellman_ford.find_negative_cycles(base)
                        if opps:
                            all_opps.extend(opps)
                    except Exception as e:
                        logger.debug(f"탐지 실패(base={base[:6]}): {e}")
                if all_opps:
                    await self._process_opportunities(all_opps)
                else:
                    logger.debug(f"탐지 결과 없음 (reason={reason}, block={block_number})")
            finally:
                if fp is not None:
                    self._last_state_fingerprint = fp

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
