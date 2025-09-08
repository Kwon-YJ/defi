import asyncio
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from web3 import Web3
import networkx as nx

from src.logger import setup_logger
from src.market_graph import DeFiMarketGraph
from src.dex_data_collector import UniswapV2Collector, SushiSwapCollector
from src.action_registry import register_default_actions, ActionRegistry
from src.real_time_collector import RealTimeDataCollector
from config.config import config

logger = setup_logger(__name__)


class BlockGraphUpdater:
    """매 블록마다 그래프 상태를 실시간으로 구축/갱신"""

    def __init__(self, market_graph: DeFiMarketGraph,
                 tokens: Optional[Dict[str, str]] = None,
                 dexes: Optional[List[str]] = None):
        self.graph = market_graph
        self.w3 = Web3(Web3.HTTPProvider(config.ethereum_mainnet_rpc))
        self.collectors: Dict[str, object] = {}
        self.fees: Dict[str, float] = {}
        self.running = False

        # 기본 토큰 셋 (메이저 4)
        self.tokens = tokens or {
            'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            'USDC': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
            'DAI':  '0x6B175474E89094C44Da98b954EedeAC495271d0F',
            'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
        }
        self.dexes = dexes or ['uniswap_v2', 'sushiswap']

        # (역호환) DEX 수집기 초기화 - 유지
        for dex in self.dexes:
            if dex == 'uniswap_v2':
                self.collectors[dex] = UniswapV2Collector(self.w3)
                self.fees[dex] = 0.003
            elif dex == 'sushiswap':
                self.collectors[dex] = SushiSwapCollector(self.w3)
                self.fees[dex] = 0.003
            else:
                logger.warning(f"지원하지 않는 DEX 무시: {dex}")

        # 블록 수신기
        self.rt = RealTimeDataCollector()
        # Protocol Action Registry (확장성)
        self.registry: ActionRegistry = register_default_actions(self.w3)

    def _major_pairs(self) -> List[Tuple[str, str]]:
        """모든 토큰 조합에서 페어 생성"""
        addrs = list(self.tokens.values())
        return list(combinations(addrs, 2))

    async def start(self):
        """블록 구독을 시작하고, 블록마다 그래프를 갱신"""
        if self.running:
            return
        self.running = True

        # Multi-graph 지원: 동일 토큰 쌍에서 여러 DEX 엣지 처리
        if not isinstance(self.graph.graph, nx.MultiDiGraph):
            try:
                self.graph.graph = nx.MultiDiGraph(self.graph.graph)
                logger.info("그래프를 MultiDiGraph로 변환하여 다중 DEX 엣지 지원")
            except Exception as e:
                logger.warning(f"MultiDiGraph 변환 실패(계속 진행): {e}")

        async def on_new_block(block_data):
            try:
                block_number = int(block_data['number'], 16)
                await self.update_via_actions(block_number)
            except Exception as e:
                logger.error(f"블록 갱신 실패: {e}")

        await self.rt.subscribe_to_blocks(on_new_block)

        # 백그라운드로 WS 리스너 실행
        asyncio.create_task(self.rt.start_websocket_listener())

        # 시작 시 1회 초기 빌드
        await self.update_via_actions()

    async def update_all_pairs(self, block_number: Optional[int] = None):
        """등록된 모든 DEX, 주요 페어에 대해 그래프 엣지 갱신"""
        pairs = self._major_pairs()
        updated_edges = 0

        for dex, collector in self.collectors.items():
            fee = self.fees.get(dex, 0.003)
            for token0, token1 in pairs:
                try:
                    pair_address = await collector.get_pair_address(token0, token1)
                    if not pair_address:
                        continue
                    reserve0, reserve1, _ = await collector.get_pool_reserves(pair_address)
                    if reserve0 == 0 or reserve1 == 0:
                        continue

                    # 그래프에 반영 (양방향)
                    self.graph.add_trading_pair(
                        token0=token0,
                        token1=token1,
                        dex=dex,
                        pool_address=pair_address,
                        reserve0=float(reserve0),
                        reserve1=float(reserve1),
                        fee=fee,
                    )
                    updated_edges += 2  # 양방향 추가
                except Exception as e:
                    logger.debug(f"{dex} {token0[:6]}-{token1[:6]} 업데이트 실패: {e}")

        if updated_edges:
            if block_number is not None:
                logger.info(f"블록 {block_number} 그래프 갱신: {updated_edges}개 엣지 업데이트")
            else:
                logger.info(f"그래프 초기 빌드 완료: {updated_edges}개 엣지 추가")

    def stop(self):
        self.running = False
        self.rt.stop()

    async def update_via_actions(self, block_number: Optional[int] = None):
        """Action Registry를 통한 확장형 그래프 갱신 (96개 actions까지 확장 가능)"""
        updated = await self.registry.update_all(self.graph, self.w3, self.tokens, block_number)
        msg = (
            f"블록 {block_number} 그래프 갱신(액션): {updated}개 엣지 업데이트"
            if block_number is not None
            else f"그래프 초기 빌드 완료(액션): {updated}개 엣지 추가"
        )
        logger.info(msg)
