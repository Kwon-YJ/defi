import asyncio
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from web3 import Web3
import networkx as nx

from src.logger import setup_logger
from src.market_graph import DeFiMarketGraph
from src.dex_data_collector import UniswapV2Collector, SushiSwapCollector
from src.dex_uniswap_v3_collector import UniswapV3Collector
from src.action_registry import register_default_actions, ActionRegistry
from src.real_time_collector import RealTimeDataCollector
from src.graph_pruner import prune_graph
from src.memory_compactor import compact_graph_attributes
from src.synth_tokens import lp_v3
from src.edge_meta import set_edge_meta
from src.data_storage import DataStorage
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
        # V3 전용 수집기 (fee accrual 추정 등에 활용)
        self.v3_collector = UniswapV3Collector(self.w3)
        # V3 포지션 매니저 (포지션 단위 조회)
        self.v3_pm = self._init_position_manager()
        # V3 통계 저장소(EMA)
        self.storage = DataStorage()
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
                # 블록 단위로 프루닝 수행 (너무 작은 유동성/지배된 엣지 제거)
                prune_graph(self.graph.graph, min_liquidity=0.1, keep_top_k=2)
                # 메모리 절감을 위한 속성 정리
                compact_graph_attributes(self.graph.graph)
            except Exception as e:
                logger.error(f"블록 갱신 실패: {e}")

        async def on_log_event(log_data):
            try:
                # Sync 이벤트: 풀 리저브 업데이트 즉시 반영
                if log_data.get('topics') and log_data['topics'][0] == self.rt.monitored_events.get('Sync'):
                    pool = log_data.get('address')
                    data_hex = log_data.get('data')
                    reserve0 = reserve1 = None
                    if isinstance(data_hex, str) and data_hex.startswith('0x') and len(data_hex) >= 2 + 64 * 2:
                        try:
                            # ABI 인코딩된 두 개의 uint256에서 상위 비트는 0으로 패딩됨 (uint112 사용)
                            r0_hex = data_hex[2:2+64]
                            r1_hex = data_hex[2+64:2+128]
                            reserve0 = int(r0_hex, 16)
                            reserve1 = int(r1_hex, 16)
                        except Exception:
                            reserve0 = reserve1 = None
                    # 일부 구현에서 토픽으로 싱크를 보낼 수 있어 예비 처리
                    if reserve0 is None or reserve1 is None:
                        topics = log_data.get('topics', [])
                        if len(topics) >= 3:
                            try:
                                reserve0 = int(topics[1], 16)
                                reserve1 = int(topics[2], 16)
                            except Exception:
                                reserve0 = reserve1 = None
                    if pool and reserve0 is not None and reserve1 is not None:
                        # 그래프 풀 데이터 즉시 갱신
                        self.graph.update_pool_data(pool, float(reserve0), float(reserve1))
                        logger.debug(f"Sync 반영: {pool} r0={reserve0} r1={reserve1}")
                        # 메모리 절감을 위한 속성 정리
                        compact_graph_attributes(self.graph.graph)
                # Swap/Mint/Burn 이벤트: 최신 리저브를 on-chain에서 조회하여 반영
                elif log_data.get('topics') and log_data['topics'][0] in (
                    self.rt.monitored_events.get('Swap'),
                    self.rt.monitored_events.get('Mint'),
                    self.rt.monitored_events.get('Burn'),
                ):
                    pool = log_data.get('address')
                    if pool:
                        try:
                            # V2/Sushi 동일 ABI 사용 가능
                            v2c = UniswapV2Collector(self.w3)
                            r0, r1, _ = await v2c.get_pool_reserves(pool)
                            if r0 and r1:
                                self.graph.update_pool_data(pool, float(r0), float(r1))
                                logger.debug(f"Event-triggered reserve refresh: {pool} r0={r0} r1={r1}")
                                compact_graph_attributes(self.graph.graph)
                        except Exception as _:
                            pass
                # Uniswap V3 Collect: 수수료 수취 이벤트 감지 시 해당 풀의 fee-collect 엣지 즉시 갱신
                elif log_data.get('topics') and log_data['topics'][0] == self.rt.monitored_events.get('V3Collect'):
                    pool = log_data.get('address')
                    if pool:
                        try:
                            state = await self.v3_collector.get_pool_core_state(pool)
                            if not state:
                                return
                            t0 = state['token0']; t1 = state['token1']
                            dec0 = state['dec0']; dec1 = state['dec1']
                            ts = int(state.get('tickSpacing', 60) or 60)
                            cur_tick = int(state.get('tick', 0) or 0)
                            tick_lower = (cur_tick // ts) * ts
                            tick_upper = tick_lower + ts
                            fee0_per_L, fee1_per_L = await self.v3_collector.estimate_fees_per_L(
                                pool, dec0, dec1, tick_lower, tick_upper, cur_tick
                            )
                            # 통계 업데이트 (EMA)
                            try:
                                await self.storage.upsert_v3_fee_stats(pool, int(state.get('fee', 0)), fee0_per_L, fee1_per_L)
                            except Exception:
                                pass
                            lp_token = lp_v3(pool)
                            base_liq = 20.0
                            # LP -> token0 (fees)
                            self.graph.add_trading_pair(
                                token0=lp_token,
                                token1=t0,
                                dex='uniswap_v3_fee_collect',
                                pool_address=pool,
                                reserve0=base_liq,
                                reserve1=base_liq * float(fee0_per_L),
                                fee=0.0,
                            )
                            set_edge_meta(
                                self.graph.graph, lp_token, t0, dex='uniswap_v3_fee_collect', pool_address=pool,
                                fee_tier=state.get('fee'), source='event', confidence=0.72,
                                extra={'tick': cur_tick, 'tick_lower': tick_lower, 'tick_upper': tick_upper,
                                       'tick_spacing': ts, 't0': t0, 't1': t1, 'fee0_per_L': float(fee0_per_L),
                                       'collect_event': True}
                            )
                            # LP -> token1 (fees)
                            self.graph.add_trading_pair(
                                token0=lp_token,
                                token1=t1,
                                dex='uniswap_v3_fee_collect',
                                pool_address=pool,
                                reserve0=base_liq,
                                reserve1=base_liq * float(fee1_per_L),
                                fee=0.0,
                            )
                            set_edge_meta(
                                self.graph.graph, lp_token, t1, dex='uniswap_v3_fee_collect', pool_address=pool,
                                fee_tier=state.get('fee'), source='event', confidence=0.72,
                                extra={'tick': cur_tick, 'tick_lower': tick_lower, 'tick_upper': tick_upper,
                                       'tick_spacing': ts, 't0': t0, 't1': t1, 'fee1_per_L': float(fee1_per_L),
                                       'collect_event': True}
                            )
                            compact_graph_attributes(self.graph.graph)
                        except Exception as _:
                            pass
                # PositionManager 이벤트 반영: 포지션 밴드/유동성 통계 업데이트 (선택적)
                elif log_data.get('topics') and log_data['topics'][0] in (
                    self.rt.monitored_events.get('V3PMIncrease'),
                    self.rt.monitored_events.get('V3PMDecrease'),
                    self.rt.monitored_events.get('V3PMCollect'),
                ):
                    try:
                        topics = log_data.get('topics', [])
                        data_hex = log_data.get('data', '') or ''
                        token_id = None
                        if len(topics) >= 2:
                            token_id = int(topics[1], 16)
                        elif isinstance(data_hex, str) and len(data_hex) >= 2+64:
                            token_id = int(data_hex[2:2+64], 16)
                        if token_id is None or not self.v3_pm:
                            return
                        pos = self.v3_pm.functions.positions(int(token_id)).call()
                        token0 = pos[2]; token1 = pos[3]; fee = int(pos[4])
                        tick_lower = int(pos[5]); tick_upper = int(pos[6])
                        liq = int(pos[7])
                        pool = await self.v3_collector.get_pool_address(token0, token1, fee)
                        if not pool:
                            return
                        # delta liquidity 파싱 (Increase/Decrease)
                        delta = None
                        if log_data['topics'][0] == self.rt.monitored_events.get('V3PMIncrease') and isinstance(data_hex, str):
                            if len(data_hex) >= 2+64:
                                delta = int(data_hex[2:2+64], 16)
                        elif log_data['topics'][0] == self.rt.monitored_events.get('V3PMDecrease') and isinstance(data_hex, str):
                            if len(data_hex) >= 2+64:
                                delta = -int(data_hex[2:2+64], 16)
                        # 현재는 로깅 수준으로 유지 (필요 시 저장/활용 가능)
                        logger.debug(f"PM pos tokenId={token_id} pool={pool} fee={fee} band=({tick_lower},{tick_upper}) L={liq} dL={delta}")
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"로그 기반 동적 업데이트 실패: {e}")

        await self.rt.subscribe_to_blocks(on_new_block)
        await self.rt.subscribe_to_logs(on_log_event)

        # 백그라운드로 WS 리스너 실행
        asyncio.create_task(self.rt.start_websocket_listener())

        # 시작 시 1회 초기 빌드
        await self.update_via_actions()
        prune_graph(self.graph.graph, min_liquidity=0.1, keep_top_k=2)
        compact_graph_attributes(self.graph.graph)

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

    def _init_position_manager(self):
        try:
            pm_addr = Web3.to_checksum_address('0xC36442b4a4522E871399CD717aBDD847Ab11FE88')
        except Exception:
            pm_addr = '0xC36442b4a4522E871399CD717aBDD847Ab11FE88'
        abi = [
            {
                "inputs": [{"internalType":"uint256","name":"tokenId","type":"uint256"}],
                "name": "positions",
                "outputs": [
                    {"internalType":"uint96","name":"nonce","type":"uint96"},
                    {"internalType":"address","name":"operator","type":"address"},
                    {"internalType":"address","name":"token0","type":"address"},
                    {"internalType":"address","name":"token1","type":"address"},
                    {"internalType":"uint24","name":"fee","type":"uint24"},
                    {"internalType":"int24","name":"tickLower","type":"int24"},
                    {"internalType":"int24","name":"tickUpper","type":"int24"},
                    {"internalType":"uint128","name":"liquidity","type":"uint128"},
                    {"internalType":"uint256","name":"feeGrowthInside0LastX128","type":"uint256"},
                    {"internalType":"uint256","name":"feeGrowthInside1LastX128","type":"uint256"},
                    {"internalType":"uint128","name":"tokensOwed0","type":"uint128"},
                    {"internalType":"uint128","name":"tokensOwed1","type":"uint128"}
                ],
                "stateMutability":"view","type":"function"
            }
        ]
        try:
            return self.w3.eth.contract(address=pm_addr, abi=abi)
        except Exception:
            return None
