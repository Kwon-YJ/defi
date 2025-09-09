import asyncio
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from web3 import Web3
import networkx as nx

from src.logger import setup_logger
from src.market_graph import DeFiMarketGraph
from src.dex_data_collector import UniswapV2Collector, SushiSwapCollector
from src.dex_uniswap_v3_collector import UniswapV3Collector
from src.dex_balancer_collector import BalancerWeightedCollector
from src.lending_collectors import AaveV2Collector, CompoundCollector
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
        self.balancer = BalancerWeightedCollector(self.w3)
        self.aave = AaveV2Collector(self.w3)
        self.comp = CompoundCollector(self.w3)
        # 이벤트 기반 미니 업데이트 큐
        self._mini_queues = {
            'v2_pools': set(),
            'balancer_pools': set(),
            'compound_ctokens': set(),
            'aave_assets': set(),
        }

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
                # 구독 주소 갱신(가능 시)
                try:
                    self.rt.log_addresses = self._compute_log_addresses()
                except Exception:
                    pass
                # 프루닝/컴팩션 전에 이벤트 기반 미니 업데이트 적용
                try:
                    await self._apply_event_minis()
                except Exception:
                    pass
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
                        pool_addr = log_data.get('address')
                        if pool_addr:
                            self._mini_queues['balancer_pools'].add(pool_addr)
                        self._mini_queues['v2_pools'].add(pool)
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
                                self._mini_queues['v2_pools'].add(pool)
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
                # Balancer: PoolBalanceChanged
                elif log_data.get('topics') and log_data['topics'][0] == self.rt.monitored_events.get('BalancerPoolBalanceChanged'):
                    try:
                        # Recompute all balancer edges (pool balance changed in vault)
                        if isinstance(self.graph.graph, nx.MultiDiGraph):
                            for u, v, k, data in list(self.graph.graph.edges(keys=True, data=True)):
                                if data.get('dex') == 'balancer':
                                    pool = data.get('pool_address')
                                    if not pool:
                                        continue
                                    try:
                                        wi, wj, bi, bj = self.balancer.get_weights_and_balances(pool, u, v)
                                        price, fee = self.balancer.get_spot_price_and_fee(pool, u, v)
                                        if price > 0 and wi > 0 and wj > 0:
                                            self.graph.graph[u][v][k]['exchange_rate'] = price
                                            self.graph.graph[u][v][k]['fee'] = float(fee)
                                            set_edge_meta(self.graph.graph, u, v, dex='balancer', pool_address=pool,
                                                          fee_tier=None, source='event', confidence=0.92,
                                                          extra={'w_in': wi, 'w_out': wj, 'b_in': bi, 'b_out': bj})
                                    except Exception:
                                        continue
                        compact_graph_attributes(self.graph.graph)
                        ctoken = log_data.get('address')
                        if ctoken:
                            self._mini_queues['compound_ctokens'].add(ctoken)
                    except Exception:
                        pass
                # Compound: AccrueInterest on cToken
                elif log_data.get('topics') and log_data['topics'][0] == self.rt.monitored_events.get('CompoundAccrueInterest'):
                    try:
                        ctoken = log_data.get('address')
                        rates = self.comp.get_rates_per_block(ctoken) or {}
                        cf = self.comp.get_collateral_factor(ctoken)
                        # Update edges referencing this cToken
                        if isinstance(self.graph.graph, nx.MultiDiGraph):
                            for u, v, k, data in list(self.graph.graph.edges(keys=True, data=True)):
                                if data.get('pool_address') == ctoken and data.get('dex') in ('compound', 'compound_borrow', 'compound_repay'):
                                    # Update meta and, for borrow, adjust reserve1 for new interest penalty
                                    from config.config import config as _cfg
                                    if data.get('dex') == 'compound_borrow':
                                        hold_blocks = int(getattr(_cfg, 'interest_hold_blocks', 100))
                                        ip = self.comp.approx_interest_penalty(ctoken, hold_blocks)
                                        eff = 1.0 / (1.0 + ip) if ip > 0 else 1.0
                                        data['exchange_rate'] = eff
                                    set_edge_meta(self.graph.graph, u, v, dex=data.get('dex'), pool_address=ctoken,
                                                  fee_tier=None, source='event', confidence=0.9,
                                                  extra={'collateralFactor': cf, 'borrowRatePerBlock': rates.get('borrowRatePerBlock'),
                                                         'supplyRatePerBlock': rates.get('supplyRatePerBlock')})
                        compact_graph_attributes(self.graph.graph)
                        topics = log_data.get('topics', [])
                        if len(topics) >= 2:
                            asset = '0x' + topics[1][-40:]
                            self._mini_queues['aave_assets'].add(asset)
                    except Exception:
                        pass
                # Aave: ReserveDataUpdated(asset,...)
                elif log_data.get('topics') and log_data['topics'][0] == self.rt.monitored_events.get('AaveReserveDataUpdated'):
                    try:
                        topics = log_data.get('topics', [])
                        if len(topics) >= 2:
                            asset = '0x' + topics[1][-40:]
                            addrs = self.aave.get_reserve_tokens(asset)
                            rates = self.aave.get_reserve_rates(asset) or {}
                            cfg = self.aave.get_reserve_configuration(asset) or {}
                            emode = self.aave.get_emode_category(asset)
                            if addrs:
                                atoken, sdebt, vdebt = addrs
                            else:
                                atoken = sdebt = vdebt = None
                            # Update edges related to this asset
                            if isinstance(self.graph.graph, nx.MultiDiGraph):
                                for u, v, k, data in list(self.graph.graph.edges(keys=True, data=True)):
                                    if asset.lower() in (u.lower(), v.lower()) or (data.get('pool_address') in (atoken, sdebt, vdebt)):
                                        set_edge_meta(self.graph.graph, u, v, dex=data.get('dex'), pool_address=data.get('pool_address'),
                                                      fee_tier=None, source='event', confidence=0.92,
                                                      extra={'ltv': cfg.get('ltv'), 'liquidationThreshold': cfg.get('liquidationThreshold'),
                                                             'variableBorrowRate': rates.get('variableBorrowRate'), 'stableBorrowRate': rates.get('stableBorrowRate'),
                                                             'liquidityRate': rates.get('liquidityRate'), 'eModeCategory': emode})
                        compact_graph_attributes(self.graph.graph)
                    except Exception:
                        pass
                # Curve LP mint/burn via ERC20 Transfer on LP token (address-filtered)
                elif log_data.get('topics') and log_data['topics'][0] == self.rt.monitored_events.get('ERC20Transfer'):
                    try:
                        lp_token_addr = log_data.get('address')
                        pool_addr = None
                        # search edges for matching lp_token meta
                        if isinstance(self.graph.graph, nx.MultiDiGraph):
                            for u, v, k, data in self.graph.graph.edges(keys=True, data=True):
                                m = data.get('meta') if isinstance(data.get('meta'), dict) else {}
                                if m.get('lp_token') and isinstance(m.get('lp_token'), str) and m.get('lp_token').lower() == lp_token_addr.lower():
                                    pool_addr = data.get('pool_address')
                                    break
                        else:
                            for u, v, data in self.graph.graph.edges(data=True):
                                m = data.get('meta') if isinstance(data.get('meta'), dict) else {}
                                if m.get('lp_token') and isinstance(m.get('lp_token'), str) and m.get('lp_token').lower() == lp_token_addr.lower():
                                    pool_addr = data.get('pool_address')
                                    break
                        if not pool_addr:
                            return
                        from src.dex_curve_collector import CurveStableSwapCollector
                        coll = CurveStableSwapCollector(self.w3)
                        coll.refresh_registry_pools(ttl_sec=0)
                        coins = coll.get_pool_coins(pool_addr)
                        if not coins:
                            return
                        n = len(coins)
                        lp_token_sym = lp_v3(pool_addr)  # reuse wrapper for naming; address remains in meta
                        # detect lp decimals
                        try:
                            from src.dex_curve_collector import CurveStableSwapCollector as _C
                            lp_dec = _C(self.w3).get_lp_decimals(lp_token_addr)
                        except Exception:
                            lp_dec = 18
                        base_liq = 150.0
                        # Recompute add/remove edges
                        for i, coin in enumerate(coins):
                            try:
                                d = coll._decimals(coin)
                                dx = int(10 ** d)
                                amts = [0] * n
                                amts[i] = dx
                                minted = coll.calc_token_amount(pool_addr, amts, True)
                                denom = float(10 ** lp_dec) if lp_dec else 1e18
                                lp_per_token = float(minted) / denom if denom else 0.0
                                if lp_per_token > 0:
                                    self.graph.add_trading_pair(
                                        token0=coin,
                                        token1=lp_token_sym,
                                        dex='curve_lp_add',
                                        pool_address=pool_addr, edge_key=f"curve_lp_add:{i}",
                                        reserve0=base_liq,
                                        reserve1=base_liq * lp_per_token,
                                        fee=0.0,
                                    )
                                    set_edge_meta(self.graph.graph, coin, lp_token_sym, dex='curve_lp_add', pool_address=pool_addr,
                                                  fee_tier=None, source='event', confidence=0.9,
                                                  extra={'n_coins': n, 'coin_index': int(i), 'lp_token': lp_token_addr, 'lp_decimals': lp_dec})
                                # remove
                                denom_i = int(10 ** lp_dec) if lp_dec else int(1e18)
                                out = coll.calc_withdraw_one_coin(pool_addr, denom_i, i)
                                if out > 0:
                                    coin_per_lp = float(out) / float(10 ** d)
                                    self.graph.add_trading_pair(
                                        token0=lp_token_sym,
                                        token1=coin,
                                        dex='curve_lp_remove',
                                        pool_address=pool_addr, edge_key=f"curve_lp_remove:{i}",
                                        reserve0=base_liq,
                                        reserve1=base_liq * coin_per_lp,
                                        fee=0.0,
                                    )
                                    set_edge_meta(self.graph.graph, lp_token_sym, coin, dex='curve_lp_remove', pool_address=pool_addr,
                                                  fee_tier=None, source='event', confidence=0.9,
                                                  extra={'n_coins': n, 'coin_index': int(i), 'lp_token': lp_token_addr, 'lp_decimals': lp_dec})
                            except Exception:
                                continue
                        compact_graph_attributes(self.graph.graph)
                    except Exception:
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
                        # 밴드 히스토그램 업데이트
                        try:
                            await self.storage.upsert_v3_band_histogram(pool, tick_lower, tick_upper, delta)
                        except Exception:
                            pass
                        logger.debug(f"PM pos tokenId={token_id} pool={pool} fee={fee} band=({tick_lower},{tick_upper}) L={liq} dL={delta}")
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"로그 기반 동적 업데이트 실패: {e}")

        await self.rt.subscribe_to_blocks(on_new_block)
        await self.rt.subscribe_to_logs(on_log_event)

        # 시작 시 1회 초기 빌드 후 주소 필터 계산 및 WS 시작
        await self.update_via_actions()
        try:
            self.rt.log_addresses = self._compute_log_addresses()
        except Exception:
            pass
        # 프루닝/컴팩션 전에 이벤트 기반 미니 업데이트 적용
        try:
            await self._apply_event_minis()
        except Exception:
            pass
        asyncio.create_task(self.rt.start_websocket_listener())
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

    async def _apply_event_minis(self):
        """이벤트 큐에 쌓인 최소 업데이트를 일괄 적용 (프루닝/컴팩션 전에 호출)."""
        # Uniswap V2 pools
        try:
            if self._mini_queues.get('v2_pools'):
                v2c = UniswapV2Collector(self.w3)
                for pool in list(self._mini_queues['v2_pools']):
                    try:
                        r0, r1, _ = await v2c.get_pool_reserves(pool)
                        if r0 and r1:
                            self.graph.update_pool_data(pool, float(r0), float(r1))
                    except Exception:
                        pass
                self._mini_queues['v2_pools'].clear()
        except Exception:
            pass
        # Balancer pools meta
        try:
            if self._mini_queues.get('balancer_pools') and isinstance(self.graph.graph, nx.MultiDiGraph):
                for pool in list(self._mini_queues['balancer_pools']):
                    for u, v, k, data in list(self.graph.graph.edges(keys=True, data=True)):
                        if data.get('dex') == 'balancer' and data.get('pool_address') == pool:
                            try:
                                wi, wj, bi, bj = self.balancer.get_weights_and_balances(pool, u, v)
                                set_edge_meta(self.graph.graph, u, v, dex='balancer', pool_address=pool,
                                              fee_tier=None, source='event', confidence=0.9,
                                              extra={'w_in': wi, 'w_out': wj, 'b_in': bi, 'b_out': bj})
                            except Exception:
                                continue
                self._mini_queues['balancer_pools'].clear()
        except Exception:
            pass
        # Compound cTokens
        try:
            if self._mini_queues.get('compound_ctokens') and isinstance(self.graph.graph, nx.MultiDiGraph):
                for ctoken in list(self._mini_queues['compound_ctokens']):
                    rates = self.comp.get_rates_per_block(ctoken) or {}
                    for u, v, k, data in list(self.graph.graph.edges(keys=True, data=True)):
                        if data.get('pool_address') == ctoken and data.get('dex') in ('compound','compound_borrow','compound_repay'):
                            set_edge_meta(self.graph.graph, u, v, dex=data.get('dex'), pool_address=ctoken,
                                          fee_tier=None, source='event', confidence=0.9,
                                          extra={'borrowRatePerBlock': rates.get('borrowRatePerBlock'),
                                                 'supplyRatePerBlock': rates.get('supplyRatePerBlock')})
                self._mini_queues['compound_ctokens'].clear()
        except Exception:
            pass
        # Aave assets
        try:
            if self._mini_queues.get('aave_assets') and isinstance(self.graph.graph, nx.MultiDiGraph):
                for asset in list(self._mini_queues['aave_assets']):
                    rates = self.aave.get_reserve_rates(asset) or {}
                    cfg = self.aave.get_reserve_configuration(asset) or {}
                    emode = self.aave.get_emode_category(asset)
                    for u, v, k, data in list(self.graph.graph.edges(keys=True, data=True)):
                        if asset.lower() in (u.lower(), v.lower()):
                            set_edge_meta(self.graph.graph, u, v, dex=data.get('dex'), pool_address=data.get('pool_address'),
                                          fee_tier=None, source='event', confidence=0.9,
                                          extra={'ltv': cfg.get('ltv'), 'liquidationThreshold': cfg.get('liquidationThreshold'),
                                                 'variableBorrowRate': rates.get('variableBorrowRate'), 'stableBorrowRate': rates.get('stableBorrowRate'),
                                                 'liquidityRate': rates.get('liquidityRate'), 'eModeCategory': emode})
                self._mini_queues['aave_assets'].clear()
        except Exception:
            pass

    def _compute_log_addresses(self) -> List[str]:
        addrs = set()
        try:
            if isinstance(self.graph.graph, nx.MultiDiGraph):
                for u, v, k, data in self.graph.graph.edges(keys=True, data=True):
                    if isinstance(data, dict):
                        pa = data.get('pool_address')
                        if pa:
                            addrs.add(pa)
                        m = data.get('meta') if isinstance(data.get('meta'), dict) else {}
                        lp = m.get('lp_token')
                        if isinstance(lp, str) and int(lp, 16) != 0:
                            addrs.add(lp)
            else:
                for u, v, data in self.graph.graph.edges(data=True):
                    pa = data.get('pool_address')
                    if pa:
                        addrs.add(pa)
                    m = data.get('meta') if isinstance(data.get('meta'), dict) else {}
                    lp = m.get('lp_token')
                    if isinstance(lp, str) and int(lp, 16) != 0:
                        addrs.add(lp)
        except Exception:
            pass
        return list(addrs)

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
