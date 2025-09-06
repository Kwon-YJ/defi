import networkx as nx
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from src.logger import setup_logger
from src.protocol_actions import ProtocolRegistry, ProtocolAction, ProtocolType

logger = setup_logger(__name__)

@dataclass
class TradingEdge:
    """거래 엣지 정보"""
    from_token: str
    to_token: str
    dex: str
    pool_address: str
    exchange_rate: float
    liquidity: float
    fee: float
    gas_cost: float
    weight: float  # -log(exchange_rate)

@dataclass
class ArbitrageOpportunity:
    """차익거래 기회"""
    path: List[str]  # 토큰 경로
    edges: List[TradingEdge]  # 거래 엣지들
    profit_ratio: float  # 수익률
    required_capital: float  # 필요 자본
    estimated_profit: float  # 예상 수익
    gas_cost: float  # 가스 비용
    net_profit: float  # 순수익
    confidence: float  # 신뢰도 (0-1)

class DeFiMarketGraph:
    def __init__(self, web3_provider=None):
        # **CRITICAL**: Multi-graph 지원을 위해 nx.MultiDiGraph 사용
        self.graph = nx.MultiDiGraph()  # 동일 토큰 쌍에서 여러 DEX edge 처리
        self.token_nodes = set()
        self.dex_pools = {}  # dex -> {(token0, token1): pool_info}
        self.last_update = 0
        
        # **Multi-graph 지원**: 동일 토큰 쌍의 여러 DEX edge 관리
        self._edge_registry = {}  # (from_token, to_token) -> {dex: edge_key}
        self._best_edges = {}     # (from_token, to_token) -> (dex, edge_key) for best rate
        
        # **CRITICAL**: 96개 protocol actions 지원을 위한 프로토콜 레지스트리
        self.protocol_registry = ProtocolRegistry(web3_provider) if web3_provider else None
        
        # **논문 기준 확장성**: 96개 protocol actions, 25개 assets 처리 최적화
        self._protocol_edge_cache = {}  # 프로토콜별 엣지 캐시
        self._supported_protocols = set()  # 지원되는 프로토콜 목록
        
    def add_token(self, token_address: str, symbol: str = None):
        """토큰 노드 추가"""
        if token_address not in self.token_nodes:
            self.graph.add_node(token_address, symbol=symbol)
            self.token_nodes.add(token_address)
            logger.debug(f"토큰 노드 추가: {symbol or token_address}")
    
    def add_trading_pair(self, token0: str, token1: str, dex: str, 
                        pool_address: str, reserve0: float, reserve1: float,
                        fee: float = 0.003):
        """거래 쌍 추가 (양방향 엣지) - Multi-graph 지원"""
        # 토큰 노드 추가
        self.add_token(token0)
        self.add_token(token1)
        
        # 유동성 확인
        if reserve0 <= 0 or reserve1 <= 0:
            logger.warning(f"유동성 부족: {pool_address}")
            return
        
        # token0 -> token1 엣지
        # 논문의 정확한 공식: w_{i,j} = -log(p^{spot}_{i,j})
        # spot price는 순수한 환율, fee는 별도로 처리
        spot_price_01 = reserve1 / reserve0
        effective_rate_01 = spot_price_01 * (1 - fee)  # fee 적용된 실제 환율
        weight_01 = self._calculate_edge_weight(spot_price_01)  # 논문 공식 사용
        
        edge_01 = TradingEdge(
            from_token=token0,
            to_token=token1,
            dex=dex,
            pool_address=pool_address,
            exchange_rate=effective_rate_01,  # 실제 거래에서 받을 수 있는 환율
            liquidity=min(reserve0, reserve1),
            fee=fee,
            gas_cost=self._estimate_gas_cost(dex),
            weight=weight_01  # 순수 spot price의 -log 값
        )
        
        # token1 -> token0 엣지
        # 논문의 정확한 공식: w_{i,j} = -log(p^{spot}_{i,j})
        spot_price_10 = reserve0 / reserve1
        effective_rate_10 = spot_price_10 * (1 - fee)  # fee 적용된 실제 환율
        weight_10 = self._calculate_edge_weight(spot_price_10)  # 논문 공식 사용
        
        edge_10 = TradingEdge(
            from_token=token1,
            to_token=token0,
            dex=dex,
            pool_address=pool_address,
            exchange_rate=effective_rate_10,  # 실제 거래에서 받을 수 있는 환율
            liquidity=min(reserve0, reserve1),
            fee=fee,
            gas_cost=self._estimate_gas_cost(dex),
            weight=weight_10  # 순수 spot price의 -log 값
        )
        
        # **Multi-graph 지원**: 동일 토큰 쌍에서 여러 DEX edge 처리
        edge_key_01 = self._add_multi_edge(token0, token1, edge_01)
        edge_key_10 = self._add_multi_edge(token1, token0, edge_10)
        
        # Edge registry 업데이트
        self._register_edge(token0, token1, dex, edge_key_01, effective_rate_01)
        self._register_edge(token1, token0, dex, edge_key_10, effective_rate_10)
        
        logger.debug(f"거래 쌍 추가 (Multi-graph): {dex} {token0}-{token1}")
    
    def _calculate_edge_weight(self, spot_price: float) -> float:
        """
        논문의 정확한 weight calculation 구현
        w_{i,j} = -log(p^{spot}_{i,j})
        
        Args:
            spot_price: 순수한 spot price (fee 적용 전)
        
        Returns:
            weight: Bellman-Ford 알고리즘에서 사용할 edge weight
        
        Note:
            - spot_price > 1인 경우: weight < 0 (negative weight)
            - spot_price < 1인 경우: weight > 0 (positive weight)
            - spot_price = 1인 경우: weight = 0 (no change)
            - 이를 통해 수익성 있는 사이클은 negative cycle로 탐지됨
        """
        if spot_price <= 0:
            logger.warning(f"Invalid spot price: {spot_price}")
            return float('inf')
        
        try:
            weight = -math.log(spot_price)
            logger.debug(f"Weight calculation: spot_price={spot_price:.6f} -> weight={weight:.6f}")
            return weight
        except (ValueError, OverflowError) as e:
            logger.error(f"Weight calculation error for spot_price={spot_price}: {e}")
            return float('inf')
    
    def _estimate_gas_cost(self, dex: str) -> float:
        """DEX별 가스 비용 추정"""
        gas_estimates = {
            'uniswap_v2': 150000,
            'uniswap_v3': 180000,
            'sushiswap': 150000,
            'curve': 200000,
            'balancer': 250000
        }
        
        gas_limit = gas_estimates.get(dex.lower(), 150000)
        gas_price = 20e9  # 20 Gwei
        eth_price = 2000  # $2000 per ETH
        
        return (gas_limit * gas_price * eth_price) / 1e18
    
    # =============================================================================
    # **Multi-graph 지원**: 동일 토큰 쌍에서 여러 DEX edge 처리
    # =============================================================================
    
    def _add_multi_edge(self, from_token: str, to_token: str, edge: TradingEdge) -> int:
        """Multi-graph에 edge 추가하고 edge key 반환"""
        edge_key = self.graph.add_edge(from_token, to_token, **edge.__dict__)
        return edge_key
    
    def _register_edge(self, from_token: str, to_token: str, dex: str, edge_key: int, rate: float):
        """Edge registry에 등록하고 최적 edge 추적"""
        # Edge registry 업데이트
        pair_key = (from_token, to_token)
        if pair_key not in self._edge_registry:
            self._edge_registry[pair_key] = {}
        self._edge_registry[pair_key][dex] = edge_key
        
        # 최적 edge 업데이트 (가장 높은 환율)
        if pair_key not in self._best_edges or rate > self._get_edge_rate(pair_key):
            self._best_edges[pair_key] = (dex, edge_key)
    
    def _get_edge_rate(self, pair_key: Tuple[str, str]) -> float:
        """현재 최적 edge의 환율 반환"""
        if pair_key not in self._best_edges:
            return 0.0
        
        dex, edge_key = self._best_edges[pair_key]
        from_token, to_token = pair_key
        
        edge_data = self.graph[from_token][to_token][edge_key]
        return edge_data.get('exchange_rate', 0.0)
    
    def get_best_edge(self, from_token: str, to_token: str) -> Optional[Dict]:
        """두 토큰 간 최적 edge 정보 반환"""
        pair_key = (from_token, to_token)
        if pair_key not in self._best_edges:
            return None
        
        dex, edge_key = self._best_edges[pair_key]
        edge_data = dict(self.graph[from_token][to_token][edge_key])
        edge_data['dex'] = dex
        edge_data['edge_key'] = edge_key
        return edge_data
    
    def get_all_edges(self, from_token: str, to_token: str) -> List[Dict]:
        """두 토큰 간 모든 edge 정보 반환 (모든 DEX)"""
        if not self.graph.has_edge(from_token, to_token):
            return []
        
        edges = []
        for edge_key, edge_data in self.graph[from_token][to_token].items():
            edge_info = dict(edge_data)
            edge_info['edge_key'] = edge_key
            edges.append(edge_info)
        
        # 환율 기준 내림차순 정렬
        edges.sort(key=lambda x: x.get('exchange_rate', 0), reverse=True)
        return edges
    
    def get_dex_count_for_pair(self, from_token: str, to_token: str) -> int:
        """특정 토큰 쌍에 대해 지원되는 DEX 개수 반환"""
        pair_key = (from_token, to_token)
        return len(self._edge_registry.get(pair_key, {}))
    
    def get_multi_graph_stats(self) -> Dict:
        """Multi-graph 통계 정보"""
        total_pairs = len(self._edge_registry)
        multi_dex_pairs = sum(1 for edges in self._edge_registry.values() if len(edges) > 1)
        max_dex_per_pair = max((len(edges) for edges in self._edge_registry.values()), default=0)
        
        return {
            'total_token_pairs': total_pairs,
            'multi_dex_pairs': multi_dex_pairs,  # 2개 이상 DEX 지원 쌍
            'single_dex_pairs': total_pairs - multi_dex_pairs,
            'max_dex_per_pair': max_dex_per_pair,
            'average_dex_per_pair': sum(len(edges) for edges in self._edge_registry.values()) / total_pairs if total_pairs > 0 else 0,
            'multi_graph_efficiency': multi_dex_pairs / total_pairs if total_pairs > 0 else 0
        }
    
    def remove_edge_by_dex(self, from_token: str, to_token: str, dex: str) -> bool:
        """특정 DEX의 edge 제거"""
        pair_key = (from_token, to_token)
        if pair_key not in self._edge_registry or dex not in self._edge_registry[pair_key]:
            return False
        
        edge_key = self._edge_registry[pair_key][dex]
        
        # 그래프에서 edge 제거 (NetworkX MultiDiGraph의 올바른 방법)
        try:
            if self.graph.has_edge(from_token, to_token, key=edge_key):
                self.graph.remove_edge(from_token, to_token, key=edge_key)
        except Exception as e:
            logger.error(f"Edge removal failed: {e}")
            return False
        
        # Registry에서 제거
        del self._edge_registry[pair_key][dex]
        
        # 최적 edge 재계산
        if pair_key in self._best_edges and self._best_edges[pair_key][0] == dex:
            self._update_best_edge(pair_key)
        
        logger.debug(f"DEX edge 제거: {dex} {from_token}-{to_token}")
        return True
    
    def _update_best_edge(self, pair_key: Tuple[str, str]):
        """특정 토큰 쌍의 최적 edge 재계산"""
        from_token, to_token = pair_key
        
        if pair_key not in self._edge_registry or not self._edge_registry[pair_key]:
            # 모든 edge가 제거된 경우
            if pair_key in self._best_edges:
                del self._best_edges[pair_key]
            return
        
        # 모든 edge 중 최고 환율 찾기
        best_rate = 0
        best_dex = None
        best_edge_key = None
        
        for dex, edge_key in self._edge_registry[pair_key].items():
            if edge_key in self.graph[from_token][to_token]:
                rate = self.graph[from_token][to_token][edge_key].get('exchange_rate', 0)
                if rate > best_rate:
                    best_rate = rate
                    best_dex = dex
                    best_edge_key = edge_key
        
        if best_dex:
            self._best_edges[pair_key] = (best_dex, best_edge_key)
    
    def update_pool_data(self, pool_address: str, reserve0: float, reserve1: float):
        """풀 데이터 업데이트 - Multi-graph 지원, 논문의 정확한 weight calculation 적용"""
        updated_pairs = []
        
        # 해당 풀과 연관된 엣지들 찾아서 업데이트 (Multi-graph 처리)
        for u, v, edge_key, data in self.graph.edges(data=True, keys=True):
            if data.get('pool_address') == pool_address:
                # 새로운 spot price 및 effective rate 계산
                if data['from_token'] == u:
                    spot_price = reserve1 / reserve0
                    effective_rate = spot_price * (1 - data['fee'])
                else:
                    spot_price = reserve0 / reserve1
                    effective_rate = spot_price * (1 - data['fee'])
                
                # 엣지 데이터 업데이트 (논문의 정확한 공식 적용)
                data['exchange_rate'] = effective_rate  # 실제 거래 환율
                data['weight'] = self._calculate_edge_weight(spot_price)  # 논문 공식 사용
                data['liquidity'] = min(reserve0, reserve1)
                
                # Best edge 재평가를 위해 쌍 기록
                pair_key = (u, v)
                if pair_key not in updated_pairs:
                    updated_pairs.append(pair_key)
        
        # 업데이트된 토큰 쌍들의 최적 edge 재계산
        for pair_key in updated_pairs:
            self._update_best_edge(pair_key)
            logger.debug(f"Pool 업데이트: {pool_address} - {pair_key[0]}-{pair_key[1]}")
    
    def get_graph_stats(self) -> Dict:
        """그래프 통계 정보 - Multi-graph 지원"""
        multi_stats = self.get_multi_graph_stats()
        
        return {
            'nodes': len(self.graph.nodes),
            'edges': len(self.graph.edges),
            'tokens': len(self.token_nodes),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'multi_graph': multi_stats
        }
    
    # =============================================================================
    # **성능 최적화**: Graph pruning 및 96개 protocol actions 처리를 위한 확장성 개선
    # =============================================================================
    
    def prune_inefficient_edges(self, min_liquidity: float = 1.0, 
                               max_fee: float = 0.01,
                               min_exchange_rate: float = 1e-6) -> int:
        """
        비효율적인 edge 자동 제거 - Multi-graph 지원, 논문의 96개 protocol actions 처리 효율성 향상
        
        Args:
            min_liquidity: 최소 유동성 임계값 (ETH 단위)
            max_fee: 최대 허용 수수료율
            min_exchange_rate: 최소 환율 (0에 가까운 환율 제거)
        
        Returns:
            제거된 edge 개수
        """
        edges_to_remove = []  # (u, v, edge_key, dex)
        
        for u, v, edge_key, data in self.graph.edges(data=True, keys=True):
            # 1. 유동성 부족한 edge 제거
            if data.get('liquidity', 0) < min_liquidity:
                edges_to_remove.append((u, v, edge_key, data.get('dex')))
                continue
            
            # 2. 수수료가 너무 높은 edge 제거
            if data.get('fee', 0) > max_fee:
                edges_to_remove.append((u, v, edge_key, data.get('dex')))
                continue
            
            # 3. 환율이 너무 낮은 edge 제거 (사실상 거래 불가능)
            if data.get('exchange_rate', 0) < min_exchange_rate:
                edges_to_remove.append((u, v, edge_key, data.get('dex')))
                continue
            
            # 4. 무한대 가중치 edge 제거
            if data.get('weight', 0) == float('inf'):
                edges_to_remove.append((u, v, edge_key, data.get('dex')))
                continue
        
        # Edge 제거 실행 (Multi-graph 처리)
        removed_count = 0
        for u, v, edge_key, dex in edges_to_remove:
            try:
                if self.graph.has_edge(u, v, key=edge_key):
                    # 그래프에서 제거
                    self.graph.remove_edge(u, v, key=edge_key)
                    removed_count += 1
                    
                    # Registry에서 제거
                    if dex:
                        pair_key = (u, v)
                        if pair_key in self._edge_registry and dex in self._edge_registry[pair_key]:
                            del self._edge_registry[pair_key][dex]
                            
                            # 최적 edge 재계산
                            if pair_key in self._best_edges and self._best_edges[pair_key][0] == dex:
                                self._update_best_edge(pair_key)
            except Exception as e:
                logger.error(f"Failed to remove edge {u}-{v} (key={edge_key}): {e}")
        
        if removed_count > 0:
            logger.info(f"Graph pruning 완료: {removed_count}개 비효율적 edge 제거")
        
        return removed_count
    
    def optimize_for_scale(self, target_actions: int = 96, target_assets: int = 25) -> Dict:
        """
        대규모 처리를 위한 그래프 최적화 - 논문 규모 달성
        96개 protocol actions, 25개 assets 처리 최적화
        """
        logger.info(f"대규모 처리 최적화 시작 - 목표: {target_actions}개 actions, {target_assets}개 assets")
        
        # 1. 비효율적 edge 제거
        removed_edges = self.prune_inefficient_edges()
        
        # 2. 고립된 노드 제거
        isolated_nodes = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolated_nodes)
        
        # 3. 메모리 사용량 최적화를 위한 데이터 압축
        self._optimize_edge_data()
        
        # 4. 최적화 결과 통계
        stats = self.get_graph_stats()
        
        optimization_result = {
            'removed_edges': removed_edges,
            'removed_isolated_nodes': len(isolated_nodes),
            'final_stats': stats,
            'scalability_score': self._calculate_scalability_score(target_actions, target_assets),
            'memory_efficient': stats['edges'] < target_actions * target_assets * 0.1  # 10% 임계값
        }
        
        logger.info(f"대규모 처리 최적화 완료 - "
                   f"제거된 edges: {removed_edges}, "
                   f"제거된 nodes: {len(isolated_nodes)}, "
                   f"확장성 점수: {optimization_result['scalability_score']:.2f}")
        
        return optimization_result
    
    def _optimize_edge_data(self):
        """Edge 데이터 메모리 최적화 - Multi-graph 지원, 불필요한 정밀도 제거"""
        for u, v, edge_key, data in self.graph.edges(data=True, keys=True):
            # 부동소수점 정밀도 최적화 (6자리까지만)
            if 'exchange_rate' in data:
                data['exchange_rate'] = round(data['exchange_rate'], 6)
            if 'weight' in data:
                data['weight'] = round(data['weight'], 6)
            if 'liquidity' in data:
                data['liquidity'] = round(data['liquidity'], 2)
            if 'fee' in data:
                data['fee'] = round(data['fee'], 5)
    
    def _calculate_scalability_score(self, target_actions: int, target_assets: int) -> float:
        """확장성 점수 계산 (0-1, 1이 최고)"""
        current_edges = len(self.graph.edges)
        current_nodes = len(self.graph.nodes)
        
        # 이상적인 그래프 크기 (완전 연결 그래프의 일정 비율)
        ideal_max_edges = target_assets * (target_assets - 1)  # 방향 그래프
        ideal_max_nodes = target_assets
        
        # 현재 그래프가 목표 규모를 얼마나 효율적으로 처리할 수 있는지 계산
        edge_efficiency = min(current_edges / (target_actions * 2), 1.0)  # 각 action당 2개 edge 가정
        node_efficiency = min(current_nodes / target_assets, 1.0)
        
        # 전체 확장성 점수
        scalability_score = (edge_efficiency + node_efficiency) / 2
        
        return scalability_score
    
    def get_optimization_recommendations(self, target_actions: int = 96) -> List[str]:
        """성능 최적화 권장사항 제공"""
        recommendations = []
        stats = self.get_graph_stats()
        
        # 1. Edge 밀도 분석
        if stats['density'] > 0.5:
            recommendations.append("그래프 밀도가 높습니다. Edge pruning을 통해 불필요한 연결을 제거하세요.")
        
        # 2. 노드 수 분석
        if stats['nodes'] > target_actions:
            recommendations.append(f"노드 수({stats['nodes']})가 목표({target_actions})를 초과합니다. 불필요한 토큰을 제거하세요.")
        
        # 3. 연결성 분석
        if not stats['is_connected']:
            recommendations.append("그래프가 연결되어 있지 않습니다. 고립된 노드들을 확인하세요.")
        
        # 4. Edge 수 분석
        expected_edges = target_actions * 2  # 각 action당 양방향 edge
        if stats['edges'] > expected_edges * 1.5:
            recommendations.append("Edge 수가 예상보다 많습니다. 중복 또는 비효율적 edge를 제거하세요.")
        
        return recommendations

    # =============================================================================
    # **CRITICAL**: 96개 Protocol Actions 지원 구현
    # =============================================================================

    def add_protocol_edges(self, protocol_name: str, token_pairs: List[Tuple[str, str]], 
                          reserves_data: Dict[str, Tuple[float, float]]) -> int:
        """
        특정 프로토콜의 모든 거래 쌍에 대해 엣지 추가
        논문의 96개 protocol actions 지원을 위한 확장성 구현
        
        Args:
            protocol_name: 프로토콜 이름 (예: "Uniswap V2", "Compound")
            token_pairs: 토큰 쌍 목록
            reserves_data: 토큰 쌍별 리저브 정보
        
        Returns:
            추가된 엣지 개수
        """
        if not self.protocol_registry:
            logger.warning("Protocol registry not initialized")
            return 0
            
        added_edges = 0
        protocol_actions = self.protocol_registry.get_actions_by_protocol(protocol_name)
        
        if not protocol_actions:
            logger.warning(f"No actions found for protocol: {protocol_name}")
            return 0
        
        logger.info(f"Adding edges for {protocol_name}: {len(protocol_actions)} actions, {len(token_pairs)} pairs")
        
        for token0, token1 in token_pairs:
            if (token0, token1) not in reserves_data:
                continue
                
            reserve0, reserve1 = reserves_data[(token0, token1)]
            
            # 각 프로토콜 액션에 대해 적절한 엣지 생성
            for action in protocol_actions:
                if self._is_swap_action(action):
                    # Swap 액션인 경우 양방향 엣지 추가
                    if self._add_protocol_swap_edge(action, token0, token1, reserve0, reserve1):
                        added_edges += 2  # 양방향
                elif self._is_lending_action(action):
                    # 렌딩 액션인 경우 단방향 엣지 추가
                    if self._add_protocol_lending_edge(action, token0, token1, reserve0, reserve1):
                        added_edges += 1
        
        self._supported_protocols.add(protocol_name)
        logger.info(f"✅ {protocol_name}: {added_edges}개 엣지 추가 완료")
        return added_edges

    def _is_swap_action(self, action: ProtocolAction) -> bool:
        """액션이 스왑/교환 액션인지 확인"""
        swap_keywords = ['swap', 'exchange', 'trade']
        return any(keyword in action.action_type.lower() for keyword in swap_keywords)

    def _is_lending_action(self, action: ProtocolAction) -> bool:
        """액션이 렌딩 액션인지 확인"""
        return action.protocol_type in [ProtocolType.LENDING, ProtocolType.CDP]

    def _add_protocol_swap_edge(self, action: ProtocolAction, token0: str, token1: str, 
                               reserve0: float, reserve1: float) -> bool:
        """프로토콜 스왑 액션에 대한 엣지 추가"""
        try:
            # 기본적인 AMM 공식 사용 (추후 각 프로토콜별로 특화 필요)
            if reserve0 <= 0 or reserve1 <= 0:
                return False
            
            # token0 -> token1
            spot_price_01 = reserve1 / reserve0
            effective_rate_01 = spot_price_01 * (1 - action.fee_rate)
            weight_01 = self._calculate_edge_weight(spot_price_01)
            
            edge_01 = TradingEdge(
                from_token=token0,
                to_token=token1,
                dex=action.protocol_name,
                pool_address=action.contract_address,
                exchange_rate=effective_rate_01,
                liquidity=min(reserve0, reserve1),
                fee=action.fee_rate,
                gas_cost=action.gas_estimate * 20e-9 * 2000,  # 20 gwei * $2000 ETH 가정
                weight=weight_01
            )
            
            # token1 -> token0
            spot_price_10 = reserve0 / reserve1
            effective_rate_10 = spot_price_10 * (1 - action.fee_rate)
            weight_10 = self._calculate_edge_weight(spot_price_10)
            
            edge_10 = TradingEdge(
                from_token=token1,
                to_token=token0,
                dex=action.protocol_name,
                pool_address=action.contract_address,
                exchange_rate=effective_rate_10,
                liquidity=min(reserve0, reserve1),
                fee=action.fee_rate,
                gas_cost=action.gas_estimate * 20e-9 * 2000,
                weight=weight_10
            )
            
            # Multi-graph 지원: edge 추가
            edge_key_01 = self._add_multi_edge(token0, token1, edge_01)
            edge_key_10 = self._add_multi_edge(token1, token0, edge_10)
            
            # Registry 업데이트
            self._register_edge(token0, token1, action.protocol_name, edge_key_01, effective_rate_01)
            self._register_edge(token1, token0, action.protocol_name, edge_key_10, effective_rate_10)
            
            return True
            
        except Exception as e:
            logger.error(f"Protocol swap edge creation failed for {action.action_id}: {e}")
            return False

    def _add_protocol_lending_edge(self, action: ProtocolAction, token0: str, token1: str,
                                  reserve0: float, reserve1: float) -> bool:
        """프로토콜 렌딩 액션에 대한 엣지 추가"""
        try:
            # 렌딩은 보통 담보 -> 대출 방향의 단방향 엣지
            if reserve0 <= 0:
                return False
            
            # 대출비율 적용 (일반적으로 70-80%)
            collateral_ratio = 0.75
            effective_rate = collateral_ratio * (1 - action.fee_rate)
            weight = self._calculate_edge_weight(effective_rate)
            
            lending_edge = TradingEdge(
                from_token=token0,  # 담보 토큰
                to_token=token1,    # 대출 토큰
                dex=f"{action.protocol_name}_lending",
                pool_address=action.contract_address,
                exchange_rate=effective_rate,
                liquidity=reserve0,
                fee=action.fee_rate,
                gas_cost=action.gas_estimate * 20e-9 * 2000,
                weight=weight
            )
            
            # Multi-graph 지원: lending edge 추가
            edge_key = self._add_multi_edge(token0, token1, lending_edge)
            self._register_edge(token0, token1, f"{action.protocol_name}_lending", edge_key, effective_rate)
            
            return True
            
        except Exception as e:
            logger.error(f"Protocol lending edge creation failed for {action.action_id}: {e}")
            return False

    def get_protocol_summary(self) -> Dict:
        """96개 protocol actions 지원 현황 요약"""
        if not self.protocol_registry:
            return {"error": "Protocol registry not initialized"}
        
        action_summary = self.protocol_registry.get_action_summary()
        
        return {
            "total_protocols": len(self._supported_protocols),
            "supported_protocols": list(self._supported_protocols),
            "total_actions": action_summary['total_actions'],
            "action_breakdown": action_summary,
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "paper_compliance": {
                "target_actions": 96,
                "current_actions": action_summary['total_actions'],
                "compliance": action_summary['total_actions'] >= 96
            }
        }

    def validate_96_protocol_support(self) -> bool:
        """논문의 96개 protocol actions 지원 여부 검증"""
        if not self.protocol_registry:
            logger.error("Protocol registry not initialized - cannot validate 96 protocol support")
            return False
            
        action_summary = self.protocol_registry.get_action_summary()
        current_actions = action_summary['total_actions']
        
        if current_actions >= 96:
            logger.info(f"✅ Paper compliance achieved: {current_actions}/96 protocol actions supported")
            return True
        else:
            logger.warning(f"❌ Paper compliance NOT achieved: {current_actions}/96 protocol actions supported")
            return False
