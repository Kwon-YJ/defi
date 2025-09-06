import networkx as nx
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from src.logger import setup_logger

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
    def __init__(self):
        self.graph = nx.DiGraph()
        self.token_nodes = set()
        self.dex_pools = {}  # dex -> {(token0, token1): pool_info}
        self.last_update = 0
        
    def add_token(self, token_address: str, symbol: str = None):
        """토큰 노드 추가"""
        if token_address not in self.token_nodes:
            self.graph.add_node(token_address, symbol=symbol)
            self.token_nodes.add(token_address)
            logger.debug(f"토큰 노드 추가: {symbol or token_address}")
    
    def add_trading_pair(self, token0: str, token1: str, dex: str, 
                        pool_address: str, reserve0: float, reserve1: float,
                        fee: float = 0.003):
        """거래 쌍 추가 (양방향 엣지)"""
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
        
        # 그래프에 엣지 추가
        self.graph.add_edge(token0, token1, **edge_01.__dict__)
        self.graph.add_edge(token1, token0, **edge_10.__dict__)
        
        logger.debug(f"거래 쌍 추가: {dex} {token0}-{token1}")
    
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
    
    def update_pool_data(self, pool_address: str, reserve0: float, reserve1: float):
        """풀 데이터 업데이트 - 논문의 정확한 weight calculation 적용"""
        # 해당 풀과 연관된 엣지들 찾아서 업데이트
        for u, v, data in self.graph.edges(data=True):
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
    
    def get_graph_stats(self) -> Dict:
        """그래프 통계 정보"""
        return {
            'nodes': len(self.graph.nodes),
            'edges': len(self.graph.edges),
            'tokens': len(self.token_nodes),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph)
        }
    
    # =============================================================================
    # **성능 최적화**: Graph pruning 및 96개 protocol actions 처리를 위한 확장성 개선
    # =============================================================================
    
    def prune_inefficient_edges(self, min_liquidity: float = 1.0, 
                               max_fee: float = 0.01,
                               min_exchange_rate: float = 1e-6) -> int:
        """
        비효율적인 edge 자동 제거 - 논문의 96개 protocol actions 처리 효율성 향상
        
        Args:
            min_liquidity: 최소 유동성 임계값 (ETH 단위)
            max_fee: 최대 허용 수수료율
            min_exchange_rate: 최소 환율 (0에 가까운 환율 제거)
        
        Returns:
            제거된 edge 개수
        """
        edges_to_remove = []
        
        for u, v, data in self.graph.edges(data=True):
            # 1. 유동성 부족한 edge 제거
            if data.get('liquidity', 0) < min_liquidity:
                edges_to_remove.append((u, v))
                continue
            
            # 2. 수수료가 너무 높은 edge 제거
            if data.get('fee', 0) > max_fee:
                edges_to_remove.append((u, v))
                continue
            
            # 3. 환율이 너무 낮은 edge 제거 (사실상 거래 불가능)
            if data.get('exchange_rate', 0) < min_exchange_rate:
                edges_to_remove.append((u, v))
                continue
            
            # 4. 무한대 가중치 edge 제거
            if data.get('weight', 0) == float('inf'):
                edges_to_remove.append((u, v))
                continue
        
        # Edge 제거 실행
        for u, v in edges_to_remove:
            if self.graph.has_edge(u, v):
                self.graph.remove_edge(u, v)
        
        removed_count = len(edges_to_remove)
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
        """Edge 데이터 메모리 최적화 - 불필요한 정밀도 제거"""
        for u, v, data in self.graph.edges(data=True):
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
