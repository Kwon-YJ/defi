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
