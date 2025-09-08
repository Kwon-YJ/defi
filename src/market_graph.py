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
        exchange_rate_01 = (reserve1 / reserve0) * (1 - fee)
        weight_01 = -math.log(exchange_rate_01) if exchange_rate_01 > 0 else float('inf')
        
        edge_01 = TradingEdge(
            from_token=token0,
            to_token=token1,
            dex=dex,
            pool_address=pool_address,
            exchange_rate=exchange_rate_01,
            liquidity=min(reserve0, reserve1),
            fee=fee,
            gas_cost=self._estimate_gas_cost(dex),
            weight=weight_01
        )
        
        # token1 -> token0 엣지
        exchange_rate_10 = (reserve0 / reserve1) * (1 - fee)
        weight_10 = -math.log(exchange_rate_10) if exchange_rate_10 > 0 else float('inf')
        
        edge_10 = TradingEdge(
            from_token=token1,
            to_token=token0,
            dex=dex,
            pool_address=pool_address,
            exchange_rate=exchange_rate_10,
            liquidity=min(reserve0, reserve1),
            fee=fee,
            gas_cost=self._estimate_gas_cost(dex),
            weight=weight_10
        )
        
        # 그래프에 엣지 추가
        self.graph.add_edge(token0, token1, **edge_01.__dict__)
        self.graph.add_edge(token1, token0, **edge_10.__dict__)
        
        logger.debug(f"거래 쌍 추가: {dex} {token0}-{token1}")
    
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
        """풀 데이터 업데이트"""
        # 해당 풀과 연관된 엣지들 찾아서 업데이트
        for u, v, data in self.graph.edges(data=True):
            if data.get('pool_address') == pool_address:
                # 새로운 환율 계산
                if data['from_token'] == u:
                    new_rate = (reserve1 / reserve0) * (1 - data['fee'])
                else:
                    new_rate = (reserve0 / reserve1) * (1 - data['fee'])
                
                # 엣지 데이터 업데이트
                data['exchange_rate'] = new_rate
                data['weight'] = -math.log(new_rate) if new_rate > 0 else float('inf')
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

    def update_graph_with_trade(self, edges: List[TradingEdge], initial_amount: float):
        """시뮬레이션된 거래로 그래프 상태를 업데이트합니다."""
        logger.info(f"{len(edges)}개의 엣지를 포함하는 거래로 그래프를 업데이트합니다. 초기 금액: {initial_amount}")
        amount = initial_amount

        for edge in edges:
            from_token, to_token = edge.from_token, edge.to_token
            
            if not self.graph.has_edge(from_token, to_token) or self.graph[from_token][to_token].get('pool_address') != edge.pool_address:
                logger.warning(f"거래 엣지를 그래프에서 찾을 수 없음: {from_token} -> {to_token} on {edge.dex}")
                continue

            # 거래 전 리저브 가져오기
            edge_data = self.graph[from_token][to_token]
            output_reserve = edge_data['liquidity']
            if edge_data['exchange_rate'] == 0: continue
            input_reserve = output_reserve / edge_data['exchange_rate']

            # 거래량 계산
            input_amount = amount * (1 - edge.fee)
            k = input_reserve * output_reserve
            if input_reserve + input_amount == 0: continue
            output_amount = output_reserve - k / (input_reserve + input_amount)

            # 새로운 리저브 계산
            new_input_reserve = input_reserve + input_amount
            new_output_reserve = output_reserve - output_amount

            # 해당 풀의 양방향 엣지 모두 업데이트
            self.update_pool_reserves(edge.pool_address, from_token, to_token, new_input_reserve, new_output_reserve)
            
            amount = output_amount

    def update_pool_reserves(self, pool_address: str, token0: str, token1: str, reserve0: float, reserve1: float):
        """특정 풀의 리저브를 업데이트하고 관련된 양방향 엣지를 모두 수정합니다."""
        if reserve0 <= 0 or reserve1 <= 0:
            logger.warning(f"업데이트 중 유동성 부족: {pool_address}")
            # 유동성이 고갈되면 엣지를 제거하거나 가중치를 무한대로 설정할 수 있습니다.
            if self.graph.has_edge(token0, token1):
                self.graph.remove_edge(token0, token1)
            if self.graph.has_edge(token1, token0):
                self.graph.remove_edge(token1, token0)
            return

        # token0 -> token1 엣지 업데이트
        if self.graph.has_edge(token0, token1) and self.graph[token0][token1].get('pool_address') == pool_address:
            edge_data = self.graph[token0][token1]
            new_rate_01 = (reserve1 / reserve0) * (1 - edge_data['fee'])
            edge_data['exchange_rate'] = new_rate_01
            edge_data['weight'] = -math.log(new_rate_01) if new_rate_01 > 0 else float('inf')
            edge_data['liquidity'] = min(reserve0, reserve1)

        # token1 -> token0 엣지 업데이트
        if self.graph.has_edge(token1, token0) and self.graph[token1][token0].get('pool_address') == pool_address:
            edge_data = self.graph[token1][token0]
            new_rate_10 = (reserve0 / reserve1) * (1 - edge_data['fee'])
            edge_data['exchange_rate'] = new_rate_10
            edge_data['weight'] = -math.log(new_rate_10) if new_rate_10 > 0 else float('inf')
            edge_data['liquidity'] = min(reserve0, reserve1)
        
        logger.debug(f"풀 업데이트: {pool_address} ({token0}/{token1}) -> 새로운 리저브: {reserve0:.4f}/{reserve1:.4f}")
