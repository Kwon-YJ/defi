import networkx as nx
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import gc
from src.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class TradingEdge:
    """거래 엣지 정보 (메모리 최적화 버전)"""
    from_token: str
    to_token: str
    dex: str
    pool_address: str
    exchange_rate: float
    liquidity: float
    fee: float
    gas_cost: float
    weight: float  # -log(exchange_rate)
    timestamp: float = 0  # 추가: 업데이트 타임스탬프
    efficiency_score: float = 1.0  # 추가: 엣지 효율성 점수
    
    def __post_init__(self):
        """객체 생성 후 메모리 최적화"""
        # 숫자 값을 필요한 정밀도로 반올림하여 메모리 사용량 줄이기
        self.exchange_rate = round(self.exchange_rate, 10)
        self.liquidity = round(self.liquidity, 6)
        self.fee = round(self.fee, 6)
        self.gas_cost = round(self.gas_cost, 6)
        self.weight = round(self.weight, 10)
        self.timestamp = round(self.timestamp, 2)
        self.efficiency_score = round(self.efficiency_score, 6)

@dataclass
class ArbitrageOpportunity:
    """차익거래 기회 (메모리 최적화 버전)"""
    path: List[str]  # 토큰 경로
    edges: List[TradingEdge]  # 거래 엣지들
    profit_ratio: float  # 수익률
    required_capital: float  # 필요 자본
    estimated_profit: float  # 예상 수익
    gas_cost: float  # 가스 비용
    net_profit: float  # 순수익
    confidence: float  # 신뢰도 (0-1)
    
    def __post_init__(self):
        """객체 생성 후 메모리 최적화"""
        # 숫자 값을 필요한 정밀도로 반올림하여 메모리 사용량 줄이기
        self.profit_ratio = round(self.profit_ratio, 10)
        self.required_capital = round(self.required_capital, 6)
        self.estimated_profit = round(self.estimated_profit, 6)
        self.gas_cost = round(self.gas_cost, 6)
        self.net_profit = round(self.net_profit, 6)
        self.confidence = round(self.confidence, 6)

class DeFiMarketGraph:
    def __init__(self):
        # Multi-graph 지원을 위한 구조 변경
        self.graph = nx.MultiDiGraph()  # MultiDiGraph로 변경하여 동일 토큰 쌍에 여러 엣지 허용
        self.token_nodes = set()
        self.dex_pools = {}  # dex -> {(token0, token1): pool_info}
        self.last_update = 0
        self.edge_efficiency_threshold = 0.1  # 효율성 임계값
        
        # 메모리 최적화를 위한 설정
        self.max_edges = 10000  # 최대 엣지 수 제한
        self.max_age_seconds = 300  # 엣지 최대 수명 (5분)
        self.cleanup_interval = 100  # 클린업 주기 (100번의 작업마다)
        self.operation_count = 0  # 작업 카운터
        
    def add_token(self, token_address: str, symbol: str = None):
        """토큰 노드 추가"""
        if token_address not in self.token_nodes:
            self.graph.add_node(token_address, symbol=symbol)
            self.token_nodes.add(token_address)
            logger.debug(f"토큰 노드 추가: {symbol or token_address}")
    
    def add_trading_pair(self, token0: str, token1: str, dex: str, 
                        pool_address: str, reserve0: float, reserve1: float,
                        fee: float = 0.003, timestamp: float = None):
        """거래 쌍 추가 (Multi-graph 지원 및 메모리 최적화)"""
        import time
        if timestamp is None:
            timestamp = time.time()
            
        # 메모리 관리: 엣지 수 제한 체크
        if self.graph.number_of_edges() >= self.max_edges:
            logger.warning(f"최대 엣지 수 도달 ({self.max_edges}), 오래된 엣지 정리")
            self.cleanup_stale_edges(self.max_age_seconds)
            
            # 여전히 엣지 수가 너무 많으면 새 엣지 추가하지 않음
            if self.graph.number_of_edges() >= self.max_edges * 0.9:
                logger.warning("여전히 엣지 수가 너무 많아 새 엣지 추가 생략")
                return
            
        # 토큰 노드 추가 (이미 존재하면 아무 작업도 하지 않음)
        self.add_token(token0)
        self.add_token(token1)
        
        # 유동성 확인
        if reserve0 <= 0 or reserve1 <= 0:
            logger.warning(f"유동성 부족: {pool_address}")
            return
            
        # token0 -> token1 엣지
        exchange_rate_01 = (reserve1 / reserve0) * (1 - fee)
        weight_01 = -math.log(exchange_rate_01) if exchange_rate_01 > 0 else float('inf')
        
        # 효율성 점수 계산 (유동성과 수수료 기반)
        efficiency_01 = self._calculate_efficiency_score(reserve0, reserve1, fee)
        
        edge_01 = TradingEdge(
            from_token=token0,
            to_token=token1,
            dex=dex,
            pool_address=pool_address,
            exchange_rate=exchange_rate_01,
            liquidity=min(reserve0, reserve1),
            fee=fee,
            gas_cost=self._estimate_gas_cost(dex),
            weight=weight_01,
            timestamp=timestamp,
            efficiency_score=efficiency_01
        )
        
        # token1 -> token0 엣지
        exchange_rate_10 = (reserve0 / reserve1) * (1 - fee)
        weight_10 = -math.log(exchange_rate_10) if exchange_rate_10 > 0 else float('inf')
        
        # 효율성 점수 계산
        efficiency_10 = self._calculate_efficiency_score(reserve1, reserve0, fee)
        
        edge_10 = TradingEdge(
            from_token=token1,
            to_token=token0,
            dex=dex,
            pool_address=pool_address,
            exchange_rate=exchange_rate_10,
            liquidity=min(reserve0, reserve1),
            fee=fee,
            gas_cost=self._estimate_gas_cost(dex),
            weight=weight_10,
            timestamp=timestamp,
            efficiency_score=efficiency_10
        )
        
        # Multi-graph 지원: 동일 토큰 쌍에 여러 DEX 엣지 추가
        # key 파라미터를 사용하여 병렬 엣지 구분
        # Use bulk operations for better performance and memory efficiency
        edges_to_add = [
            (token0, token1, f"{dex}_{pool_address}", asdict(edge_01)),
            (token1, token0, f"{dex}_{pool_address}", asdict(edge_10))
        ]
        
        self.graph.add_edges_from(edges_to_add)
        
        logger.debug(f"거래 쌍 추가 (Multi-graph 지원): {dex} {token0}-{token1}")
        
        # 주기적으로 메모리 클린업
        self._increment_operation_and_cleanup()
    
    def _calculate_efficiency_score(self, reserve_from: float, reserve_to: float, fee: float) -> float:
        """엣지 효율성 점수 계산"""
        # 유동성과 수수료를 기반으로 효율성 점수 계산
        liquidity_score = min(reserve_from, reserve_to) / 1000000  # 1M을 기준으로 정규화
        fee_penalty = fee * 10  # 수수료 페널티
        return max(0.0, min(1.0, liquidity_score - fee_penalty))
    
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
    
    def update_pool_data(self, pool_address: str, reserve0: float, reserve1: float, timestamp: float = None):
        """풀 데이터 업데이트 (Dynamic graph update 및 메모리 최적화)"""
        import time
        if timestamp is None:
            timestamp = time.time()
            
        updated_edges = 0
        # 해당 풀과 연관된 모든 엣지들 찾아서 업데이트
        # Create a list of edges to update to avoid modifying the graph during iteration
        edges_to_update = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if data.get('pool_address') == pool_address:
                edges_to_update.append((u, v, key, data))
        
        # Batch update all edges
        for u, v, key, data in edges_to_update:
            # 새로운 환율 계산
            fee = data.get('fee', 0.003)
            if data['from_token'] == u:
                new_rate = (reserve1 / reserve0) * (1 - fee)
            else:
                new_rate = (reserve0 / reserve1) * (1 - fee)
            
            # 엣지 데이터 업데이트
            data['exchange_rate'] = new_rate
            data['weight'] = -math.log(new_rate) if new_rate > 0 else float('inf')
            data['liquidity'] = min(reserve0, reserve1)
            data['timestamp'] = timestamp
            
            # 효율성 점수 업데이트
            from_token = data['from_token']
            to_token = data['to_token']
            if from_token == u:
                data['efficiency_score'] = self._calculate_efficiency_score(reserve0, reserve1, fee)
            else:
                data['efficiency_score'] = self._calculate_efficiency_score(reserve1, reserve0, fee)
            
            updated_edges += 1
        
        if updated_edges > 0:
            self.last_update = timestamp
            logger.debug(f"풀 데이터 업데이트 완료: {pool_address}, 업데이트된 엣지 수: {updated_edges}")
        
        # 주기적으로 메모리 클린업
        self._increment_operation_and_cleanup()
    
    def prune_inefficient_edges(self):
        """비효율적인 엣지 자동 제거 (Graph pruning 및 메모리 최적화)"""
        edges_to_remove = []
        
        # 효율성 점수가 임계값 이하인 엣지 식별
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            efficiency_score = data.get('efficiency_score', 1.0)
            if efficiency_score < self.edge_efficiency_threshold:
                edges_to_remove.append((u, v, key))
        
        # 엣지 제거
        for u, v, key in edges_to_remove:
            self.graph.remove_edge(u, v, key=key)
        
        if len(edges_to_remove) > 0:
            logger.info(f"비효율적인 엣지 {len(edges_to_remove)}개 제거 완료")
        
        # 메모리 회수
        if len(edges_to_remove) > 0:
            gc.collect()
        
        return len(edges_to_remove)
    
    def get_best_edge(self, from_token: str, to_token: str) -> Optional[TradingEdge]:
        """특정 토큰 쌍의 최고 엣지 반환 (가장 효율적인 거래 경로)"""
        if not self.graph.has_edge(from_token, to_token):
            return None
        
        best_edge = None
        best_efficiency = -1
        
        # 동일 토큰 쌍의 모든 엣지 중 가장 효율적인 것 선택
        for key, data in self.graph[from_token][to_token].items():
            efficiency = data.get('efficiency_score', 0)
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_edge = TradingEdge(**data)
        
        return best_edge
    
    def get_all_edges_between(self, from_token: str, to_token: str) -> List[TradingEdge]:
        """특정 토큰 쌍의 모든 엣지 반환 (Multi-graph 지원)"""
        edges = []
        if self.graph.has_edge(from_token, to_token):
            for key, data in self.graph[from_token][to_token].items():
                edges.append(TradingEdge(**data))
        return edges
    
    def get_graph_stats(self) -> Dict:
        """그래프 통계 정보 (메모리 효율성 포함)"""
        # 메모리 사용량 추정 (더 효율적인 방법)
        try:
            # Use more efficient memory estimation
            edge_count = self.graph.number_of_edges()
            node_count = self.graph.number_of_nodes()
            
            # Estimate memory usage based on counts rather than individual object sizes
            # Rough estimation: each edge ~200 bytes, each node ~100 bytes
            estimated_edge_memory = edge_count * 200
            estimated_node_memory = node_count * 100
            estimated_memory = estimated_edge_memory + estimated_node_memory
            
            return {
                'nodes': node_count,
                'edges': edge_count,
                'tokens': len(self.token_nodes),
                'density': nx.density(self.graph),
                'is_connected': nx.is_weakly_connected(self.graph) if node_count > 0 else False,
                'memory_usage_bytes': estimated_memory,
                'parallel_edges': self._count_parallel_edges(),
                'max_edges_limit': self.max_edges,
                'cleanup_interval': self.cleanup_interval
            }
        except Exception as e:
            logger.error(f"그래프 통계 계산 중 오류 발생: {e}")
            return {
                'nodes': 0,
                'edges': 0,
                'tokens': 0,
                'density': 0,
                'is_connected': False,
                'memory_usage_bytes': 0,
                'parallel_edges': 0,
                'max_edges_limit': self.max_edges,
                'cleanup_interval': self.cleanup_interval,
                'error': str(e)
            }
    
    def _count_parallel_edges(self) -> int:
        """병렬 엣지 수 계산"""
        parallel_count = 0
        for u, v in self.graph.edges():
            if self.graph.number_of_edges(u, v) > 1:
                parallel_count += 1
        return parallel_count
    
    def cleanup_stale_edges(self, max_age_seconds: float = 300):
        """오래된 엣지 정리 (기본 5분)"""
        import time
        current_time = time.time()
        edges_to_remove = []
        
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            timestamp = data.get('timestamp', 0)
            if current_time - timestamp > max_age_seconds:
                edges_to_remove.append((u, v, key))
        
        for u, v, key in edges_to_remove:
            self.graph.remove_edge(u, v, key=key)
        
        if len(edges_to_remove) > 0:
            logger.info(f"오래된 엣지 {len(edges_to_remove)}개 정리 완료")
            
            # 메모리 회수
            gc.collect()
        
        return len(edges_to_remove)
    
    def _increment_operation_and_cleanup(self):
        """작업 카운터 증가 및 주기적 클린업"""
        self.operation_count += 1
        
        # 설정된 주기마다 클린업 수행
        if self.operation_count >= self.cleanup_interval:
            self.operation_count = 0
            
            # 오래된 엣지 정리
            cleaned_count = self.cleanup_stale_edges(self.max_age_seconds)
            
            # 메모리 회수
            if cleaned_count > 0:
                gc.collect()
                
            logger.debug(f"주기적 메모리 클린업 완료: {cleaned_count}개 엣지 정리")
    
    def optimize_memory_usage(self):
        """메모리 사용량 최적화"""
        # 1. 오래된 엣지 정리
        cleaned_count = self.cleanup_stale_edges(self.max_age_seconds)
        
        # 2. 비효율적인 엣지 정리
        pruned_count = self.prune_inefficient_edges()
        
        # 3. 가비지 컬렉션
        gc.collect()
        
        logger.info(f"메모리 최적화 완료: {cleaned_count}개 오래된 엣지, {pruned_count}개 비효율 엣지 정리")
        
        return cleaned_count + pruned_count