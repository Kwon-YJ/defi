import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from src.market_graph import DeFiMarketGraph, ArbitrageOpportunity, TradingEdge
from src.local_search import LocalSearch
from src.logger import setup_logger

logger = setup_logger(__name__)

class BellmanFordArbitrage:
    def __init__(self, market_graph: DeFiMarketGraph):
        self.graph = market_graph
        self.distances = {}
        self.predecessors = {}
        self.local_search = LocalSearch()
        
    def find_negative_cycles(self, source_token: str, 
                           max_path_length: int = 4) -> List[ArbitrageOpportunity]:
        """음의 사이클 탐지를 통한 차익거래 기회 발견"""
        opportunities = []
        
        # 1. Bellman-Ford 알고리즘 실행
        if not self._bellman_ford(source_token, max_path_length):
            logger.info("음의 사이클이 발견되었습니다!")
            
            # 2. 음의 사이클 추출
            cycles = self._extract_negative_cycles(source_token)
            
            # 3. 차익거래 기회로 변환
            for cycle in cycles:
                opportunity = self._cycle_to_opportunity(cycle)
                if opportunity and opportunity.net_profit > 0:
                    # 4. Local Search를 통한 최적화 (논문의 "perform a local search and repeat" 구현)
                    optimized_opportunity = self.local_search.multi_start_search(opportunity)
                    opportunities.append(optimized_opportunity)
        
        return sorted(opportunities, key=lambda x: x.net_profit, reverse=True)
    
    def _bellman_ford(self, source: str, max_iterations: int) -> bool:
        """Bellman-Ford 알고리즘 실행 - 논문의 정확한 구현"""
        # 초기화
        self.distances = {node: float('inf') for node in self.graph.graph.nodes}
        self.predecessors = {node: None for node in self.graph.graph.nodes}
        self.distances[source] = 0
        
        # 거리 완화 (Relaxation) - 논문에 맞게 최적화
        for i in range(max_iterations):
            updated = False
            
            # Multi-graph 지원: 동일 토큰 쌍의 엣지 중 가장 좋은 것만 사용
            best_edges = self._get_best_edges()
            
            # 모든 최고 엣지에 대해 거리 완화 수행
            for (u, v), data in best_edges.items():
                weight = data.get('weight', float('inf'))
                
                # 무한대가 아닌 경우에만 업데이트
                if self.distances[u] != float('inf'):
                    new_distance = self.distances[u] + weight
                    
                    # 더 짧은 경로를 찾은 경우 업데이트
                    if new_distance < self.distances[v]:
                        self.distances[v] = new_distance
                        self.predecessors[v] = (u, data)  # predecessor에 엣지 정보도 저장
                        updated = True
            
            # 업데이트가 없으면 조기 종료
            if not updated:
                break
        
        # 음의 사이클 검사
        best_edges = self._get_best_edges()
        for (u, v), data in best_edges.items():
            weight = data.get('weight', float('inf'))
            
            # 거리 완화가 더 가능하면 음의 사이클 존재
            if (self.distances[u] != float('inf') and 
                self.distances[u] + weight < self.distances[v]):
                return False  # 음의 사이클 존재
        
        return True  # 음의 사이클 없음
    
    def _get_best_edges(self) -> Dict[Tuple[str, str], Dict]:
        """각 토큰 쌍의 최고 엣지 반환 (Multi-graph 지원)"""
        best_edges = {}
        
        # MultiDiGraph의 모든 엣지 처리
        for u, v, data in self.graph.graph.edges(data=True):
            edge_key = (u, v)
            current_weight = data.get('weight', float('inf'))
            
            # 동일 토큰 쌍의 엣지 중 가장 가중치가 낮은 것 선택
            if edge_key not in best_edges or current_weight < best_edges[edge_key].get('weight', float('inf')):
                best_edges[edge_key] = data
                
        return best_edges
    
    def _extract_negative_cycles(self, source: str) -> List[List[str]]:
        """음의 사이클 추출 - 최적화된 구현"""
        cycles = []
        visited = set()
        
        # 모든 노드에서 시작하여 사이클 탐지
        for node in self.graph.graph.nodes:
            if node in visited:
                continue
                
            # 사이클 탐지
            cycle = self._find_cycle_from_node(node, visited)
            if cycle and len(cycle) >= 3:  # 최소 3개 노드 (실제 사이클)
                cycles.append(cycle)
        
        return cycles
    
    def _find_cycle_from_node(self, start_node: str, visited: set) -> Optional[List[str]]:
        """특정 노드에서 시작하는 사이클 찾기 - 개선된 알고리즘"""
        path = []
        current = start_node
        path_set = set()
        
        # predecessor 체인을 따라가며 사이클 탐지
        while current is not None and current not in visited:
            if current in path_set:
                # 사이클 발견
                cycle_start = path.index(current)
                cycle = path[cycle_start:] + [current]
                
                # 방문 처리
                for node in path:
                    visited.add(node)
                
                return cycle
            
            path.append(current)
            path_set.add(current)
            # predecessor가 (node, edge_data) 튜플인 경우 처리
            pred_info = self.predecessors.get(current)
            current = pred_info[0] if isinstance(pred_info, tuple) else pred_info
        
        # 방문 처리
        for node in path:
            visited.add(node)
        
        return None
    
    def _cycle_to_opportunity(self, cycle: List[str]) -> Optional[ArbitrageOpportunity]:
        """사이클을 차익거래 기회로 변환"""
        if len(cycle) < 3:
            return None
        
        edges = []
        total_gas_cost = 0
        total_fee = 0
        
        # 사이클의 각 엣지 정보 수집
        for i in range(len(cycle) - 1):
            from_token = cycle[i]
            to_token = cycle[i + 1]
            
            # Multi-graph 지원: 최고 엣지 가져오기
            best_edge = self.graph.get_best_edge(from_token, to_token)
            if best_edge is None:
                logger.warning(f"엣지가 존재하지 않음: {from_token} -> {to_token}")
                return None
            
            edges.append(best_edge)
            total_gas_cost += best_edge.gas_cost
            total_fee += best_edge.fee
        
        # 수익률 계산
        profit_ratio = self._calculate_profit_ratio(edges)
        
        if profit_ratio <= 1.0:
            return None  # 수익 없음
        
        # 필요 자본 추정 (가장 큰 유동성 제약 기준)
        min_liquidity = min(edge.liquidity for edge in edges)
        required_capital = min(min_liquidity * 0.1, 10.0)  # 최대 10 ETH
        
        estimated_profit = required_capital * (profit_ratio - 1)
        net_profit = estimated_profit - total_gas_cost
        
        # 신뢰도 계산 (유동성, 가격 안정성 기준)
        confidence = self._calculate_confidence(edges)
        
        return ArbitrageOpportunity(
            path=cycle,
            edges=edges,
            profit_ratio=profit_ratio,
            required_capital=required_capital,
            estimated_profit=estimated_profit,
            gas_cost=total_gas_cost,
            net_profit=net_profit,
            confidence=confidence
        )
    
    def _calculate_profit_ratio(self, edges: List[TradingEdge]) -> float:
        """수익률 계산"""
        total_ratio = 1.0
        
        for edge in edges:
            total_ratio *= edge.exchange_rate
        
        return total_ratio
    
    def _calculate_confidence(self, edges: List[TradingEdge]) -> float:
        """신뢰도 계산"""
        # 유동성 기반 신뢰도
        min_liquidity = min(edge.liquidity for edge in edges)
        liquidity_score = min(min_liquidity / 100.0, 1.0)  # 100 ETH 기준
        
        # 경로 길이 기반 신뢰도 (짧을수록 좋음)
        path_score = max(0.5, 1.0 - (len(edges) - 2) * 0.1)
        
        # DEX 다양성 기반 신뢰도
        unique_dexes = len(set(edge.dex for edge in edges))
        diversity_score = min(unique_dexes / len(edges), 1.0)
        
        return (liquidity_score * 0.5 + path_score * 0.3 + diversity_score * 0.2)
    
    def update_graph_state(self, executed_opportunity: ArbitrageOpportunity):
        """
        실행된 기회에 따라 그래프 상태 업데이트
        논문에서 언급된 "update the graph g after every action" 구현
        """
        # 각 엣지의 유동성 업데이트
        for edge in executed_opportunity.edges:
            # 실제 구현에서는 풀의 리저브를 업데이트해야 함
            # 여기서는 간단히 예시로만 구현
            if self.graph.graph.has_edge(edge.from_token, edge.to_token):
                # 엣지 데이터 업데이트
                # Note: Multi-graph에서는 모든 병렬 엣지를 업데이트해야 함
                for u, v, key, data in self.graph.graph.edges(keys=True, data=True):
                    if (u == edge.from_token and v == edge.to_token and 
                        data.get('pool_address') == edge.pool_address):
                        data['liquidity'] = max(0, data['liquidity'] - executed_opportunity.required_capital)
                
        logger.info(f"Graph state updated after executing opportunity: {executed_opportunity.path}")