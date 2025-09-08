import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from src.market_graph import DeFiMarketGraph, ArbitrageOpportunity, TradingEdge
from src.logger import setup_logger

logger = setup_logger(__name__)

class BellmanFordArbitrage:
    def __init__(self, market_graph: DeFiMarketGraph):
        self.graph = market_graph
        self.distances = {}
        self.predecessors = {}
        
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
                    opportunities.append(opportunity)
        
        return sorted(opportunities, key=lambda x: x.net_profit, reverse=True)
    
    def _bellman_ford(self, source: str, max_iterations: int) -> bool:
        """Bellman-Ford 알고리즘 실행"""
        # 초기화
        self.distances = {node: float('inf') for node in self.graph.graph.nodes}
        self.predecessors = {node: None for node in self.graph.graph.nodes}
        self.distances[source] = 0
        
        # 거리 완화 (Relaxation)
        for i in range(max_iterations):
            updated = False
            
            for u, v, data in self.graph.graph.edges(data=True):
                weight = data.get('weight', float('inf'))
                
                if self.distances[u] != float('inf'):
                    new_distance = self.distances[u] + weight
                    
                    if new_distance < self.distances[v]:
                        self.distances[v] = new_distance
                        self.predecessors[v] = u
                        updated = True
            
            if not updated:
                break
        
        # 음의 사이클 검사
        for u, v, data in self.graph.graph.edges(data=True):
            weight = data.get('weight', float('inf'))
            
            if (self.distances[u] != float('inf') and 
                self.distances[u] + weight < self.distances[v]):
                return False  # 음의 사이클 존재
        
        return True  # 음의 사이클 없음
    
    def _extract_negative_cycles(self, source: str) -> List[List[str]]:
        """음의 사이클 추출"""
        cycles = []
        visited = set()
        
        for node in self.graph.graph.nodes:
            if node in visited:
                continue
                
            # 사이클 탐지
            cycle = self._find_cycle_from_node(node, visited)
            if cycle and len(cycle) >= 3:  # 최소 3개 노드
                cycles.append(cycle)
        
        return cycles
    
    def _find_cycle_from_node(self, start_node: str, visited: set) -> Optional[List[str]]:
        """특정 노드에서 시작하는 사이클 찾기"""
        path = []
        current = start_node
        path_set = set()
        
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
            current = self.predecessors.get(current)
        
        # 방문 처리
        for node in path:
            visited.add(node)
        
        return None
    
    def _cycle_to_opportunity(self, cycle: List[str]) -> Optional[ArbitrageOpportunity]:
        """사이클을 차익거래 기회로 변환하고 최적화합니다."""
        if len(cycle) < 3:
            return None
        
        edges = []
        total_gas_cost = 0
        
        for i in range(len(cycle) - 1):
            from_token = cycle[i]
            to_token = cycle[i + 1]
            
            if not self.graph.graph.has_edge(from_token, to_token):
                logger.warning(f"엣지가 존재하지 않음: {from_token} -> {to_token}")
                return None
            
            edge_data = self.graph.graph[from_token][to_token]
            edge = TradingEdge(**edge_data)
            edges.append(edge)
            total_gas_cost += edge.gas_cost
        
        # Local Search를 통해 최적 거래량 및 수익 계산
        optimal_amount, max_profit = self._optimize_trade_volume(edges)
        
        if max_profit <= total_gas_cost:
            return None # 가스비를 제하고 나면 수익이 없음

        net_profit = max_profit - total_gas_cost
        profit_ratio = (optimal_amount + net_profit) / optimal_amount if optimal_amount > 0 else 0

        confidence = self._calculate_confidence(edges)
        
        return ArbitrageOpportunity(
            path=cycle,
            edges=edges,
            profit_ratio=profit_ratio,
            required_capital=optimal_amount,
            estimated_profit=max_profit, # 이제 이게 총 수익
            gas_cost=total_gas_cost,
            net_profit=net_profit, # 가스비 제외 순수익
            confidence=confidence
        )

    def _simulate_trade(self, edges: List[TradingEdge], initial_amount: float) -> float:
        """Slippage를 고려하여 거래를 시뮬레이션하고 최종 결과 금액을 반환합니다."""
        amount = initial_amount
        for edge in edges:
            # Constant product formula: (x + dx) * (y - dy) = k
            # dy = y - k / (x + dx)
            # dx is the input amount, dy is the output amount
            # k = x * y
            
            # A more accurate model would use the actual pool reserves.
            # For now, we use a simplified model based on liquidity and exchange rate.
            # Let's assume liquidity represents the reserve of the output token.
            output_reserve = edge.liquidity
            if edge.exchange_rate == 0: return 0
            input_reserve = output_reserve / edge.exchange_rate

            # Input amount for this step
            input_amount = amount * (1 - edge.fee)

            # Calculate output amount with slippage
            k = input_reserve * output_reserve
            if input_reserve + input_amount == 0: return 0
            
            output_amount = output_reserve - k / (input_reserve + input_amount)
            
            if output_amount <= 0:
                return 0 # 거래 불가능
            
            amount = output_amount
            
        return amount

    def _optimize_trade_volume(self, edges: List[TradingEdge]) -> Tuple[float, float]:
        """Hill Climbing을 사용하여 최적의 거래량을 찾습니다."""
        best_profit = 0.0
        optimal_amount = 0.0
        
        # 탐색 범위와 스텝 설정
        step = 0.1  # Start with 0.1 ETH
        max_amount = 50.0 # Don't search beyond 50 ETH for performance
        
        current_amount = step
        
        while current_amount <= max_amount:
            output_amount = self._simulate_trade(edges, current_amount)
            profit = output_amount - current_amount
            
            if profit > best_profit:
                best_profit = profit
                optimal_amount = current_amount
            # If profit starts to decrease, we might have passed the peak.
            # A more robust implementation would check for this and stop.
            
            # Increase step size for larger amounts to speed up search
            if current_amount >= 1.0:
                step = 0.5
            if current_amount >= 5.0:
                step = 1.0
                
            current_amount += step

        return optimal_amount, best_profit

    def _calculate_profit_ratio(self, edges: List[TradingEdge]) -> float:
        """수익률 계산 (참고용, 실제 수익은 시뮬레이션으로)"""
        total_ratio = 1.0
        
        for edge in edges:
            total_ratio *= edge.exchange_rate
        
        return total_ratio
    
    def _calculate_confidence(self, edges: List[TradingEdge]) -> float:
        """신뢰도 계산"""
        # 유동성 기반 신뢰도
        min_liquidity = min(edge.liquidity for edge in edges if edge.liquidity > 0) if any(e.liquidity > 0 for e in edges) else 1.0
        liquidity_score = min(min_liquidity / 100.0, 1.0)  # 100 ETH 기준
        
        # 경로 길이 기반 신뢰도 (짧을수록 좋음)
        path_score = max(0.5, 1.0 - (len(edges) - 2) * 0.1)
        
        # DEX 다양성 기반 신뢰도
        unique_dexes = len(set(edge.dex for edge in edges))
        diversity_score = min(unique_dexes / len(edges), 1.0)
        
        return (liquidity_score * 0.5 + path_score * 0.3 + diversity_score * 0.2)
