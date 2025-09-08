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
        
        # 1. Bellman-Ford 알고리즘 실행 및 음의 사이클 노드 찾기
        negative_cycle_node = self._bellman_ford(source_token, max_path_length)
        
        if negative_cycle_node:
            logger.info(f"음의 사이클이 노드 {negative_cycle_node} 근처에서 발견되었습니다!")
            
            # 2. 특정 노드에서 시작하는 음의 사이클 추출
            # 벨만-포드 알고리즘의 특성상, predecessor 체인을 따라가면 사이클을 찾을 수 있습니다.
            cycle = self._extract_cycle_from_node(negative_cycle_node)
            
            if cycle:
                # 3. 차익거래 기회로 변환
                opportunity = self._cycle_to_opportunity(cycle)
                if opportunity and opportunity.net_profit > 0:
                    opportunities.append(opportunity)
        
        return sorted(opportunities, key=lambda x: x.net_profit, reverse=True)
    
    def _bellman_ford(self, source: str, max_iterations: int) -> Optional[str]:
        """Bellman-Ford 알고리즘을 실행하고, 음의 사이클이 발견되면 해당 사이클의 노드를 반환합니다."""
        # 초기화
        self.distances = {node: float('inf') for node in self.graph.graph.nodes}
        self.predecessors = {node: None for node in self.graph.graph.nodes}
        self.distances[source] = 0
        
        # 거리 완화 (Relaxation)
        for i in range(max_iterations):
            updated_in_iteration = False
            for u, v, data in self.graph.graph.edges(data=True):
                weight = data.get('weight', float('inf'))
                
                if self.distances[u] != float('inf') and self.distances[u] + weight < self.distances[v]:
                    self.distances[v] = self.distances[u] + weight
                    self.predecessors[v] = u
                    updated_in_iteration = True
            
            if not updated_in_iteration:
                break # 더 이상 업데이트가 없으면 조기 종료
        
        # 음의 사이클 검사
        for u, v, data in self.graph.graph.edges(data=True):
            weight = data.get('weight', float('inf'))
            if self.distances[u] != float('inf') and self.distances[u] + weight < self.distances[v]:
                logger.debug(f"음의 사이클 감지: {u} -> {v}")
                return v  # 음의 사이클에 포함된 노드 반환
        
        return None  # 음의 사이클 없음

    def _extract_cycle_from_node(self, start_node: str) -> Optional[List[str]]:
        """주어진 노드에서 시작하여 predecessor 체인을 역추적하여 사이클을 찾습니다."""
        path = []
        current = start_node
        
        # 사이클을 찾기 위해 일정 횟수만큼만 역추적 (무한 루프 방지)
        # 그래프의 노드 수만큼 추적하면 반드시 사이클을 찾게 됨
        for _ in range(self.graph.graph.number_of_nodes()):
            if current is None:
                return None # 사이클을 찾기 전에 경로가 끊김
            current = self.predecessors[current]
        
        # `current`는 이제 사이클 내의 한 노드입니다.
        # 사이클 경로를 재구성합니다.
        cycle_node = current
        while True:
            path.append(cycle_node)
            cycle_node = self.predecessors[cycle_node]
            if cycle_node == current:
                path.append(cycle_node) # 시작 노드를 마지막에 추가하여 사이클 완성
                break
            if len(path) > self.graph.graph.number_of_nodes():
                 logger.warning("사이클 재구성 중 무한 루프 의심")
                 return None # 무한 루프 방지

        # 경로를 뒤집어서 올바른 순서로 만듭니다.
        return path[::-1]
    
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
        """Hill Climbing을 사용하여 최적의 거래량을 찾습니다.
        수익이 감소하기 시작하면 탐색을 중단하여 더 효율적으로 최적점을 찾습니다.
        """
        best_profit = 0.0
        optimal_amount = 0.0
        
        # 탐색 범위와 스텝 설정
        step = 0.1  # 0.1 ETH로 탐색 시작
        max_amount = 50.0 # 성능을 위해 50 ETH 이상은 탐색하지 않음
        
        current_amount = step
        last_profit = -1.0

        while current_amount <= max_amount:
            output_amount = self._simulate_trade(edges, current_amount)
            profit = output_amount - current_amount
            
            if profit > best_profit:
                best_profit = profit
                optimal_amount = current_amount
            
            # 수익이 감소하기 시작하면 탐색 중단 (최적점을 지났을 가능성 높음)
            if profit < last_profit and last_profit > 0:
                break
            
            last_profit = profit

            # 탐색 속도를 높이기 위해 거래량에 따라 스텝 크기 조정
            if current_amount >= 5.0:
                step = 1.0
            elif current_amount >= 1.0:
                step = 0.5
                
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
