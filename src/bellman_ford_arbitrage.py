import math
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
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
            
            # 3. 차익거래 기회로 변환 + Local Search 반복 수행
            for cycle in cycles:
                opportunity = self._perform_local_search_and_repeat(cycle)
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
        for _ in range(max_iterations):
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
        """사이클을 차익거래 기회로 변환 + Local Search 최적화"""
        if len(cycle) < 3:
            return None
        
        edges: List[TradingEdge] = []
        total_gas_cost = 0.0
        
        # 사이클의 각 엣지 정보 수집
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
        
        # Local Search로 최적 시작 자본 및 수익 계산
        ls_result = self._local_search_optimize_amount(edges, total_gas_cost)
        if ls_result is None:
            return None
        required_capital, final_amount, profit_ratio, net_profit = ls_result
        
        confidence = self._calculate_confidence(edges)
        
        return ArbitrageOpportunity(
            path=cycle,
            edges=edges,
            profit_ratio=profit_ratio,
            required_capital=required_capital,
            estimated_profit=final_amount - required_capital,
            gas_cost=total_gas_cost,
            net_profit=net_profit,
            confidence=confidence
        )
    
    def _calculate_profit_ratio(self, edges: List[TradingEdge]) -> float:
        """수익률 계산 (수수료가 exchange_rate에 포함됨)"""
        total_ratio = 1.0
        for edge in edges:
            total_ratio *= edge.exchange_rate
        return total_ratio
    
    def _calculate_confidence(self, edges: List[TradingEdge]) -> float:
        """신뢰도 계산"""
        min_liquidity = min(edge.liquidity for edge in edges)
        liquidity_score = min(min_liquidity / 100.0, 1.0)
        path_score = max(0.5, 1.0 - (len(edges) - 2) * 0.1)
        unique_dexes = len(set(edge.dex for edge in edges))
        diversity_score = min(unique_dexes / len(edges), 1.0)
        return (liquidity_score * 0.5 + path_score * 0.3 + diversity_score * 0.2)

    def _local_search_optimize_amount(self, edges: List[TradingEdge], total_gas_cost: float,
                                     start_factors: Optional[Tuple[float, ...]] = None
                                     ) -> Optional[Tuple[float, float, float, float]]:
        """힐 클라이밍 기반 Local Search로 최적 시작 자본 탐색
        반환: (required_capital, final_amount, profit_ratio, net_profit)
        """
        if not edges:
            return None
        try:
            min_liquidity = max(1e-6, min(edge.liquidity for edge in edges))
            max_capital = min(10.0, min_liquidity * 0.2)
            if max_capital <= 0:
                return None
            min_capital = max_capital * 0.01

            if start_factors is None:
                start_factors = (0.2, 0.4, 0.6, 0.8)
            starts = [max(min_capital, max_capital * f) for f in start_factors]
            best: Optional[Tuple[float, float, float]] = None  # (net_profit, required_capital, final_amount)

            def optimize_from_start(a0: float) -> Tuple[float, float, float]:
                a = a0
                step = max_capital * 0.25
                final_amt = self._simulate_final_amount(edges, a)
                net = (final_amt - a) - total_gas_cost
                for _ in range(20):
                    improved = False
                    up = min(max_capital, a + step)
                    down = max(min_capital, a - step)
                    for cand in (up, down):
                        fa = self._simulate_final_amount(edges, cand)
                        n = (fa - cand) - total_gas_cost
                        if n > net + 1e-9:
                            a, net, final_amt = cand, n, fa
                            improved = True
                    if not improved:
                        step *= 0.5
                        if step < max_capital * 0.005:
                            break
                return (net, a, final_amt)

            # 병렬 실행: 여러 시작점에 대해 동시에 최적화
            max_workers = min(len(starts), 8) if len(starts) > 0 else 0
            if max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for net, a, final_amt in executor.map(optimize_from_start, starts):
                        if best is None or net > best[0]:
                            best = (net, a, final_amt)
            elif max_workers == 1:
                best = optimize_from_start(starts[0])

            if best is None:
                return None

            net_profit, req_cap, final_amount = best
            if final_amount <= 0 or req_cap <= 0:
                return None
            profit_ratio = final_amount / req_cap if req_cap > 0 else 0.0
            if profit_ratio <= 1.0:
                return None
            return (req_cap, final_amount, profit_ratio, net_profit)
        except Exception:
            return None

    def _perform_local_search_and_repeat(self, cycle: List[str]) -> Optional[ArbitrageOpportunity]:
        """논문 Figure 1의 4단계: perform a local search and repeat
        다양한 시작점 팩터로 반복 수행하여 최적 순이익을 찾음
        """
        if len(cycle) < 3:
            return None

        # 엣지/가스 수집
        edges: List[TradingEdge] = []
        total_gas_cost = 0.0
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i + 1]
            if not self.graph.graph.has_edge(u, v):
                return None
            data = self.graph.graph[u][v]
            edge = TradingEdge(**data)
            edges.append(edge)
            total_gas_cost += edge.gas_cost

        # 반복적으로 시작점 세트를 바꿔가며 탐색
        start_sets = [
            (0.2, 0.4, 0.6, 0.8),
            (0.1, 0.3, 0.5, 0.7, 0.9),
            (0.15, 0.35, 0.55, 0.75, 0.95)
        ]

        best: Optional[Tuple[float, float, float, float]] = None  # (req_cap, final_amt, ratio, net)
        for starts in start_sets:
            res = self._local_search_optimize_amount(edges, total_gas_cost, starts)
            if res is None:
                continue
            req_cap, final_amt, ratio, net = res
            if best is None or net > best[3]:
                best = (req_cap, final_amt, ratio, net)

        if best is None:
            return None

        req_cap, final_amt, ratio, net = best
        confidence = self._calculate_confidence(edges)
        return ArbitrageOpportunity(
            path=cycle,
            edges=edges,
            profit_ratio=ratio,
            required_capital=req_cap,
            estimated_profit=final_amt - req_cap,
            gas_cost=total_gas_cost,
            net_profit=net,
            confidence=confidence
        )

    def _simulate_final_amount(self, edges: List[TradingEdge], start_amount: float) -> float:
        """슬리피지를 고려하여 경로 실행 후 최종 금액 계산"""
        amount = start_amount
        for edge in edges:
            slippage = self._calculate_slippage(amount, edge.liquidity)
            amount = amount * edge.exchange_rate * (1 - slippage)
        return amount

    def _calculate_slippage(self, trade_amount: float, liquidity: float) -> float:
        """단순 슬리피지 모델 (SimulationExecutor와 정합 유지)"""
        if liquidity <= 0:
            return 0.1
        impact_ratio = trade_amount / liquidity
        if impact_ratio < 0.01:
            return 0.001
        elif impact_ratio < 0.05:
            return 0.005
        elif impact_ratio < 0.1:
            return 0.02
        else:
            return 0.05
