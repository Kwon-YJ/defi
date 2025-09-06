import math
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random
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
                    
                    # 4. **CRITICAL**: Local Search 알고리즘 적용 (논문의 핵심 알고리즘)
                    optimized_opportunity = self._perform_local_search(opportunity)
                    if optimized_opportunity:
                        opportunities.append(optimized_opportunity)
        
        # 5. **CRITICAL**: Best revenue transaction 선택 로직 구현
        best_opportunities = self._select_best_revenue_transactions(opportunities)
        
        return best_opportunities
    
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
            
            if not self.graph.graph.has_edge(from_token, to_token):
                logger.warning(f"엣지가 존재하지 않음: {from_token} -> {to_token}")
                return None
            
            edge_data = self.graph.graph[from_token][to_token]
            edge = TradingEdge(**edge_data)
            edges.append(edge)
            
            total_gas_cost += edge.gas_cost
            total_fee += edge.fee
        
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

    # =============================================================================
    # **CRITICAL**: Local Search Algorithm Implementation (논문의 핵심 알고리즘)
    # =============================================================================
    
    def _perform_local_search(self, opportunity: ArbitrageOpportunity, 
                            max_iterations: int = 50,
                            num_starting_points: int = 5,
                            max_workers: int = 3) -> Optional[ArbitrageOpportunity]:
        """
        Local Search 알고리즘 - 논문 Figure 1의 4단계
        Hill climbing을 통한 거래량 최적화
        Multiple starting points에서 병렬 실행 (PARALLEL IMPLEMENTATION)
        """
        logger.debug(f"Local search 시작 (병렬 실행): {' -> '.join(opportunity.path)}")
        
        best_opportunity = opportunity
        best_revenue = opportunity.net_profit
        
        # ThreadPoolExecutor를 사용한 병렬 실행
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 각 starting point에서 hill climbing 작업 제출
            future_to_start_point = {}
            
            for start_point in range(num_starting_points):
                initial_amount = self._generate_initial_amount(opportunity, start_point)
                future = executor.submit(
                    self._hill_climbing_optimization, 
                    opportunity, 
                    initial_amount, 
                    max_iterations
                )
                future_to_start_point[future] = start_point
            
            # 결과 수집 및 최적 솔루션 선택
            for future in as_completed(future_to_start_point):
                start_point = future_to_start_point[future]
                try:
                    optimized = future.result()
                    if optimized and optimized.net_profit > best_revenue:
                        best_opportunity = optimized
                        best_revenue = optimized.net_profit
                        logger.debug(f"병렬 search에서 더 좋은 solution 발견 (start_point {start_point}): {best_revenue:.6f} ETH")
                        
                except Exception as e:
                    logger.warning(f"Start point {start_point}에서 hill climbing 실패: {e}")
        
        if best_revenue > opportunity.net_profit:
            logger.info(f"병렬 Local search 최적화 완료: {opportunity.net_profit:.6f} -> {best_revenue:.6f} ETH")
            return best_opportunity
        
        return opportunity
    
    async def _perform_local_search_async(self, opportunity: ArbitrageOpportunity, 
                                        max_iterations: int = 50,
                                        num_starting_points: int = 5,
                                        max_workers: int = 3) -> Optional[ArbitrageOpportunity]:
        """
        비동기 Local Search 알고리즘 - 메인 이벤트 루프를 블로킹하지 않음
        Multiple starting points에서 병렬 실행 (ASYNC PARALLEL IMPLEMENTATION)
        """
        logger.debug(f"비동기 Local search 시작 (병렬 실행): {' -> '.join(opportunity.path)}")
        
        best_opportunity = opportunity
        best_revenue = opportunity.net_profit
        
        # 이벤트 루프에서 ThreadPoolExecutor 실행
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 각 starting point에서 hill climbing 작업 생성
            tasks = []
            
            for start_point in range(num_starting_points):
                initial_amount = self._generate_initial_amount(opportunity, start_point)
                task = loop.run_in_executor(
                    executor,
                    self._hill_climbing_optimization, 
                    opportunity, 
                    initial_amount, 
                    max_iterations
                )
                tasks.append((task, start_point))
            
            # 모든 작업 완료 대기
            for task, start_point in tasks:
                try:
                    optimized = await task
                    if optimized and optimized.net_profit > best_revenue:
                        best_opportunity = optimized
                        best_revenue = optimized.net_profit
                        logger.debug(f"비동기 병렬 search에서 더 좋은 solution 발견 (start_point {start_point}): {best_revenue:.6f} ETH")
                        
                except Exception as e:
                    logger.warning(f"비동기 Start point {start_point}에서 hill climbing 실패: {e}")
        
        if best_revenue > opportunity.net_profit:
            logger.info(f"비동기 병렬 Local search 최적화 완료: {opportunity.net_profit:.6f} -> {best_revenue:.6f} ETH")
            return best_opportunity
        
        return opportunity
    
    def _generate_initial_amount(self, opportunity: ArbitrageOpportunity, 
                               start_point: int) -> float:
        """다양한 시작점에서 초기 거래량 생성"""
        min_liquidity = min(edge.liquidity for edge in opportunity.edges)
        max_amount = min_liquidity * 0.5  # 유동성의 50%까지
        
        if start_point == 0:
            # 보수적 시작점
            return max_amount * 0.1
        elif start_point == 1:
            # 중간 시작점
            return max_amount * 0.3
        elif start_point == 2:
            # 적극적 시작점
            return max_amount * 0.7
        else:
            # 랜덤 시작점
            return max_amount * random.uniform(0.05, 0.9)
    
    def _hill_climbing_optimization(self, opportunity: ArbitrageOpportunity,
                                  initial_amount: float, max_iterations: int) -> Optional[ArbitrageOpportunity]:
        """Hill climbing을 통한 거래량 최적화"""
        current_amount = initial_amount
        current_revenue = self._calculate_revenue_for_amount(opportunity, current_amount)
        
        if current_revenue <= 0:
            return None
        
        # Hill climbing 실행
        for iteration in range(max_iterations):
            # Neighborhood 탐색을 위한 step sizes
            step_sizes = [0.01, 0.05, 0.1, 0.2]  # 다양한 step size
            
            best_neighbor_amount = current_amount
            best_neighbor_revenue = current_revenue
            
            # 각 step size로 neighbor 탐색
            for step_size in step_sizes:
                # 증가 방향
                increase_amount = current_amount * (1 + step_size)
                increase_revenue = self._calculate_revenue_for_amount(opportunity, increase_amount)
                
                if increase_revenue > best_neighbor_revenue:
                    best_neighbor_amount = increase_amount
                    best_neighbor_revenue = increase_revenue
                
                # 감소 방향
                decrease_amount = max(current_amount * (1 - step_size), 0.001)
                decrease_revenue = self._calculate_revenue_for_amount(opportunity, decrease_amount)
                
                if decrease_revenue > best_neighbor_revenue:
                    best_neighbor_amount = decrease_amount
                    best_neighbor_revenue = decrease_revenue
            
            # Local maximum 도달 확인
            if best_neighbor_revenue <= current_revenue:
                logger.debug(f"Local maximum 도달 (iteration {iteration}): {current_revenue:.6f} ETH")
                break
            
            # 더 좋은 neighbor로 이동
            current_amount = best_neighbor_amount
            current_revenue = best_neighbor_revenue
        
        # 최적화된 opportunity 생성
        return self._create_optimized_opportunity(opportunity, current_amount, current_revenue)
    
    def _calculate_revenue_for_amount(self, opportunity: ArbitrageOpportunity, 
                                    amount: float) -> float:
        """특정 거래량에서의 수익 계산 (AMM slippage 고려)"""
        if amount <= 0:
            return 0
        
        current_amount = amount
        total_gas_cost = 0
        
        try:
            # 각 edge를 통한 거래 시뮬레이션
            for edge in opportunity.edges:
                # AMM slippage 계산 (Constant Product Model)
                output_amount = self._calculate_amm_output(current_amount, edge)
                
                if output_amount <= 0:
                    return 0  # 유효하지 않은 거래
                
                current_amount = output_amount * (1 - edge.fee)  # 수수료 차감
                total_gas_cost += edge.gas_cost
            
            # 순수익 계산
            net_revenue = current_amount - amount - total_gas_cost
            return max(net_revenue, 0)
            
        except Exception as e:
            logger.debug(f"Revenue 계산 오류: {e}")
            return 0
    
    def _calculate_amm_output(self, input_amount: float, edge: TradingEdge) -> float:
        """AMM에서 slippage를 고려한 출력량 계산"""
        # Constant Product Model: x * y = k
        # 더 정확한 계산을 위해 유동성과 slippage 고려
        
        # 간단한 slippage 모델 (실제로는 더 복잡함)
        slippage_factor = min(input_amount / edge.liquidity, 0.1)  # 최대 10% slippage
        effective_rate = edge.exchange_rate * (1 - slippage_factor)
        
        return input_amount * effective_rate
    
    def _create_optimized_opportunity(self, original: ArbitrageOpportunity,
                                    optimal_amount: float, optimal_revenue: float) -> ArbitrageOpportunity:
        """최적화된 opportunity 객체 생성"""
        return ArbitrageOpportunity(
            path=original.path,
            edges=original.edges,
            profit_ratio=optimal_revenue / optimal_amount + 1,
            required_capital=optimal_amount,
            estimated_profit=optimal_revenue + original.gas_cost,  # gas 제외한 gross profit
            gas_cost=original.gas_cost,
            net_profit=optimal_revenue,  # 이미 gas 차감된 net profit
            confidence=original.confidence
        )
    
    def _select_best_revenue_transactions(self, opportunities: List[ArbitrageOpportunity],
                                        max_concurrent: int = 3,
                                        min_profit_threshold: float = 0.01) -> List[ArbitrageOpportunity]:
        """
        **CRITICAL**: Best revenue transaction 선택 로직 구현
        - 최고 수익 거래 선택
        - 자원 충돌 방지 (같은 DEX 풀 동시 사용 방지)
        - 위험도 기반 포트폴리오 구성
        """
        if not opportunities:
            return []
        
        logger.debug(f"Best revenue transaction 선택: {len(opportunities)}개 후보에서 선택")
        
        # 1. 수익률 기준 정렬 (이미 정렬되어 있지만 확실히)
        sorted_opportunities = sorted(opportunities, key=lambda x: x.net_profit, reverse=True)
        
        # 2. 최소 수익 임계값 필터링
        filtered_opportunities = [
            opp for opp in sorted_opportunities 
            if opp.net_profit >= min_profit_threshold
        ]
        
        if not filtered_opportunities:
            logger.info("수익 임계값을 만족하는 거래가 없습니다.")
            return []
        
        # 3. 자원 충돌 검사 및 최적 조합 선택
        selected_opportunities = []
        used_dex_pools = set()
        
        for opp in filtered_opportunities:
            # DEX 풀 충돌 검사
            opp_dex_pools = set()
            for edge in opp.edges:
                pool_key = f"{edge.dex}_{edge.from_token}_{edge.to_token}"
                opp_dex_pools.add(pool_key)
            
            # 충돌 검사
            if opp_dex_pools.isdisjoint(used_dex_pools):
                selected_opportunities.append(opp)
                used_dex_pools.update(opp_dex_pools)
                
                logger.info(f"선택된 거래: {' -> '.join(opp.path)}, "
                          f"수익: {opp.net_profit:.6f} ETH, "
                          f"신뢰도: {opp.confidence:.2f}")
                
                # 최대 동시 거래 수 제한
                if len(selected_opportunities) >= max_concurrent:
                    break
            else:
                logger.debug(f"자원 충돌로 인한 거래 제외: {' -> '.join(opp.path)}")
        
        # 4. 최종 선택된 거래들을 수익률 순으로 정렬
        final_selection = sorted(selected_opportunities, key=lambda x: x.net_profit, reverse=True)
        
        total_revenue = sum(opp.net_profit for opp in final_selection)
        logger.info(f"최종 선택: {len(final_selection)}개 거래, 총 예상 수익: {total_revenue:.6f} ETH")
        
        return final_selection
