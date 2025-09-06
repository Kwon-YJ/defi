import math
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import random
import time
import numpy as np
from src.market_graph import DeFiMarketGraph, ArbitrageOpportunity, TradingEdge
from src.logger import setup_logger

logger = setup_logger(__name__)

class BellmanFordArbitrage:
    def __init__(self, market_graph: DeFiMarketGraph):
        self.graph = market_graph
        self.distances = {}
        self.predecessors = {}
        
        # **성능 최적화**: 메모리 효율성을 위한 캐시 및 버퍼
        self._edge_cache = None  # 사전 계산된 edge 리스트 캐시
        self._node_cache = None  # 노드 리스트 캐시
        self._last_graph_update = 0  # 그래프 업데이트 타임스탬프
        
        # **논문 성능 기준**: 6.43초 내 처리를 위한 성능 모니터링
        self.performance_metrics = {
            'total_processing_time': 0.0,
            'bellman_ford_time': 0.0,
            'cycle_extraction_time': 0.0,
            'local_search_time': 0.0,
            'cycles_found': 0,
            'opportunities_generated': 0
        }
        
    def find_negative_cycles(self, source_token: str, 
                           max_path_length: int = 4) -> List[ArbitrageOpportunity]:
        """
        최적화된 음의 사이클 탐지를 통한 차익거래 기회 발견
        논문 성능 기준: 평균 6.43초 내 처리 완료
        96개 protocol actions, 25개 assets 처리 가능한 확장성 확보
        """
        start_time = time.time()
        logger.info(f"음의 사이클 탐지 시작: source={source_token}, max_path_length={max_path_length}")
        opportunities = []
        
        # **최적화 13**: 그래프 변경 사항 확인 및 캐시 무효화
        self._update_cache_if_needed()
        
        # **최적화 14**: 소스 노드 유효성 검증
        if source_token not in self.graph.graph.nodes:
            logger.warning(f"소스 토큰 {source_token}이 그래프에 존재하지 않습니다.")
            return []
        
        # 1. **최적화된 Bellman-Ford 알고리즘 실행**
        bf_start_time = time.time()
        has_negative_cycle = not self._bellman_ford(source_token, max_path_length)
        self.performance_metrics['bellman_ford_time'] = time.time() - bf_start_time
        
        if has_negative_cycle:
            logger.info(f"음의 사이클 발견! Bellman-Ford 실행시간: {self.performance_metrics['bellman_ford_time']:.3f}초")
            
            # 2. **최적화된 음의 사이클 추출**
            cycle_start_time = time.time()
            cycles = self._extract_negative_cycles(source_token)
            self.performance_metrics['cycle_extraction_time'] = time.time() - cycle_start_time
            self.performance_metrics['cycles_found'] = len(cycles)
            
            logger.info(f"사이클 추출 완료: {len(cycles)}개 발견, 실행시간: {self.performance_metrics['cycle_extraction_time']:.3f}초")
            
            # 3. 차익거래 기회로 변환
            for cycle in cycles:
                opportunity = self._cycle_to_opportunity(cycle)
                if opportunity and opportunity.net_profit > 0:
                    
                    # 4. **CRITICAL**: Local Search 알고리즘 적용 (논문의 핵심 알고리즘)
                    ls_start_time = time.time()
                    optimized_opportunity = self._perform_local_search(opportunity)
                    if optimized_opportunity:
                        opportunities.append(optimized_opportunity)
                        self.performance_metrics['opportunities_generated'] += 1
                    self.performance_metrics['local_search_time'] += time.time() - ls_start_time
        
        # 5. **CRITICAL**: Best revenue transaction 선택 로직 구현
        best_opportunities = self._select_best_revenue_transactions(opportunities)
        
        # **성능 모니터링**: 총 처리 시간 기록
        total_time = time.time() - start_time
        self.performance_metrics['total_processing_time'] = total_time
        
        # **논문 성능 기준 검증**: 6.43초 목표 달성 확인
        if total_time > 6.43:
            logger.warning(f"성능 목표 미달성! 실행시간: {total_time:.3f}초 > 6.43초 (논문 기준)")
        else:
            logger.info(f"성능 목표 달성! 실행시간: {total_time:.3f}초 ≤ 6.43초 (논문 기준)")
        
        # **상세 성능 로그**
        logger.info(f"성능 분석 - BF: {self.performance_metrics['bellman_ford_time']:.3f}초, "
                   f"사이클추출: {self.performance_metrics['cycle_extraction_time']:.3f}초, "
                   f"로컬서치: {self.performance_metrics['local_search_time']:.3f}초")
        
        return best_opportunities
    
    def _bellman_ford(self, source: str, max_iterations: int) -> bool:
        """최적화된 Bellman-Ford 알고리즘 실행 - 논문 성능 기준 달성을 위한 최적화"""
        # 초기화 - Dictionary comprehension 최적화
        nodes = list(self.graph.graph.nodes)
        self.distances = {node: float('inf') for node in nodes}
        self.predecessors = {node: None for node in nodes}
        self.distances[source] = 0
        
        # **최적화 1**: Edge 리스트 사전 계산 (매번 iterator 생성 방지)
        edges_list = [(u, v, data.get('weight', float('inf'))) 
                     for u, v, data in self.graph.graph.edges(data=True)
                     if data.get('weight', float('inf')) != float('inf')]
        
        # **최적화 2**: Early termination with change tracking
        for i in range(max_iterations):
            updated = False
            
            # **최적화 3**: 사전 계산된 edge 리스트 사용
            for u, v, weight in edges_list:
                if self.distances[u] != float('inf'):
                    new_distance = self.distances[u] + weight
                    
                    # **최적화 4**: 부동소수점 비교 최적화
                    if new_distance < self.distances[v] - 1e-10:  # 수치 안정성
                        self.distances[v] = new_distance
                        self.predecessors[v] = u
                        updated = True
            
            # **최적화 5**: Early termination (수렴 시 조기 종료)
            if not updated:
                logger.debug(f"Bellman-Ford 조기 수렴 완료: {i+1}회 반복")
                break
        
        # **최적화 6**: 음의 사이클 검사 - 이미 계산된 edge 리스트 재사용
        for u, v, weight in edges_list:
            if (self.distances[u] != float('inf') and 
                self.distances[u] + weight < self.distances[v] - 1e-10):
                logger.debug(f"음의 사이클 탐지: {u} -> {v}, weight={weight}")
                return False  # 음의 사이클 존재
        
        return True  # 음의 사이클 없음
    
    def _extract_negative_cycles(self, source: str) -> List[List[str]]:
        """최적화된 음의 사이클 추출 - 성능 최적화 및 메모리 효율성 개선"""
        cycles = []
        visited = set()
        
        # **최적화 7**: 음의 거리를 가진 노드만 검사 (pruning)
        negative_distance_nodes = [
            node for node, dist in self.distances.items() 
            if dist != float('inf') and dist < -1e-10
        ]
        
        logger.debug(f"음의 거리 노드 {len(negative_distance_nodes)}개 발견, 전체 {len(self.graph.graph.nodes)}개 중")
        
        # **최적화 8**: 우선순위 기반 사이클 탐지 (가장 음의 값이 큰 노드부터)
        sorted_nodes = sorted(negative_distance_nodes, key=lambda n: self.distances[n])
        
        for node in sorted_nodes:
            if node in visited:
                continue
                
            # **최적화 9**: 깊이 제한을 통한 탐색 공간 축소
            cycle = self._find_cycle_from_node_optimized(node, visited, max_depth=6)
            if cycle and len(cycle) >= 3:  # 최소 3개 노드
                # **최적화 10**: 사이클 품질 검증 (실제로 음의 가중치인지 확인)
                if self._validate_negative_cycle(cycle):
                    cycles.append(cycle)
                    logger.debug(f"유효한 음의 사이클 발견: {' -> '.join(cycle)}")
                else:
                    logger.debug(f"무효한 사이클 제외: {' -> '.join(cycle)}")
        
        return cycles
    
    def _find_cycle_from_node_optimized(self, start_node: str, visited: set, max_depth: int = 6) -> Optional[List[str]]:
        """최적화된 사이클 탐지 - 깊이 제한 및 메모리 효율성 개선"""
        path = []
        current = start_node
        path_set = set()
        depth = 0
        
        # **최적화 11**: 깊이 제한으로 무한루프 및 과도한 탐색 방지
        while current is not None and current not in visited and depth < max_depth:
            if current in path_set:
                # 사이클 발견
                cycle_start = path.index(current)
                cycle = path[cycle_start:] + [current]
                
                # **최적화 12**: 방문 처리 최적화
                visited.update(path)
                
                logger.debug(f"사이클 발견 (깊이 {depth}): {' -> '.join(cycle)}")
                return cycle
            
            path.append(current)
            path_set.add(current)
            current = self.predecessors.get(current)
            depth += 1
        
        # 방문 처리
        visited.update(path)
        
        if depth >= max_depth:
            logger.debug(f"최대 깊이 {max_depth} 도달, 탐색 중단")
        
        return None
    
    def _validate_negative_cycle(self, cycle: List[str]) -> bool:
        """사이클이 실제로 음의 가중치를 가지는지 검증"""
        if len(cycle) < 3:
            return False
        
        total_weight = 0.0
        
        # 사이클의 각 엣지 가중치 합계 계산
        for i in range(len(cycle) - 1):
            from_token = cycle[i]
            to_token = cycle[i + 1]
            
            if not self.graph.graph.has_edge(from_token, to_token):
                logger.debug(f"엣지 없음: {from_token} -> {to_token}")
                return False
            
            edge_data = self.graph.graph[from_token][to_token]
            weight = edge_data.get('weight', float('inf'))
            
            if weight == float('inf'):
                return False
            
            total_weight += weight
        
        # 음의 사이클인지 확인
        is_negative = total_weight < -1e-10
        logger.debug(f"사이클 가중치 합계: {total_weight:.6f}, 음의 사이클: {is_negative}")
        
        return is_negative
    
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
    
    # =============================================================================
    # **성능 최적화**: 캐시 및 메모리 효율성 최적화 메서드들
    # =============================================================================
    
    def _update_cache_if_needed(self):
        """그래프 변경 사항 확인 및 캐시 업데이트"""
        current_update_time = getattr(self.graph, 'last_update', 0)
        
        if current_update_time > self._last_graph_update or self._edge_cache is None:
            logger.debug(f"캐시 업데이트 필요 - 그래프 업데이트: {current_update_time} > {self._last_graph_update}")
            
            # Edge 리스트 캐시 갱신
            self._edge_cache = [
                (u, v, data.get('weight', float('inf'))) 
                for u, v, data in self.graph.graph.edges(data=True)
                if data.get('weight', float('inf')) != float('inf')
            ]
            
            # Node 리스트 캐시 갱신  
            self._node_cache = list(self.graph.graph.nodes)
            
            # 업데이트 시점 기록
            self._last_graph_update = current_update_time
            
            logger.debug(f"캐시 업데이트 완료 - Edges: {len(self._edge_cache)}, Nodes: {len(self._node_cache)}")
    
    def get_performance_metrics(self) -> Dict:
        """성능 메트릭 조회 - 논문 성능 기준 6.43초 달성 확인"""
        total_time = self.performance_metrics['total_processing_time']
        target_time = 6.43  # 논문 목표 시간
        
        return {
            **self.performance_metrics,
            'target_time_seconds': target_time,
            'performance_ratio': total_time / target_time if target_time > 0 else 0,
            'meets_paper_requirement': total_time <= target_time,
            'efficiency_score': max(0, (target_time - total_time) / target_time) if target_time > 0 else 0
        }
    
    def reset_performance_metrics(self):
        """성능 메트릭 초기화"""
        self.performance_metrics = {
            'total_processing_time': 0.0,
            'bellman_ford_time': 0.0,
            'cycle_extraction_time': 0.0,
            'local_search_time': 0.0,
            'cycles_found': 0,
            'opportunities_generated': 0
        }
