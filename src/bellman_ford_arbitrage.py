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
        """음의 사이클 탐지를 통한 차익거래 기회 발견 (최적화된 버전)"""
        opportunities = []
        
        # 성능 제한: 너무 많은 노드가 있는 경우 조기 종료
        node_count = len(self.graph.graph.nodes)
        if node_count > 1000:  # 1000개 노드 이상이면 경고
            logger.warning(f"그래프 노드 수가 너무 많습니다: {node_count} nodes")
            # 노드 수를 제한하여 처리
            max_nodes = 500
            nodes = list(self.graph.graph.nodes)[:max_nodes]
        else:
            nodes = list(self.graph.graph.nodes)
        
        # 1. Bellman-Ford 알고리즘 실행
        if not self._bellman_ford(source_token, max_path_length):
            logger.info("음의 사이클이 발견되었습니다!")
            
            # 2. 음의 사이클 추출
            cycles = self._extract_negative_cycles(source_token)
            
            # 성능 제한: 너무 많은 사이클이 발견되면 상위 N개만 처리
            max_cycles = 50
            if len(cycles) > max_cycles:
                logger.warning(f"발견된 사이클 수가 너무 많습니다: {len(cycles)} cycles, 상위 {max_cycles}개만 처리")
                cycles = sorted(cycles, key=len)[:max_cycles]
            
            # 3. 차익거래 기회로 변환
            processed_opportunities = 0
            max_opportunities = 100  # 최대 100개의 기회만 처리
            
            for cycle in cycles:
                # 처리된 기회 수 제한
                if processed_opportunities >= max_opportunities:
                    logger.info(f"최대 처리 기회 수 도달: {max_opportunities}")
                    break
                
                opportunity = self._cycle_to_opportunity(cycle)
                if opportunity and opportunity.net_profit > 0:
                    # 4. Local Search를 통한 최적화 (논문의 "perform a local search and repeat" 구현)
                    # Use the new asynchronous multi-start search for better performance
                    try:
                        # For now, we'll use the synchronous version for compatibility
                        optimized_opportunity = self.local_search.multi_start_search(opportunity)
                        opportunities.append(optimized_opportunity)
                    except Exception as e:
                        logger.error(f"Local search optimization failed for opportunity: {e}")
                        # Fall back to original opportunity
                        opportunities.append(opportunity)
                    processed_opportunities += 1
            
            # 수익이 높은 순으로 정렬 (최대 20개만 반환)
            opportunities = sorted(opportunities, key=lambda x: x.net_profit, reverse=True)[:20]
        
        return opportunities
    
    def _bellman_ford(self, source: str, max_iterations: int) -> bool:
        """Bellman-Ford 알고리즘 실행 - 논문의 정확한 구현 (최적화 버전)"""
        # 초기화 - 더 효율적인 방식으로
        nodes = list(self.graph.graph.nodes)
        self.distances = {node: float('inf') for node in nodes}
        self.predecessors = {node: None for node in nodes}
        self.distances[source] = 0
        
        # 미리 계산된 최고 엣지들 (반복 계산 방지)
        best_edges_cache = {}
        
        # 거리 완화 (Relaxation) - 논문에 맞게 최적화
        for i in range(max_iterations):
            updated = False
            
            # Multi-graph 지원: 동일 토큰 쌍의 엣지 중 가장 좋은 것만 사용
            # 캐싱을 통해 반복 계산 방지
            if i == 0 or not best_edges_cache:
                best_edges = self._get_best_edges_cached()
                best_edges_cache = best_edges
            else:
                best_edges = best_edges_cache
            
            # 벡터화된 업데이트를 위한 준비
            updates = []
            
            # 모든 최고 엣지에 대해 거리 완화 수행
            for (u, v), data in best_edges.items():
                weight = data.get('weight', float('inf'))
                
                # 무한대가 아닌 경우에만 업데이트
                if self.distances[u] != float('inf'):
                    new_distance = self.distances[u] + weight
                    
                    # 더 짧은 경로를 찾은 경우 업데이트 예약
                    if new_distance < self.distances[v]:
                        updates.append((v, new_distance, (u, data)))
                        updated = True
            
            # 벡터화된 업데이트 적용
            for v, new_distance, predecessor in updates:
                self.distances[v] = new_distance
                self.predecessors[v] = predecessor
            
            # 업데이트가 없으면 조기 종료
            if not updated:
                break
        
        # 음의 사이클 검사 - 최적화된 버전
        if best_edges_cache:
            best_edges = best_edges_cache
        else:
            best_edges = self._get_best_edges_cached()
            best_edges_cache = best_edges
        
        # 벡터화된 음의 사이클 검사
        negative_cycle_found = False
        for (u, v), data in best_edges.items():
            weight = data.get('weight', float('inf'))
            
            # 거리 완화가 더 가능하면 음의 사이클 존재
            if (self.distances[u] != float('inf') and 
                self.distances[u] + weight < self.distances[v]):
                negative_cycle_found = True
                break  # 조기 종료
        
        return not negative_cycle_found  # 음의 사이클이 없으면 True 반환
    
    def _get_best_edges_cached(self) -> Dict[Tuple[str, str], Dict]:
        """각 토큰 쌍의 최고 엣지 반환 (캐싱된 버전)"""
        # 캐시 키 생성 (간단한 해시 기반)
        cache_key = f"best_edges_{hash(tuple(self.graph.graph.edges()))}"
        
        # 캐시된 결과가 있는지 확인
        if hasattr(self, '_cached_best_edges') and hasattr(self, '_cache_key'):
            if self._cache_key == cache_key:
                return self._cached_best_edges
        
        # 캐시되지 않은 경우 새로 계산
        best_edges = {}
        
        # MultiDiGraph의 모든 엣지 처리 (효율적인 방식)
        edges_by_pair = defaultdict(list)
        
        # 먼저 모든 엣지를 토큰 쌍별로 그룹화
        for u, v, data in self.graph.graph.edges(data=True):
            edges_by_pair[(u, v)].append(data)
        
        # 각 토큰 쌍별로 최고 엣지 선택
        for (u, v), edge_data_list in edges_by_pair.items():
            if len(edge_data_list) == 1:
                # 엣지가 하나뿐인 경우 바로 사용
                best_edges[(u, v)] = edge_data_list[0]
            else:
                # 여러 엣지 중 가장 가중치가 낮은 것 선택
                best_edge = min(edge_data_list, key=lambda x: x.get('weight', float('inf')))
                best_edges[(u, v)] = best_edge
        
        # 캐시 저장
        self._cached_best_edges = best_edges
        self._cache_key = cache_key
        
        return best_edges
    
    def _get_best_edges(self) -> Dict[Tuple[str, str], Dict]:
        """각 토큰 쌍의 최고 엣지 반환 (Multi-graph 지원) - 최적화된 버전"""
        # 캐시된 결과가 있는지 확인
        cache_key = f"best_edges_{hash(tuple(self.graph.graph.edges()))}"
        
        if hasattr(self, '_cached_best_edges_v2') and hasattr(self, '_cache_key_v2'):
            if self._cache_key_v2 == cache_key:
                return self._cached_best_edges_v2
        
        # 최적화된 방식으로 최고 엣지 찾기
        best_edges = {}
        
        # 토큰 쌍별 엣지 그룹화 (한 번의 순회로 처리)
        edge_groups = defaultdict(list)
        
        for u, v, data in self.graph.graph.edges(data=True):
            edge_key = (u, v)
            edge_groups[edge_key].append(data)
        
        # 각 토큰 쌍별로 최고 엣지 선택 (벡터화된 처리)
        for edge_key, edge_data_list in edge_groups.items():
            if len(edge_data_list) == 1:
                # 엣지가 하나뿐인 경우
                best_edges[edge_key] = edge_data_list[0]
            else:
                # 여러 엣지 중 최고의 것을 선택
                # 가중치 기준으로 정렬하여 최적의 엣지 선택
                sorted_edges = sorted(
                    edge_data_list, 
                    key=lambda x: x.get('weight', float('inf'))
                )
                best_edges[edge_key] = sorted_edges[0]  # 가장 낮은 가중치
        
        # 캐시 저장
        self._cached_best_edges_v2 = best_edges
        self._cache_key_v2 = cache_key
        
        return best_edges
    
    def _extract_negative_cycles(self, source: str) -> List[List[str]]:
        """음의 사이클 추출 - 최적화된 구현"""
        cycles = []
        visited = set()
        
        # 모든 노드에서 시작하여 사이클 탐지 (병렬 처리 가능)
        nodes = list(self.graph.graph.nodes)
        
        # 병렬 처리를 위한 청크 분할
        chunk_size = max(1, len(nodes) // 4)  # 4개의 청크로 분할
        node_chunks = [nodes[i:i + chunk_size] for i in range(0, len(nodes), chunk_size)]
        
        # 각 청크별로 사이클 탐지
        for chunk in node_chunks:
            chunk_cycles = []
            chunk_visited = set(visited)  # 청크별 방문 집합
            
            for node in chunk:
                if node in chunk_visited:
                    continue
                    
                # 사이클 탐지
                cycle = self._find_cycle_from_node_optimized(node, chunk_visited)
                if cycle and len(cycle) >= 3:  # 최소 3개 노드 (실제 사이클)
                    chunk_cycles.append(cycle)
            
            # 청크 결과 병합
            cycles.extend(chunk_cycles)
            visited.update(chunk_visited)
        
        return cycles
    
    def _find_cycle_from_node_optimized(self, start_node: str, visited: set) -> Optional[List[str]]:
        """특정 노드에서 시작하는 사이클 찾기 - 개선된 알고리즘 (최적화 버전)"""
        # 제한된 깊이로 탐색 (성능 향상)
        max_depth = 10
        path = []
        current = start_node
        path_dict = {}  # 리스트 대신 딕셔너리 사용으로 O(1) 검색
        
        depth = 0
        while current is not None and current not in visited and depth < max_depth:
            if current in path_dict:
                # 사이클 발견
                cycle_start_idx = path_dict[current]
                cycle = path[cycle_start_idx:] + [current]
                
                # 방문 처리
                for node in path:
                    visited.add(node)
                
                return cycle
            
            path.append(current)
            path_dict[current] = len(path) - 1
            depth += 1
            
            # predecessor가 (node, edge_data) 튜플인 경우 처리
            pred_info = self.predecessors.get(current)
            current = pred_info[0] if isinstance(pred_info, tuple) else pred_info
        
        # 방문 처리
        for node in path:
            visited.add(node)
        
        return None
    
    def _find_cycle_from_node(self, start_node: str, visited: set) -> Optional[List[str]]:
        """특정 노드에서 시작하는 사이클 찾기 - 개선된 알고리즘"""
        # 이 메소드는 더 이상 사용되지 않음. 대신 _find_cycle_from_node_optimized 사용
        return self._find_cycle_from_node_optimized(start_node, visited)
    
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