import math
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
from src.market_graph import DeFiMarketGraph, ArbitrageOpportunity, TradingEdge
from src.slippage import (
    amount_out_uniswap_v2,
    amount_out_balancer_weighted,
    amount_out_curve_stable_approx,
    amount_out_cpmm,
)
from src.logger import setup_logger
from config.config import config
from functools import lru_cache

logger = setup_logger(__name__)

class BellmanFordArbitrage:
    def __init__(self, market_graph: DeFiMarketGraph):
        self.graph = market_graph
        self.distances = {}
        self.predecessors = {}
        
    def find_negative_cycles(self, source_token: str, 
                           max_path_length: int = 4) -> List[ArbitrageOpportunity]:
        """음의 사이클 탐지를 통한 차익거래 기회 발견 (SPFA 우선)"""
        opportunities: List[ArbitrageOpportunity] = []

        # 1) SPFA로 빠르게 음의 사이클 탐지 시도
        cycles = self._spfa_detect_negative_cycles(source_token)

        # 2) 실패 시 Bellman-Ford 보조 확인
        if not cycles:
            if not self._bellman_ford(source_token, max_path_length):
                logger.info("음의 사이클이 발견되었습니다!")
                cycles = self._extract_negative_cycles(source_token)

        # 3) 사이클을 기회로 변환 후 Local Search 반복 수행
        for cycle in cycles:
            opportunity = self._perform_local_search_and_repeat(cycle)
            if opportunity and opportunity.net_profit > 0:
                opportunities.append(opportunity)
        
        # 최고 수익 기회 우선 반환 (Best revenue transaction per source)
        opportunities.sort(key=lambda x: x.net_profit, reverse=True)
        # 기본적으로 상위 1개만 반환하여 상위 기회에 집중
        return opportunities[:1]

    def _spfa_detect_negative_cycles(self, source: str) -> List[List[str]]:
        """SPFA 기반 음의 사이클 탐지 (성능 최적화)
        - 도달 가능한 노드만 대상으로 큐 기반 완화를 수행
        - 노드별 완화 횟수가 노드 수 이상이 되면 음의 사이클 존재
        """
        # 인접 리스트 구성 (환율>0 인 엣지만 사용)
        adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for u, v, data in self.graph.graph.edges(data=True):
            rate = data.get('exchange_rate', 0.0)
            if rate and rate > 0:
                w = -math.log(rate)
                adj[u].append((v, w))

        # source로부터 도달 가능한 노드만 추림
        reachable = set()
        dq = deque([source])
        while dq:
            u = dq.popleft()
            if u in reachable:
                continue
            reachable.add(u)
            for v, _ in adj.get(u, []):
                if v not in reachable:
                    dq.append(v)
        if not reachable:
            return []

        dist = {node: float('inf') for node in reachable}
        pred: Dict[str, Optional[str]] = {node: None for node in reachable}
        inq = {node: False for node in reachable}
        relax_cnt = {node: 0 for node in reachable}

        q = deque([source])
        dist[source] = 0.0
        inq[source] = True
        n = len(reachable)

        while q:
            u = q.popleft()
            inq[u] = False
            for v, w in adj.get(u, []):
                if v not in reachable:
                    continue
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd
                    pred[v] = u
                    if not inq[v]:
                        q.append(v)
                        inq[v] = True
                        relax_cnt[v] += 1
                        if relax_cnt[v] >= n:  # 음의 사이클 발견
                            cyc = self._reconstruct_cycle(pred, v)
                            if cyc:
                                return [cyc]
        return []

    def _reconstruct_cycle(self, pred: Dict[str, Optional[str]], start: str) -> Optional[List[str]]:
        """predecessor로부터 사이클 복원"""
        if start not in pred:
            return None
        x = start
        for _ in range(len(pred)):
            if pred.get(x) is None:
                return None
            x = pred[x]
        cycle = []
        cur = x
        while True:
            cycle.append(cur)
            cur = pred.get(cur)
            if cur is None or cur in cycle:
                break
        if cur is None:
            return None
        try:
            idx = cycle.index(cur)
            cyc = cycle[idx:] + [cur]
            cyc = list(reversed(cyc))
            return cyc
        except ValueError:
            return None
    
    def _bellman_ford(self, source: str, max_iterations: int) -> bool:
        """Bellman-Ford 알고리즘 실행"""
        # 초기화 (source와 관련된 노드/엣지로 축소)
        # 인접/역인접 리스트 빌드
        adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        radj: Dict[str, List[str]] = defaultdict(list)
        nodes = set()
        for u, v, data in self.graph.graph.edges(data=True):
            rate = data.get('exchange_rate', 0.0)
            if not rate or rate <= 0:
                continue
            w = -math.log(rate)
            adj[u].append((v, w))
            radj[v].append(u)
            nodes.add(u); nodes.add(v)
        # source에서 도달 가능 + source로 도달 가능한 교집합만 유지
        reach_out = set()
        from collections import deque
        dq = deque([source])
        while dq:
            u = dq.popleft()
            if u in reach_out:
                continue
            reach_out.add(u)
            for v, _ in adj.get(u, []):
                dq.append(v)
        reach_in = set()
        dq = deque([source])
        while dq:
            u = dq.popleft()
            if u in reach_in:
                continue
            reach_in.add(u)
            for v in radj.get(u, []):
                dq.append(v)
        cand_nodes = list(reach_out & reach_in) if reach_out and reach_in else list(nodes)
        if source not in cand_nodes:
            cand_nodes.append(source)
        # 거리/선행자 초기화
        self.distances = {node: float('inf') for node in cand_nodes}
        self.predecessors = {node: None for node in cand_nodes}
        self.distances[source] = 0
        # Cand edges 목록
        cand_edges: List[Tuple[str, str, float]] = []
        cand_set = set(cand_nodes)
        for u in cand_nodes:
            for v, w in adj.get(u, []):
                if v in cand_set:
                    cand_edges.append((u, v, w))

        # 거리 완화 (Relaxation)
        for _ in range(max_iterations):
            updated = False
            for u, v, w in cand_edges:
                if self.distances.get(u, float('inf')) != float('inf'):
                    nd = self.distances[u] + w
                    if nd < self.distances.get(v, float('inf')):
                        self.distances[v] = nd
                        self.predecessors[v] = u
                        updated = True
            
            if not updated:
                break
        # 음의 사이클 검사 (축소된 엣지 집합)
        for u, v, w in cand_edges:
            if (self.distances.get(u, float('inf')) != float('inf') and 
                self.distances[u] + w < self.distances.get(v, float('inf'))):
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

            edge = self._pick_edge(from_token, to_token)
            if edge is None:
                logger.warning(f"엣지 선택 실패: {from_token} -> {to_token}")
                return None
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
            edge = self._pick_edge(u, v)
            if edge is None:
                return None
            edges.append(edge)
            total_gas_cost += edge.gas_cost

        # 경로 유효성 검사: 부채/담보 제약 및 포지션 균형
        if not self._route_satisfies_constraints(edges):
            return None

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

    def _route_satisfies_constraints(self, edges: List[TradingEdge]) -> bool:
        """경로가 부채/담보 제약을 만족하는지 검증 + Health Factor 근사 계산.

        규칙/근사:
        - Aave: supply(underlying->aToken)로 담보 용량(capacity)을 적립: amount * liquidationThreshold 합산
        - Aave: borrow(edge: debt->underlying)로 부채 증가; repay(underlying->debt)로 감소
        - 경로 중 어느 시점이든 debt <= capacity 유지 (HF = capacity / debt >= 1)
        - usageAsCollateralEnabled=False 담보는 capacity 미반영
        - borrowingEnabled=False 자산은 borrow 엣지 무효
        - debt 토큰(\"debt:*\")/LP/BPT 잔고는 경로 종료 시 0이어야 함 (기존 규칙 유지)
        - 단위/가격은 근사적으로 1:1 가정(상대적 제약 목적)
        """
        if not edges:
            return False

        debt_counts: Dict[str, int] = defaultdict(int)
        lp_counts: Dict[str, int] = defaultdict(int)
        have_supply = {
            'aave': False,
            'compound': False,
        }

        # Aave HF 근사 상태
        aave_capacity = 0.0  # sum(amount * liquidationThreshold)
        aave_debt = 0.0      # sum(borrowed)
        amount = 1.0         # 기준 시작 자본 단위
        # Compound HF 근사 상태
        comp_capacity = 0.0
        comp_debt = 0.0
        comp_incent = 1.0
        comp_close = 0.5
        # dYdX leverage constraint (desired vs max)
        dydx_desired_lev = float(getattr(config, 'dydx_desired_leverage', 3.0))

        for e in edges:
            dex = (e.dex or '').lower()
            u = e.from_token or ''
            v = e.to_token or ''
            # 엣지 메타 조회
            meta = {}
            try:
                ed = self.graph.graph.get_edge_data(u, v)
                data = None
                if isinstance(ed, dict) and 'exchange_rate' in ed:
                    data = ed
                elif isinstance(ed, dict):
                    for _, d in ed.items():
                        if not isinstance(d, dict):
                            continue
                        if d.get('dex') == e.dex and d.get('pool_address') == e.pool_address:
                            data = d; break
                if isinstance(data, dict):
                    meta = data.get('meta', {}) or {}
            except Exception:
                meta = {}

            # Supply detection
            if dex == 'aave':
                have_supply['aave'] = True
            if dex == 'compound':
                have_supply['compound'] = True

            # LP / BPT balances
            if v.startswith('lp:') or v.startswith('bpt:'):
                lp_counts[v] += 1
            if u.startswith('lp:') or u.startswith('bpt:'):
                lp_counts[u] -= 1
                if lp_counts[u] < 0:
                    return False

            # Debt balances via dex types and token prefix (legacy)
            if dex in ('aave_borrow', 'compound_borrow'):
                # require prior supply for that protocol
                proto = 'aave' if dex.startswith('aave') else 'compound'
                if not have_supply.get(proto, False):
                    return False
                # Increase outstanding debt for the synthetic debt token id in from_token
                if u.startswith('debt:'):
                    debt_counts[u] += 1
            elif dex in ('aave_repay', 'compound_repay'):
                # Decrease outstanding debt for to_token (which is the debt token)
                if v.startswith('debt:'):
                    debt_counts[v] -= 1
                    if debt_counts[v] < 0:
                        return False

            # Aave Health Factor 근사
            try:
                if dex == 'aave':
                    # supply: treat as depositing `amount` units; add to capacity with liquidationThreshold
                    lt = float(meta.get('liquidationThreshold', 0.5) or 0.5)
                    # eMode override mapping (optional)
                    try:
                        mapping = getattr(config, 'aave_emode_ltv_overrides', '') or ''
                        mp = {}
                        if mapping:
                            for item in mapping.split(','):
                                if ':' in item:
                                    k, v = item.split(':')
                                    mp[int(k.strip())] = float(v.strip())
                        cat = meta.get('eModeCategory')
                        if cat is not None:
                            try:
                                cat = int(cat)
                                if cat in mp:
                                    lt = max(lt, mp[cat])
                            except Exception:
                                pass
                    except Exception:
                        pass
                    uac = bool(meta.get('usageAsCollateralEnabled', True))
                    if uac:
                        aave_capacity += max(0.0, amount) * max(0.0, min(lt, 1.0))
                    amount = self._amount_out_with_slippage(amount, e)
                elif dex in ('aave_borrow_variable', 'aave_borrow_stable'):
                    # require supply first
                    if not have_supply.get('aave', False):
                        return False
                    # borrowing enabled?
                    if dex.endswith('stable') and meta and not bool(meta.get('stableBorrowRateEnabled', True)):
                        return False
                    if meta and not bool(meta.get('borrowingEnabled', True)):
                        return False
                    out = self._amount_out_with_slippage(amount, e)
                    aave_debt += max(0.0, out)
                    # HF >= 1 requirement
                    if aave_debt > aave_capacity + 1e-9:
                        return False
                    amount = out
                elif dex in ('aave_repay_variable', 'aave_repay_stable'):
                    # repay uses underlying input amount to reduce debt
                    aave_debt -= max(0.0, amount)
                    if aave_debt < -1e-6:
                        return False
                    amount = self._amount_out_with_slippage(amount, e)
                # Compound HF: supply/borrow/repay
                elif dex == 'compound':
                    # supply underlying -> cToken: add to capacity using collateralFactor
                    cf = float(meta.get('collateralFactor', 0.5) or 0.5)
                    li = float(meta.get('liquidationIncentive', 1.0) or 1.0)
                    comp_incent = li if li > 0 else 1.0
                    try:
                        comp_close = float(meta.get('closeFactor', comp_close) or comp_close)
                    except Exception:
                        pass
                    # conservative: adjust capacity by liquidation incentive (require margin)
                    cap_add = max(0.0, amount) * max(0.0, min(cf, 1.0)) / max(1.0, comp_incent)
                    comp_capacity += cap_add
                    amount = self._amount_out_with_slippage(amount, e)
                elif dex == 'compound_borrow':
                    if not have_supply.get('compound', False):
                        return False
                    out = self._amount_out_with_slippage(amount, e)
                    comp_debt += max(0.0, out)
                    if comp_debt > comp_capacity + 1e-9:
                        return False
                    amount = out
                elif dex == 'compound_repay':
                    comp_debt -= max(0.0, amount)
                    if comp_debt < -1e-6:
                        return False
                    amount = self._amount_out_with_slippage(amount, e)
                else:
                    amount = self._amount_out_with_slippage(amount, e)
                # dYdX leverage/margin feasibility check
                if dex == 'dydx':
                    try:
                        maxLev = float(meta.get('maxLeverage', 10.0) or 10.0)
                        initM = float(meta.get('initialMargin', 0.1) or 0.1)
                        # reject if desired leverage exceeds max
                        if dydx_desired_lev > maxLev + 1e-9:
                            return False
                        # simple budget check: require initial margin fraction to be affordable
                        # amount represents current capital; require amount * initM <= amount (trivially true),
                        # but we can enforce minimal trade size by ensuring margin >= tiny threshold
                        if initM >= 1.0:
                            return False
                    except Exception:
                        return False
            except Exception:
                # 안전하게 실패 시 경로 거부
                return False

        # All LP/BPT must be fully unwound
        for k, c in lp_counts.items():
            if c != 0:
                return False

        # All debts must be repaid
        for k, c in debt_counts.items():
            if c != 0:
                return False

        # All debts must be repaid (legacy counter also ensures)
        if aave_debt > 1e-6 or comp_debt > 1e-6:
            return False

        return True

    def _simulate_final_amount(self, edges: List[TradingEdge], start_amount: float) -> float:
        """프로토콜별 슬리피지를 고려하여 경로 실행 후 최종 금액 계산"""
        amount = start_amount
        for edge in edges:
            amount = self._amount_out_with_slippage(amount, edge)
        return amount

    def _amount_out_with_slippage(self, amount_in: float, edge: TradingEdge) -> float:
        try:
            u, v = edge.from_token, edge.to_token
            ed = self.graph.graph.get_edge_data(u, v)
            data = None
            if isinstance(ed, dict) and 'exchange_rate' in ed:
                data = ed
            elif isinstance(ed, dict):
                # MultiDiGraph: find matching dex/pool
                for _, d in ed.items():
                    if not isinstance(d, dict):
                        continue
                    if d.get('dex') == edge.dex and d.get('pool_address') == edge.pool_address:
                        data = d; break
            if data is None:
                # fallback
                return amount_in * edge.exchange_rate * (1 - self._calculate_slippage(amount_in, edge.liquidity))

            dex = edge.dex.lower()
            fee = float(edge.fee)
            meta = data.get('meta', {}) or {}

            # Uniswap V2/Sushiswap CPMM
            if dex in ('uniswap_v2', 'sushiswap'):
                t0 = meta.get('t0'); t1 = meta.get('t1')
                r0 = float(meta.get('r0', 0.0)); r1 = float(meta.get('r1', 0.0))
                if t0 and t1 and r0 > 0 and r1 > 0:
                    if u == t0 and v == t1:
                        return amount_out_uniswap_v2(amount_in, r0, r1, fee)
                    elif u == t1 and v == t0:
                        return amount_out_uniswap_v2(amount_in, r1, r0, fee)
            # Uniswap V3 approximate via CPMM with pseudo reserves
            if dex == 'uniswap_v3':
                t0 = meta.get('t0'); t1 = meta.get('t1')
                r0 = float(meta.get('r0', 0.0)); r1 = float(meta.get('r1', 0.0))
                if t0 and t1 and r0 > 0 and r1 > 0:
                    if u == t0 and v == t1:
                        return amount_out_cpmm(amount_in, r0, r1, fee)
                    elif u == t1 and v == t0:
                        return amount_out_cpmm(amount_in, r1, r0, fee)
            # Curve stableswap approximate (reduced price impact)
            if dex == 'curve' or meta.get('stableswap'):
                # Use pseudo reserves if present; otherwise liquidity as proxy
                r0 = float(meta.get('r0', edge.liquidity))
                r1 = float(meta.get('r1', edge.liquidity))
                amp = float(meta.get('amp', 100.0))
                return amount_out_curve_stable_approx(amount_in, r0, r1, fee, amp)
            # Balancer weighted
            if dex == 'balancer':
                w_in = float(meta.get('w_in', 0.5)); w_out = float(meta.get('w_out', 0.5))
                b_in = float(meta.get('b_in', edge.liquidity)); b_out = float(meta.get('b_out', edge.liquidity))
                return amount_out_balancer_weighted(amount_in, b_in, b_out, w_in, w_out, fee)

            # Default fallback
            out = amount_in * edge.exchange_rate * (1 - self._calculate_slippage(amount_in, edge.liquidity))
            # Apply conservative penalties for risky tokens (fee-on-transfer/rebase)
            fot = bool(meta.get('risk_fot'))
            rbs = bool(meta.get('risk_rebase'))
            if fot:
                out *= 0.99  # 1% extra fee per transfer
            if rbs:
                out *= 0.995  # minor penalty due to supply changes risk
            return out
        except Exception:
            return amount_in * edge.exchange_rate * (1 - self._calculate_slippage(amount_in, edge.liquidity))

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

    def _pick_edge(self, u: str, v: str) -> Optional[TradingEdge]:
        """동일 토큰 쌍에서 여러 DEX 엣지가 있을 경우 최적 엣지 선택
        기본 정책: exchange_rate가 가장 큰 엣지 선택
        """
        try:
            ed = self.graph.graph.get_edge_data(u, v)
            if ed is None:
                return None
            # DiGraph: ed는 단일 속성 dict
            if isinstance(ed, dict) and 'exchange_rate' in ed and 'dex' in ed:
                return TradingEdge(
                    from_token=u,
                    to_token=v,
                    dex=ed.get('dex',''),
                    pool_address=ed.get('pool_address',''),
                    exchange_rate=ed.get('exchange_rate',0.0),
                    liquidity=ed.get('liquidity',0.0),
                    fee=ed.get('fee',0.0),
                    gas_cost=ed.get('gas_cost',0.0),
                    weight=0.0,
                )
            # MultiDiGraph: ed는 key->data dict
            candidates = []
            if isinstance(ed, dict):
                for _, data in ed.items():
                    if isinstance(data, dict) and 'exchange_rate' in data:
                        candidates.append(data)
            if not candidates:
                return None
            best = max(candidates, key=lambda d: d.get('exchange_rate', 0))
            return TradingEdge(
                from_token=u,
                to_token=v,
                dex=best.get('dex',''),
                pool_address=best.get('pool_address',''),
                exchange_rate=best.get('exchange_rate',0.0),
                liquidity=best.get('liquidity',0.0),
                fee=best.get('fee',0.0),
                gas_cost=best.get('gas_cost',0.0),
                weight=0.0,
            )
        except Exception:
            return None
