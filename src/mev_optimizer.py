#!/usr/bin/env python3
"""
MEV (Maximal Extractable Value) Optimizer
논문 "[2103.02228] On the Just-In-Time Discovery of Profit-Generating Transactions in DeFi Protocols" 기반 구현

논문의 핵심 MEV 개념:
- MEV threshold 계산 (마이너가 블록체인을 포크할 임계값)
- MDP (Markov Decision Process)를 통한 최적 전략
- bZx 공격 사례: 874× 블록 보상 초과 수익
- 10% 해시레이트 마이너: 4× 블록 보상 초과 시 포크 결정

주요 기능:
1. MEV 기회 탐지 및 분류
2. MDP 기반 최적 전략 계산
3. 블록 보상 대비 MEV 수익률 분석
4. 마이너 인센티브 분석 (포크 vs 정직한 마이닝)
5. 실시간 MEV 스코어링 시스템
"""

import asyncio
import time
import json
import math
from typing import Dict, List, Optional, Tuple, Union, NamedTuple
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from enum import Enum
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict, deque
import statistics

from web3 import Web3
from eth_account import Account

from src.logger import setup_logger
from src.market_graph import DeFiMarketGraph, ArbitrageOpportunity
from src.economic_state_exploiter import EconomicStateExploiter, ExploitationType
from src.flash_loan_manager import FlashLoanManager
from src.transaction_pool_monitor import TransactionPoolMonitor
from src.protocol_actions import ProtocolRegistry

# Set high precision for financial calculations
getcontext().prec = 28

logger = setup_logger(__name__)

class MEVType(Enum):
    """MEV 기회의 종류"""
    ARBITRAGE = "arbitrage"  # 기본 차익거래
    SANDWICH = "sandwich"    # 샌드위치 공격
    LIQUIDATION = "liquidation"  # 청산
    FRONT_RUNNING = "front_running"  # 선행 거래
    BACK_RUNNING = "back_running"   # 후행 거래
    JUST_IN_TIME = "just_in_time"   # JIT 유동성
    ATOMIC_ARBITRAGE = "atomic_arbitrage"  # 원자적 차익거래
    ECONOMIC_EXPLOIT = "economic_exploit"  # 경제적 익스플로잇 (bZx 스타일)

@dataclass
class MEVOpportunity:
    """MEV 기회 데이터 구조"""
    opportunity_id: str
    mev_type: MEVType
    potential_profit: Decimal  # Wei 단위
    required_capital: Decimal  # Wei 단위
    success_probability: float  # 0.0 - 1.0
    time_sensitivity: int  # 초 단위
    block_dependency: int  # 의존하는 블록 수
    affected_tokens: List[str]
    affected_protocols: List[str]
    execution_cost: Decimal  # Gas cost in Wei
    mev_score: float  # 0.0 - 1.0 정규화된 점수
    competition_level: float  # 0.0 - 1.0 (경쟁 강도)
    discovery_timestamp: float
    metadata: Dict = field(default_factory=dict)

@dataclass 
class MEVThreshold:
    """논문의 MEV threshold 모델"""
    miner_hash_rate: float  # 0.0 - 1.0 (10% = 0.1)
    block_reward: Decimal  # Wei 단위
    stale_block_rate: float  # 5.72% = 0.0572 (논문 기준)
    threshold_multiplier: float  # 4.0 (논문: 10% 해시레이트에서 4× 임계값)
    fork_threshold: Decimal  # Wei 단위 (threshold_multiplier × block_reward)

class MDP_State(Enum):
    """MDP 상태 정의"""
    HONEST_MINING = "honest"
    PRIVATE_CHAIN = "private" 
    FORK_SUCCESS = "fork_success"
    FORK_FAILED = "fork_failed"

@dataclass
class MDP_Strategy:
    """논문의 MDP 최적 전략"""
    hash_rate: float
    mev_value: Decimal
    optimal_action: str  # "fork" or "honest"
    expected_reward: Decimal
    fork_probability: float
    strategy_confidence: float

class MEVOptimizer:
    """
    MEV (Maximal Extractable Value) 최적화 엔진
    논문의 MDP 모델과 MEV threshold 분석을 구현
    """
    
    def __init__(
        self,
        web3: Web3,
        market_graph: DeFiMarketGraph,
        economic_exploiter: EconomicStateExploiter,
        flash_loan_manager: FlashLoanManager,
        transaction_pool_monitor: TransactionPoolMonitor,
        protocol_registry: ProtocolRegistry
    ):
        self.w3 = web3
        self.market_graph = market_graph
        self.economic_exploiter = economic_exploiter
        self.flash_loan_manager = flash_loan_manager
        self.tx_pool_monitor = transaction_pool_monitor
        self.protocol_registry = protocol_registry
        
        # 논문 기준 설정
        self.ethereum_block_reward = Web3.to_wei(2.0, 'ether')  # 약 2 ETH (2023년 기준)
        self.ethereum_stale_rate = 0.0572  # 5.72% (논문 데이터)
        self.bzx_benchmark_profit = Web3.to_wei(1193.69, 'ether')  # 논문의 bZx 공격 수익
        
        # MEV threshold 계산을 위한 MDP 파라미터
        self.mdp_parameters = {
            'stale_block_rate': self.ethereum_stale_rate,
            'network_propagation': 0.0,  # γ = 0 (논문 설정)
            'eclipse_attack_param': 0.0,  # ω = 0 (논문 설정)
            'cutoff_blocks': 20,  # 최대 고려 블록 수
            'search_precision': 0.1  # 이진 탐색 정밀도
        }
        
        # MEV 기회 저장소
        self.active_opportunities: Dict[str, MEVOpportunity] = {}
        self.opportunity_history: deque = deque(maxlen=10000)
        self.mev_thresholds: Dict[float, MEVThreshold] = {}  # hash_rate -> threshold
        
        # 성능 메트릭
        self.performance_metrics = {
            'total_mev_detected': 0,
            'total_mev_value': Decimal('0'),
            'largest_mev_opportunity': Decimal('0'),
            'bzx_benchmark_comparison': 0.0,
            'average_mev_score': 0.0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'threshold_violations': 0,  # MEV threshold 초과 횟수
            'potential_fork_incentives': 0,
            'detection_accuracy': 0.0
        }
        
        # 실시간 모니터링
        self.real_time_scoring = True
        self.score_update_interval = 1.0  # 1초마다 스코어 업데이트
        
        # 논문 벤치마크 추적
        self.paper_benchmarks = {
            'weekly_revenue_target': Web3.to_wei(191.48, 'ether'),  # DeFiPoser-ARB 목표
            'max_single_profit_target': Web3.to_wei(81.31, 'ether'),  # 논문 최고 수익
            'bzx_profit_ratio': 874,  # 874× 블록 보상 (논문 기준)
            'processing_time_target': 6.43  # 초 (논문 성능 목표)
        }
        
        # 초기 MEV threshold 계산
        asyncio.create_task(self._initialize_mev_thresholds())
        
        logger.info("MEV Optimizer 초기화 완료")
    
    async def _initialize_mev_thresholds(self):
        """논문 기반 MEV threshold 초기 계산"""
        try:
            # 논문에서 분석한 해시레이트별 threshold 계산
            hash_rates = [0.01, 0.05, 0.1, 0.2, 0.3]  # 1%, 5%, 10%, 20%, 30%
            
            for hash_rate in hash_rates:
                threshold = await self._calculate_mev_threshold(hash_rate)
                self.mev_thresholds[hash_rate] = threshold
                
                logger.info(
                    f"MEV Threshold ({hash_rate*100:.0f}% 해시레이트): "
                    f"{Web3.from_wei(threshold.fork_threshold, 'ether'):.2f} ETH "
                    f"({threshold.threshold_multiplier:.1f}× 블록 보상)"
                )
        
        except Exception as e:
            logger.error(f"MEV threshold 초기화 실패: {e}")
    
    async def _calculate_mev_threshold(self, hash_rate: float) -> MEVThreshold:
        """
        논문의 MDP 모델을 사용한 MEV threshold 계산
        
        논문 결과: 10% 해시레이트 마이너는 4× 블록 보상 초과 시 포크
        """
        try:
            # MDP 파라미터 설정
            mdp_params = self.mdp_parameters.copy()
            mdp_params['miner_hash_rate'] = hash_rate
            mdp_params['mining_cost'] = hash_rate  # c_m = α (논문 설정)
            
            # 이진 탐색으로 최소 MEV 값 찾기
            threshold_multiplier = await self._binary_search_mev_threshold(mdp_params)
            
            fork_threshold = self.ethereum_block_reward * Decimal(str(threshold_multiplier))
            
            return MEVThreshold(
                miner_hash_rate=hash_rate,
                block_reward=self.ethereum_block_reward,
                stale_block_rate=self.ethereum_stale_rate,
                threshold_multiplier=threshold_multiplier,
                fork_threshold=fork_threshold
            )
            
        except Exception as e:
            logger.error(f"MEV threshold 계산 실패 (해시레이트 {hash_rate}): {e}")
            # Fallback to paper's empirical result
            return MEVThreshold(
                miner_hash_rate=hash_rate,
                block_reward=self.ethereum_block_reward,
                stale_block_rate=self.ethereum_stale_rate,
                threshold_multiplier=4.0,  # 논문 기준값
                fork_threshold=self.ethereum_block_reward * Decimal('4.0')
            )
    
    async def _binary_search_mev_threshold(self, mdp_params: Dict) -> float:
        """이진 탐색으로 MEV threshold 찾기"""
        try:
            left, right = 1.0, 20.0  # 1× ~ 20× 블록 보상 범위
            precision = mdp_params['search_precision']
            
            while right - left > precision:
                mid = (left + right) / 2.0
                
                # MDP 전략 계산
                fork_reward = await self._calculate_fork_expected_reward(mid, mdp_params)
                honest_reward = await self._calculate_honest_expected_reward(mdp_params)
                
                if fork_reward > honest_reward:
                    right = mid  # MEV 값이 너무 높음
                else:
                    left = mid   # MEV 값이 너무 낮음
            
            return (left + right) / 2.0
            
        except Exception as e:
            logger.error(f"이진 탐색 실패: {e}")
            return 4.0  # 논문의 기본값
    
    async def _calculate_fork_expected_reward(
        self, 
        mev_multiplier: float, 
        mdp_params: Dict
    ) -> float:
        """포크 전략의 기댓값 계산"""
        try:
            hash_rate = mdp_params['miner_hash_rate']
            stale_rate = mdp_params['stale_block_rate']
            mining_cost = mdp_params['mining_cost']
            
            # 논문의 MDP 모델 기반 계산
            # 간소화된 모델: P(fork_success) ≈ α × (1 - stale_rate)
            fork_success_prob = hash_rate * (1.0 - stale_rate)
            
            mev_reward = mev_multiplier * float(Web3.from_wei(self.ethereum_block_reward, 'ether'))
            block_reward = float(Web3.from_wei(self.ethereum_block_reward, 'ether'))
            
            expected_reward = (
                fork_success_prob * (mev_reward + block_reward) +
                (1 - fork_success_prob) * 0.0 -  # 포크 실패 시 보상 없음
                mining_cost * block_reward  # 마이닝 비용
            )
            
            return expected_reward
            
        except Exception as e:
            logger.error(f"포크 기댓값 계산 실패: {e}")
            return 0.0
    
    async def _calculate_honest_expected_reward(self, mdp_params: Dict) -> float:
        """정직한 마이닝 전략의 기댓값 계산"""
        try:
            hash_rate = mdp_params['miner_hash_rate']
            mining_cost = mdp_params['mining_cost']
            block_reward = float(Web3.from_wei(self.ethereum_block_reward, 'ether'))
            
            # 정직한 마이닝: E[R] = α × block_reward - cost
            expected_reward = hash_rate * block_reward - mining_cost * block_reward
            
            return expected_reward
            
        except Exception as e:
            logger.error(f"정직한 마이닝 기댓값 계산 실패: {e}")
            return 0.0
    
    async def detect_mev_opportunities(
        self, 
        block_number: int, 
        pending_transactions: List[Dict] = None
    ) -> List[MEVOpportunity]:
        """
        MEV 기회 탐지 (실시간)
        논문의 "Just-in-Time Discovery" 구현
        """
        try:
            start_time = time.time()
            opportunities = []
            
            # 1. 차익거래 MEV 탐지
            arbitrage_mevs = await self._detect_arbitrage_mev(block_number)
            opportunities.extend(arbitrage_mevs)
            
            # 2. 청산 MEV 탐지
            liquidation_mevs = await self._detect_liquidation_mev(block_number)
            opportunities.extend(liquidation_mevs)
            
            # 3. Sandwich 공격 MEV 탐지
            if pending_transactions:
                sandwich_mevs = await self._detect_sandwich_mev(pending_transactions)
                opportunities.extend(sandwich_mevs)
            
            # 4. 경제적 익스플로잇 MEV 탐지 (bZx 스타일)
            economic_mevs = await self._detect_economic_exploit_mev(block_number)
            opportunities.extend(economic_mevs)
            
            # 5. MEV 스코어 계산 및 필터링
            scored_opportunities = await self._calculate_mev_scores(opportunities)
            viable_opportunities = self._filter_viable_opportunities(scored_opportunities)
            
            detection_time = time.time() - start_time
            
            logger.info(
                f"블록 {block_number}: {len(viable_opportunities)}개 MEV 기회 탐지 "
                f"({detection_time:.3f}s)"
            )
            
            # 성능 메트릭 업데이트
            self._update_performance_metrics(viable_opportunities)
            
            return viable_opportunities
            
        except Exception as e:
            logger.error(f"MEV 기회 탐지 실패: {e}")
            return []
    
    async def _detect_arbitrage_mev(self, block_number: int) -> List[MEVOpportunity]:
        """차익거래 MEV 탐지"""
        mev_opportunities = []
        
        try:
            # 시장 그래프에서 차익거래 기회 검색
            market_opportunities = self.market_graph.find_arbitrage_opportunities()
            
            for opp in market_opportunities:
                if opp.net_profit > Web3.to_wei(0.01, 'ether'):  # 0.01 ETH 이상
                    mev_opp = MEVOpportunity(
                        opportunity_id=f"arb_{block_number}_{hash(opp.path)}",
                        mev_type=MEVType.ARBITRAGE,
                        potential_profit=Decimal(str(opp.net_profit)),
                        required_capital=Decimal(str(opp.required_capital)),
                        success_probability=opp.confidence,
                        time_sensitivity=30,  # 30초 유효
                        block_dependency=1,
                        affected_tokens=[token for token in opp.path],
                        affected_protocols=[edge.dex for edge in opp.edges],
                        execution_cost=Web3.to_wei(0.02, 'ether'),  # 가스 비용
                        mev_score=0.0,  # 계산 예정
                        competition_level=0.5,  # 중간 경쟁 수준
                        discovery_timestamp=time.time(),
                        metadata={'arbitrage_path': opp.path, 'profit_ratio': opp.profit_ratio}
                    )
                    mev_opportunities.append(mev_opp)
            
        except Exception as e:
            logger.error(f"차익거래 MEV 탐지 실패: {e}")
        
        return mev_opportunities
    
    async def _detect_liquidation_mev(self, block_number: int) -> List[MEVOpportunity]:
        """청산 MEV 탐지"""
        mev_opportunities = []
        
        try:
            # Economic State Exploiter에서 청산 기회 검색
            vulnerabilities = await self.economic_exploiter.scan_for_economic_vulnerabilities()
            
            for vulnerability in vulnerabilities:
                if vulnerability.vulnerability_type == ExploitationType.LIQUIDATION_CASCADING:
                    mev_opp = MEVOpportunity(
                        opportunity_id=f"liq_{block_number}_{vulnerability.protocol}",
                        mev_type=MEVType.LIQUIDATION,
                        potential_profit=vulnerability.potential_profit,
                        required_capital=Decimal(str(Web3.to_wei(1, 'ether'))),  # 최소 자본
                        success_probability=0.8,  # 청산은 비교적 확실
                        time_sensitivity=vulnerability.time_window_seconds,
                        block_dependency=1,
                        affected_tokens=vulnerability.affected_assets,
                        affected_protocols=[vulnerability.protocol],
                        execution_cost=Web3.to_wei(0.03, 'ether'),  # 청산 가스비
                        mev_score=0.0,
                        competition_level=0.7,  # 청산은 경쟁이 치열
                        discovery_timestamp=time.time(),
                        metadata={'vulnerability': vulnerability.description}
                    )
                    mev_opportunities.append(mev_opp)
            
        except Exception as e:
            logger.error(f"청산 MEV 탐지 실패: {e}")
        
        return mev_opportunities
    
    async def _detect_sandwich_mev(self, pending_transactions: List[Dict]) -> List[MEVOpportunity]:
        """Sandwich 공격 MEV 탐지"""
        mev_opportunities = []
        
        try:
            for tx in pending_transactions:
                # 대형 거래를 샌드위치 공격 대상으로 식별
                tx_value = tx.get('value', 0)
                if tx_value > Web3.to_wei(10, 'ether'):  # 10 ETH 이상 거래
                    
                    # 잠재적 샌드위치 수익 계산
                    estimated_profit = await self._estimate_sandwich_profit(tx)
                    
                    if estimated_profit > Web3.to_wei(0.05, 'ether'):  # 0.05 ETH 이상
                        mev_opp = MEVOpportunity(
                            opportunity_id=f"sand_{tx['hash']}",
                            mev_type=MEVType.SANDWICH,
                            potential_profit=Decimal(str(estimated_profit)),
                            required_capital=Decimal(str(tx_value * 2)),  # 2배 자본 필요
                            success_probability=0.6,  # 중간 성공률
                            time_sensitivity=15,  # 15초 내 실행 필요
                            block_dependency=1,
                            affected_tokens=self._extract_tokens_from_tx(tx),
                            affected_protocols=self._extract_protocols_from_tx(tx),
                            execution_cost=Web3.to_wei(0.08, 'ether'),  # 2개 트랜잭션 가스
                            mev_score=0.0,
                            competition_level=0.9,  # 매우 경쟁적
                            discovery_timestamp=time.time(),
                            metadata={'target_tx': tx['hash'], 'target_value': tx_value}
                        )
                        mev_opportunities.append(mev_opp)
            
        except Exception as e:
            logger.error(f"Sandwich MEV 탐지 실패: {e}")
        
        return mev_opportunities
    
    async def _detect_economic_exploit_mev(self, block_number: int) -> List[MEVOpportunity]:
        """경제적 익스플로잇 MEV 탐지 (bZx 스타일)"""
        mev_opportunities = []
        
        try:
            # Economic State Exploiter에서 경제적 익스플로잇 기회 검색
            vulnerabilities = await self.economic_exploiter.scan_for_economic_vulnerabilities()
            
            for vulnerability in vulnerabilities:
                if vulnerability.vulnerability_type == ExploitationType.PUMP_AND_ARBITRAGE:
                    # bZx 스타일 공격은 매우 높은 수익 잠재력
                    mev_opp = MEVOpportunity(
                        opportunity_id=f"exploit_{block_number}_{vulnerability.protocol}",
                        mev_type=MEVType.ECONOMIC_EXPLOIT,
                        potential_profit=vulnerability.potential_profit,
                        required_capital=Decimal(str(Web3.to_wei(0.1, 'ether'))),  # 플래시론 사용
                        success_probability=vulnerability.exploitation_complexity / 10.0,
                        time_sensitivity=vulnerability.time_window_seconds,
                        block_dependency=vulnerability.time_window_seconds // 13,  # 블록 수
                        affected_tokens=vulnerability.affected_assets,
                        affected_protocols=[vulnerability.protocol],
                        execution_cost=Web3.to_wei(0.1, 'ether'),  # 복잡한 거래
                        mev_score=0.0,
                        competition_level=0.3,  # 복잡해서 경쟁 적음
                        discovery_timestamp=time.time(),
                        metadata={
                            'exploit_type': vulnerability.vulnerability_type.value,
                            'complexity': vulnerability.exploitation_complexity
                        }
                    )
                    mev_opportunities.append(mev_opp)
            
        except Exception as e:
            logger.error(f"경제적 익스플로잇 MEV 탐지 실패: {e}")
        
        return mev_opportunities
    
    async def _calculate_mev_scores(self, opportunities: List[MEVOpportunity]) -> List[MEVOpportunity]:
        """MEV 기회들의 점수 계산"""
        try:
            for opp in opportunities:
                # 기본 점수 계산 요소들
                profit_score = self._calculate_profit_score(opp.potential_profit)
                success_score = opp.success_probability
                time_score = self._calculate_time_sensitivity_score(opp.time_sensitivity)
                competition_score = 1.0 - opp.competition_level
                capital_efficiency_score = self._calculate_capital_efficiency_score(opp)
                
                # 가중 평균으로 최종 점수 계산
                opp.mev_score = (
                    profit_score * 0.35 +
                    success_score * 0.25 +
                    time_score * 0.15 +
                    competition_score * 0.15 +
                    capital_efficiency_score * 0.10
                )
                
                # bZx 벤치마크와 비교 (논문 기준)
                if opp.potential_profit > self.bzx_benchmark_profit * Decimal('0.01'):  # 1% 이상
                    opp.mev_score = min(opp.mev_score * 1.2, 1.0)  # 20% 보너스
                
            return opportunities
            
        except Exception as e:
            logger.error(f"MEV 점수 계산 실패: {e}")
            return opportunities
    
    def _calculate_profit_score(self, profit: Decimal) -> float:
        """수익 기반 점수 계산 (0.0 - 1.0)"""
        try:
            profit_eth = float(Web3.from_wei(int(profit), 'ether'))
            
            # 로그 스케일로 정규화 (0.01 ETH = 0.1, 10 ETH = 1.0)
            if profit_eth <= 0.01:
                return 0.0
            elif profit_eth >= 10.0:
                return 1.0
            else:
                return math.log10(profit_eth / 0.01) / math.log10(1000)  # log scale
                
        except Exception:
            return 0.0
    
    def _calculate_time_sensitivity_score(self, time_sensitivity: int) -> float:
        """시간 민감도 점수 (짧을수록 높은 점수)"""
        try:
            if time_sensitivity <= 10:
                return 1.0  # 10초 이하는 최고 점수
            elif time_sensitivity >= 300:
                return 0.1  # 5분 이상은 최저 점수
            else:
                return 1.0 - ((time_sensitivity - 10) / 290) * 0.9
        except Exception:
            return 0.5
    
    def _calculate_capital_efficiency_score(self, opp: MEVOpportunity) -> float:
        """자본 효율성 점수"""
        try:
            if opp.required_capital == 0:
                return 1.0  # 무자본 (플래시론)
            
            profit_ratio = float(opp.potential_profit) / float(opp.required_capital)
            
            if profit_ratio >= 1.0:
                return 1.0  # 100% 이상 수익률
            elif profit_ratio >= 0.1:
                return profit_ratio  # 10% ~ 100%
            else:
                return profit_ratio * 0.5  # 10% 미만은 페널티
                
        except Exception:
            return 0.5
    
    def _filter_viable_opportunities(self, opportunities: List[MEVOpportunity]) -> List[MEVOpportunity]:
        """실행 가능한 MEV 기회만 필터링"""
        try:
            viable = []
            
            for opp in opportunities:
                # 최소 조건 체크
                if (opp.mev_score >= 0.3 and  # 최소 점수
                    opp.potential_profit > Web3.to_wei(0.01, 'ether') and  # 최소 수익
                    opp.success_probability >= 0.2):  # 최소 성공률
                    
                    viable.append(opp)
            
            # MEV 점수 순으로 정렬
            viable.sort(key=lambda x: x.mev_score, reverse=True)
            
            return viable
            
        except Exception as e:
            logger.error(f"MEV 기회 필터링 실패: {e}")
            return opportunities
    
    async def optimize_mev_extraction(
        self, 
        opportunities: List[MEVOpportunity]
    ) -> List[MEVOpportunity]:
        """
        MEV 추출 최적화
        논문의 MDP 모델을 사용하여 최적 전략 결정
        """
        try:
            optimized_opportunities = []
            
            for opp in opportunities:
                # MEV threshold 체크
                threshold_analysis = await self._analyze_mev_threshold(opp)
                
                if threshold_analysis['should_extract']:
                    # 최적 전략 계산
                    optimal_strategy = await self._calculate_optimal_strategy(opp)
                    
                    # 전략 정보를 메타데이터에 추가
                    opp.metadata['optimal_strategy'] = optimal_strategy
                    opp.metadata['threshold_analysis'] = threshold_analysis
                    
                    optimized_opportunities.append(opp)
                    
                    # Threshold 위반 통계
                    if threshold_analysis['threshold_violation']:
                        self.performance_metrics['threshold_violations'] += 1
                        self.performance_metrics['potential_fork_incentives'] += 1
            
            return optimized_opportunities
            
        except Exception as e:
            logger.error(f"MEV 추출 최적화 실패: {e}")
            return opportunities
    
    async def _analyze_mev_threshold(self, opportunity: MEVOpportunity) -> Dict:
        """MEV threshold 분석 (논문의 핵심 기여)"""
        try:
            # 기본 10% 해시레이트 threshold 사용 (논문 기준)
            base_threshold = self.mev_thresholds.get(0.1)
            
            if not base_threshold:
                return {'should_extract': True, 'threshold_violation': False}
            
            # MEV 가치와 threshold 비교
            mev_value_multiplier = float(opportunity.potential_profit) / float(base_threshold.block_reward)
            threshold_violation = mev_value_multiplier > base_threshold.threshold_multiplier
            
            # 추출 결정 (논문의 rational miner 모델)
            should_extract = True  # 기본적으로 추출
            
            # 매우 높은 MEV는 포크 인센티브 제공
            fork_incentive_risk = mev_value_multiplier > base_threshold.threshold_multiplier
            
            analysis = {
                'should_extract': should_extract,
                'threshold_violation': threshold_violation,
                'fork_incentive_risk': fork_incentive_risk,
                'mev_multiplier': mev_value_multiplier,
                'threshold_multiplier': base_threshold.threshold_multiplier,
                'analysis_timestamp': time.time()
            }
            
            if threshold_violation:
                logger.warning(
                    f"MEV threshold 초과: {mev_value_multiplier:.1f}× vs {base_threshold.threshold_multiplier:.1f}× "
                    f"(포크 인센티브 위험)"
                )
            
            return analysis
            
        except Exception as e:
            logger.error(f"MEV threshold 분석 실패: {e}")
            return {'should_extract': True, 'threshold_violation': False}
    
    async def _calculate_optimal_strategy(self, opportunity: MEVOpportunity) -> MDP_Strategy:
        """MDP 기반 최적 전략 계산"""
        try:
            # 기본 해시레이트 가정 (10%)
            hash_rate = 0.1
            mev_value = opportunity.potential_profit
            
            # MDP 계산
            fork_reward = await self._calculate_fork_expected_reward(
                float(mev_value) / float(self.ethereum_block_reward),
                {**self.mdp_parameters, 'miner_hash_rate': hash_rate}
            )
            honest_reward = await self._calculate_honest_expected_reward(
                {**self.mdp_parameters, 'miner_hash_rate': hash_rate}
            )
            
            # 최적 행동 결정
            optimal_action = "fork" if fork_reward > honest_reward else "honest"
            expected_reward = Decimal(str(max(fork_reward, honest_reward)))
            fork_probability = hash_rate * (1.0 - self.ethereum_stale_rate)
            
            strategy_confidence = min(
                abs(fork_reward - honest_reward) / max(fork_reward, honest_reward, 1.0),
                1.0
            )
            
            return MDP_Strategy(
                hash_rate=hash_rate,
                mev_value=mev_value,
                optimal_action=optimal_action,
                expected_reward=expected_reward,
                fork_probability=fork_probability,
                strategy_confidence=strategy_confidence
            )
            
        except Exception as e:
            logger.error(f"최적 전략 계산 실패: {e}")
            return MDP_Strategy(
                hash_rate=0.1,
                mev_value=opportunity.potential_profit,
                optimal_action="honest",
                expected_reward=Decimal('0'),
                fork_probability=0.0,
                strategy_confidence=0.0
            )
    
    def _update_performance_metrics(self, opportunities: List[MEVOpportunity]):
        """성능 메트릭 업데이트"""
        try:
            self.performance_metrics['total_mev_detected'] += len(opportunities)
            
            total_value = sum(opp.potential_profit for opp in opportunities)
            self.performance_metrics['total_mev_value'] += total_value
            
            if opportunities:
                largest_opp = max(opportunities, key=lambda x: x.potential_profit)
                if largest_opp.potential_profit > self.performance_metrics['largest_mev_opportunity']:
                    self.performance_metrics['largest_mev_opportunity'] = largest_opp.potential_profit
                
                # bZx 벤치마크 비교
                bzx_ratio = float(largest_opp.potential_profit) / float(self.bzx_benchmark_profit)
                if bzx_ratio > self.performance_metrics['bzx_benchmark_comparison']:
                    self.performance_metrics['bzx_benchmark_comparison'] = bzx_ratio
                
                # 평균 MEV 점수
                avg_score = statistics.mean(opp.mev_score for opp in opportunities)
                self.performance_metrics['average_mev_score'] = avg_score
            
        except Exception as e:
            logger.error(f"성능 메트릭 업데이트 실패: {e}")
    
    def get_mev_performance_report(self) -> Dict:
        """종합 MEV 성능 보고서"""
        try:
            total_mev_eth = float(Web3.from_wei(int(self.performance_metrics['total_mev_value']), 'ether'))
            largest_mev_eth = float(Web3.from_wei(int(self.performance_metrics['largest_mev_opportunity']), 'ether'))
            
            return {
                'detection_metrics': {
                    'total_opportunities_detected': self.performance_metrics['total_mev_detected'],
                    'total_mev_value_eth': total_mev_eth,
                    'largest_opportunity_eth': largest_mev_eth,
                    'average_mev_score': self.performance_metrics['average_mev_score']
                },
                'extraction_metrics': {
                    'successful_extractions': self.performance_metrics['successful_extractions'],
                    'failed_extractions': self.performance_metrics['failed_extractions'],
                    'extraction_success_rate': (
                        self.performance_metrics['successful_extractions'] / 
                        max(self.performance_metrics['successful_extractions'] + 
                            self.performance_metrics['failed_extractions'], 1) * 100
                    )
                },
                'threshold_analysis': {
                    'threshold_violations': self.performance_metrics['threshold_violations'],
                    'potential_fork_incentives': self.performance_metrics['potential_fork_incentives'],
                    'current_thresholds': {
                        f"{int(rate*100)}%_hashrate": f"{Web3.from_wei(threshold.fork_threshold, 'ether'):.2f} ETH"
                        for rate, threshold in self.mev_thresholds.items()
                    }
                },
                'paper_benchmarks': {
                    'bzx_benchmark_ratio': self.performance_metrics['bzx_benchmark_comparison'],
                    'bzx_profit_comparison': {
                        'paper_bzx_profit_eth': float(Web3.from_wei(self.bzx_benchmark_profit, 'ether')),
                        'our_largest_eth': largest_mev_eth,
                        'ratio_achieved': self.performance_metrics['bzx_benchmark_comparison']
                    },
                    'weekly_target_progress': {
                        'target_eth': float(Web3.from_wei(self.paper_benchmarks['weekly_revenue_target'], 'ether')),
                        'current_total_eth': total_mev_eth,
                        'progress_percentage': (total_mev_eth / float(Web3.from_wei(self.paper_benchmarks['weekly_revenue_target'], 'ether'))) * 100
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"성능 보고서 생성 실패: {e}")
            return {'error': str(e)}
    
    # Helper methods (simplified implementations)
    async def _estimate_sandwich_profit(self, tx: Dict) -> int:
        """Sandwich 공격 수익 추정"""
        try:
            # 간소화된 계산: 거래 규모의 0.1% 수익 가정
            tx_value = tx.get('value', 0)
            return int(tx_value * 0.001)  # 0.1% 수익
        except Exception:
            return 0
    
    def _extract_tokens_from_tx(self, tx: Dict) -> List[str]:
        """트랜잭션에서 토큰 추출"""
        # 간소화된 구현
        return ['ETH', 'USDC']
    
    def _extract_protocols_from_tx(self, tx: Dict) -> List[str]:
        """트랜잭션에서 프로토콜 추출"""
        # 간소화된 구현
        return ['uniswap_v2']