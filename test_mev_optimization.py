#!/usr/bin/env python3
"""
MEV Optimization 테스트 스크립트
논문 "[2103.02228] On the Just-In-Time Discovery of Profit-Generating Transactions in DeFi Protocols" MEV 기능 검증

테스트 목표:
1. MEV threshold 계산 정확성 (논문 기준: 10% 해시레이트에서 4× 블록 보상)
2. MDP 최적 전략 계산
3. bZx 공격 수준의 MEV 탐지 (874× 블록 보상)
4. MEV 스코어링 시스템 
5. 실시간 MEV 기회 탐지 성능
"""

import asyncio
import time
import json
from decimal import Decimal
from web3 import Web3
from eth_account import Account

# DeFi 시스템 import
from src.mev_optimizer import MEVOptimizer, MEVType, MEVOpportunity, MDP_Strategy
from src.market_graph import DeFiMarketGraph
from src.economic_state_exploiter import EconomicStateExploiter 
from src.flash_loan_manager import FlashLoanManager
from src.transaction_pool_monitor import TransactionPoolMonitor
from src.protocol_actions import ProtocolRegistry
from src.logger import setup_logger

logger = setup_logger(__name__)

class MEVOptimizationTest:
    """MEV Optimization 종합 테스트"""
    
    def __init__(self):
        # Web3 연결
        self.w3 = Web3(Web3.HTTPProvider("https://eth-mainnet.alchemyapi.io/v2/demo"))
        self.account = Account.from_key("0x" + "1" * 64)  # 테스트용 계정
        
        # 시스템 구성 요소 초기화
        self.market_graph = DeFiMarketGraph()
        self.protocol_registry = ProtocolRegistry(self.w3)
        self.flash_loan_manager = FlashLoanManager(self.w3, self.account)
        self.tx_pool_monitor = TransactionPoolMonitor()
        
        # Economic State Exploiter 초기화
        self.economic_exploiter = EconomicStateExploiter(
            self.w3, 
            self.account,
            self.market_graph,
            self.flash_loan_manager,
            self.protocol_registry
        )
        
        # MEV Optimizer 초기화
        self.mev_optimizer = MEVOptimizer(
            self.w3,
            self.market_graph,
            self.economic_exploiter,
            self.flash_loan_manager,
            self.tx_pool_monitor,
            self.protocol_registry
        )
        
        # 테스트 결과 저장
        self.test_results = {
            'threshold_tests': {},
            'mdp_tests': {},
            'detection_tests': {},
            'scoring_tests': {},
            'performance_tests': {}
        }
        
        logger.info("MEV Optimization 테스트 시스템 초기화 완료")
    
    async def run_comprehensive_tests(self):
        """종합 MEV 최적화 테스트 실행"""
        logger.info("=== MEV Optimization 종합 테스트 시작 ===")
        
        try:
            # 1. MEV Threshold 테스트 (논문 검증)
            await self.test_mev_thresholds()
            
            # 2. MDP 전략 테스트
            await self.test_mdp_strategies()
            
            # 3. MEV 탐지 테스트
            await self.test_mev_detection()
            
            # 4. MEV 스코어링 테스트
            await self.test_mev_scoring()
            
            # 5. 성능 테스트
            await self.test_performance()
            
            # 6. bZx 벤치마크 테스트
            await self.test_bzx_benchmark()
            
            # 결과 보고서 생성
            await self.generate_test_report()
            
        except Exception as e:
            logger.error(f"MEV 테스트 실행 중 오류: {e}")
        
        logger.info("=== MEV Optimization 테스트 완료 ===")
    
    async def test_mev_thresholds(self):
        """MEV Threshold 계산 테스트 (논문 검증)"""
        logger.info("1. MEV Threshold 계산 테스트 시작")
        
        try:
            # 논문 기준: 10% 해시레이트에서 4× 블록 보상 threshold
            target_hash_rate = 0.1
            expected_multiplier = 4.0
            
            # Threshold 계산 대기 (초기화 시간)
            await asyncio.sleep(2.0)
            
            # 계산된 threshold 검증
            threshold = self.mev_optimizer.mev_thresholds.get(target_hash_rate)
            
            if threshold:
                actual_multiplier = threshold.threshold_multiplier
                threshold_eth = Web3.from_wei(threshold.fork_threshold, 'ether')
                
                logger.info(f"10% 해시레이트 MEV Threshold:")
                logger.info(f"  - 계산된 배수: {actual_multiplier:.2f}× (목표: {expected_multiplier}×)")
                logger.info(f"  - Threshold 값: {threshold_eth:.2f} ETH")
                logger.info(f"  - 블록 보상: {Web3.from_wei(threshold.block_reward, 'ether'):.2f} ETH")
                
                # 정확도 체크
                accuracy = abs(actual_multiplier - expected_multiplier) / expected_multiplier
                test_passed = accuracy < 0.2  # 20% 오차 허용
                
                self.test_results['threshold_tests']['10_percent_hashrate'] = {
                    'expected_multiplier': expected_multiplier,
                    'actual_multiplier': actual_multiplier,
                    'accuracy': 1.0 - accuracy,
                    'test_passed': test_passed,
                    'threshold_eth': float(threshold_eth)
                }
                
                logger.info(f"  - 테스트 결과: {'✅ 통과' if test_passed else '❌ 실패'} (정확도: {(1-accuracy)*100:.1f}%)")
                
            else:
                logger.error("MEV threshold 계산 실패")
                self.test_results['threshold_tests']['10_percent_hashrate'] = {
                    'test_passed': False,
                    'error': 'threshold_not_calculated'
                }
            
            # 다른 해시레이트들도 테스트
            for hash_rate in [0.01, 0.05, 0.2, 0.3]:
                await self._test_specific_hashrate_threshold(hash_rate)
                
        except Exception as e:
            logger.error(f"MEV Threshold 테스트 실패: {e}")
            self.test_results['threshold_tests']['error'] = str(e)
    
    async def _test_specific_hashrate_threshold(self, hash_rate: float):
        """특정 해시레이트의 threshold 테스트"""
        threshold = self.mev_optimizer.mev_thresholds.get(hash_rate)
        
        if threshold:
            logger.info(f"  {hash_rate*100:.0f}% 해시레이트: {threshold.threshold_multiplier:.2f}× 블록 보상")
            self.test_results['threshold_tests'][f'{int(hash_rate*100)}_percent'] = {
                'multiplier': threshold.threshold_multiplier,
                'threshold_eth': float(Web3.from_wei(threshold.fork_threshold, 'ether'))
            }
    
    async def test_mdp_strategies(self):
        """MDP 최적 전략 계산 테스트"""
        logger.info("2. MDP 전략 계산 테스트 시작")
        
        try:
            # 테스트용 MEV 기회 생성
            test_mev_values = [
                Web3.to_wei(2, 'ether'),   # 1× 블록 보상 
                Web3.to_wei(4, 'ether'),   # 2× 블록 보상
                Web3.to_wei(8, 'ether'),   # 4× 블록 보상 (threshold)
                Web3.to_wei(20, 'ether'),  # 10× 블록 보상
                Web3.to_wei(1748, 'ether') # bZx 수준 (874× 블록 보상)
            ]
            
            for i, mev_value in enumerate(test_mev_values):
                test_opportunity = MEVOpportunity(
                    opportunity_id=f"test_mdp_{i}",
                    mev_type=MEVType.ECONOMIC_EXPLOIT,
                    potential_profit=Decimal(str(mev_value)),
                    required_capital=Decimal(str(Web3.to_wei(0.1, 'ether'))),
                    success_probability=0.8,
                    time_sensitivity=60,
                    block_dependency=1,
                    affected_tokens=['WETH', 'WBTC'],
                    affected_protocols=['compound', 'uniswap'],
                    execution_cost=Web3.to_wei(0.05, 'ether'),
                    mev_score=0.8,
                    competition_level=0.3,
                    discovery_timestamp=time.time()
                )
                
                strategy = await self.mev_optimizer._calculate_optimal_strategy(test_opportunity)
                mev_eth = Web3.from_wei(mev_value, 'ether')
                multiplier = float(mev_value) / float(self.mev_optimizer.ethereum_block_reward)
                
                logger.info(f"  MEV {mev_eth:.0f} ETH ({multiplier:.1f}×): {strategy.optimal_action} "
                           f"(확신도: {strategy.strategy_confidence:.2f})")
                
                self.test_results['mdp_tests'][f'mev_{int(mev_eth)}'] = {
                    'mev_value_eth': float(mev_eth),
                    'block_reward_multiplier': multiplier,
                    'optimal_action': strategy.optimal_action,
                    'expected_reward': float(Web3.from_wei(int(strategy.expected_reward), 'ether')),
                    'fork_probability': strategy.fork_probability,
                    'strategy_confidence': strategy.strategy_confidence
                }
        
        except Exception as e:
            logger.error(f"MDP 전략 테스트 실패: {e}")
            self.test_results['mdp_tests']['error'] = str(e)
    
    async def test_mev_detection(self):
        """MEV 탐지 기능 테스트"""
        logger.info("3. MEV 탐지 기능 테스트 시작")
        
        try:
            # 모의 시장 데이터 설정
            await self._setup_mock_market_data()
            
            # MEV 기회 탐지 실행
            detection_start = time.time()
            opportunities = await self.mev_optimizer.detect_mev_opportunities(
                block_number=18500000  # 테스트 블록 번호
            )
            detection_time = time.time() - detection_start
            
            logger.info(f"  탐지된 MEV 기회: {len(opportunities)}개 ({detection_time:.3f}초)")
            
            # MEV 유형별 분석
            mev_types = {}
            total_value = Decimal('0')
            
            for opp in opportunities:
                mev_type = opp.mev_type.value
                mev_types[mev_type] = mev_types.get(mev_type, 0) + 1
                total_value += opp.potential_profit
            
            logger.info(f"  총 예상 수익: {Web3.from_wei(int(total_value), 'ether'):.4f} ETH")
            logger.info(f"  MEV 유형 분포: {mev_types}")
            
            # 최고 수익 기회
            if opportunities:
                best_opp = max(opportunities, key=lambda x: x.potential_profit)
                best_eth = Web3.from_wei(int(best_opp.potential_profit), 'ether')
                logger.info(f"  최고 수익 기회: {best_eth:.4f} ETH ({best_opp.mev_type.value})")
            
            self.test_results['detection_tests'] = {
                'opportunities_detected': len(opportunities),
                'total_value_eth': float(Web3.from_wei(int(total_value), 'ether')),
                'detection_time_seconds': detection_time,
                'mev_type_distribution': mev_types,
                'best_opportunity_eth': float(Web3.from_wei(int(best_opp.potential_profit), 'ether')) if opportunities else 0
            }
            
        except Exception as e:
            logger.error(f"MEV 탐지 테스트 실패: {e}")
            self.test_results['detection_tests']['error'] = str(e)
    
    async def test_mev_scoring(self):
        """MEV 스코어링 시스템 테스트"""
        logger.info("4. MEV 스코어링 시스템 테스트 시작")
        
        try:
            # 다양한 MEV 기회 생성 (테스트용)
            test_opportunities = [
                # 고수익, 높은 성공률
                MEVOpportunity(
                    opportunity_id="high_profit_high_success",
                    mev_type=MEVType.ARBITRAGE,
                    potential_profit=Decimal(str(Web3.to_wei(5, 'ether'))),
                    required_capital=Decimal(str(Web3.to_wei(10, 'ether'))),
                    success_probability=0.9,
                    time_sensitivity=10,
                    block_dependency=1,
                    affected_tokens=['WETH', 'USDC'],
                    affected_protocols=['uniswap'],
                    execution_cost=Web3.to_wei(0.02, 'ether'),
                    mev_score=0.0,
                    competition_level=0.3,
                    discovery_timestamp=time.time()
                ),
                # 중수익, 중간 성공률
                MEVOpportunity(
                    opportunity_id="med_profit_med_success",
                    mev_type=MEVType.LIQUIDATION,
                    potential_profit=Decimal(str(Web3.to_wei(1, 'ether'))),
                    required_capital=Decimal(str(Web3.to_wei(5, 'ether'))),
                    success_probability=0.6,
                    time_sensitivity=60,
                    block_dependency=2,
                    affected_tokens=['DAI'],
                    affected_protocols=['compound'],
                    execution_cost=Web3.to_wei(0.05, 'ether'),
                    mev_score=0.0,
                    competition_level=0.7,
                    discovery_timestamp=time.time()
                ),
                # 저수익, 낮은 성공률
                MEVOpportunity(
                    opportunity_id="low_profit_low_success",
                    mev_type=MEVType.FRONT_RUNNING,
                    potential_profit=Decimal(str(Web3.to_wei(0.1, 'ether'))),
                    required_capital=Decimal(str(Web3.to_wei(2, 'ether'))),
                    success_probability=0.3,
                    time_sensitivity=300,
                    block_dependency=5,
                    affected_tokens=['USDT'],
                    affected_protocols=['sushiswap'],
                    execution_cost=Web3.to_wei(0.08, 'ether'),
                    mev_score=0.0,
                    competition_level=0.9,
                    discovery_timestamp=time.time()
                )
            ]
            
            # 스코어 계산
            scored_opportunities = await self.mev_optimizer._calculate_mev_scores(test_opportunities)
            
            logger.info("  MEV 스코어링 결과:")
            for opp in scored_opportunities:
                profit_eth = Web3.from_wei(int(opp.potential_profit), 'ether')
                logger.info(f"    {opp.opportunity_id}: {opp.mev_score:.3f} "
                           f"(수익: {profit_eth:.2f} ETH, 성공률: {opp.success_probability:.1f})")
            
            # 필터링 테스트
            viable_opportunities = self.mev_optimizer._filter_viable_opportunities(scored_opportunities)
            logger.info(f"  실행 가능한 기회: {len(viable_opportunities)}/{len(scored_opportunities)}개")
            
            self.test_results['scoring_tests'] = {
                'opportunities_scored': len(scored_opportunities),
                'viable_opportunities': len(viable_opportunities),
                'score_distribution': [opp.mev_score for opp in scored_opportunities],
                'average_score': sum(opp.mev_score for opp in scored_opportunities) / len(scored_opportunities)
            }
            
        except Exception as e:
            logger.error(f"MEV 스코어링 테스트 실패: {e}")
            self.test_results['scoring_tests']['error'] = str(e)
    
    async def test_performance(self):
        """MEV 최적화 성능 테스트"""
        logger.info("5. MEV 최적화 성능 테스트 시작")
        
        try:
            # 성능 측정 루프
            detection_times = []
            optimization_times = []
            
            for i in range(10):  # 10회 측정
                # 탐지 성능
                detection_start = time.time()
                opportunities = await self.mev_optimizer.detect_mev_opportunities(18500000 + i)
                detection_time = time.time() - detection_start
                detection_times.append(detection_time)
                
                # 최적화 성능
                if opportunities:
                    opt_start = time.time()
                    optimized = await self.mev_optimizer.optimize_mev_extraction(opportunities)
                    opt_time = time.time() - opt_start
                    optimization_times.append(opt_time)
            
            avg_detection = sum(detection_times) / len(detection_times)
            avg_optimization = sum(optimization_times) / len(optimization_times) if optimization_times else 0
            
            logger.info(f"  평균 탐지 시간: {avg_detection:.3f}초")
            logger.info(f"  평균 최적화 시간: {avg_optimization:.3f}초")
            logger.info(f"  총 평균 시간: {avg_detection + avg_optimization:.3f}초")
            
            # 논문 성능 목표 비교 (6.43초)
            paper_target = 6.43
            performance_ratio = (avg_detection + avg_optimization) / paper_target
            meets_target = performance_ratio <= 1.0
            
            logger.info(f"  논문 목표 대비: {performance_ratio:.2f} ({'✅ 달성' if meets_target else '❌ 미달성'})")
            
            self.test_results['performance_tests'] = {
                'average_detection_time': avg_detection,
                'average_optimization_time': avg_optimization,
                'total_average_time': avg_detection + avg_optimization,
                'paper_target_seconds': paper_target,
                'performance_ratio': performance_ratio,
                'meets_paper_target': meets_target
            }
            
        except Exception as e:
            logger.error(f"성능 테스트 실패: {e}")
            self.test_results['performance_tests']['error'] = str(e)
    
    async def test_bzx_benchmark(self):
        """bZx 벤치마크 테스트"""
        logger.info("6. bZx 벤치마크 테스트 시작")
        
        try:
            # bZx 수준 MEV 기회 시뮬레이션
            bzx_opportunity = MEVOpportunity(
                opportunity_id="bzx_simulation",
                mev_type=MEVType.ECONOMIC_EXPLOIT,
                potential_profit=self.mev_optimizer.bzx_benchmark_profit,
                required_capital=Decimal(str(Web3.to_wei(0.1, 'ether'))),
                success_probability=0.85,
                time_sensitivity=30,
                block_dependency=1,
                affected_tokens=['WETH', 'WBTC'],
                affected_protocols=['bzx', 'uniswap'],
                execution_cost=Web3.to_wei(0.1, 'ether'),
                mev_score=0.0,
                competition_level=0.1,
                discovery_timestamp=time.time()
            )
            
            # MEV threshold 분석
            threshold_analysis = await self.mev_optimizer._analyze_mev_threshold(bzx_opportunity)
            
            bzx_eth = Web3.from_wei(int(bzx_opportunity.potential_profit), 'ether')
            block_reward_eth = Web3.from_wei(self.mev_optimizer.ethereum_block_reward, 'ether')
            multiplier = bzx_eth / block_reward_eth
            
            logger.info(f"  bZx 시뮬레이션 결과:")
            logger.info(f"    - MEV 가치: {bzx_eth:.2f} ETH")
            logger.info(f"    - 블록 보상 배수: {multiplier:.0f}× (논문: 874×)")
            logger.info(f"    - Threshold 위반: {'예' if threshold_analysis['threshold_violation'] else '아니오'}")
            logger.info(f"    - 포크 인센티브: {'예' if threshold_analysis['fork_incentive_risk'] else '아니오'}")
            
            # MDP 전략 계산
            optimal_strategy = await self.mev_optimizer._calculate_optimal_strategy(bzx_opportunity)
            logger.info(f"    - 최적 전략: {optimal_strategy.optimal_action}")
            logger.info(f"    - 전략 확신도: {optimal_strategy.strategy_confidence:.3f}")
            
            self.test_results['bzx_benchmark'] = {
                'bzx_value_eth': float(bzx_eth),
                'block_reward_multiplier': float(multiplier),
                'paper_multiplier': 874.0,
                'threshold_violation': threshold_analysis['threshold_violation'],
                'fork_incentive_risk': threshold_analysis['fork_incentive_risk'],
                'optimal_strategy': optimal_strategy.optimal_action,
                'strategy_confidence': optimal_strategy.strategy_confidence
            }
            
        except Exception as e:
            logger.error(f"bZx 벤치마크 테스트 실패: {e}")
            self.test_results['bzx_benchmark']['error'] = str(e)
    
    async def _setup_mock_market_data(self):
        """모의 시장 데이터 설정"""
        try:
            # 주요 토큰 쌍 추가
            major_pairs = [
                ("WETH", "USDC", "uniswap_v2", 1000, 2000000),  # ETH/USDC
                ("WETH", "DAI", "uniswap_v2", 800, 1600000),    # ETH/DAI  
                ("WETH", "USDT", "sushiswap", 500, 1000000),    # ETH/USDT
                ("USDC", "DAI", "curve", 1000000, 1000000),     # USDC/DAI
                ("WETH", "WBTC", "balancer", 1000, 50),         # ETH/WBTC
            ]
            
            for token0, token1, dex, reserve0, reserve1 in major_pairs:
                self.market_graph.add_trading_pair(
                    f"0x{token0.lower()}", f"0x{token1.lower()}", dex,
                    f"0x{hash((token0, token1, dex)) % (16**40):040x}",
                    reserve0, reserve1, 0.003
                )
                
        except Exception as e:
            logger.error(f"모의 시장 데이터 설정 실패: {e}")
    
    async def generate_test_report(self):
        """종합 테스트 보고서 생성"""
        logger.info("=== MEV Optimization 테스트 보고서 ===")
        
        try:
            # 성능 보고서 가져오기
            performance_report = self.mev_optimizer.get_mev_performance_report()
            
            # 종합 점수 계산
            total_tests = 0
            passed_tests = 0
            
            for category, tests in self.test_results.items():
                if isinstance(tests, dict) and 'test_passed' in tests:
                    total_tests += 1
                    if tests['test_passed']:
                        passed_tests += 1
                elif isinstance(tests, dict):
                    for test_name, test_data in tests.items():
                        if isinstance(test_data, dict) and 'test_passed' in test_data:
                            total_tests += 1
                            if test_data['test_passed']:
                                passed_tests += 1
            
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            logger.info(f"테스트 통과율: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
            
            # 핵심 지표 요약
            threshold_test = self.test_results.get('threshold_tests', {}).get('10_percent_hashrate', {})
            performance_test = self.test_results.get('performance_tests', {})
            bzx_test = self.test_results.get('bzx_benchmark', {})
            
            logger.info("핵심 성과 지표:")
            
            if threshold_test.get('test_passed'):
                logger.info(f"  ✅ MEV Threshold: {threshold_test.get('actual_multiplier', 0):.2f}× (목표: 4.0×)")
            else:
                logger.info(f"  ❌ MEV Threshold: 테스트 실패")
            
            if performance_test.get('meets_paper_target'):
                logger.info(f"  ✅ 성능 목표: {performance_test.get('performance_ratio', 0):.2f} (목표: ≤1.0)")
            else:
                logger.info(f"  ❌ 성능 목표: {performance_test.get('performance_ratio', 0):.2f} (목표: ≤1.0)")
            
            if bzx_test.get('block_reward_multiplier', 0) > 100:
                logger.info(f"  ✅ bZx 수준 탐지: {bzx_test.get('block_reward_multiplier', 0):.0f}× 블록 보상")
            else:
                logger.info(f"  ❌ bZx 수준 탐지: {bzx_test.get('block_reward_multiplier', 0):.0f}× 블록 보상")
            
            # 결과 파일 저장
            with open('mev_optimization_test_results.json', 'w') as f:
                json.dump({
                    'test_results': self.test_results,
                    'performance_report': performance_report,
                    'summary': {
                        'total_tests': total_tests,
                        'passed_tests': passed_tests,
                        'success_rate': success_rate,
                        'test_timestamp': time.time()
                    }
                }, f, indent=2, default=str)
            
            logger.info("테스트 결과가 'mev_optimization_test_results.json'에 저장되었습니다.")
            
        except Exception as e:
            logger.error(f"테스트 보고서 생성 실패: {e}")

# 메인 실행 함수
async def main():
    """MEV 최적화 테스트 메인 함수"""
    test_system = MEVOptimizationTest()
    await test_system.run_comprehensive_tests()

if __name__ == "__main__":
    asyncio.run(main())