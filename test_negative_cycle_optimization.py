#!/usr/bin/env python3
"""
Negative Cycle Detection 최적화 성능 테스트
논문 기준 6.43초 달성 검증
"""

import sys
import os
import time
import asyncio
from typing import List, Dict

# 프로젝트 루트 디렉토리 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.market_graph import DeFiMarketGraph
from src.bellman_ford_arbitrage import BellmanFordArbitrage
from src.logger import setup_logger

logger = setup_logger(__name__)

class NegativeCycleOptimizationTest:
    """Negative Cycle Detection 최적화 성능 테스트"""
    
    def __init__(self):
        self.graph = DeFiMarketGraph()
        self.bellman_ford = BellmanFordArbitrage(self.graph)
        self.test_results = []
        
    def create_test_graph(self, num_tokens: int = 25, num_edges: int = 96):
        """테스트용 그래프 생성 (논문 기준 규모)"""
        logger.info(f"테스트 그래프 생성: {num_tokens}개 토큰, {num_edges}개 엣지")
        
        # 메인 토큰들 (논문에서 사용한 토큰들과 유사)
        main_tokens = [
            'ETH', 'WETH', 'USDC', 'USDT', 'DAI', 'WBTC', 'UNI', 'SUSHI', 
            'COMP', 'AAVE', 'CRV', 'BAL', 'YFI', 'MKR', 'LINK', 'SNX', 
            'MATIC', 'FTT', 'BNT', 'ZRX', 'LRC', 'ENJ', 'MANA', 'BAT', 'KNC'
        ][:num_tokens]
        
        # 토큰 추가
        for token in main_tokens:
            self.graph.add_token(token)
            
        # DEX 목록
        dexes = ['Uniswap_V2', 'Sushiswap', 'Bancor', 'Balancer', 'Curve', '1inch']
        
        # 다양한 거래쌍 생성
        edge_count = 0
        for i, token0 in enumerate(main_tokens):
            if edge_count >= num_edges:
                break
                
            for j, token1 in enumerate(main_tokens):
                if i == j or edge_count >= num_edges:
                    continue
                    
                # 여러 DEX에 동일한 쌍이 있을 수 있음 (Multi-graph)
                dex = dexes[edge_count % len(dexes)]
                
                # 가격 변동을 시뮬레이션한 환율 설정
                base_rate = 1.0 + (i - j) * 0.001  # 작은 기본 차이
                volatility = 0.005 * (1 + edge_count % 10)  # 변동성
                exchange_rate = base_rate + volatility
                
                # 유동성 및 reserve 설정
                reserve0 = 1000.0 + (edge_count * 100)
                reserve1 = reserve0 * exchange_rate
                
                self.graph.add_trading_pair(
                    token0=token0,
                    token1=token1,
                    dex=dex,
                    pool_address=f"pool_{token0}_{token1}_{dex}_{edge_count}",
                    reserve0=reserve0,
                    reserve1=reserve1,
                    fee=0.003  # 0.3% fee
                )
                
                edge_count += 1
        
        logger.info(f"테스트 그래프 완성: {len(self.graph.token_nodes)}개 토큰, "
                   f"{self.graph.graph.number_of_edges()}개 엣지")
        
        # 몇 개의 negative cycle을 의도적으로 생성
        self._inject_negative_cycles()
    
    def _inject_negative_cycles(self):
        """음의 사이클 의도적 생성 (테스트용)"""
        logger.info("테스트용 음의 사이클 주입")
        
        # ETH -> USDC -> WBTC -> ETH 사이클
        if all(token in self.graph.token_nodes for token in ['ETH', 'USDC', 'WBTC']):
            # 약간의 차익거래 기회 생성
            self.graph.add_trading_pair(
                'ETH', 'USDC', 'TestDEX1', 'test_pool_1',
                1000, 3000, 0.001  # 낮은 수수료로 차익거래 기회 생성
            )
            self.graph.add_trading_pair(
                'USDC', 'WBTC', 'TestDEX2', 'test_pool_2',
                3000, 0.1, 0.002
            )
            self.graph.add_trading_pair(
                'WBTC', 'ETH', 'TestDEX3', 'test_pool_3',
                0.1, 1.01, 0.001  # 약간의 이익 여지
            )
    
    async def run_performance_test(self, num_runs: int = 10) -> Dict:
        """성능 테스트 실행"""
        logger.info(f"성능 테스트 시작: {num_runs}회 실행")
        
        results = {
            'total_times': [],
            'bellman_ford_times': [],
            'cycle_extraction_times': [],
            'local_search_times': [],
            'opportunities_found': [],
            'meets_paper_requirement': []
        }
        
        target_time = 6.43  # 논문 목표 시간 (초)
        
        for run in range(num_runs):
            logger.info(f"테스트 실행 {run + 1}/{num_runs}")
            
            # 성능 메트릭 초기화
            self.bellman_ford.reset_performance_metrics()
            
            # ETH를 소스로 차익거래 기회 탐지
            start_time = time.time()
            opportunities = self.bellman_ford.find_negative_cycles('ETH', max_path_length=5)
            total_time = time.time() - start_time
            
            # 성능 메트릭 수집
            metrics = self.bellman_ford.get_performance_metrics()
            
            results['total_times'].append(total_time)
            results['bellman_ford_times'].append(metrics['bellman_ford_time'])
            results['cycle_extraction_times'].append(metrics['cycle_extraction_time'])
            results['local_search_times'].append(metrics['local_search_time'])
            results['opportunities_found'].append(len(opportunities))
            results['meets_paper_requirement'].append(total_time <= target_time)
            
            logger.info(f"실행 {run + 1} 완료: {total_time:.3f}초, "
                       f"{len(opportunities)}개 기회 발견, "
                       f"목표 달성: {'✅' if total_time <= target_time else '❌'}")
        
        # 통계 계산
        avg_total_time = sum(results['total_times']) / len(results['total_times'])
        avg_bf_time = sum(results['bellman_ford_times']) / len(results['bellman_ford_times'])
        avg_cycle_time = sum(results['cycle_extraction_times']) / len(results['cycle_extraction_times'])
        avg_local_search_time = sum(results['local_search_times']) / len(results['local_search_times'])
        success_rate = sum(results['meets_paper_requirement']) / len(results['meets_paper_requirement'])
        
        summary = {
            'average_total_time': avg_total_time,
            'average_bellman_ford_time': avg_bf_time,
            'average_cycle_extraction_time': avg_cycle_time,
            'average_local_search_time': avg_local_search_time,
            'success_rate': success_rate,
            'target_time': target_time,
            'meets_requirement': avg_total_time <= target_time,
            'performance_improvement': max(0, (target_time - avg_total_time) / target_time * 100),
            'raw_results': results
        }
        
        return summary
    
    def print_performance_report(self, summary: Dict):
        """성능 테스트 결과 출력"""
        print("\n" + "="*80)
        print("NEGATIVE CYCLE DETECTION 최적화 성능 테스트 결과")
        print("="*80)
        print(f"📊 테스트 환경:")
        print(f"   - 토큰 수: {len(self.graph.token_nodes)}개")
        print(f"   - 엣지 수: {self.graph.graph.number_of_edges()}개")
        print(f"   - 목표 시간: {summary['target_time']:.3f}초 (논문 기준)")
        print()
        
        print(f"⏱️  평균 성능 결과:")
        print(f"   - 전체 실행 시간: {summary['average_total_time']:.3f}초")
        print(f"   - Bellman-Ford: {summary['average_bellman_ford_time']:.3f}초")
        print(f"   - 사이클 추출: {summary['average_cycle_extraction_time']:.3f}초") 
        print(f"   - 로컬 서치: {summary['average_local_search_time']:.3f}초")
        print()
        
        print(f"✅ 성능 목표 달성:")
        success_icon = "🎯" if summary['meets_requirement'] else "❌"
        print(f"   {success_icon} 논문 기준 달성: {summary['meets_requirement']}")
        print(f"   📈 성공률: {summary['success_rate']*100:.1f}%")
        
        if summary['meets_requirement']:
            print(f"   🚀 성능 개선: {summary['performance_improvement']:.1f}%")
            print(f"   💡 논문 목표 대비 {summary['target_time'] - summary['average_total_time']:.3f}초 빠름")
        else:
            deficit = summary['average_total_time'] - summary['target_time']
            print(f"   ⚠️  목표 미달: {deficit:.3f}초 초과")
        print()
        
        print(f"🔧 최적화 효과:")
        print(f"   - SPFA 하이브리드 알고리즘 적용")
        print(f"   - 큐 기반 업데이트로 불필요한 연산 제거")
        print(f"   - 캐시된 엣지 리스트 사용")
        print(f"   - 음의 사이클 조기 감지")
        print("="*80)

async def main():
    """메인 테스트 함수"""
    print("🚀 Negative Cycle Detection 최적화 테스트 시작")
    
    # 테스트 인스턴스 생성
    test = NegativeCycleOptimizationTest()
    
    # 논문 기준 규모의 테스트 그래프 생성
    test.create_test_graph(num_tokens=25, num_edges=96)
    
    # 성능 테스트 실행
    summary = await test.run_performance_test(num_runs=5)
    
    # 결과 출력
    test.print_performance_report(summary)
    
    # 추가 상세 분석
    if summary['meets_requirement']:
        print("\n✅ 최적화 성공! 논문의 6.43초 기준을 달성했습니다.")
        print(f"📈 평균 {summary['average_total_time']:.3f}초로 {summary['performance_improvement']:.1f}% 성능 향상")
    else:
        print(f"\n⚠️  추가 최적화 필요: 평균 {summary['average_total_time']:.3f}초")
        print("💡 다음 최적화 고려사항:")
        print("   - 그래프 pruning 강화")
        print("   - 병렬 처리 확장")
        print("   - 메모리 캐싱 개선")
    
    return summary

if __name__ == "__main__":
    # rye 환경에서 실행
    try:
        summary = asyncio.run(main())
        
        # 성공적으로 최적화된 경우에만 TODO 업데이트 신호
        if summary and summary['meets_requirement']:
            print(f"\n🎉 TODO 업데이트 대상: Negative cycle detection 알고리즘 최적화 완료!")
            exit(0)  # 성공 코드
        else:
            print(f"\n❌ 최적화 목표 미달성")
            exit(1)  # 실패 코드
            
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        exit(1)