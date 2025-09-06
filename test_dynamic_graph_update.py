#!/usr/bin/env python3
"""
Dynamic Graph Update 기능 테스트
실시간 상태 변화 반영 시스템 검증

논문 요구사항 검증:
- Dynamic graph update: 실시간 상태 변화 반영
- 업데이트 큐 시스템을 통한 효율적인 처리
- 상태 변화 감지 및 알림 시스템
- 멀티그래프 지원과 함께 작동
"""

import asyncio
import time
import random
from typing import Dict
from src.market_graph import DeFiMarketGraph
from src.logger import setup_logger

logger = setup_logger(__name__)

class DynamicGraphUpdateTester:
    """Dynamic Graph Update 기능 테스트"""
    
    def __init__(self):
        self.graph = DeFiMarketGraph()
        self.state_changes_received = []
        self.test_results = {}
    
    async def test_state_change_listener(self):
        """상태 변화 리스너 테스트"""
        logger.info("=== 상태 변화 리스너 테스트 시작 ===")
        
        # 리스너 등록
        self.graph.register_state_change_listener(self._state_change_callback)
        
        # 초기 토큰 추가
        tokens = [
            ("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "WETH"),
            ("0xA0b86a91c6218b36c1d19D4a2e9Eb0cE3606eB48", "USDC"),
            ("0x6B175474E89094C44Da98b954EedeAC495271d0F", "DAI"),
            ("0xdAC17F958D2ee523a2206206994597C13D831ec7", "USDT")
        ]
        
        for token_addr, symbol in tokens:
            self.graph.add_token(token_addr, symbol)
        
        # 거래 쌍 추가 (상태 변화 발생)
        pairs = [
            ("WETH", "USDC"), ("WETH", "DAI"), ("WETH", "USDT"),
            ("USDC", "DAI"), ("USDC", "USDT"), ("DAI", "USDT")
        ]
        
        token_map = {symbol: addr for addr, symbol in tokens}
        
        for token0_symbol, token1_symbol in pairs:
            token0 = token_map[token0_symbol]
            token1 = token_map[token1_symbol]
            
            # 여러 DEX에 동일 쌍 추가 (Multi-graph 테스트)
            for dex in ["uniswap_v2", "sushiswap"]:
                pool_address = f"0x{hash((token0, token1, dex)) % (16**40):040x}"
                reserve0 = random.uniform(100, 10000)
                reserve1 = random.uniform(100000, 50000000)
                
                if token0_symbol == "WETH":
                    reserve1 = reserve0 * 2000 * random.uniform(0.95, 1.05)
                
                self.graph.add_trading_pair(
                    token0, token1, dex, pool_address, reserve0, reserve1
                )
                
                await asyncio.sleep(0.1)  # 상태 변화 감지를 위한 짧은 대기
        
        # 결과 확인
        received_notifications = len(self.state_changes_received)
        logger.info(f"상태 변화 알림 수신: {received_notifications}개")
        
        self.test_results['state_listener'] = {
            'notifications_received': received_notifications,
            'success': received_notifications > 0
        }
        
        return received_notifications > 0
    
    async def test_update_queue_system(self):
        """업데이트 큐 시스템 테스트"""
        logger.info("=== 업데이트 큐 시스템 테스트 시작 ===")
        
        # 큐 초기화
        self.graph.clear_update_queue()
        
        # 다양한 우선순위의 업데이트 추가
        updates = [
            ('pool_update', {'pool_address': '0x123', 'reserve0': 1000, 'reserve1': 2000000}, 1),
            ('pool_update', {'pool_address': '0x456', 'reserve0': 2000, 'reserve1': 4000000}, 3),
            ('token_add', {'token_address': '0x789', 'symbol': 'TEST'}, 5),
            ('pool_update', {'pool_address': '0xabc', 'reserve0': 3000, 'reserve1': 6000000}, 2),
        ]
        
        # 큐에 업데이트 추가
        for update_type, data, priority in updates:
            self.graph.queue_update(update_type, data, priority)
        
        # 큐 상태 확인
        initial_queue_size = self.graph.get_update_stats()['queue_size']
        logger.info(f"큐에 추가된 업데이트: {initial_queue_size}개")
        
        # 큐 처리
        start_time = time.time()
        processed_count = self.graph.process_update_queue()
        processing_time = time.time() - start_time
        
        # 결과 확인
        final_queue_size = self.graph.get_update_stats()['queue_size']
        update_stats = self.graph.get_update_stats()
        
        logger.info(f"처리된 업데이트: {processed_count}개")
        logger.info(f"처리 시간: {processing_time:.3f}초")
        logger.info(f"남은 큐 크기: {final_queue_size}개")
        logger.info(f"총 업데이트 통계: {update_stats['total_updates']}개")
        
        self.test_results['update_queue'] = {
            'initial_queue_size': initial_queue_size,
            'processed_count': processed_count,
            'final_queue_size': final_queue_size,
            'processing_time': processing_time,
            'success': processed_count == initial_queue_size and final_queue_size == 0
        }
        
        return processed_count == initial_queue_size and final_queue_size == 0
    
    async def test_state_change_detection(self):
        """상태 변화 감지 테스트"""
        logger.info("=== 상태 변화 감지 테스트 시작 ===")
        
        # 초기 상태 해시 기록
        initial_change = self.graph.detect_state_changes()  # 첫 실행은 None 반환
        logger.info(f"초기 상태 해시 설정: {initial_change}")
        
        # 첫 번째 상태 변화 없음 확인
        no_change = self.graph.detect_state_changes()
        logger.info(f"변화 없음 확인: {no_change}")
        
        # 상태 변화 발생시키기 (기존 풀 데이터 업데이트)
        # 먼저 테스트 풀을 생성
        test_pool = f"0x{hash(('test_token1', 'test_token2', 'test_dex')) % (16**40):040x}"
        self.graph.add_trading_pair(
            "0xtest1", "0xtest2", "test_dex", 
            test_pool, 1000, 1000000, 0.003
        )
        
        # 이제 해당 풀을 업데이트 (상태 변화 발생)
        self.graph.update_pool_data(test_pool, 2000, 4000000)
        
        # 상태 변화 감지
        change_detected = self.graph.detect_state_changes()
        logger.info(f"상태 변화 감지: {change_detected}")
        
        self.test_results['state_detection'] = {
            'initial_setup': initial_change is None,
            'no_change_detected': not no_change.get('changed', True),
            'change_detected': change_detected and change_detected.get('changed', False),
            'success': (initial_change is None and 
                       not no_change.get('changed', True) and
                       change_detected and change_detected.get('changed', False))
        }
        
        return self.test_results['state_detection']['success']
    
    async def test_real_time_summary(self):
        """실시간 요약 정보 테스트"""
        logger.info("=== 실시간 요약 정보 테스트 시작 ===")
        
        summary = self.graph.get_real_time_summary()
        
        logger.info("실시간 요약:")
        logger.info(f"  그래프: {summary['graph']['nodes']}개 노드, {summary['graph']['edges']}개 엣지")
        logger.info(f"  업데이트: {summary['updates']['total_updates']}개 총 업데이트")
        logger.info(f"  상태: {summary['state']['listeners_active']}개 리스너 활성")
        logger.info(f"  논문 준수: {summary['paper_compliance']}")
        
        # 논문 준수 사항 확인
        compliance = summary['paper_compliance']
        success = (compliance['real_time_updates'] and
                  compliance['multi_graph_support'] and
                  compliance['state_change_detection'])
        
        self.test_results['real_time_summary'] = {
            'summary': summary,
            'paper_compliance': compliance,
            'success': success
        }
        
        return success
    
    async def test_multi_graph_dynamic_update(self):
        """멀티그래프 동적 업데이트 테스트"""
        logger.info("=== 멀티그래프 동적 업데이트 테스트 시작 ===")
        
        # 동일 토큰 쌍에 여러 DEX 추가 (새로운 토큰 쌍 사용)
        token0 = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"  # WBTC 
        token1 = "0x514910771AF9Ca656af840dff83E8264EcF986CA"  # LINK
        
        dexes = ["uniswap_v2", "uniswap_v3", "sushiswap", "curve"]
        
        # 각 DEX에 풀 추가
        for i, dex in enumerate(dexes):
            pool_address = f"0x{(hash((token0, token1, dex)) % (16**40)):040x}"
            reserve0 = 1000 + i * 100
            reserve1 = (1000 + i * 100) * 2000
            
            self.graph.add_trading_pair(
                token0, token1, dex, pool_address, reserve0, reserve1
            )
        
        # 멀티그래프 통계 확인
        multi_stats = self.graph.get_multi_graph_stats()
        logger.info(f"멀티그래프 통계: {multi_stats}")
        
        # 동적 업데이트 테스트 (첫 번째 DEX 풀 업데이트)
        first_pool = f"0x{(hash((token0, token1, dexes[0])) % (16**40)):040x}"
        
        # 큐를 통한 동적 업데이트
        self.graph.queue_update('pool_update', {
            'pool_address': first_pool,
            'reserve0': 5000,
            'reserve1': 10000000
        }, priority=1)
        
        # 업데이트 처리
        processed = self.graph.process_update_queue()
        
        # 최적 엣지 확인
        best_edge = self.graph.get_best_edge(token0, token1)
        all_edges_01 = self.graph.get_all_edges(token0, token1)
        all_edges_10 = self.graph.get_all_edges(token1, token0)
        
        logger.info(f"최적 엣지: {best_edge['dex'] if best_edge else 'None'}")
        logger.info(f"총 엣지 수 (token0->token1): {len(all_edges_01)}개")
        logger.info(f"총 엣지 수 (token1->token0): {len(all_edges_10)}개")
        
        # 각 방향마다 4개 DEX의 엣지가 있어야 함
        success = (multi_stats['multi_dex_pairs'] > 0 and
                  processed > 0 and
                  len(all_edges_01) == len(dexes) and
                  len(all_edges_10) == len(dexes))
        
        self.test_results['multi_graph_dynamic'] = {
            'multi_stats': multi_stats,
            'processed_updates': processed,
            'best_edge_dex': best_edge['dex'] if best_edge else None,
            'total_edges_01': len(all_edges_01),
            'total_edges_10': len(all_edges_10),
            'success': success
        }
        
        return success
    
    async def _state_change_callback(self, notification: Dict):
        """상태 변화 콜백"""
        self.state_changes_received.append(notification)
        logger.debug(f"상태 변화 알림 수신: {notification['type']}")
    
    async def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("=== Dynamic Graph Update 기능 테스트 시작 ===")
        start_time = time.time()
        
        tests = [
            ("상태 변화 리스너", self.test_state_change_listener),
            ("업데이트 큐 시스템", self.test_update_queue_system), 
            ("상태 변화 감지", self.test_state_change_detection),
            ("실시간 요약 정보", self.test_real_time_summary),
            ("멀티그래프 동적 업데이트", self.test_multi_graph_dynamic_update)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\n--- {test_name} 테스트 시작 ---")
                result = await test_func()
                results[test_name] = "✅ 성공" if result else "❌ 실패"
                logger.info(f"{test_name}: {results[test_name]}")
            except Exception as e:
                results[test_name] = f"❌ 오류: {e}"
                logger.error(f"{test_name} 오류: {e}")
        
        total_time = time.time() - start_time
        
        # 최종 결과
        logger.info("\n" + "="*60)
        logger.info("Dynamic Graph Update 테스트 결과:")
        logger.info("="*60)
        
        for test_name, result in results.items():
            logger.info(f"{test_name}: {result}")
        
        success_count = sum(1 for r in results.values() if "✅" in r)
        total_tests = len(results)
        
        logger.info(f"\n성공률: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")
        logger.info(f"총 실행 시간: {total_time:.2f}초")
        
        # 상세 테스트 결과
        logger.info(f"\n상세 결과: {self.test_results}")
        
        # 논문 요구사항 준수 확인
        paper_compliance = self._check_paper_compliance()
        logger.info(f"\n논문 요구사항 준수:")
        for requirement, status in paper_compliance.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"  {requirement}: {status_icon}")
        
        return success_count == total_tests
    
    def _check_paper_compliance(self) -> Dict[str, bool]:
        """논문 요구사항 준수 여부 확인"""
        return {
            "실시간 상태 변화 반영": self.test_results.get('state_detection', {}).get('success', False),
            "멀티그래프 지원": self.test_results.get('multi_graph_dynamic', {}).get('success', False),
            "업데이트 큐 시스템": self.test_results.get('update_queue', {}).get('success', False),
            "상태 변화 감지 시스템": self.test_results.get('state_listener', {}).get('success', False),
            "실시간 요약 정보": self.test_results.get('real_time_summary', {}).get('success', False)
        }

async def main():
    """메인 테스트 실행"""
    tester = DynamicGraphUpdateTester()
    
    try:
        success = await tester.run_all_tests()
        
        if success:
            logger.info("\n🎉 모든 Dynamic Graph Update 테스트 통과!")
            logger.info("논문의 '실시간 상태 변화 반영' 요구사항이 성공적으로 구현되었습니다.")
        else:
            logger.warning("\n⚠️  일부 테스트 실패")
            logger.warning("Dynamic Graph Update 구현을 점검해주세요.")
            
    except Exception as e:
        logger.error(f"테스트 실행 중 오류 발생: {e}")
        return False
    
    return success

if __name__ == "__main__":
    asyncio.run(main())