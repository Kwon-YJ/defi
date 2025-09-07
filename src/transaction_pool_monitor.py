#!/usr/bin/env python3
"""
Transaction Pool Monitor
실시간 트랜잭션 풀 모니터링 및 상태 변화 감지

논문 요구사항 구현:
- Transaction pool monitoring
- State change detection 및 즉시 대응
- Gas price tracking for MEV opportunities
- 실시간 mempool analysis
"""

import asyncio
import json
import time
from typing import Dict, List, Callable, Set, Optional
from datetime import datetime, timedelta
from web3 import Web3
from collections import deque, defaultdict
import statistics

from src.logger import setup_logger
from src.memory_storage import get_memory_storage
from config.config import config

logger = setup_logger(__name__)

class TransactionPoolMonitor:
    """
    트랜잭션 풀 모니터
    논문의 "Transaction pool monitoring" 요구사항 구현
    """
    
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(config.ethereum_mainnet_rpc))
        self.storage = get_memory_storage()
        self.running = False
        
        # Mempool 상태
        self.mempool_transactions: Dict[str, Dict] = {}
        self.pending_arbitrage_txs: Set[str] = set()
        self.confirmed_transactions: Set[str] = set()
        
        # Gas price 추적
        self.gas_price_history: deque = deque(maxlen=1000)  # 최근 1000개 샘플
        self.gas_price_stats = {
            'current_median': 0,
            'current_average': 0,
            'current_max': 0,
            'current_min': 0,
            'last_update': None,
            'trend': 'stable'  # 'rising', 'falling', 'stable'
        }
        
        # 상태 변화 감지
        self.state_change_listeners: List[Callable] = []
        self.last_state_hash = None
        self.state_changes_count = 0
        
        # MEV 기회 탐지
        self.potential_mev_txs: Dict[str, Dict] = {}
        self.arbitrage_patterns = [
            'swap',           # DEX swap functions
            'multicall',      # Batch transactions
            'flashLoan',      # Flash loan calls
            'arbitrage',      # Direct arbitrage calls
        ]
        
        # 성능 메트릭스
        self.metrics = {
            'total_txs_monitored': 0,
            'arbitrage_txs_detected': 0,
            'mev_opportunities_found': 0,
            'state_changes_detected': 0,
            'average_processing_time': 0.0
        }
    
    async def start_monitoring(self):
        """트랜잭션 풀 모니터링 시작"""
        self.running = True
        logger.info("=== 트랜잭션 풀 모니터링 시작 ===")
        logger.info("기능: mempool analysis, gas price tracking, MEV detection")
        
        # 병렬 모니터링 태스크들
        monitoring_tasks = [
            asyncio.create_task(self._monitor_pending_transactions()),
            asyncio.create_task(self._monitor_gas_prices()),
            asyncio.create_task(self._detect_state_changes()),
            asyncio.create_task(self._analyze_mev_opportunities()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        try:
            await asyncio.gather(*monitoring_tasks)
        except Exception as e:
            logger.error(f"모니터링 시스템 오류: {e}")
        finally:
            for task in monitoring_tasks:
                task.cancel()
    
    async def _monitor_pending_transactions(self):
        """펜딩 트랜잭션 모니터링"""
        logger.info("펜딩 트랜잭션 모니터링 시작")
        
        while self.running:
            try:
                start_time = time.time()
                
                # 현재 mempool에서 pending 트랜잭션들 조회
                pending_txs = await self._get_pending_transactions()
                
                # 새로운 트랜잭션들 분석
                new_txs = []
                for tx_hash in pending_txs:
                    if tx_hash not in self.mempool_transactions:
                        tx_data = await self._get_transaction_details(tx_hash)
                        if tx_data:
                            self.mempool_transactions[tx_hash] = tx_data
                            new_txs.append((tx_hash, tx_data))
                
                # 새 트랜잭션 분석
                if new_txs:
                    await self._analyze_new_transactions(new_txs)
                
                # 성능 메트릭 업데이트
                processing_time = time.time() - start_time
                self._update_processing_metrics(processing_time, len(new_txs))
                
                # 잠시 대기 (너무 자주 폴링하지 않도록)
                await asyncio.sleep(1.0)  # 1초 간격
                
            except Exception as e:
                logger.error(f"펜딩 트랜잭션 모니터링 오류: {e}")
                await asyncio.sleep(5.0)
    
    async def _get_pending_transactions(self) -> List[str]:
        """현재 mempool의 pending 트랜잭션 조회"""
        try:
            # web3.py를 통한 pending 트랜잭션 조회
            loop = asyncio.get_event_loop()
            
            # eth_pendingTransactions 또는 txpool_content 사용
            # 실제 구현에서는 노드 제공업체에 따라 다름
            pending_block = await loop.run_in_executor(
                None, 
                self.w3.eth.get_block, 
                'pending', 
                True
            )
            
            if pending_block and pending_block.transactions:
                return [tx.hash.hex() for tx in pending_block.transactions[:100]]  # 최대 100개
            
            return []
            
        except Exception as e:
            logger.debug(f"펜딩 트랜잭션 조회 실패: {e}")
            return []
    
    async def _get_transaction_details(self, tx_hash: str) -> Optional[Dict]:
        """트랜잭션 상세 정보 조회"""
        try:
            loop = asyncio.get_event_loop()
            tx = await loop.run_in_executor(
                None, 
                self.w3.eth.get_transaction, 
                tx_hash
            )
            
            if tx:
                return {
                    'hash': tx_hash,
                    'from': tx['from'],
                    'to': tx['to'],
                    'value': tx['value'],
                    'gas': tx['gas'],
                    'gasPrice': tx['gasPrice'],
                    'data': tx['input'],
                    'nonce': tx['nonce'],
                    'timestamp': datetime.now(),
                    'processed': False
                }
            
        except Exception as e:
            logger.debug(f"트랜잭션 상세 정보 조회 실패 {tx_hash}: {e}")
        
        return None
    
    async def _analyze_new_transactions(self, new_txs: List[tuple]):
        """새로운 트랜잭션들 분석"""
        for tx_hash, tx_data in new_txs:
            # 차익거래 패턴 탐지
            if await self._is_potential_arbitrage_tx(tx_data):
                self.pending_arbitrage_txs.add(tx_hash)
                self.metrics['arbitrage_txs_detected'] += 1
                
                logger.info(f"차익거래 트랜잭션 감지: {tx_hash[:10]}...")
                
                # 리스너들에게 알림
                await self._notify_arbitrage_detected(tx_hash, tx_data)
            
            # MEV 기회 분석
            mev_score = await self._calculate_mev_score(tx_data)
            if mev_score > 0.7:  # 70% 이상 확률
                self.potential_mev_txs[tx_hash] = {
                    'score': mev_score,
                    'tx_data': tx_data,
                    'detected_at': datetime.now()
                }
                self.metrics['mev_opportunities_found'] += 1
                
                logger.info(f"MEV 기회 감지 (점수: {mev_score:.2f}): {tx_hash[:10]}...")
            
            # Gas price 통계 업데이트
            if tx_data.get('gasPrice'):
                self.gas_price_history.append(tx_data['gasPrice'])
                await self._update_gas_price_stats()
        
        self.metrics['total_txs_monitored'] += len(new_txs)
    
    async def _is_potential_arbitrage_tx(self, tx_data: Dict) -> bool:
        """차익거래 트랜잭션인지 판단"""
        try:
            data = tx_data.get('data', '0x')
            
            # 함수 시그니처 확인 (첫 4바이트)
            if len(data) >= 10:
                function_sig = data[:10]
                
                # 알려진 차익거래 함수 시그니처들
                arbitrage_signatures = {
                    '0x38ed1739': 'swapExactTokensForTokens',      # Uniswap V2
                    '0x8803dbee': 'swapTokensForExactTokens',      # Uniswap V2
                    '0x7ff36ab5': 'swapExactETHForTokens',         # Uniswap V2
                    '0x02751cec': 'removeLiquidity',               # Uniswap V2
                    '0xab834bab': 'atomicArbitrage',               # Custom arbitrage
                    '0x1e83409a': 'call',                         # Multicall
                    '0xac9650d8': 'multicall'                     # Multicall V2
                }
                
                if function_sig in arbitrage_signatures:
                    return True
            
            # Gas price 기반 판단 (MEV 봇들은 높은 gas price 사용)
            gas_price = tx_data.get('gasPrice', 0)
            if gas_price > self.gas_price_stats.get('current_median', 0) * 2:
                return True
            
            # 값이 큰 트랜잭션들 (potential profit)
            value = tx_data.get('value', 0)
            if value > Web3.to_wei(10, 'ether'):  # 10 ETH 이상
                return True
                
        except Exception as e:
            logger.debug(f"차익거래 패턴 분석 오류: {e}")
        
        return False
    
    async def _calculate_mev_score(self, tx_data: Dict) -> float:
        """MEV 점수 계산 (0.0 - 1.0)"""
        try:
            score = 0.0
            
            # Gas price factor (높은 gas price = 높은 MEV 가능성)
            gas_price = tx_data.get('gasPrice', 0)
            current_median = self.gas_price_stats.get('current_median', gas_price)
            if current_median > 0:
                gas_ratio = gas_price / current_median
                score += min(gas_ratio / 10, 0.4)  # 최대 40% 기여
            
            # Transaction value factor
            value = tx_data.get('value', 0)
            if value > 0:
                eth_value = Web3.from_wei(value, 'ether')
                score += min(float(eth_value) / 100, 0.3)  # 최대 30% 기여
            
            # Data complexity factor (복잡한 데이터 = MEV 가능성)
            data = tx_data.get('data', '0x')
            data_complexity = len(data) / 1000  # 복잡성 점수
            score += min(data_complexity, 0.2)  # 최대 20% 기여
            
            # Timing factor (빠른 nonce = front-running 가능성)
            # 이 부분은 더 정교한 구현 필요
            score += 0.1  # 기본 점수
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug(f"MEV 점수 계산 오류: {e}")
            return 0.0
    
    async def _monitor_gas_prices(self):
        """Gas price 모니터링"""
        logger.info("Gas price 모니터링 시작")
        
        while self.running:
            try:
                # 현재 gas price 조회
                loop = asyncio.get_event_loop()
                current_gas_price = await loop.run_in_executor(
                    None, 
                    self.w3.eth.gas_price
                )
                
                # 히스토리에 추가
                self.gas_price_history.append(current_gas_price)
                
                # 통계 업데이트
                await self._update_gas_price_stats()
                
                # 10초마다 업데이트
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Gas price 모니터링 오류: {e}")
                await asyncio.sleep(30)
    
    async def _update_gas_price_stats(self):
        """Gas price 통계 업데이트"""
        try:
            if not self.gas_price_history:
                return
            
            prices = list(self.gas_price_history)
            
            # 기본 통계
            self.gas_price_stats.update({
                'current_median': statistics.median(prices),
                'current_average': statistics.mean(prices),
                'current_max': max(prices),
                'current_min': min(prices),
                'last_update': datetime.now()
            })
            
            # 트렌드 분석 (최근 10개 vs 이전 10개)
            if len(prices) >= 20:
                recent_avg = statistics.mean(prices[-10:])
                previous_avg = statistics.mean(prices[-20:-10])
                
                if recent_avg > previous_avg * 1.1:
                    self.gas_price_stats['trend'] = 'rising'
                elif recent_avg < previous_avg * 0.9:
                    self.gas_price_stats['trend'] = 'falling'
                else:
                    self.gas_price_stats['trend'] = 'stable'
            
            # 주기적으로 통계 로깅 (5분마다)
            if datetime.now().minute % 5 == 0:
                self._log_gas_price_stats()
                
        except Exception as e:
            logger.error(f"Gas price 통계 업데이트 오류: {e}")
    
    def _log_gas_price_stats(self):
        """Gas price 통계 로깅"""
        stats = self.gas_price_stats
        logger.info("=== Gas Price 통계 ===")
        logger.info(f"현재 중간값: {Web3.from_wei(stats['current_median'], 'gwei'):.1f} Gwei")
        logger.info(f"현재 평균값: {Web3.from_wei(stats['current_average'], 'gwei'):.1f} Gwei")
        logger.info(f"최고/최저: {Web3.from_wei(stats['current_max'], 'gwei'):.1f} / {Web3.from_wei(stats['current_min'], 'gwei'):.1f} Gwei")
        logger.info(f"트렌드: {stats['trend']}")
    
    async def _detect_state_changes(self):
        """블록체인 상태 변화 감지"""
        logger.info("상태 변화 감지 시작")
        
        while self.running:
            try:
                # 현재 블록 번호 조회
                loop = asyncio.get_event_loop()
                current_block = await loop.run_in_executor(
                    None, 
                    self.w3.eth.block_number
                )
                
                # 새 블록 감지
                if current_block > self.last_block_number:
                    block_diff = current_block - self.last_block_number
                    logger.info(f"새 블록 감지: {current_block} (+{block_diff})")
                    
                    # 상태 변화 알림
                    await self._notify_state_change({
                        'type': 'new_block',
                        'block_number': current_block,
                        'previous_block': self.last_block_number,
                        'timestamp': datetime.now()
                    })
                    
                    self.last_block_number = current_block
                    self.state_changes_count += block_diff
                    self.metrics['state_changes_detected'] += block_diff
                
                # 5초마다 체크
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"상태 변화 감지 오류: {e}")
                await asyncio.sleep(10)
    
    async def _analyze_mev_opportunities(self):
        """MEV 기회 분석"""
        while self.running:
            try:
                current_time = datetime.now()
                
                # 오래된 MEV 기회들 정리
                expired_txs = []
                for tx_hash, mev_data in self.potential_mev_txs.items():
                    if current_time - mev_data['detected_at'] > timedelta(minutes=5):
                        expired_txs.append(tx_hash)
                
                for tx_hash in expired_txs:
                    del self.potential_mev_txs[tx_hash]
                
                # 현재 MEV 기회 분석
                if self.potential_mev_txs:
                    await self._analyze_current_mev_opportunities()
                
                # 30초마다 분석
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"MEV 기회 분석 오류: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_current_mev_opportunities(self):
        """현재 MEV 기회들 분석"""
        high_value_opportunities = []
        
        for tx_hash, mev_data in self.potential_mev_txs.items():
            if mev_data['score'] > 0.8:  # 높은 점수의 기회들
                high_value_opportunities.append((tx_hash, mev_data))
        
        if high_value_opportunities:
            logger.info(f"{len(high_value_opportunities)}개의 고가치 MEV 기회 감지")
            
            for tx_hash, mev_data in high_value_opportunities:
                # 상세 분석 및 알림
                await self._notify_mev_opportunity(tx_hash, mev_data)
    
    async def _cleanup_old_data(self):
        """오래된 데이터 정리"""
        while self.running:
            try:
                current_time = datetime.now()
                cleanup_threshold = current_time - timedelta(hours=1)
                
                # 오래된 mempool 트랜잭션 정리
                old_txs = []
                for tx_hash, tx_data in self.mempool_transactions.items():
                    if tx_data['timestamp'] < cleanup_threshold:
                        old_txs.append(tx_hash)
                
                for tx_hash in old_txs:
                    del self.mempool_transactions[tx_hash]
                    self.pending_arbitrage_txs.discard(tx_hash)
                
                if old_txs:
                    logger.debug(f"{len(old_txs)}개의 오래된 트랜잭션 정리")
                
                # 10분마다 정리
                await asyncio.sleep(600)
                
            except Exception as e:
                logger.error(f"데이터 정리 오류: {e}")
                await asyncio.sleep(600)
    
    def register_state_change_listener(self, callback: Callable):
        """상태 변화 리스너 등록"""
        self.state_change_listeners.append(callback)
        logger.debug("상태 변화 리스너 등록됨")
    
    def remove_state_change_listener(self, callback: Callable):
        """상태 변화 리스너 제거"""
        if callback in self.state_change_listeners:
            self.state_change_listeners.remove(callback)
            logger.debug("상태 변화 리스너 제거됨")
    
    async def _notify_state_change(self, change_data: Dict):
        """상태 변화 리스너들에게 알림"""
        for listener in self.state_change_listeners:
            try:
                await listener(change_data)
            except Exception as e:
                logger.error(f"상태 변화 리스너 알림 실패: {e}")
    
    async def _notify_arbitrage_detected(self, tx_hash: str, tx_data: Dict):
        """차익거래 트랜잭션 감지 알림"""
        logger.info(f"차익거래 감지 알림: {tx_hash}")
        # 여기서 다른 시스템 컴포넌트들에게 알림 가능
    
    async def _notify_mev_opportunity(self, tx_hash: str, mev_data: Dict):
        """MEV 기회 알림"""
        logger.info(f"MEV 기회 알림: {tx_hash} (점수: {mev_data['score']:.2f})")
        # 여기서 차익거래 탐지 시스템에 알림 가능
    
    def _update_processing_metrics(self, processing_time: float, tx_count: int):
        """처리 성능 메트릭 업데이트"""
        # 평균 처리 시간 계산
        if self.metrics['average_processing_time'] == 0:
            self.metrics['average_processing_time'] = processing_time
        else:
            # 지수 이동 평균
            alpha = 0.1
            self.metrics['average_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.metrics['average_processing_time']
            )
    
    def get_metrics(self) -> Dict:
        """현재 메트릭스 반환"""
        return {
            **self.metrics,
            'mempool_size': len(self.mempool_transactions),
            'pending_arbitrage_count': len(self.pending_arbitrage_txs),
            'potential_mev_count': len(self.potential_mev_txs),
            'gas_price_stats': self.gas_price_stats.copy()
        }
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.running = False
        logger.info("트랜잭션 풀 모니터링 중지")

# 사용 예시
async def main():
    monitor = TransactionPoolMonitor()
    
    # 상태 변화 리스너 등록
    async def on_state_change(change_data):
        print(f"상태 변화: {change_data['type']} - 블록 {change_data['block_number']}")
    
    monitor.register_state_change_listener(on_state_change)
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        logger.info("프로그램 종료")

if __name__ == "__main__":
    asyncio.run(main())