"""
Historical Data Backfilling Module
TODO requirement completion: Historical data backfilling for analysis

이 모듈은 논문 [2103.02228]의 DeFiPoser-ARB 시스템 재현을 위한
과거 데이터 백필링 시스템을 구현합니다.

Features:
- Historical price data collection from multiple sources
- Block-by-block historical data reconstruction
- Data gap filling and interpolation
- Historical arbitrage opportunity analysis
- Performance analysis over 150 days (paper requirement)
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import sqlite3
from web3 import Web3

from src.logger import setup_logger
from src.token_manager import TokenManager
from src.real_time_price_feeds import PriceFeed
from config.config import config

logger = setup_logger(__name__)

@dataclass
class HistoricalPricePoint:
    """과거 가격 데이터 포인트"""
    token_address: str
    symbol: str
    price_usd: float
    timestamp: int
    block_number: int
    source: str
    volume_24h: float = 0.0
    market_cap: float = 0.0

@dataclass
class BlockData:
    """블록 데이터"""
    number: int
    timestamp: int
    hash: str
    prices: Dict[str, HistoricalPricePoint]

class HistoricalDataBackfill:
    """
    과거 데이터 백필링 관리자
    
    논문의 150일간 분석을 위한 과거 데이터 수집 및 관리
    블록 9,100,000 ~ 10,050,000 구간 (150일) 데이터 재현
    """
    
    def __init__(self, token_manager: Optional[TokenManager] = None):
        self.token_manager = token_manager or TokenManager()
        self.db_path = "historical_data.db"
        
        # 논문의 분석 구간 (150일)
        self.start_block = 9_100_000
        self.end_block = 10_050_000
        self.total_blocks = self.end_block - self.start_block
        
        # 데이터 소스들
        self.data_sources = {
            'coingecko': 'https://api.coingecko.com/api/v3',
            'cryptocompare': 'https://min-api.cryptocompare.com/data',
            'coinmarketcap': 'https://pro-api.coinmarketcap.com/v1'
        }
        
        # Web3 연결
        if hasattr(config, 'ethereum_mainnet_rpc') and config.ethereum_mainnet_rpc:
            self.w3 = Web3(Web3.HTTPProvider(config.ethereum_mainnet_rpc))
        else:
            self.w3 = None
            logger.warning("Web3 연결이 설정되지 않았습니다")
            
        # DB 초기화
        self._init_database()
        
    def _init_database(self):
        """SQLite 데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 가격 데이터 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price_usd REAL NOT NULL,
                    timestamp INTEGER NOT NULL,
                    block_number INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    volume_24h REAL DEFAULT 0,
                    market_cap REAL DEFAULT 0,
                    UNIQUE(token_address, block_number, source)
                )
            ''')
            
            # 블록 데이터 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS blocks (
                    number INTEGER PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    hash TEXT NOT NULL,
                    processed BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # 인덱스 생성
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_prices_block ON historical_prices(block_number)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_prices_token ON historical_prices(token_address)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_prices_timestamp ON historical_prices(timestamp)')
            
            conn.commit()
            conn.close()
            
            logger.info("과거 데이터 데이터베이스 초기화 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
            
    async def start_backfill(self, start_block: Optional[int] = None, end_block: Optional[int] = None):
        """과거 데이터 백필링 시작"""
        start_block = start_block or self.start_block
        end_block = end_block or self.end_block
        
        logger.info(f"과거 데이터 백필링 시작: 블록 {start_block:,} ~ {end_block:,}")
        
        # 1. 블록 정보 수집
        await self._collect_block_info(start_block, end_block)
        
        # 2. 가격 데이터 수집
        await self._collect_historical_prices(start_block, end_block)
        
        # 3. 데이터 검증 및 보간
        await self._validate_and_interpolate(start_block, end_block)
        
        # 4. 분석 리포트 생성
        await self._generate_analysis_report(start_block, end_block)
        
        logger.info("과거 데이터 백필링 완료")
        
    async def _collect_block_info(self, start_block: int, end_block: int):
        """블록 정보 수집"""
        if not self.w3 or not self.w3.is_connected():
            logger.warning("Web3 연결이 없어 블록 정보 수집을 건너뜁니다")
            return
            
        logger.info(f"블록 정보 수집: {start_block:,} ~ {end_block:,}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 이미 수집된 블록 확인
        cursor.execute('SELECT number FROM blocks WHERE number BETWEEN ? AND ?', 
                      (start_block, end_block))
        existing_blocks = {row[0] for row in cursor.fetchall()}
        
        # 수집할 블록들
        blocks_to_collect = set(range(start_block, end_block + 1)) - existing_blocks
        total_blocks = len(blocks_to_collect)
        
        if not blocks_to_collect:
            logger.info("모든 블록 정보가 이미 수집되어 있습니다")
            conn.close()
            return
            
        logger.info(f"{total_blocks:,}개 블록 정보를 수집합니다")
        
        batch_size = 100
        processed = 0
        
        for i in range(0, len(blocks_to_collect), batch_size):
            batch = list(blocks_to_collect)[i:i + batch_size]
            
            try:
                # 병렬로 블록 정보 수집
                tasks = [self._get_block_info(block_num) for block_num in batch]
                block_infos = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 데이터베이스에 저장
                block_data = []
                for block_num, block_info in zip(batch, block_infos):
                    if isinstance(block_info, Exception):
                        logger.error(f"블록 {block_num} 정보 수집 실패: {block_info}")
                        continue
                        
                    if block_info:
                        block_data.append((
                            block_info['number'],
                            block_info['timestamp'], 
                            block_info['hash'],
                            False
                        ))
                        
                if block_data:
                    cursor.executemany(
                        'INSERT OR REPLACE INTO blocks (number, timestamp, hash, processed) VALUES (?, ?, ?, ?)',
                        block_data
                    )
                    conn.commit()
                    
                processed += len(batch)
                if processed % 1000 == 0:
                    progress = (processed / total_blocks) * 100
                    logger.info(f"블록 정보 수집 진행률: {processed:,}/{total_blocks:,} ({progress:.1f}%)")
                    
            except Exception as e:
                logger.error(f"블록 배치 처리 실패: {e}")
                
        conn.close()
        logger.info(f"블록 정보 수집 완료: {processed:,}개 블록")
        
    async def _get_block_info(self, block_number: int) -> Optional[Dict]:
        """개별 블록 정보 조회"""
        try:
            block = self.w3.eth.get_block(block_number, full_transactions=False)
            return {
                'number': block.number,
                'timestamp': block.timestamp,
                'hash': block.hash.hex()
            }
        except Exception as e:
            logger.debug(f"블록 {block_number} 정보 조회 실패: {e}")
            return None
            
    async def _collect_historical_prices(self, start_block: int, end_block: int):
        """과거 가격 데이터 수집"""
        logger.info("과거 가격 데이터 수집 시작")
        
        # 블록 타임스탬프 조회
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT number, timestamp FROM blocks 
            WHERE number BETWEEN ? AND ? 
            ORDER BY number
        ''', (start_block, end_block))
        
        block_timestamps = dict(cursor.fetchall())
        conn.close()
        
        if not block_timestamps:
            logger.warning("블록 타임스탬프 정보가 없습니다. 먼저 블록 정보를 수집하세요.")
            return
            
        logger.info(f"{len(block_timestamps):,}개 블록에 대한 가격 데이터 수집")
        
        # 날짜별로 그룹화 (API 효율성을 위해)
        daily_blocks = {}
        for block_num, timestamp in block_timestamps.items():
            date_key = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
            if date_key not in daily_blocks:
                daily_blocks[date_key] = []
            daily_blocks[date_key].append((block_num, timestamp))
            
        logger.info(f"{len(daily_blocks)}일치 데이터를 수집합니다")
        
        # 일별로 가격 데이터 수집
        for date_str, blocks in daily_blocks.items():
            try:
                await self._collect_daily_prices(date_str, blocks)
                await asyncio.sleep(1)  # API rate limiting
            except Exception as e:
                logger.error(f"{date_str} 가격 데이터 수집 실패: {e}")
                
    async def _collect_daily_prices(self, date_str: str, blocks: List[Tuple[int, int]]):
        """일별 가격 데이터 수집"""
        try:
            # CoinGecko에서 일별 가격 데이터 수집
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # 주요 토큰들의 CoinGecko ID 수집
            token_ids = []
            token_map = {}  # coingecko_id -> token_address
            
            for address, token in self.token_manager.tokens.items():
                if token.coingecko_id:
                    token_ids.append(token.coingecko_id)
                    token_map[token.coingecko_id] = address
                    
            if not token_ids:
                return
                
            # CoinGecko API 호출
            url = f"{self.data_sources['coingecko']}/coins/{token_ids[0]}/history"
            params = {
                'date': date_obj.strftime('%d-%m-%Y'),
                'localization': 'false'
            }
            
            price_data = {}
            
            # 각 토큰별로 가격 수집 (배치 API가 없어서 개별 호출)
            for i, token_id in enumerate(token_ids[:20]):  # API 제한으로 20개만
                try:
                    url = f"{self.data_sources['coingecko']}/coins/{token_id}/history"
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                current_price = data.get('market_data', {}).get('current_price', {}).get('usd')
                                market_cap = data.get('market_data', {}).get('market_cap', {}).get('usd', 0)
                                total_volume = data.get('market_data', {}).get('total_volume', {}).get('usd', 0)
                                
                                if current_price and current_price > 0:
                                    token_address = token_map.get(token_id)
                                    if token_address:
                                        price_data[token_address] = {
                                            'price': current_price,
                                            'market_cap': market_cap,
                                            'volume': total_volume
                                        }
                                        
                            elif response.status == 429:
                                logger.warning("CoinGecko rate limit reached")
                                break
                                
                    # Rate limiting
                    if i % 10 == 0 and i > 0:
                        await asyncio.sleep(2)
                        
                except Exception as e:
                    logger.debug(f"토큰 {token_id} 가격 수집 실패: {e}")
                    
            # 수집된 가격 데이터를 각 블록에 적용
            if price_data:
                await self._save_historical_prices(blocks, price_data, 'coingecko')
                logger.info(f"{date_str}: {len(price_data)}개 토큰 가격 데이터 저장")
                
        except Exception as e:
            logger.error(f"일별 가격 수집 실패 {date_str}: {e}")
            
    async def _save_historical_prices(self, blocks: List[Tuple[int, int]], 
                                    price_data: Dict, source: str):
        """과거 가격 데이터 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            price_records = []
            
            for block_num, timestamp in blocks:
                for token_address, data in price_data.items():
                    token_info = self.token_manager.tokens.get(token_address)
                    if not token_info:
                        continue
                        
                    price_records.append((
                        token_address,
                        token_info.symbol,
                        data['price'],
                        timestamp,
                        block_num,
                        source,
                        data.get('volume', 0),
                        data.get('market_cap', 0)
                    ))
                    
            if price_records:
                cursor.executemany('''
                    INSERT OR REPLACE INTO historical_prices 
                    (token_address, symbol, price_usd, timestamp, block_number, source, volume_24h, market_cap)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', price_records)
                
                conn.commit()
                
            conn.close()
            
        except Exception as e:
            logger.error(f"과거 가격 데이터 저장 실패: {e}")
            
    async def _validate_and_interpolate(self, start_block: int, end_block: int):
        """데이터 검증 및 보간"""
        logger.info("데이터 검증 및 보간 시작")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 토큰별 데이터 커버리지 분석
        cursor.execute('''
            SELECT token_address, symbol, COUNT(*) as price_count,
                   MIN(block_number) as first_block, MAX(block_number) as last_block
            FROM historical_prices 
            WHERE block_number BETWEEN ? AND ?
            GROUP BY token_address
            ORDER BY price_count DESC
        ''', (start_block, end_block))
        
        coverage_stats = cursor.fetchall()
        
        logger.info("토큰별 데이터 커버리지:")
        for token_addr, symbol, count, first, last in coverage_stats[:10]:
            coverage_pct = (count / self.total_blocks) * 100
            logger.info(f"  {symbol}: {count:,} 포인트 ({coverage_pct:.1f}% 커버리지)")
            
        # 데이터 갭 식별 및 보간
        gap_filled_count = 0
        
        for token_addr, symbol, count, first, last in coverage_stats:
            if count < self.total_blocks * 0.1:  # 10% 미만 커버리지는 건너뜀
                continue
                
            # 데이터 갭 찾기
            cursor.execute('''
                SELECT block_number, price_usd 
                FROM historical_prices 
                WHERE token_address = ? AND block_number BETWEEN ? AND ?
                ORDER BY block_number
            ''', (token_addr, start_block, end_block))
            
            price_points = cursor.fetchall()
            if len(price_points) < 2:
                continue
                
            # 간단한 선형 보간으로 갭 채우기
            interpolated_points = []
            
            for i in range(len(price_points) - 1):
                current_block, current_price = price_points[i]
                next_block, next_price = price_points[i + 1]
                
                block_gap = next_block - current_block
                if block_gap > 100:  # 100 블록 이상 갭이 있으면 보간
                    price_diff = next_price - current_price
                    
                    for j in range(1, min(block_gap, 1000)):  # 최대 1000블록까지만
                        interp_block = current_block + j
                        interp_price = current_price + (price_diff * j / block_gap)
                        
                        # 타임스탬프 조회
                        cursor.execute('SELECT timestamp FROM blocks WHERE number = ?', (interp_block,))
                        timestamp_row = cursor.fetchone()
                        if timestamp_row:
                            interpolated_points.append((
                                token_addr, symbol, interp_price, timestamp_row[0], 
                                interp_block, 'interpolated', 0, 0
                            ))
                            
            # 보간된 포인트들 저장
            if interpolated_points:
                cursor.executemany('''
                    INSERT OR IGNORE INTO historical_prices 
                    (token_address, symbol, price_usd, timestamp, block_number, source, volume_24h, market_cap)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', interpolated_points)
                
                gap_filled_count += len(interpolated_points)
                
        conn.commit()
        conn.close()
        
        logger.info(f"데이터 보간 완료: {gap_filled_count:,}개 포인트 추가")
        
    async def _generate_analysis_report(self, start_block: int, end_block: int):
        """분석 리포트 생성"""
        logger.info("분석 리포트 생성")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 전체 통계
        cursor.execute('''
            SELECT COUNT(*) as total_records,
                   COUNT(DISTINCT token_address) as unique_tokens,
                   COUNT(DISTINCT block_number) as covered_blocks,
                   MIN(timestamp) as start_time,
                   MAX(timestamp) as end_time
            FROM historical_prices
            WHERE block_number BETWEEN ? AND ?
        ''', (start_block, end_block))
        
        stats = cursor.fetchone()
        total_records, unique_tokens, covered_blocks, start_time, end_time = stats
        
        # 기간 계산
        duration_days = (end_time - start_time) / (24 * 3600)
        coverage_pct = (covered_blocks / self.total_blocks) * 100
        
        logger.info("=" * 60)
        logger.info("📊 과거 데이터 백필링 분석 리포트")
        logger.info("=" * 60)
        logger.info(f"분석 기간: {duration_days:.1f}일")
        logger.info(f"블록 범위: {start_block:,} ~ {end_block:,}")
        logger.info(f"총 가격 레코드: {total_records:,}개")
        logger.info(f"커버된 토큰 수: {unique_tokens}개")
        logger.info(f"커버된 블록 수: {covered_blocks:,}개 ({coverage_pct:.1f}%)")
        
        # 토큰별 상위 통계
        cursor.execute('''
            SELECT symbol, COUNT(*) as records, 
                   MIN(price_usd) as min_price, 
                   MAX(price_usd) as max_price,
                   AVG(price_usd) as avg_price
            FROM historical_prices 
            WHERE block_number BETWEEN ? AND ?
            GROUP BY token_address, symbol
            ORDER BY records DESC
            LIMIT 10
        ''', (start_block, end_block))
        
        top_tokens = cursor.fetchall()
        
        logger.info("\n상위 토큰 통계:")
        for symbol, records, min_price, max_price, avg_price in top_tokens:
            price_range = ((max_price - min_price) / min_price) * 100 if min_price > 0 else 0
            logger.info(f"  {symbol}: {records:,} 레코드, "
                       f"가격 범위: ${min_price:.6f} ~ ${max_price:.6f} "
                       f"({price_range:.1f}% 변동)")
        
        # 일별 데이터 커버리지
        cursor.execute('''
            SELECT DATE(timestamp, 'unixepoch') as date, 
                   COUNT(DISTINCT token_address) as tokens_count,
                   COUNT(*) as total_records
            FROM historical_prices 
            WHERE block_number BETWEEN ? AND ?
            GROUP BY DATE(timestamp, 'unixepoch')
            ORDER BY date
            LIMIT 10
        ''', (start_block, end_block))
        
        daily_coverage = cursor.fetchall()
        
        if daily_coverage:
            logger.info("\n일별 데이터 커버리지 (첫 10일):")
            for date, tokens, records in daily_coverage:
                logger.info(f"  {date}: {tokens}개 토큰, {records:,}개 레코드")
                
        conn.close()
        
        # 성공 지표 확인
        paper_target_days = 150
        paper_target_tokens = 25
        
        success_criteria = {
            'duration': duration_days >= paper_target_days * 0.8,  # 80% of target duration
            'tokens': unique_tokens >= paper_target_tokens * 0.8,   # 80% of target tokens  
            'coverage': coverage_pct >= 50.0  # 50% block coverage
        }
        
        all_success = all(success_criteria.values())
        
        logger.info("\n📈 논문 재현 목표 달성도:")
        logger.info(f"  기간 목표 (150일): {'✅' if success_criteria['duration'] else '❌'} "
                   f"{duration_days:.1f}일")
        logger.info(f"  토큰 목표 (25개): {'✅' if success_criteria['tokens'] else '❌'} "
                   f"{unique_tokens}개")
        logger.info(f"  커버리지 목표: {'✅' if success_criteria['coverage'] else '❌'} "
                   f"{coverage_pct:.1f}%")
        
        if all_success:
            logger.info("🎉 과거 데이터 백필링 목표 달성!")
        else:
            logger.warning("⚠️  일부 목표가 미달성되었습니다")
            
        return all_success
        
    async def get_historical_prices(self, token_address: str, start_block: int, 
                                  end_block: int) -> List[HistoricalPricePoint]:
        """특정 토큰의 과거 가격 데이터 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT token_address, symbol, price_usd, timestamp, block_number, 
                       source, volume_24h, market_cap
                FROM historical_prices
                WHERE token_address = ? AND block_number BETWEEN ? AND ?
                ORDER BY block_number
            ''', (token_address.lower(), start_block, end_block))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                HistoricalPricePoint(
                    token_address=row[0],
                    symbol=row[1],
                    price_usd=row[2],
                    timestamp=row[3],
                    block_number=row[4],
                    source=row[5],
                    volume_24h=row[6],
                    market_cap=row[7]
                )
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"과거 가격 조회 실패: {e}")
            return []
            
    async def get_block_prices(self, block_number: int) -> Dict[str, HistoricalPricePoint]:
        """특정 블록의 모든 토큰 가격 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT token_address, symbol, price_usd, timestamp, block_number,
                       source, volume_24h, market_cap
                FROM historical_prices
                WHERE block_number = ?
            ''', (block_number,))
            
            rows = cursor.fetchall()
            conn.close()
            
            prices = {}
            for row in rows:
                price_point = HistoricalPricePoint(
                    token_address=row[0],
                    symbol=row[1],
                    price_usd=row[2],
                    timestamp=row[3],
                    block_number=row[4],
                    source=row[5],
                    volume_24h=row[6],
                    market_cap=row[7]
                )
                prices[row[0]] = price_point
                
            return prices
            
        except Exception as e:
            logger.error(f"블록 가격 조회 실패: {e}")
            return {}

# 사용 예시
async def main():
    # 토큰 매니저 초기화
    token_manager = TokenManager()
    
    # 과거 데이터 백필러 초기화
    backfill = HistoricalDataBackfill(token_manager)
    
    # 샘플 기간으로 백필링 테스트 (1000 블록)
    test_start = 9_100_000
    test_end = 9_101_000
    
    logger.info(f"테스트 백필링 시작: 블록 {test_start} ~ {test_end}")
    await backfill.start_backfill(test_start, test_end)
    
    # 결과 확인
    eth_address = token_manager.get_address_by_symbol('ETH')
    if eth_address:
        prices = await backfill.get_historical_prices(eth_address, test_start, test_end)
        logger.info(f"ETH 가격 데이터 {len(prices)}개 수집됨")

if __name__ == "__main__":
    asyncio.run(main())