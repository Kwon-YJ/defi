import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from src.logger import setup_logger
from src.token_manager import TokenManager
from src.data_storage import DataStorage

logger = setup_logger(__name__)

@dataclass
class PriceFeed:
    """가격 피드 데이터 클래스"""
    token_address: str
    symbol: str
    price_usd: float
    source: str
    timestamp: float

class RealTimePriceFeeds:
    """실시간 가격 피드 관리자"""
    
    def __init__(self):
        self.token_manager = TokenManager()
        self.data_storage = DataStorage()
        self.price_feeds: Dict[str, PriceFeed] = {}
        self.subscribers: List[Callable] = []
        self.running = False
        self.session = None
        
        # CoinGecko API endpoint
        self.coingecko_api_base = "https://api.coingecko.com/api/v3"
        
        # 토큰 심볼과 CoinGecko ID 매핑
        self.token_coingecko_ids = self._build_coingecko_mapping()
        
    def _build_coingecko_mapping(self) -> Dict[str, str]:
        """토큰 심볼과 CoinGecko ID 매핑 생성"""
        # 실제 CoinGecko ID 매핑 (주요 토큰들)
        mapping = {
            'ETH': 'ethereum',
            'WETH': 'weth',
            'USDC': 'usd-coin',
            'USDT': 'tether',
            'DAI': 'dai',
            'WBTC': 'wrapped-bitcoin',
            'UNI': 'uniswap',
            'SUSHI': 'sushi',
            'COMP': 'compound-governance-token',
            'AAVE': 'aave',
            'CRV': 'curve-dao-token',
            'BAL': 'balancer',
            'YFI': 'yearn-finance',
            'MKR': 'maker',
            'BNT': 'bancor',
            'BAT': 'basic-attention-token',
            'KNC': 'kyber-network-crystal',
            'MANA': 'decentraland',
            'GNO': 'gnosis',
            'RLC': 'iexec-rlc',
            'UBT': 'unibright',
            'SAI': 'sai',  # MakerDAO legacy token
            'cETH': 'compound-ether',
            'cUSDC': 'compound-usd-coin',
            'aETH': 'aave-eth',
            'aUSDC': 'aave-usdc',
            'sUSD': 'nusd',
            'sETH': 'seth',
            'sBTC': 'sbtc',
            'sLINK': 'slink',
            'sDEFI': 'sdefi',
            'sXAU': 'sxau',
            'sXAG': 'sxag'
        }
        return mapping
    
    async def initialize(self):
        """초기화"""
        self.session = aiohttp.ClientSession()
        logger.info("실시간 가격 피드 초기화 완료")
    
    async def cleanup(self):
        """정리"""
        if self.session:
            await self.session.close()
        logger.info("실시간 가격 피드 정리 완료")
    
    async def start_price_feed(self):
        """가격 피드 시작"""
        self.running = True
        logger.info("실시간 가격 피드 시작")
        
        # 초기 가격 데이터 가져오기
        await self._update_all_prices()
        
        # 주기적으로 가격 업데이트 (60초마다)
        while self.running:
            try:
                await asyncio.sleep(60)
                await self._update_all_prices()
            except Exception as e:
                logger.error(f"가격 업데이트 중 오류 발생: {e}")
    
    def stop_price_feed(self):
        """가격 피드 중지"""
        self.running = False
        logger.info("실시간 가격 피드 중지")
    
    async def _update_all_prices(self):
        """모든 토큰의 가격 업데이트"""
        try:
            # 지원되는 모든 토큰 가져오기
            all_tokens = self.token_manager.get_all_asset_symbols()
            
            # CoinGecko ID가 있는 토큰 필터링
            tokens_with_ids = {
                symbol: self.token_coingecko_ids[symbol] 
                for symbol in all_tokens 
                if symbol in self.token_coingecko_ids
            }
            
            if not tokens_with_ids:
                logger.warning("가격 피드를 위한 토큰 ID가 없습니다")
                return
            
            # CoinGecko API에서 가격 데이터 가져오기
            prices = await self._fetch_coingecko_prices(list(tokens_with_ids.values()))
            
            if not prices:
                logger.warning("CoinGecko에서 가격 데이터를 가져오지 못했습니다")
                return
            
            # 각 토큰의 가격 업데이트
            updated_count = 0
            for symbol, cg_id in tokens_with_ids.items():
                if cg_id in prices and 'usd' in prices[cg_id]:
                    price_usd = prices[cg_id]['usd']
                    token_address = self.token_manager.get_asset_address(symbol)
                    
                    price_feed = PriceFeed(
                        token_address=token_address,
                        symbol=symbol,
                        price_usd=price_usd,
                        source='coingecko',
                        timestamp=datetime.now().timestamp()
                    )
                    
                    self.price_feeds[symbol] = price_feed
                    await self._store_price_feed(price_feed)
                    updated_count += 1
                    
                    logger.debug(f"가격 업데이트: {symbol} = ${price_usd}")
            
            logger.info(f"총 {updated_count}개 토큰의 가격 업데이트 완료")
            
            # 구독자들에게 알림
            await self._notify_subscribers()
            
        except Exception as e:
            logger.error(f"모든 토큰 가격 업데이트 중 오류 발생: {e}")
    
    async def _fetch_coingecko_prices(self, coin_ids: List[str]) -> Optional[Dict]:
        """CoinGecko API에서 가격 데이터 가져오기"""
        try:
            if not self.session:
                await self.initialize()
            
            # 최대 250개의 코인 ID를 한 번에 요청 가능
            ids_param = ','.join(coin_ids)
            url = f"{self.coingecko_api_base}/simple/price?ids={ids_param}&vs_currencies=usd"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"CoinGecko API 요청 실패: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"CoinGecko API 요청 중 오류 발생: {e}")
            return None
    
    async def _store_price_feed(self, price_feed: PriceFeed):
        """가격 피드 데이터 저장"""
        try:
            # Redis에 현재 가격 저장
            key = f"price:{price_feed.symbol}"
            data = json.dumps(asdict(price_feed), default=str)
            self.data_storage.redis_client.setex(key, 300, data)  # 5분 TTL
            
            # 시계열 데이터도 저장
            timestamp = datetime.now().isoformat()
            ts_key = f"price_history:{price_feed.symbol}:{timestamp}"
            self.data_storage.redis_client.setex(ts_key, 3600, data)  # 1시간 보관
            
        except Exception as e:
            logger.error(f"가격 피드 데이터 저장 중 오류 발생: {e}")
    
    async def _notify_subscribers(self):
        """구독자들에게 가격 업데이트 알림"""
        for callback in self.subscribers:
            try:
                await callback(self.price_feeds)
            except Exception as e:
                logger.error(f"구독자 콜백 실행 중 오류 발생: {e}")
    
    def subscribe_to_price_updates(self, callback: Callable):
        """가격 업데이트 구독"""
        self.subscribers.append(callback)
        logger.debug("새로운 가격 업데이트 구독자 추가")
    
    def get_price_feed(self, symbol: str) -> Optional[PriceFeed]:
        """특정 토큰의 가격 피드 반환"""
        return self.price_feeds.get(symbol)
    
    def get_all_price_feeds(self) -> Dict[str, PriceFeed]:
        """모든 가격 피드 반환"""
        return self.price_feeds.copy()
    
    async def get_price_history(self, symbol: str, hours: int = 24) -> List[Dict]:
        """토큰의 가격 히스토리 조회"""
        try:
            # Redis에서 시계열 데이터 조회
            pattern = f"price_history:{symbol}:*"
            keys = self.data_storage.redis_client.keys(pattern)
            
            history = []
            for key in keys:
                # 키에서 타임스탬프 추출
                timestamp_str = key.decode().split(':')[-1]
                try:
                    data = self.data_storage.redis_client.get(key)
                    if data:
                        price_data = json.loads(data)
                        price_data['timestamp'] = timestamp_str
                        history.append(price_data)
                except ValueError:
                    continue
            
            # 시간순 정렬
            history.sort(key=lambda x: x['timestamp'])
            return history
            
        except Exception as e:
            logger.error(f"가격 히스토리 조회 중 오류 발생: {e}")
            return []

# 사용 예시
async def main():
    """메인 함수 예시"""
    price_feeds = RealTimePriceFeeds()
    
    # 콜백 함수 정의
    async def on_price_update(price_feeds_dict):
        print(f"가격 업데이트: {len(price_feeds_dict)}개 토큰")
        for symbol, feed in list(price_feeds_dict.items())[:3]:  # 처음 3개만 출력
            print(f"  {symbol}: ${feed.price_usd}")
    
    # 구독 설정
    price_feeds.subscribe_to_price_updates(on_price_update)
    
    # 시작
    await price_feeds.initialize()
    await price_feeds.start_price_feed()

if __name__ == "__main__":
    asyncio.run(main())