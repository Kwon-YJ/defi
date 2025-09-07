import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import numpy as np
from src.logger import setup_logger
from src.token_manager import TokenManager
from src.data_storage import DataStorage
from src.data_validator import DataValidator

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
        self.data_validator = DataValidator(self.data_storage)
        self.price_feeds: Dict[str, PriceFeed] = {}
        self.subscribers: List[Callable] = []
        self.running = False
        self.session = None
        
        # Multiple price feed sources for redundancy
        self.price_sources = [
            {
                'name': 'coingecko',
                'api_base': 'https://api.coingecko.com/api/v3',
                'enabled': True
            },
            {
                'name': 'coinpaprika',
                'api_base': 'https://api.coinpaprika.com/v1',
                'enabled': True
            },
            {
                'name': 'cryptocompare',
                'api_base': 'https://min-api.cryptocompare.com/data',
                'enabled': True
            }
        ]
        
        # 토큰 심볼과 각 소스의 ID 매핑
        self.token_source_ids = self._build_token_source_mapping()
        
    def _build_token_source_mapping(self) -> Dict[str, Dict[str, str]]:
        """각 소스별 토큰 심볼과 ID 매핑 생성"""
        # CoinGecko ID 매핑
        coingecko_mapping = {
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
        
        # Coinpaprika ID 매핑
        coinpaprika_mapping = {
            'ETH': 'eth-ethereum',
            'WETH': 'weth-wrapped-ether',
            'USDC': 'usdc-usd-coin',
            'USDT': 'usdt-tether',
            'DAI': 'dai-dai',
            'WBTC': 'wbtc-wrapped-bitcoin',
            'UNI': 'uni-uniswap',
            'SUSHI': 'sushi-sushi',
            'COMP': 'comp-compoundd',
            'AAVE': 'aave-aave',
            'CRV': 'crv-curve-dao-token',
            'BAL': 'bal-balancer',
            'YFI': 'yfi-yearn-finance',
            'MKR': 'mkr-maker',
            'BNT': 'bnt-bancor',
            'BAT': 'bat-basic-attention-token',
            'KNC': 'knc-kyber-network-crystal',
            'MANA': 'mana-decentraland',
            'GNO': 'gno-gnosis',
            'RLC': 'rlc-iexec-rlc',
            'UBT': 'ubt-unibright',
            'SAI': 'sai-single-collateral-dai',
            'cETH': 'ceth-compound-ether',
            'cUSDC': 'cusdc-compound-usd-coin',
            'aETH': 'aeth-aave-eth',
            'aUSDC': 'ausdc-aave-usdc',
            'sUSD': 'susd-susd',
            'sETH': 'seth-seth',
            'sBTC': 'sbtc-sbtc',
            'sLINK': 'slink-slink',
            'sDEFI': 'sdefi-sdefi',
            'sXAU': 'sxau-sxau',
            'sXAG': 'sxag-sxag'
        }
        
        # CryptoCompare symbol mapping (same as token symbol for most)
        cryptocompare_mapping = {
            symbol: symbol for symbol in coingecko_mapping.keys()
        }
        
        # Combine all mappings
        mapping = {}
        for symbol in coingecko_mapping.keys():
            mapping[symbol] = {
                'coingecko': coingecko_mapping.get(symbol),
                'coinpaprika': coinpaprika_mapping.get(symbol),
                'cryptocompare': cryptocompare_mapping.get(symbol)
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
        """모든 토큰의 가격 업데이트 (여러 소스에서 가져와서 비교)"""
        try:
            # 지원되는 모든 토큰 가져오기
            all_tokens = self.token_manager.get_all_asset_symbols()
            
            # 각 소스별로 가격 데이터 가져오기
            price_data_by_source = {}
            
            for source in self.price_sources:
                if not source['enabled']:
                    continue
                    
                try:
                    if source['name'] == 'coingecko':
                        prices = await self._fetch_coingecko_prices(all_tokens)
                    elif source['name'] == 'coinpaprika':
                        prices = await self._fetch_coinpaprika_prices(all_tokens)
                    elif source['name'] == 'cryptocompare':
                        prices = await self._fetch_cryptocompare_prices(all_tokens)
                    else:
                        continue
                        
                    if prices:
                        price_data_by_source[source['name']] = prices
                        logger.debug(f"{source['name']}에서 {len(prices)}개 토큰의 가격 데이터 가져옴")
                except Exception as e:
                    logger.error(f"{source['name']}에서 가격 데이터 가져오기 실패: {e}")
            
            if not price_data_by_source:
                logger.warning("모든 가격 소스에서 데이터를 가져오지 못했습니다")
                return
            
            # 각 토큰에 대해 여러 소스의 데이터를 비교하고 최선의 데이터 선택
            updated_count = 0
            for symbol in all_tokens:
                best_price_data = self._select_best_price_data(symbol, price_data_by_source)
                
                if best_price_data:
                    price_usd = best_price_data['price']
                    source = best_price_data['source']
                    token_address = self.token_manager.get_asset_address(symbol)
                    
                    price_feed = PriceFeed(
                        token_address=token_address,
                        symbol=symbol,
                        price_usd=price_usd,
                        source=source,
                        timestamp=datetime.now().timestamp()
                    )
                    
                    self.price_feeds[symbol] = price_feed
                    await self._store_price_feed(price_feed)
                    updated_count += 1
                    
                    logger.debug(f"가격 업데이트: {symbol} = ${price_usd} (소스: {source})")
            
            logger.info(f"총 {updated_count}개 토큰의 가격 업데이트 완료 (다중 소스 사용)")
            
            # 구독자들에게 알림
            await self._notify_subscribers()
            
        except Exception as e:
            logger.error(f"모든 토큰 가격 업데이트 중 오류 발생: {e}")
    
    def _select_best_price_data(self, symbol: str, price_data_by_source: Dict[str, Dict]) -> Optional[Dict]:
        """여러 소스의 가격 데이터 중 최선의 것을 선택"""
        available_prices = []
        
        # 각 소스에서 해당 토큰의 가격 데이터 수집
        for source_name, price_data in price_data_by_source.items():
            if symbol in price_data and price_data[symbol] is not None:
                available_prices.append({
                    'source': source_name,
                    'price': price_data[symbol],
                    'timestamp': datetime.now().timestamp()
                })
        
        if not available_prices:
            return None
        
        # 여러 소스가 있는 경우, 가장 최근의 데이터 선택
        # 또는 평균값 계산 (더 정확한 방법)
        if len(available_prices) == 1:
            return available_prices[0]
        else:
            # 여러 소스의 평균값 사용 (이상치 제거를 위해 중간값 사용)
            prices = [p['price'] for p in available_prices]
            prices.sort()
            median_price = prices[len(prices) // 2]  # 중간값
            
            # 중간값에 가장 가까운 소스 선택
            best_source = min(available_prices, key=lambda x: abs(x['price'] - median_price))
            best_source['price'] = median_price  # 평균값으로 업데이트
            
            return best_source
    
    async def _fetch_coingecko_prices(self, symbols: List[str]) -> Optional[Dict[str, float]]:
        """CoinGecko API에서 가격 데이터 가져오기"""
        try:
            if not self.session:
                await self.initialize()
            
            # CoinGecko ID가 있는 토큰 필터링
            tokens_with_ids = {
                symbol: self.token_source_ids[symbol]['coingecko'] 
                for symbol in symbols 
                if symbol in self.token_source_ids and self.token_source_ids[symbol]['coingecko']
            }
            
            if not tokens_with_ids:
                return None
            
            # 최대 250개의 코인 ID를 한 번에 요청 가능
            ids_param = ','.join(tokens_with_ids.values())
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids_param}&vs_currencies=usd"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # 토큰 심볼로 변환
                    result = {}
                    for symbol, cg_id in tokens_with_ids.items():
                        if cg_id in data and 'usd' in data[cg_id]:
                            result[symbol] = data[cg_id]['usd']
                    return result
                else:
                    logger.error(f"CoinGecko API 요청 실패: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"CoinGecko API 요청 중 오류 발생: {e}")
            return None
    
    async def _fetch_coinpaprika_prices(self, symbols: List[str]) -> Optional[Dict[str, float]]:
        """Coinpaprika API에서 가격 데이터 가져오기"""
        try:
            if not self.session:
                await self.initialize()
            
            # Coinpaprika ID가 있는 토큰 필터링
            tokens_with_ids = {
                symbol: self.token_source_ids[symbol]['coinpaprika'] 
                for symbol in symbols 
                if symbol in self.token_source_ids and self.token_source_ids[symbol]['coinpaprika']
            }
            
            if not tokens_with_ids:
                return None
            
            # 각 토큰에 대해 개별 요청
            result = {}
            for symbol, cp_id in tokens_with_ids.items():
                try:
                    url = f"https://api.coinpaprika.com/v1/tickers/{cp_id}"
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'quotes' in data and 'USD' in data['quotes']:
                                result[symbol] = data['quotes']['USD']['price']
                        else:
                            logger.debug(f"Coinpaprika API 요청 실패 ({symbol}): {response.status}")
                except Exception as e:
                    logger.debug(f"Coinpaprika API 요청 중 오류 발생 ({symbol}): {e}")
                    continue
            
            return result if result else None
        except Exception as e:
            logger.error(f"Coinpaprika API 요청 중 오류 발생: {e}")
            return None
    
    async def _fetch_cryptocompare_prices(self, symbols: List[str]) -> Optional[Dict[str, float]]:
        """CryptoCompare API에서 가격 데이터 가져오기"""
        try:
            if not self.session:
                await self.initialize()
            
            # CryptoCompare에서 지원하는 토큰 필터링
            supported_symbols = {
                symbol: self.token_source_ids[symbol]['cryptocompare'] 
                for symbol in symbols 
                if symbol in self.token_source_ids and self.token_source_ids[symbol]['cryptocompare']
            }
            
            if not supported_symbols:
                return None
            
            # 여러 토큰의 가격을 한 번에 요청
            fsyms = ','.join(supported_symbols.values())
            url = f"https://min-api.cryptocompare.com/data/pricemulti?fsyms={fsyms}&tsyms=USD"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # 토큰 심볼로 변환
                    result = {}
                    for symbol, cc_symbol in supported_symbols.items():
                        if cc_symbol in data and 'USD' in data[cc_symbol]:
                            result[symbol] = data[cc_symbol]['USD']
                    return result
                else:
                    logger.error(f"CryptoCompare API 요청 실패: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"CryptoCompare API 요청 중 오류 발생: {e}")
            return None
    
    async def _store_price_feed(self, price_feed: PriceFeed):
        """가격 피드 데이터 저장"""
        try:
            # 데이터 검증
            validation_result = self.data_validator.validate_price_data(
                price_feed.symbol, price_feed.price_usd, price_feed.timestamp
            )
            
            if not validation_result['valid']:
                logger.warning(f"가격 데이터 검증 실패 ({price_feed.symbol}): {validation_result['reason']}")
                if validation_result['severity'] in ['critical', 'error']:
                    return  # 심각한 오류는 저장하지 않음
            
            # 검증을 통과한 데이터만 저장
            # Redis에 현재 가격 저장
            key = f"price:{price_feed.symbol}"
            data = json.dumps(asdict(price_feed), default=str)
            self.data_storage.redis_client.setex(key, 300, data)  # 5분 TTL
            
            # 시계열 데이터도 저장
            timestamp = datetime.now().isoformat()
            ts_key = f"price_history:{price_feed.symbol}:{timestamp}"
            self.data_storage.redis_client.setex(ts_key, 3600, data)  # 1시간 보관
            
            logger.debug(f"가격 피드 저장 완료: {price_feed.symbol} (검증: {validation_result['severity']})")
            
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