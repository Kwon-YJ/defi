import json
import aiohttp
from typing import Dict, Optional
from dataclasses import dataclass
from src.logger import setup_logger

logger = setup_logger(__name__ )

@dataclass
class TokenInfo:
    address: str
    symbol: str
    name: str
    decimals: int
    price_usd: float = 0.0
    market_cap: float = 0.0
    volume_24h: float = 0.0

class TokenManager:
    def __init__(self):
        self.tokens: Dict[str, TokenInfo] = {}
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        self._load_common_tokens( )
    
    def _load_common_tokens(self):
        """주요 토큰 정보 로드"""
        common_tokens = {
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": TokenInfo(
                address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                symbol="WETH",
                name="Wrapped Ether", 
                decimals=18
            ),
            "0xA0b73E1Ff0B80914AB6fe0444E65848C4C34450b": TokenInfo(
                address="0xA0b73E1Ff0B80914AB6fe0444E65848C4C34450b",
                symbol="USDC",
                name="USD Coin",
                decimals=6
            ),
            "0x6B175474E89094C44Da98b954EedeAC495271d0F": TokenInfo(
                address="0x6B175474E89094C44Da98b954EedeAC495271d0F",
                symbol="DAI",
                name="Dai Stablecoin",
                decimals=18
            ),
            "0xdAC17F958D2ee523a2206206994597C13D831ec7": TokenInfo(
                address="0xdAC17F958D2ee523a2206206994597C13D831ec7",
                symbol="USDT",
                name="Tether USD",
                decimals=6
            )
        }
        self.tokens.update(common_tokens)
    
    async def get_token_info(self, address: str) -> Optional[TokenInfo]:
        """토큰 정보 조회"""
        if address in self.tokens:
            return self.tokens[address]
        
        # 온체인에서 토큰 정보 조회
        token_info = await self._fetch_token_info_onchain(address)
        if token_info:
            self.tokens[address] = token_info
            return token_info
        
        return None
    
    async def _fetch_token_info_onchain(self, address: str) -> Optional[TokenInfo]:
        """온체인에서 토큰 정보 조회"""
        try:
            # ERC20 컨트랙트 인터페이스를 통한 정보 조회
            # symbol(), name(), decimals() 함수 호출
            # 실제 구현에서는 Web3 연결 필요
            pass
        except Exception as e:
            logger.error(f"토큰 정보 조회 실패 {address}: {e}")
            return None
    
    async def update_token_prices(self):
        """CoinGecko API를 통한 가격 정보 업데이트"""
        try:
            token_addresses = list(self.tokens.keys())
            addresses_str = ",".join(token_addresses)
            
            async with aiohttp.ClientSession( ) as session:
                url = f"{self.coingecko_api}/simple/token_price/ethereum"
                params = {
                    "contract_addresses": addresses_str,
                    "vs_currencies": "usd",
                    "include_market_cap": "true",
                    "include_24hr_vol": "true"
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for address, price_data in data.items():
                            if address in self.tokens:
                                token = self.tokens[address]
                                token.price_usd = price_data.get('usd', 0.0)
                                token.market_cap = price_data.get('usd_market_cap', 0.0)
                                token.volume_24h = price_data.get('usd_24h_vol', 0.0)
                                
        except Exception as e:
            logger.error(f"가격 정보 업데이트 실패: {e}")
