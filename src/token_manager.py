import json
import aiohttp
from typing import Dict, Optional, List
from dataclasses import dataclass
from web3 import Web3
from src.logger import setup_logger
from config.config import config
from src.constants import ETH_NATIVE_ADDRESS

logger = setup_logger(__name__)

@dataclass
class TokenInfo:
    address: str
    symbol: str
    name: str
    decimals: int
    price_usd: float = 0.0
    market_cap: float = 0.0
    volume_24h: float = 0.0
    coingecko_id: str = ""

class TokenManager:
    def __init__(self, web3_provider_url: Optional[str] = None):
        self.tokens: Dict[str, TokenInfo] = {}
        self.symbol_to_address: Dict[str, str] = {}
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        
        # RPC URL 설정 - 파라미터가 없으면 config에서 가져옴
        rpc_url = web3_provider_url or config.ethereum_mainnet_rpc
        self.w3 = Web3(Web3.HTTPProvider(rpc_url)) if rpc_url else None
        self.erc20_abi = [
            {
                "constant": True,
                "inputs": [],
                "name": "name",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "symbol", 
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            }
        ]
        self._load_common_tokens()
    
    def _load_common_tokens(self):
        """주요 토큰 정보 로드 - 실제 주소로 업데이트 필요"""
        common_tokens = {
            # 네이티브 ETH (플레이스홀더)
            ETH_NATIVE_ADDRESS: TokenInfo(
                address=ETH_NATIVE_ADDRESS,
                symbol="ETH",
                name="Ether (native)",
                decimals=18,
                coingecko_id="ethereum"
            ),
            # Compound cTokens (selected)
            "0x39AA39c021dfbaE8faC545936693aC917d5E7563": TokenInfo(
                address="0x39AA39c021dfbaE8faC545936693aC917d5E7563",
                symbol="cUSDC",
                name="Compound USD Coin",
                decimals=8,
                coingecko_id="compound-usd-coin"
            ),
            "0x4DdC2D193948926d02f9B1fE9e1daa0718270ED5": TokenInfo(
                address="0x4DdC2D193948926d02f9B1fE9e1daa0718270ED5",
                symbol="cETH",
                name="Compound Ether",
                decimals=8,
                coingecko_id="compound-ether"
            ),
            # Aave aTokens (selected)
            "0xBcca60bB61934080951369a648Fb03DF4F96263C": TokenInfo(
                address="0xBcca60bB61934080951369a648Fb03DF4F96263C",
                symbol="aUSDC",
                name="Aave interest bearing USDC",
                decimals=6,
                coingecko_id="aave-usdc"
            ),
            # 주요/디파이 에코 토큰들
            "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599": TokenInfo(
                address="0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
                symbol="WBTC",
                name="Wrapped Bitcoin",
                decimals=8,
                coingecko_id="wrapped-bitcoin"
            ),
            "0xc00e94Cb662C3520282E6f5717214004A7f26888": TokenInfo(
                address="0xc00e94Cb662C3520282E6f5717214004A7f26888",
                symbol="COMP",
                name="Compound",
                decimals=18,
                coingecko_id="compound-governance-token"
            ),
            "0xD533a949740bb3306d119CC777fa900bA034cd52": TokenInfo(
                address="0xD533a949740bb3306d119CC777fa900bA034cd52",
                symbol="CRV",
                name="Curve DAO Token",
                decimals=18,
                coingecko_id="curve-dao-token"
            ),
            "0xba100000625a3754423978a60c9317c58a424e3D": TokenInfo(
                address="0xba100000625a3754423978a60c9317c58a424e3D",
                symbol="BAL",
                name="Balancer",
                decimals=18,
                coingecko_id="balancer"
            ),
            "0x0bc529c00C6401aEF6D220BE8C6Ea1667F6Ad93e": TokenInfo(
                address="0x0bc529c00C6401aEF6D220BE8C6Ea1667F6Ad93e",
                symbol="YFI",
                name="yearn.finance",
                decimals=18,
                coingecko_id="yearn-finance"
            ),
            # 메인 토큰들
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": TokenInfo(
                address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                symbol="WETH",
                name="Wrapped Ether", 
                decimals=18,
                coingecko_id="weth"
            ),
            "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": TokenInfo(
                address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                symbol="USDC",
                name="USD Coin",
                decimals=6,
                coingecko_id="usd-coin"
            ),
            "0x6B175474E89094C44Da98b954EedeAC495271d0F": TokenInfo(
                address="0x6B175474E89094C44Da98b954EedeAC495271d0F",
                symbol="DAI",
                name="Dai Stablecoin",
                decimals=18,
                coingecko_id="dai"
            ),
            "0xdAC17F958D2ee523a2206206994597C13D831ec7": TokenInfo(
                address="0xdAC17F958D2ee523a2206206994597C13D831ec7",
                symbol="USDT",
                name="Tether USD",
                decimals=6,
                coingecko_id="tether"
            ),
            # DeFi 토큰들
            "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984": TokenInfo(
                address="0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
                symbol="UNI",
                name="Uniswap",
                decimals=18,
                coingecko_id="uniswap"
            ),
            "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9": TokenInfo(
                address="0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
                symbol="AAVE",
                name="Aave Token",
                decimals=18,
                coingecko_id="aave"
            ),
            "0x6B3595068778DD592e39A122f4f5a5cF09C90fE2": TokenInfo(
                address="0x6B3595068778DD592e39A122f4f5a5cF09C90fE2",
                symbol="SUSHI",
                name="SushiToken",
                decimals=18,
                coingecko_id="sushi"
            ),
            "0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F": TokenInfo(
                address="0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F",
                symbol="SNX",
                name="Synthetix Network Token",
                decimals=18,
                coingecko_id="havven"
            ),
            # 스테이블코인들
            "0x4Fabb145d64652a948d72533023f6E7A623C7C53": TokenInfo(
                address="0x4Fabb145d64652a948d72533023f6E7A623C7C53",
                symbol="BUSD",
                name="Binance USD",
                decimals=18,
                coingecko_id="binance-usd"
            ),
            "0x853d955aCEf822Db058eb8505911ED77F175b99e": TokenInfo(
                address="0x853d955aCEf822Db058eb8505911ED77F175b99e",
                symbol="FRAX",
                name="Frax",
                decimals=18,
                coingecko_id="frax"
            )
        }
        
        self.tokens.update(common_tokens)
        
        # 심볼 -> 주소 매핑 생성
        for address, token_info in common_tokens.items():
            self.symbol_to_address[token_info.symbol] = address
    
    async def get_token_info(self, address: str) -> Optional[TokenInfo]:
        """토큰 정보 조회"""
        address = address.lower()
        if address in self.tokens:
            return self.tokens[address]
        
        # 온체인에서 토큰 정보 조회 시도
        token_info = await self._fetch_token_info_onchain(address)
        if token_info:
            self.tokens[address] = token_info
            return token_info
        
        return None
    
    def get_address_by_symbol(self, symbol: str) -> Optional[str]:
        """심볼로 주소 조회"""
        return self.symbol_to_address.get(symbol.upper())
    
    async def update_prices(self):
        """CoinGecko에서 가격 정보 업데이트"""
        try:
            # CoinGecko ID 목록 생성
            coingecko_ids = [
                token.coingecko_id for token in self.tokens.values() 
                if token.coingecko_id
            ]
            
            if not coingecko_ids:
                return
            
            # API 호출
            ids_str = ",".join(coingecko_ids)
            url = f"{self.coingecko_api}/simple/price"
            params = {
                'ids': ids_str,
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # 가격 정보 업데이트
                        for token in self.tokens.values():
                            if token.coingecko_id in data:
                                price_data = data[token.coingecko_id]
                                token.price_usd = price_data.get('usd', 0)
                                token.market_cap = price_data.get('usd_market_cap', 0)
                                token.volume_24h = price_data.get('usd_24h_vol', 0)
                        
                        logger.info(f"{len(data)}개 토큰 가격 정보 업데이트 완료")
                    else:
                        logger.error(f"CoinGecko API 오류: {response.status}")
                        
        except Exception as e:
            logger.error(f"가격 정보 업데이트 실패: {e}")
    
    async def _fetch_token_info_onchain(self, address: str) -> Optional[TokenInfo]:
        """온체인에서 토큰 정보 조회"""
        if not self.w3 or not self.w3.is_connected():
            logger.warning("Web3 연결이 설정되지 않았습니다")
            return None
            
        try:
            # 주소 체크섬 변환
            checksum_address = self.w3.to_checksum_address(address)
            
            # ERC20 컨트랙트 생성
            contract = self.w3.eth.contract(address=checksum_address, abi=self.erc20_abi)
            
            # 토큰 정보 조회
            symbol = contract.functions.symbol().call()
            name = contract.functions.name().call()
            decimals = contract.functions.decimals().call()
            
            token_info = TokenInfo(
                address=checksum_address,
                symbol=symbol,
                name=name,
                decimals=decimals
            )
            
            logger.info(f"온체인 토큰 정보 조회 성공: {symbol} ({name})")
            return token_info
            
        except Exception as e:
            logger.error(f"온체인 토큰 정보 조회 실패 {address}: {e}")
            return None
    
    def get_major_trading_pairs(self) -> List[tuple]:
        """주요 거래 쌍 목록 반환"""
        major_pairs = [
            ("WETH", "USDC"),
            ("WETH", "DAI"),
            ("WETH", "USDT"),
            ("USDC", "DAI"),
            ("USDC", "USDT"),
            ("DAI", "USDT"),
            ("WETH", "UNI"),
            ("WETH", "AAVE"),
            ("WETH", "SUSHI"),
            ("USDC", "UNI"),
            ("USDC", "AAVE")
        ]
        
        # 주소로 변환
        address_pairs = []
        for symbol0, symbol1 in major_pairs:
            addr0 = self.get_address_by_symbol(symbol0)
            addr1 = self.get_address_by_symbol(symbol1)
            if addr0 and addr1:
                address_pairs.append((addr0, addr1))
        
        return address_pairs
