import json
import aiohttp
from typing import Dict, Optional, List
from dataclasses import dataclass
from web3 import Web3
from src.logger import setup_logger
from config.config import config

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
        }

        paper_tokens = {
            "0x89d24a6b4ccb1b6faa2625fe562bdd9a23260359": TokenInfo(address="0x89d24a6b4ccb1b6faa2625fe562bdd9a23260359", symbol="SAI", name="Sai Stablecoin v1.0", decimals=18, coingecko_id="sai"),
            "0x1f573d6fb3f13d689ff844b4ce37794d79a7ff1c": TokenInfo(address="0x1f573d6fb3f13d689ff844b4ce37794d79a7ff1c", symbol="BNT", name="Bancor Network Token", decimals=18, coingecko_id="bancor-network-token"),
            "0x0d8775f648430679a709e98d2b0cb6250d2887ef": TokenInfo(address="0x0d8775f648430679a709e98d2b0cb6250d2887ef", symbol="BAT", name="Basic Attention Token", decimals=18, coingecko_id="basic-attention-token"),
            "0xf629cbd94d3791c9250152bd8dfbdf380e2a3b9c": TokenInfo(address="0xf629cbd94d3791c9250152bd8dfbdf380e2a3b9c", symbol="ENJ", name="Enjin Coin", decimals=18, coingecko_id="enjin-coin"),
            "0x744d70fdbe2ba4cf95131626614a1763df805b9e": TokenInfo(address="0x744d70fdbe2ba4cf95131626614a1763df805b9e", symbol="SNT", name="Status Network Token", decimals=18, coingecko_id="status"),
            "0xdefa4e8a7bcba345f687a2f1456f5edd9ce97202": TokenInfo(address="0xdefa4e8a7bcba345f687a2f1456f5edd9ce97202", symbol="KNC", name="Kyber Network Crystal", decimals=18, coingecko_id="kyber-network-crystal"),
            "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2": TokenInfo(address="0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2", symbol="MKR", name="Maker", decimals=18, coingecko_id="maker"),
            "0x8f693ca8d21b157107184d29d398a8d082b38b76": TokenInfo(address="0x8f693ca8d21b157107184d29d398a8d082b38b76", symbol="DATA", name="Streamr", decimals=18, coingecko_id="streamr"),
            "0x0f5d2fb29fb7d3cfee444a200298f468908cc942": TokenInfo(address="0x0f5d2fb29fb7d3cfee444a200298f468908cc942", symbol="MANA", name="Decentraland", decimals=18, coingecko_id="decentraland"),
            "0xa117000000f279d81a1d3cc75430faa017fa5a2e": TokenInfo(address="0xa117000000f279d81a1d3cc75430faa017fa5a2e", symbol="ANT", name="Aragon", decimals=18, coingecko_id="aragon"),
            "0x607f4c5bb672230e8672085532f7e901544a7375": TokenInfo(address="0x607f4c5bb672230e8672085532f7e901544a7375", symbol="RLC", name="iExec RLC", decimals=9, coingecko_id="iexec-rlc"),
            "0xf970b8e3d081a77092b7793bde317ee64d9a712b": TokenInfo(address="0xf970b8e3d081a77092b7793bde317ee64d9a712b", symbol="RCN", name="Ripio Credit Network", decimals=18, coingecko_id="ripio-credit-network"),
            "0x8400d94a5cb0fa0d041a3788e395285d61c9ee5e": TokenInfo(address="0x8400d94a5cb0fa0d041a3788e395285d61c9ee5e", symbol="UBT", name="Unibright", decimals=8, coingecko_id="unibright"),
            "0x6810e776880c02933d47db1b9fc05908e5386b96": TokenInfo(address="0x6810e776880c02933d47db1b9fc05908e5386b96", symbol="GNO", name="Gnosis", decimals=18, coingecko_id="gnosis"),
            "0x255aa6df07540cb5d3d297f0d0d4d84cb52bc8e6": TokenInfo(address="0x255aa6df07540cb5d3d297f0d0d4d84cb52bc8e6", symbol="RDN", name="Raiden Network Token", decimals=18, coingecko_id="raiden-network-token"),
            "0xaaaf91d9b90df800df4f55c205fd6989c977e73a": TokenInfo(address="0xaaaf91d9b90df800df4f55c205fd6989c977e73a", symbol="TKN", name="Monolith", decimals=8, coingecko_id="tokencard"),
            "0x4a57e687b9126435a9b19e4a802113e266adebde": TokenInfo(address="0x4a57e687b9126435a9b19e4a802113e266adebde", symbol="FXC", name="Flexacoin", decimals=18, coingecko_id="flexacoin"),
            "0x7c5a0ce9267ed19b22f8cae653f198e3e8daf098": TokenInfo(address="0x7c5a0ce9267ed19b22f8cae653f198e3e8daf098", symbol="SAN", name="Santiment Network Token", decimals=18, coingecko_id="santiment-network-token"),
            "0xd46ba6d942050d489dbd938a2c909a5d5039a161": TokenInfo(address="0xd46ba6d942050d489dbd938a2c909a5d5039a161", symbol="AMPL", name="Ampleforth", decimals=9, coingecko_id="ampleforth"),
            "0xf1290473e210b2108a85237fbcd7b6eb42cc654f": TokenInfo(address="0xf1290473e210b2108a85237fbcd7b6eb42cc654f", symbol="HEDG", name="HedgeTrade", decimals=18, coingecko_id="hedgetrade"),
        }
        
        self.tokens.update(common_tokens)
        self.tokens.update(paper_tokens)
        
        # 심볼 -> 주소 매핑 생성
        for address, token_info in self.tokens.items():
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
        paper_token_symbols = [
            "SAI", "BNT", "BAT", "ENJ", "SNT", "KNC", "MKR", "DATA", "MANA",
            "ANT", "RLC", "RCN", "UBT", "GNO", "RDN", "TKN", "FXC", "SAN",
            "AMPL", "HEDG", "DAI"
        ]

        major_pairs = []

        # Uniswap pairs (WETH vs all paper tokens)
        for symbol in paper_token_symbols:
            if symbol != "WETH":
                major_pairs.append(("WETH", symbol))

        # Bancor pairs (BNT vs all paper tokens except BNT)
        for symbol in paper_token_symbols:
            if symbol != "BNT":
                major_pairs.append(("BNT", symbol))

        # MakerDAO pair
        major_pairs.append(("DAI", "SAI"))

        # 주소로 변환
        address_pairs = []
        for symbol0, symbol1 in major_pairs:
            addr0 = self.get_address_by_symbol(symbol0)
            addr1 = self.get_address_by_symbol(symbol1)
            if addr0 and addr1:
                # 중복 및 역순 쌍 제거
                if (addr0, addr1) not in address_pairs and (addr1, addr0) not in address_pairs:
                    address_pairs.append((addr0, addr1))
        
        return address_pairs