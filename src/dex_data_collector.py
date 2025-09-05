import json
from web3 import Web3
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from src.logger import setup_logger
from config.config import config

logger = setup_logger(__name__)

@dataclass
class PoolInfo:
    address: str
    token0: str
    token1: str
    reserve0: int
    reserve1: int
    dex: str
    fee: float
    last_updated: int

class UniswapV2Collector:
    def __init__(self, web3_provider: Web3):
        self.w3 = web3_provider
        self.factory_address = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"
        self.factory_abi = self._get_factory_abi()
        self.pair_abi = self._get_pair_abi()
        self.factory_contract = self.w3.eth.contract(
            address=self.factory_address, 
            abi=self.factory_abi
        )
        self.pools: Dict[str, PoolInfo] = {}
        
    def _get_factory_abi(self) -> List[Dict]:
        """Factory ABI 로드"""
        try:
            with open('abi/uniswap_v2_factory.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # 최소 ABI
            return [
                {
                    "constant": True,
                    "inputs": [
                        {"name": "tokenA", "type": "address"},
                        {"name": "tokenB", "type": "address"}
                    ],
                    "name": "getPair",
                    "outputs": [{"name": "pair", "type": "address"}],
                    "type": "function"
                }
            ]
    
    def _get_pair_abi(self) -> List[Dict]:
        """Pair ABI 로드"""
        try:
            with open('abi/uniswap_v2_pair.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # 최소 ABI
            return [
                {
                    "constant": True,
                    "inputs": [],
                    "name": "getReserves",
                    "outputs": [
                        {"name": "reserve0", "type": "uint112"},
                        {"name": "reserve1", "type": "uint112"},
                        {"name": "blockTimestampLast", "type": "uint32"}
                    ],
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "token0",
                    "outputs": [{"name": "", "type": "address"}],
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "token1",
                    "outputs": [{"name": "", "type": "address"}],
                    "type": "function"
                }
            ]
    
    async def get_pair_address(self, token0: str, token1: str) -> Optional[str]:
        """토큰 쌍의 풀 주소 조회"""
        try:
            pair_address = self.factory_contract.functions.getPair(token0, token1).call()
            
            # 0x0 주소는 풀이 존재하지 않음을 의미
            if pair_address == "0x0000000000000000000000000000000000000000":
                return None
                
            return pair_address
            
        except Exception as e:
            logger.error(f"페어 주소 조회 실패 {token0}-{token1}: {e}")
            return None
    
    async def get_pool_reserves(self, pool_address: str) -> Tuple[int, int, int]:
        """풀의 리저브 정보 조회"""
        try:
            pair_contract = self.w3.eth.contract(
                address=pool_address, 
                abi=self.pair_abi
            )
            
            reserves = pair_contract.functions.getReserves().call()
            return reserves[0], reserves[1], reserves[2]  # reserve0, reserve1, timestamp
            
        except Exception as e:
            logger.error(f"리저브 조회 실패 {pool_address}: {e}")
            return 0, 0, 0
    
    async def get_pool_tokens(self, pool_address: str) -> Tuple[str, str]:
        """풀의 토큰 주소 조회"""
        try:
            pair_contract = self.w3.eth.contract(
                address=pool_address, 
                abi=self.pair_abi
            )
            
            token0 = pair_contract.functions.token0().call()
            token1 = pair_contract.functions.token1().call()
            
            return token0, token1
            
        except Exception as e:
            logger.error(f"토큰 주소 조회 실패 {pool_address}: {e}")
            return "", ""
    
    async def update_pool_data(self, token0: str, token1: str) -> Optional[PoolInfo]:
        """풀 데이터 업데이트"""
        try:
            # 풀 주소 조회
            pool_address = await self.get_pair_address(token0, token1)
            if not pool_address:
                return None
            
            # 리저브 정보 조회
            reserve0, reserve1, timestamp = await self.get_pool_reserves(pool_address)
            
            if reserve0 == 0 or reserve1 == 0:
                return None
            
            # PoolInfo 생성
            pool_info = PoolInfo(
                address=pool_address,
                token0=token0,
                token1=token1,
                reserve0=reserve0,
                reserve1=reserve1,
                dex="uniswap_v2",
                fee=0.003,  # 0.3%
                last_updated=timestamp
            )
            
            self.pools[pool_address] = pool_info
            return pool_info
            
        except Exception as e:
            logger.error(f"풀 데이터 업데이트 실패 {token0}-{token1}: {e}")
            return None
    
    def calculate_price(self, reserve0: int, reserve1: int, 
                       decimals0: int = 18, decimals1: int = 18) -> float:
        """가격 계산 (token1/token0)"""
        if reserve0 == 0:
            return 0
        
        # 소수점 조정
        adjusted_reserve0 = reserve0 / (10 ** decimals0)
        adjusted_reserve1 = reserve1 / (10 ** decimals1)
        
        return adjusted_reserve1 / adjusted_reserve0

class SushiSwapCollector(UniswapV2Collector):
    """SushiSwap 데이터 수집기"""
    def __init__(self, web3_provider: Web3):
        super().__init__(web3_provider)
        self.factory_address = "0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac"
        self.factory_contract = self.w3.eth.contract(
            address=self.factory_address, 
            abi=self.factory_abi
        )

class CurveCollector:
    """Curve Finance 데이터 수집기"""
    def __init__(self, web3_provider: Web3):
        self.w3 = web3_provider
        self.registry_address = "0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5"
        self.pools: Dict[str, PoolInfo] = {}
        
    async def get_curve_pools(self) -> List[str]:
        """Curve 풀 목록 조회"""
        # TODO: Curve Registry 컨트랙트 구현
        return []
    
    async def get_curve_price(self, pool_address: str, i: int, j: int, 
                            amount: int = 10**18) -> float:
        """Curve 풀에서 토큰 i를 j로 교환할 때의 가격"""
        # TODO: Curve 특화 가격 계산 로직 구현
        return 0.0