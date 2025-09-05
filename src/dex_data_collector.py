import asyncio
import json
from web3 import Web3
from web3.contract import Contract
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
    fee: int
    reserve0: int = 0
    reserve1: int = 0
    price: float = 0.0
    last_updated: int = 0

class UniswapV2Collector:
    def __init__(self, web3_provider: Web3):
        self.w3 = web3_provider
        self.factory_address = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"
        self.factory_abi = self._load_abi("uniswap_v2_factory.json")
        self.pair_abi = self._load_abi("uniswap_v2_pair.json")
        self.factory_contract = self.w3.eth.contract(
            address=self.factory_address, 
            abi=self.factory_abi
        )
        self.pools: Dict[str, PoolInfo] = {}
        
    def _load_abi(self, filename: str) -> List[Dict]:
        """ABI 파일 로드"""
        try:
            with open(f"abi/{filename}", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"ABI 파일을 찾을 수 없습니다: {filename}")
            return []
    
    async def get_pair_address(self, token0: str, token1: str) -> Optional[str]:
        """토큰 쌍의 풀 주소 조회"""
        try:
            pair_address = self.factory_contract.functions.getPair(
                token0, token1
            ).call()
            
            if pair_address == "0x0000000000000000000000000000000000000000":
                return None
            return pair_address
        except Exception as e:
            logger.error(f"페어 주소 조회 실패: {e}")
            return None
    
    async def get_pool_reserves(self, pool_address: str) -> Tuple[int, int]:
        """풀의 리저브 정보 조회"""
        try:
            pair_contract = self.w3.eth.contract(
                address=pool_address, 
                abi=self.pair_abi
            )
            reserves = pair_contract.functions.getReserves().call()
            return reserves[0], reserves[1]  # reserve0, reserve1
        except Exception as e:
            logger.error(f"리저브 조회 실패 {pool_address}: {e}")
            return 0, 0
    
    async def calculate_price(self, reserve0: int, reserve1: int, 
                            decimals0: int = 18, decimals1: int = 18) -> float:
        """가격 계산 (token1/token0)"""
        if reserve0 == 0 or reserve1 == 0:
            return 0.0
        
        # 소수점 조정
        adjusted_reserve0 = reserve0 / (10 ** decimals0)
        adjusted_reserve1 = reserve1 / (10 ** decimals1)
        
        return adjusted_reserve1 / adjusted_reserve0
    
    async def update_pool_data(self, pool_info: PoolInfo) -> bool:
        """풀 데이터 업데이트"""
        try:
            reserve0, reserve1 = await self.get_pool_reserves(pool_info.address)
            
            if reserve0 > 0 and reserve1 > 0:
                pool_info.reserve0 = reserve0
                pool_info.reserve1 = reserve1
                pool_info.price = await self.calculate_price(reserve0, reserve1)
                pool_info.last_updated = self.w3.eth.get_block('latest').timestamp
                return True
            return False
        except Exception as e:
            logger.error(f"풀 데이터 업데이트 실패 {pool_info.address}: {e}")
            return False

class SushiSwapCollector(UniswapV2Collector):
    """SushiSwap 데이터 수집기 (Uniswap V2와 동일한 인터페이스)"""
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
        # Curve Registry 컨트랙트를 통한 풀 목록 조회 구현
        pass
    
    async def get_curve_price(self, pool_address: str, i: int, j: int) -> float:
        """Curve 풀에서 토큰 i를 j로 교환할 때의 가격"""
        # Curve 특화 가격 계산 로직 구현
        pass
