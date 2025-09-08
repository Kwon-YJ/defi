import json
from typing import Dict, List, Optional, Tuple
from web3 import Web3
from src.logger import setup_logger

logger = setup_logger(__name__)


class UniswapV2LPCollector:
    """Uniswap V2 LP collector (minimal ABI)

    Provides pair discovery, reserves, tokens, and totalSupply for LP rate modeling.
    """

    FACTORY_ADDRESS = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"

    def __init__(self, w3: Web3):
        self.w3 = w3
        self.factory_abi = self._factory_abi()
        self.pair_abi = self._pair_abi()
        self.factory = self.w3.eth.contract(address=self.FACTORY_ADDRESS, abi=self.factory_abi)

    def _factory_abi(self) -> List[Dict]:
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

    def _pair_abi(self) -> List[Dict]:
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
            {"constant": True, "inputs": [], "name": "token0", "outputs": [{"name": "", "type": "address"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "token1", "outputs": [{"name": "", "type": "address"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
        ]

    async def get_pair_address(self, token0: str, token1: str) -> Optional[str]:
        try:
            pair = self.factory.functions.getPair(token0, token1).call()
            if pair == "0x0000000000000000000000000000000000000000":
                return None
            return pair
        except Exception as e:
            logger.debug(f"V2 LP getPair failed {token0[:6]}-{token1[:6]}: {e}")
            return None

    async def get_pool_reserves(self, pair: str) -> Tuple[int, int, int]:
        try:
            c = self.w3.eth.contract(address=pair, abi=self.pair_abi)
            r0, r1, ts = c.functions.getReserves().call()
            return int(r0), int(r1), int(ts)
        except Exception as e:
            logger.debug(f"V2 LP getReserves failed {pair[:6]}: {e}")
            return 0, 0, 0

    async def get_pool_tokens(self, pair: str) -> Tuple[str, str]:
        try:
            c = self.w3.eth.contract(address=pair, abi=self.pair_abi)
            t0 = c.functions.token0().call()
            t1 = c.functions.token1().call()
            return t0, t1
        except Exception as e:
            logger.debug(f"V2 LP tokens failed {pair[:6]}: {e}")
            return "", ""

    async def get_total_supply(self, pair: str) -> int:
        try:
            c = self.w3.eth.contract(address=pair, abi=self.pair_abi)
            ts = c.functions.totalSupply().call()
            return int(ts)
        except Exception as e:
            logger.debug(f"V2 LP totalSupply failed {pair[:6]}: {e}")
            return 0

