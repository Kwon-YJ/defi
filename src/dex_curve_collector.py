import json
from typing import Dict, Optional, Tuple
from web3 import Web3
from src.logger import setup_logger

logger = setup_logger(__name__)


class CurveStableSwapCollector:
    """Curve Finance stableswap collector (minimal, resilient).

    - Uses a static list of known mainnet pools to avoid registry complexity.
    - Computes price via pool.get_dy(i, j, dx) when RPC is available.
    - Falls back to 1.0 price if calls fail or RPC unavailable.
    """

    # Known mainnet pools (3pool)
    POOLS = [
        {
            'address': '0xbEbc44782C7dB0a1A60Cb6fe97d0a2fEdcBcd44',  # 3pool (DAI/USDC/USDT)
            'coins': [
                '0x6B175474E89094C44Da98b954EedeAC495271d0F',  # DAI
                '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',  # USDC
                '0xdAC17F958D2ee523a2206206994597C13D831ec7',  # USDT
            ]
        }
    ]

    def __init__(self, w3: Web3):
        self.w3 = w3
        # Minimal ABIs
        self.pool_abi = [
            {
                "name": "get_dy",
                "inputs": [
                    {"name": "i", "type": "int128"},
                    {"name": "j", "type": "int128"},
                    {"name": "dx", "type": "uint256"}
                ],
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
        ]
        try:
            with open('abi/erc20.json', 'r') as f:
                self.erc20_abi = json.load(f)
        except FileNotFoundError:
            self.erc20_abi = [
                {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
                {"constant": True, "inputs": [], "name": "symbol",   "outputs": [{"name": "", "type": "string"}], "type": "function"},
                {"constant": True, "inputs": [], "name": "name",     "outputs": [{"name": "", "type": "string"}], "type": "function"}
            ]

    def _decimals(self, token: str) -> int:
        try:
            if not self.w3 or not self.w3.is_connected():
                return 18
            c = self.w3.eth.contract(address=token, abi=self.erc20_abi)
            return int(c.functions.decimals().call())
        except Exception:
            return 18

    def find_pool_for_pair(self, tokenA: str, tokenB: str) -> Optional[Tuple[str, int, int]]:
        """Return (pool_address, i, j) if a known pool contains both tokens."""
        a, b = tokenA.lower(), tokenB.lower()
        for p in self.POOLS:
            coins = [c.lower() for c in p['coins']]
            if a in coins and b in coins:
                i = coins.index(a)
                j = coins.index(b)
                return (p['address'], i, j)
        return None

    def get_price(self, pool: str, i: int, j: int, token_i: str, token_j: str) -> float:
        """Compute token_j/token_i price via get_dy over 1 unit of token_i."""
        try:
            if not self.w3 or not self.w3.is_connected():
                return 1.0
            dec_i = self._decimals(token_i)
            dec_j = self._decimals(token_j)
            dx = 10 ** dec_i
            c = self.w3.eth.contract(address=pool, abi=self.pool_abi)
            dy = c.functions.get_dy(i, j, dx).call()
            if dy <= 0:
                return 0.0
            return float(dy) / float(10 ** dec_j)
        except Exception as e:
            logger.debug(f"Curve get_dy failed pool {pool[:6]} i={i} j={j}: {e}")
            return 1.0

