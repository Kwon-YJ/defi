import json
from typing import Dict, Optional, Tuple, List
from web3 import Web3
from src.logger import setup_logger

logger = setup_logger(__name__)


class CurveStableSwapCollector:
    """Curve Finance stableswap collector (minimal, resilient).

    - Uses a static list of known mainnet pools to avoid registry complexity.
    - Computes price via pool.get_dy(i, j, dx) when RPC is available.
    - Falls back to 1.0 price if calls fail or RPC unavailable.
    """

    # Known mainnet pools (3pool) fallback
    FALLBACK_POOLS = [
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
        self.pools: List[Dict] = []
        self._last_refresh = 0
        # Address Provider / Registry (mainnet)
        self.ADDRESS_PROVIDER = "0x0000000022D53366457F9d5E68Ec105046FC4383"
        self.FALLBACK_REGISTRY = "0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5"
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
        self.addr_provider_abi = [
            {"name": "get_registry", "inputs": [], "outputs": [{"type": "address", "name": ""}], "stateMutability": "view", "type": "function"}
        ]
        self.registry_abi = [
            {"name": "pool_count", "inputs": [], "outputs": [{"type": "uint256", "name": ""}], "stateMutability": "view", "type": "function"},
            {"name": "pool_list",  "inputs": [{"type":"uint256","name":"i"}], "outputs": [{"type": "address", "name": ""}], "stateMutability": "view", "type": "function"},
            {"name": "get_coins",  "inputs": [{"type":"address","name":"pool"}], "outputs": [{"type": "address[8]", "name": "coins"}, {"type": "address[8]", "name": "underlying"}], "stateMutability": "view", "type": "function"},
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

    def _registry(self):
        try:
            ap = self.w3.eth.contract(address=self.ADDRESS_PROVIDER, abi=self.addr_provider_abi)
            reg_addr = ap.functions.get_registry().call()
            if reg_addr == "0x0000000000000000000000000000000000000000":
                reg_addr = self.FALLBACK_REGISTRY
            return self.w3.eth.contract(address=reg_addr, abi=self.registry_abi)
        except Exception:
            try:
                return self.w3.eth.contract(address=self.FALLBACK_REGISTRY, abi=self.registry_abi)
            except Exception:
                return None

    def refresh_registry_pools(self, ttl_sec: int = 1800) -> None:
        """Curve Registry에서 모든 풀/메타풀을 탐색하여 pools 캐시에 저장. 실패 시 fallback 유지."""
        import time
        try:
            now = int(time.time())
            if self._last_refresh and now - self._last_refresh < int(ttl_sec):
                return
            if not self.w3 or not self.w3.is_connected():
                # 네트워크 미가용 시 fallback 유지
                if not self.pools:
                    self.pools = list(self.FALLBACK_POOLS)
                return
            reg = self._registry()
            if not reg:
                if not self.pools:
                    self.pools = list(self.FALLBACK_POOLS)
                return
            count = 0
            try:
                count = int(reg.functions.pool_count().call())
            except Exception:
                count = 0
            pools: List[Dict] = []
            for i in range(max(0, count)):
                try:
                    pool = reg.functions.pool_list(i).call()
                    if pool == "0x0000000000000000000000000000000000000000":
                        continue
                    coins = []
                    try:
                        cc = reg.functions.get_coins(pool).call()
                        if isinstance(cc, (list, tuple)):
                            # some registries return (coins, underlying)
                            arr = cc[0] if len(cc) > 0 else cc
                            for t in arr:
                                if isinstance(t, str) and int(t, 16) != 0:
                                    coins.append(t)
                        else:
                            pass
                    except Exception:
                        pass
                    if coins:
                        pools.append({'address': pool, 'coins': coins})
                except Exception:
                    continue
            # 최소 1개라도 성공 시 교체, 아니면 fallback
            if pools:
                self.pools = pools
            elif not self.pools:
                self.pools = list(self.FALLBACK_POOLS)
            self._last_refresh = now
        except Exception as e:
            logger.debug(f"Curve registry refresh 실패: {e}")
            if not self.pools:
                self.pools = list(self.FALLBACK_POOLS)

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
        # 먼저 동적 캐시에서 검색
        for p in self.pools or []:
            coins = [c.lower() for c in p['coins']]
            if a in coins and b in coins:
                i = coins.index(a)
                j = coins.index(b)
                return (p['address'], i, j)
        # 실패 시 fallback 목록에서 검색
        for p in self.FALLBACK_POOLS:
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
