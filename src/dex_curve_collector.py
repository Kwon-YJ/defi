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

    # Known mainnet pools fallback (3pool, sUSD pool)
    FALLBACK_POOLS = [
        {
            'address': '0xbEbc44782C7dB0a1A60Cb6fe97d0a2fEdcBcd44',  # 3pool (DAI/USDC/USDT)
            'coins': [
                '0x6B175474E89094C44Da98b954EedeAC495271d0F',  # DAI
                '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',  # USDC
                '0xdAC17F958D2ee523a2206206994597C13D831ec7',  # USDT
            ]
        },
        {
            'address': '0xA5407eAE9Ba41422680e2e00537571bcC53efBfD',  # sUSD pool (DAI/USDC/USDT/sUSD)
            'coins': [
                '0x6B175474E89094C44Da98b954EedeAC495271d0F',  # DAI
                '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',  # USDC
                '0xdAC17F958D2ee523a2206206994597C13D831ec7',  # USDT
                '0x57ab1ec28d129707052df4df418d58a2d46d5f51',  # sUSD
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
            {
                "name": "calc_withdraw_one_coin",
                "inputs": [
                    {"name": "_token_amount", "type": "uint256"},
                    {"name": "i", "type": "int128"}
                ],
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        self.addr_provider_abi = [
            {"name": "get_registry", "inputs": [], "outputs": [{"type": "address", "name": ""}], "stateMutability": "view", "type": "function"}
        ]
        self.registry_abi = [
            {"name": "pool_count", "inputs": [], "outputs": [{"type": "uint256", "name": ""}], "stateMutability": "view", "type": "function"},
            {"name": "pool_list",  "inputs": [{"type":"uint256","name":"i"}], "outputs": [{"type": "address", "name": ""}], "stateMutability": "view", "type": "function"},
            {"name": "get_coins",  "inputs": [{"type":"address","name":"pool"}], "outputs": [{"type": "address[8]", "name": "coins"}, {"type": "address[8]", "name": "underlying"}], "stateMutability": "view", "type": "function"},
            {"name": "get_lp_token", "inputs": [{"type":"address","name":"pool"}], "outputs": [{"type":"address","name":"lp"}], "stateMutability": "view", "type": "function"},
        ]
        # Pool params ABIs (best-effort across pool variants)
        self.pool_params_abi = [
            {"name": "A", "inputs": [], "outputs": [{"type": "uint256", "name": ""}], "stateMutability": "view", "type": "function"},
            {"name": "get_A", "inputs": [], "outputs": [{"type": "uint256", "name": ""}], "stateMutability": "view", "type": "function"},
            {"name": "A_precise", "inputs": [], "outputs": [{"type": "uint256", "name": ""}], "stateMutability": "view", "type": "function"},
            {"name": "fee", "inputs": [], "outputs": [{"type": "uint256", "name": ""}], "stateMutability": "view", "type": "function"},
            {"name": "admin_fee", "inputs": [], "outputs": [{"type": "uint256", "name": ""}], "stateMutability": "view", "type": "function"},
        ]
        try:
            with open('abi/erc20.json', 'r') as f:
                self.erc20_abi = json.load(f)
        except FileNotFoundError:
            self.erc20_abi = [
                {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
                {"constant": True, "inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
                {"constant": True, "inputs": [], "name": "symbol",   "outputs": [{"name": "", "type": "string"}], "type": "function"},
                {"constant": True, "inputs": [], "name": "name",     "outputs": [{"name": "", "type": "string"}], "type": "function"}
            ]
        # pool methods to retrieve LP token
        self.pool_lp_token_abi = [
            {"name": "token", "inputs": [], "outputs": [{"type":"address","name":""}], "stateMutability":"view", "type":"function"},
            {"name": "lp_token", "inputs": [], "outputs": [{"type":"address","name":""}], "stateMutability":"view", "type":"function"},
        ]
        # Pool whitelist from config (addresses only). Empty means allow all.
        try:
            from config.config import config
            wl = getattr(config, 'curve_pool_whitelist', '') or ''
            self.whitelist = {a.strip().lower() for a in wl.split(',') if a.strip()}
        except Exception:
            self.whitelist = set()

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
                    # apply whitelist if set
                    if self.whitelist and pool.lower() not in self.whitelist:
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

    # --- Helpers for coins / indices ---
    def get_pool_coins(self, pool: str) -> List[str]:
        """Return list of active coins for a pool (filtered zero addresses)."""
        # First try from cached pools
        for p in self.pools or []:
            if p.get('address', '').lower() == pool.lower():
                return [c for c in p.get('coins', []) if isinstance(c, str) and int(c, 16) != 0]
        # Fallback: from registry directly
        reg = self._registry()
        if reg:
            try:
                coins, _under = reg.functions.get_coins(pool).call()
                out = []
                for t in coins:
                    if isinstance(t, str) and int(t, 16) != 0:
                        out.append(t)
                if out:
                    return out
            except Exception:
                pass
        # Final fallback: in fallback pools table
        for p in self.FALLBACK_POOLS:
            if p.get('address', '').lower() == pool.lower():
                return [c for c in p.get('coins', [])]
        return []

    def coin_index(self, pool: str, token: str) -> Optional[int]:
        coins = [c.lower() for c in self.get_pool_coins(pool)]
        t = token.lower()
        if t in coins:
            return coins.index(t)
        return None

    def get_lp_token(self, pool: str) -> Optional[str]:
        """Try registry.get_lp_token(pool), or pool.token()/lp_token()."""
        # via registry
        try:
            reg = self._registry()
            if reg:
                lp = reg.functions.get_lp_token(pool).call()
                if isinstance(lp, str) and int(lp, 16) != 0:
                    return lp
        except Exception:
            pass
        # via pool direct methods
        try:
            c = self.w3.eth.contract(address=pool, abi=self.pool_lp_token_abi)
            for name in ("token", "lp_token"):
                try:
                    lp = getattr(c.functions, name)().call()
                    if isinstance(lp, str) and int(lp, 16) != 0:
                        return lp
                except Exception:
                    continue
        except Exception:
            pass
        # fallback: sometimes pool token equals pool address (rare)
        try:
            _ = self.w3.eth.get_code(pool)
            return pool
        except Exception:
            return None

    def get_lp_decimals(self, lp_token: str) -> int:
        try:
            c = self.w3.eth.contract(address=lp_token, abi=self.erc20_abi)
            return int(c.functions.decimals().call())
        except Exception:
            return 18

    def get_lp_total_supply(self, lp_token: str) -> int:
        try:
            c = self.w3.eth.contract(address=lp_token, abi=self.erc20_abi)
            return int(c.functions.totalSupply().call())
        except Exception:
            return 0

    # --- Accurate calc functions ---
    def _calc_token_amount_abi(self, n: int) -> List[Dict]:
        return [{
            "name": "calc_token_amount",
            "inputs": [
                {"name": "amounts", "type": f"uint256[{n}]"},
                {"name": "is_deposit", "type": "bool"}
            ],
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function"
        }]

    def calc_token_amount(self, pool: str, amounts: List[int], is_deposit: bool) -> int:
        """Call calc_token_amount with a statically typed array sized by len(amounts). Fallback safe."""
        try:
            n = max(0, len(amounts))
            if n <= 0:
                return 0
            abi = self._calc_token_amount_abi(n)
            c = self.w3.eth.contract(address=pool, abi=abi)
            return int(c.functions.calc_token_amount(amounts, bool(is_deposit)).call())
        except Exception as e:
            logger.debug(f"Curve calc_token_amount fallback pool {pool[:6]}: {e}")
            # Fallback: sum(amounts) as rough LP (notional)
            try:
                return int(sum(int(x) for x in amounts))
            except Exception:
                return 0

    def calc_withdraw_one_coin(self, pool: str, lp_amount: int, i: int) -> int:
        """Call calc_withdraw_one_coin. Fallback safe."""
        try:
            c = self.w3.eth.contract(address=pool, abi=self.pool_abi)
            return int(c.functions.calc_withdraw_one_coin(int(lp_amount), int(i)).call())
        except Exception as e:
            logger.debug(f"Curve calc_withdraw_one_coin fallback pool {pool[:6]}: {e}")
            return 0

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

    def get_pool_params(self, pool: str) -> Dict:
        """Fetch A, fee, admin_fee (best-effort). Returns normalized values: A(int), fee(float), admin_fee(float)."""
        params = {"A": None, "fee": None, "admin_fee": None}
        try:
            if not self.w3 or not self.w3.is_connected():
                return params
            c = self.w3.eth.contract(address=pool, abi=self.pool_params_abi)
            A = 0
            for name in ("A", "get_A", "A_precise"):
                try:
                    A = int(getattr(c.functions, name)().call())
                    if name == "A_precise" and A > 0:
                        # many pools: A_precise = A * 100
                        if A % 100 == 0:
                            A = A // 100
                    break
                except Exception:
                    continue
            params["A"] = int(A) if A else None
            fee = None
            admin = None
            try:
                fee = int(c.functions.fee().call())
            except Exception:
                fee = None
            try:
                admin = int(c.functions.admin_fee().call())
            except Exception:
                admin = None
            # Normalize to fraction if looks like 1e10-denominated
            def _norm(x):
                if x is None:
                    return None
                # Heuristic: common denom is 1e10 for stableswap
                if x > 1_000_000:
                    return float(x) / 1e10
                # If already small, assume it's a fraction
                if x <= 1_000:
                    return float(x)
                return float(x) / 1e10
            params["fee"] = _norm(fee)
            params["admin_fee"] = _norm(admin)
        except Exception as e:
            logger.debug(f"Curve pool params fetch failed {pool[:6]}: {e}")
        return params
