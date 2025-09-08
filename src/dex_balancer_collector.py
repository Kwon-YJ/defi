import json
from typing import Dict, Optional, Tuple
from src.slippage import amount_out_balancer_weighted
from config.config import config
from web3 import Web3
from src.logger import setup_logger

logger = setup_logger(__name__)


class BalancerWeightedCollector:
    """Balancer V2 Weighted Pool collector (minimal, resilient).

    - Static list of representative weighted pools using core tokens (WETH, USDC, DAI).
    - Tries to compute spot price via on-chain Vault/Pool if RPC available; falls back to 1.0.
    - Provides pool fee when readable; else uses a reasonable default (0.1%).
    """

    # Representative (placeholder) pools for integration; replace/extend with actual pools as needed
    POOLS = [
        {
            'address': '0x1111111111111111111111111111111111111111',  # placeholder
            'tokens': [
                '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',  # WETH
                '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',  # USDC
            ],
        },
        {
            'address': '0x2222222222222222222222222222222222222222',  # placeholder
            'tokens': [
                '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',  # WETH
                '0x6B175474E89094C44Da98b954EedeAC495271d0F',  # DAI
            ],
        },
    ]

    def __init__(self, w3: Web3):
        self.w3 = w3
        # Minimal ABIs
        self.pool_abi = [
            {"inputs": [], "name": "getNormalizedWeights", "outputs": [{"internalType": "uint256[]", "name": "", "type": "uint256[]"}], "stateMutability": "view", "type": "function"},
            {"inputs": [], "name": "getSwapFeePercentage", "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
            {"inputs": [], "name": "getVault", "outputs": [{"internalType": "address", "name": "", "type": "address"}], "stateMutability": "view", "type": "function"},
            {"inputs": [], "name": "getPoolId", "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}], "stateMutability": "view", "type": "function"},
        ]
        self.vault_abi = [
            {
                "inputs": [{"internalType": "bytes32", "name": "poolId", "type": "bytes32"}],
                "name": "getPoolTokens",
                "outputs": [
                    {"internalType": "address[]", "name": "tokens", "type": "address[]"},
                    {"internalType": "uint256[]", "name": "balances", "type": "uint256[]"},
                    {"internalType": "uint256", "name": "lastChangeBlock", "type": "uint256"},
                ],
                "stateMutability": "view",
                "type": "function"
            }
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

    def _decimals(self, token: str) -> int:
        try:
            if not self.w3 or not self.w3.is_connected():
                return 18
            c = self.w3.eth.contract(address=token, abi=self.erc20_abi)
            return int(c.functions.decimals().call())
        except Exception:
            return 18

    def _total_supply(self, token: str) -> int:
        try:
            if not self.w3 or not self.w3.is_connected():
                return 0
            c = self.w3.eth.contract(address=token, abi=self.erc20_abi)
            return int(c.functions.totalSupply().call())
        except Exception:
            return 0

    def find_pool_for_pair(self, tokenA: str, tokenB: str) -> Optional[str]:
        a, b = tokenA.lower(), tokenB.lower()
        for p in self.POOLS:
            ts = [t.lower() for t in p['tokens']]
            if a in ts and b in ts:
                return p['address']
        return None

    def get_weights_and_balances(self, pool_addr: str, token0: str, token1: str) -> Tuple[float, float, float, float]:
        """Return (w_in, w_out, b_in, b_out) normalized (weights sum to 1, balances in human units).

        Falls back to (0.5, 0.5, 0.0, 0.0) on failure.
        """
        try:
            if not self.w3 or not self.w3.is_connected():
                return 0.5, 0.5, 0.0, 0.0
            pool = self.w3.eth.contract(address=pool_addr, abi=self.pool_abi)
            vault_addr = pool.functions.getVault().call()
            pool_id = pool.functions.getPoolId().call()
            weights = pool.functions.getNormalizedWeights().call()  # 1e18 scale
            vault = self.w3.eth.contract(address=vault_addr, abi=self.vault_abi)
            tokens, balances, _ = vault.functions.getPoolTokens(pool_id).call()
            tokens_l = [t.lower() for t in tokens]
            i = tokens_l.index(token0.lower())
            j = tokens_l.index(token1.lower())
            wi = float(weights[i]) / 1e18
            wj = float(weights[j]) / 1e18
            di = 10 ** self._decimals(token0)
            dj = 10 ** self._decimals(token1)
            bi = float(balances[i]) / di
            bj = float(balances[j]) / dj
            return wi, wj, bi, bj
        except Exception as e:
            logger.debug(f"Balancer weights/balances failed {pool_addr[:6]}: {e}")
            return 0.5, 0.5, 0.0, 0.0

    def get_spot_price_and_fee(self, pool_addr: str, token0: str, token1: str) -> Tuple[float, float]:
        """Return (price1_per_0, fee_fraction). Fallback to (1.0, 0.001)."""
        try:
            if not self.w3 or not self.w3.is_connected():
                return 1.0, 0.001
            pool = self.w3.eth.contract(address=pool_addr, abi=self.pool_abi)
            vault_addr = pool.functions.getVault().call()
            pool_id = pool.functions.getPoolId().call()
            weights = pool.functions.getNormalizedWeights().call()  # 1e18 scale
            fee_pct = pool.functions.getSwapFeePercentage().call()  # 1e18 scale
            vault = self.w3.eth.contract(address=vault_addr, abi=self.vault_abi)
            tokens, balances, _ = vault.functions.getPoolTokens(pool_id).call()

            # Map indices
            tokens_l = [t.lower() for t in tokens]
            i = tokens_l.index(token0.lower())
            j = tokens_l.index(token1.lower())
            wi = float(weights[i]) / 1e18
            wj = float(weights[j]) / 1e18
            bi_raw = float(balances[i])
            bj_raw = float(balances[j])
            di = 10 ** self._decimals(token0)
            dj = 10 ** self._decimals(token1)
            bi = bi_raw / di
            bj = bj_raw / dj
            # Spot price without fee
            price = (bj / wj) / (bi / wi) if bi > 0 and wi > 0 and wj > 0 else 0.0
            fee_frac = float(fee_pct) / 1e18
            return (price if price > 0 else 1.0), (fee_frac if fee_frac > 0 else 0.001)
        except Exception as e:
            logger.debug(f"Balancer spot price read failed {pool_addr[:6]}: {e}")
            return 1.0, 0.001

    def effective_rate_for_fraction(self, pool_addr: str, token0: str, token1: str,
                                    trade_fraction: Optional[float] = None) -> Tuple[float, float, float, float, float]:
        """Compute effective price for a fractional trade size of token0 reserves.

        Returns: (eff_rate, fee, wi, wj, bi)
        """
        wi, wj, bi, bj = self.get_weights_and_balances(pool_addr, token0, token1)
        price, fee = self.get_spot_price_and_fee(pool_addr, token0, token1)
        f = trade_fraction if trade_fraction is not None else float(getattr(config, 'slippage_trade_fraction', 0.01))
        try:
            f = max(1e-6, min(float(f), 0.5))
        except Exception:
            f = 0.01
        if bi <= 0 or bj <= 0 or wi <= 0 or wj <= 0:
            return price, fee, wi, wj, bi
        amount_in = float(bi) * f
        out = amount_out_balancer_weighted(amount_in, bi, bj, wi, wj, fee)
        eff = (out / amount_in) if amount_in > 0 and out > 0 else price
        return eff, fee, wi, wj, bi

    # --- Weighted join/exit math (single asset) ---
    def get_bpt_info(self, pool_addr: str) -> Tuple[int, int]:
        """Return (bpt_decimals, total_supply_raw)."""
        dec = self._decimals(pool_addr)
        ts = self._total_supply(pool_addr)
        return int(dec), int(ts)

    def bpt_out_per_token_in(self, pool_addr: str, token_in: str, amount_in_hu: float) -> float:
        """Single-asset join: BPT out per amount_in (human units)."""
        try:
            wi, _, bi, _ = self.get_weights_and_balances(pool_addr, token_in, token_in)  # hack to get wi, bi ignoring out
        except Exception:
            wi, bi = 0.5, 0.0
        _, fee = self.get_spot_price_and_fee(pool_addr, token_in, token_in)
        bpt_dec, ts_raw = self.get_bpt_info(pool_addr)
        ts = float(ts_raw) / float(10 ** bpt_dec) if bpt_dec else float(ts_raw) / 1e18
        if wi <= 0 or bi <= 0 or ts <= 0:
            return 0.0
        # fee applied to non-proportional part: amount_in_after_fee = amount*(1 - fee*(1-wi))
        amount_after_fee = float(amount_in_hu) * (1.0 - float(fee) * (1.0 - float(wi)))
        ratio = (1.0 + (amount_after_fee / float(bi))) ** float(wi) - 1.0
        bpt_out = ts * ratio
        return max(0.0, float(bpt_out))

    def token_out_per_bpt_in(self, pool_addr: str, token_out: str, bpt_in_hu: float) -> float:
        """Single-asset exit: token out per BPT in (human units)."""
        try:
            wi, _, bi, _ = self.get_weights_and_balances(pool_addr, token_out, token_out)  # wi of token_out
        except Exception:
            wi, bi = 0.5, 0.0
        _, fee = self.get_spot_price_and_fee(pool_addr, token_out, token_out)
        bpt_dec, ts_raw = self.get_bpt_info(pool_addr)
        ts = float(ts_raw) / float(10 ** bpt_dec) if bpt_dec else float(ts_raw) / 1e18
        if wi <= 0 or bi <= 0 or ts <= 0:
            return 0.0
        frac = float(bpt_in_hu) / ts
        if frac <= 0 or frac >= 1:
            return 0.0
        gross = float(bi) * (1.0 - (1.0 - frac) ** (1.0 / float(wi)))
        net = gross * (1.0 - float(fee) * (1.0 - float(wi)))
        return max(0.0, float(net))
