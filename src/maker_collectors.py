from typing import Optional, Dict
from web3 import Web3
from src.logger import setup_logger
from src.dex_data_collector import UniswapV2Collector

logger = setup_logger(__name__)


class MakerCollector:
    """Minimal MakerDAO collector for CDP minting approximation.

    - Focuses on WETH -> DAI minting using a conservative collateralization ratio.
    - Attempts to read WETH price in USDC from Uniswap V2; falls back to 2000 DAI/ETH.
    - Uses a collateralization ratio of 150% and a safety factor (0.95) to avoid edge overstating.
    """

    def __init__(self, w3: Web3, weth: str, dai: str, usdc: Optional[str] = None):
        self.w3 = w3
        self.weth = weth
        self.dai = dai
        self.usdc = usdc or "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
        self.v2 = UniswapV2Collector(w3)
        # Core Maker contracts (mainnet addresses)
        self.VAT = "0x35D1b3F3D7966A1DFe207aa4514C12a259A0492B"
        self.JUG = "0x19c0976f590D67707E62397C87829d896Dc0f1F1"
        self.vat_abi = [
            {
                "name": "ilks",
                "inputs": [{"internalType": "bytes32", "name": "ilk", "type": "bytes32"}],
                "outputs": [
                    {"internalType": "uint256", "name": "Art", "type": "uint256"},
                    {"internalType": "uint256", "name": "rate", "type": "uint256"},
                    {"internalType": "uint256", "name": "spot", "type": "uint256"},
                    {"internalType": "uint256", "name": "line", "type": "uint256"},
                    {"internalType": "uint256", "name": "dust", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        self.jug_abi = [
            {
                "name": "ilks",
                "inputs": [{"internalType": "bytes32", "name": "ilk", "type": "bytes32"}],
                "outputs": [
                    {"internalType": "uint256", "name": "duty", "type": "uint256"},
                    {"internalType": "uint256", "name": "rho", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {"name": "base", "inputs": [], "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"}
        ]
        self.ilk_map: Dict[str, str] = {
            # underlying address(lower) -> ilk bytes32 (as hex string via web3.toBytes)
            # Minimal support for WETH, WBTC
        }

    def _price_dai_per_weth(self) -> float:
        """Estimate DAI per WETH via Uniswap V2 WETH/USDC; fallback to 2000."""
        try:
            pair = self.v2.factory_contract.functions.getPair(self.weth, self.usdc).call()
            if pair and pair != "0x0000000000000000000000000000000000000000":
                r0, r1, _ = self.v2.get_pool_reserves(pair)  # sync call to async method ok for our minimal impl
                if r0 > 0 and r1 > 0:
                    # Determine ordering
                    t0, t1 = self.v2.get_pool_tokens(pair)  # may return empty on error
                    if t0 and t1:
                        if t0.lower() == self.weth.lower() and t1.lower() == self.usdc.lower():
                            # USDC has 6 decimals, WETH 18
                            return (r1 / 1e6) / (r0 / 1e18)
                        elif t0.lower() == self.usdc.lower() and t1.lower() == self.weth.lower():
                            return (r0 / 1e6) / (r1 / 1e18)
        except Exception as e:
            logger.debug(f"Maker price fetch failed; fallback to 2000 DAI/ETH: {e}")
        return 2000.0

    def mintable_dai_per_weth(self, collateral_ratio: float = 1.5, safety_factor: float = 0.95) -> float:
        price = self._price_dai_per_weth()
        if collateral_ratio <= 0:
            collateral_ratio = 1.5
        if safety_factor <= 0 or safety_factor > 1:
            safety_factor = 0.95
        return (price / collateral_ratio) * safety_factor

    # --- OSM/Vat/Jug based parameters ---
    def _to_ilk(self, symbol: str) -> bytes:
        return Web3.to_bytes(text=symbol)

    def get_ilk_params(self, ilk_symbol: str) -> Dict:
        params = {"spot": None, "duty": None, "line": None}
        try:
            if not self.w3 or not self.w3.is_connected():
                return params
            vat = self.w3.eth.contract(address=self.VAT, abi=self.vat_abi)
            jug = self.w3.eth.contract(address=self.JUG, abi=self.jug_abi)
            ilk = self._to_ilk(ilk_symbol)
            v = vat.functions.ilks(ilk).call()
            if isinstance(v, (list, tuple)) and len(v) >= 5:
                params["spot"] = float(v[2]) / 1e27
                params["line"] = float(v[3]) / 1e45
            try:
                duty, _ = jug.functions.ilks(ilk).call()
                base = jug.functions.base().call()
                params["duty"] = (float(duty) + float(base)) / 1e27  # per-second rate in ray → fraction
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"Maker ilk params failed: {e}")
        return params

    def mintable_dai_per_collateral_via_ilk(self, ilk_symbol: str, safety_factor: float = 0.95) -> float:
        p = self.get_ilk_params(ilk_symbol)
        if p.get('spot') and p['spot'] > 0:
            return float(p['spot']) * max(0.0, min(safety_factor, 1.0))
        return 0.0

