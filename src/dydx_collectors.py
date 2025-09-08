import json
from typing import Dict, Optional, Tuple
from web3 import Web3
from src.logger import setup_logger

logger = setup_logger(__name__)


class DyDxCollector:
    """Minimal dYdX spot-price helper for margin action modeling.

    - Uses Uniswap V2 WETH/USDC for price proxy (fallback 2000).
    - Provides basic fee approximation to reflect trading costs.
    - This is an approximation to surface margin trading edges in the graph.
    """

    WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
    DAI  = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
    USDT = "0xdAC17F958D2ee523a2206206994597C13D831ec7"

    def __init__(self, w3: Web3):
        self.w3 = w3
        self.v2_factory = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"
        self.factory_abi = [
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
        self.pair_abi = [
            {"constant": True, "inputs": [], "name": "getReserves", "outputs": [
                {"name": "reserve0", "type": "uint112"},
                {"name": "reserve1", "type": "uint112"},
                {"name": "blockTimestampLast", "type": "uint32"}
            ], "type": "function"},
            {"constant": True, "inputs": [], "name": "token0", "outputs": [{"name": "", "type": "address"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "token1", "outputs": [{"name": "", "type": "address"}], "type": "function"},
        ]
        self.factory = self.w3.eth.contract(address=self.v2_factory, abi=self.factory_abi)

    def _price_usdc_per_weth(self) -> float:
        try:
            pair = self.factory.functions.getPair(self.WETH, self.USDC).call()
            if pair and pair != "0x0000000000000000000000000000000000000000":
                c = self.w3.eth.contract(address=pair, abi=self.pair_abi)
                t0 = c.functions.token0().call()
                t1 = c.functions.token1().call()
                r0, r1, _ = c.functions.getReserves().call()
                if t0.lower() == self.WETH.lower() and t1.lower() == self.USDC.lower():
                    return (r1 / 1e6) / (r0 / 1e18)
                elif t0.lower() == self.USDC.lower() and t1.lower() == self.WETH.lower():
                    return (r0 / 1e6) / (r1 / 1e18)
        except Exception as e:
            logger.debug(f"dYdX price fetch failed; fallback 2000: {e}")
        return 2000.0

    def get_price_and_fee(self, token0: str, token1: str) -> Tuple[float, float]:
        """Return (price token1 per 1 token0, fee fraction). Supports WETH/USDC and stables.

        Fee approximated at 0.1% (0.001) to reflect trading and funding costs.
        """
        fee = 0.001
        a = token0.lower(); b = token1.lower()
        # Stable pairs valued at ~1
        stables = {self.USDC.lower(), self.DAI.lower(), self.USDT.lower()}
        if a in stables and b in stables:
            return 1.0, fee
        # WETH vs USDC
        if a == self.WETH.lower() and b == self.USDC.lower():
            return self._price_usdc_per_weth(), fee
        if a == self.USDC.lower() and b == self.WETH.lower():
            p = self._price_usdc_per_weth()
            return (1.0 / p) if p > 0 else 0.0, fee
        # Unsupported pairs
        return 0.0, fee

