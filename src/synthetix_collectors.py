import json
from typing import Dict
from web3 import Web3
from src.logger import setup_logger

logger = setup_logger(__name__)


class SynthetixCollector:
    """Minimal Synthetix collector for synthetic asset exchange.

    - Provides sUSD and sETH addresses (mainnet proxies).
    - Estimates sUSD per sETH using Uniswap V2 WETH/USDC price (fallback 2000).
    - Uses a typical Synthetix exchange fee of ~0.3%.
    """

    SUSD = "0x57ab1e02fee23774580c119740129eac7081e9d3"  # sUSD Proxy (18)
    SETH = "0x5e74c9036fb86bd7ecdcb084a0673efc32ea31cb"  # sETH Proxy (18)
    SNX  = "0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F"  # SNX (18)
    WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

    def __init__(self, w3: Web3):
        self.w3 = w3
        # Minimal ABIs for Uniswap V2 factory/pair to fetch price
        self.v2_factory_address = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"
        self.v2_factory_abi = [
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
        self.v2_pair_abi = [
            {"constant": True, "inputs": [], "name": "getReserves", "outputs": [
                {"name": "reserve0", "type": "uint112"},
                {"name": "reserve1", "type": "uint112"},
                {"name": "blockTimestampLast", "type": "uint32"}
            ], "type": "function"},
            {"constant": True, "inputs": [], "name": "token0", "outputs": [{"name": "", "type": "address"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "token1", "outputs": [{"name": "", "type": "address"}], "type": "function"},
        ]
        self.v2_factory = self.w3.eth.contract(address=self.v2_factory_address, abi=self.v2_factory_abi)

    def get_synths(self) -> Dict[str, str]:
        return {"sUSD": self.SUSD, "sETH": self.SETH, "SNX": self.SNX}

    def get_exchange_fee(self) -> float:
        return 0.003  # 0.3%

    def price_susd_per_seth(self) -> float:
        """Estimate sUSD/1 sETH using V2 WETH/USDC reserves; fallback to 2000."""
        try:
            pair = self.v2_factory.functions.getPair(self.WETH, self.USDC).call()
            if pair and pair != "0x0000000000000000000000000000000000000000":
                c = self.w3.eth.contract(address=pair, abi=self.v2_pair_abi)
                t0 = c.functions.token0().call()
                t1 = c.functions.token1().call()
                r0, r1, _ = c.functions.getReserves().call()
                # USDC 6 decimals, WETH 18
                if t0.lower() == self.WETH.lower() and t1.lower() == self.USDC.lower():
                    return (r1 / 1e6) / (r0 / 1e18)
                elif t0.lower() == self.USDC.lower() and t1.lower() == self.WETH.lower():
                    return (r0 / 1e6) / (r1 / 1e18)
        except Exception as e:
            logger.debug(f"Synthetix price fetch failed; using fallback 2000: {e}")
        return 2000.0

    def _price_tokenB_per_tokenA_v2(self, tokenA: str, tokenB: str, decA: int, decB: int) -> float:
        """Generic Uniswap V2 price for tokenB per 1 tokenA with known decimals"""
        try:
            pair = self.v2_factory.functions.getPair(tokenA, tokenB).call()
            if pair and pair != "0x0000000000000000000000000000000000000000":
                c = self.w3.eth.contract(address=pair, abi=self.v2_pair_abi)
                t0 = c.functions.token0().call()
                t1 = c.functions.token1().call()
                r0, r1, _ = c.functions.getReserves().call()
                if t0.lower() == tokenA.lower() and t1.lower() == tokenB.lower():
                    return (r1 / (10 ** decB)) / (r0 / (10 ** decA))
                elif t0.lower() == tokenB.lower() and t1.lower() == tokenA.lower():
                    return (r0 / (10 ** decB)) / (r1 / (10 ** decA))
        except Exception as e:
            logger.debug(f"V2 price fetch failed {tokenA[:6]}->{tokenB[:6]}: {e}")
        return 0.0

    def price_susd_per_snx(self) -> float:
        """Estimate sUSD per SNX by chaining SNX/WETH * WETH/USDC."""
        try:
            snx_per_weth = self._price_tokenB_per_tokenA_v2(self.WETH, self.SNX, 18, 18)
            # If we queried WETH->SNX, we need SNX per WETH; above returns SNX per WETH directly.
            if snx_per_weth <= 0:
                # Try SNX->WETH
                weth_per_snx = self._price_tokenB_per_tokenA_v2(self.SNX, self.WETH, 18, 18)
                snx_per_weth = (1.0 / weth_per_snx) if weth_per_snx > 0 else 0.0
            usdc_per_weth = self._price_tokenB_per_tokenA_v2(self.WETH, self.USDC, 18, 6)
            if snx_per_weth > 0 and usdc_per_weth > 0:
                usdc_per_snx = usdc_per_weth / snx_per_weth
                return usdc_per_snx  # sUSD ~ USDC
        except Exception as e:
            logger.debug(f"SNX price via V2 failed; fallback 2.0: {e}")
        return 2.0

    def mintable_susd_per_snx(self, collateral_ratio: float = 5.0, safety_factor: float = 0.95) -> float:
        price = self.price_susd_per_snx()
        if collateral_ratio <= 0:
            collateral_ratio = 5.0
        if safety_factor <= 0 or safety_factor > 1:
            safety_factor = 0.95
        return (price / collateral_ratio) * safety_factor

    def unlockable_snx_per_susd(self, collateral_ratio: float = 5.0, safety_factor: float = 0.95) -> float:
        price = self.price_susd_per_snx()
        if price <= 0:
            return 0.0
        if collateral_ratio <= 0:
            collateral_ratio = 5.0
        if safety_factor <= 0 or safety_factor > 1:
            safety_factor = 0.95
        # inverse of mintable ratio
        return (collateral_ratio / (price)) * (1.0 / safety_factor)
