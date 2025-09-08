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

    SUSD = "0x57ab1e02fee23774580c119740129eac7081e9d3"  # sUSD Proxy
    SETH = "0x5e74c9036fb86bd7ecdcb084a0673efc32ea31cb"  # sETH Proxy
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
        return {"sUSD": self.SUSD, "sETH": self.SETH}

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

