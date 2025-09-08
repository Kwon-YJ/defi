from typing import Dict, Optional
from web3 import Web3
from src.logger import setup_logger

logger = setup_logger(__name__)


class YearnV2Collector:
    """Minimal Yearn v2 collector for vault pricePerShare and mapping.

    - Provides a small static mapping of underlying -> yvToken (DAI, USDC).
    - Reads pricePerShare (underlying per 1 yvToken) when RPC available; fallback 1.0.
    """

    VAULTS: Dict[str, str] = {
        # DAI -> yvDAI v2
        "0x6B175474E89094C44Da98b954EedeAC495271d0F": "0x19D3364A399d251E894aC732651be8B0E4e85001",
        # USDC -> yvUSDC v2
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "0x5f18C75AbDAe578b483E5F43f12a39cF75B973a9",
    }

    def __init__(self, w3: Web3):
        self.w3 = w3
        self.vault_abi = [
            {"inputs": [], "name": "pricePerShare", "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
            {"inputs": [], "name": "decimals", "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}], "stateMutability": "view", "type": "function"},
        ]

    def get_vault(self, underlying: str) -> Optional[str]:
        return self.VAULTS.get(underlying.lower())

    def get_shares_per_underlying(self, underlying: str, yv_token: str) -> float:
        """Return yvToken shares per 1 underlying unit.

        pricePerShare ≈ underlying per 1 share, scaled by 1e(decimals).
        shares per underlying ≈ 1 / (pricePerShare / 1e(decimals)).
        Fallback to 1.0 if on-chain read unavailable.
        """
        try:
            if not self.w3 or not self.w3.is_connected():
                return 1.0
            c = self.w3.eth.contract(address=yv_token, abi=self.vault_abi)
            pps = float(c.functions.pricePerShare().call())
            dec = int(c.functions.decimals().call())
            if pps <= 0:
                return 1.0
            underlying_per_share = pps / (10 ** dec)
            return 1.0 / underlying_per_share if underlying_per_share > 0 else 1.0
        except Exception as e:
            logger.debug(f"Yearn pricePerShare read failed {yv_token[:6]}: {e}")
            return 1.0

