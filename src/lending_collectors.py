import json
from typing import Dict, Optional
from web3 import Web3
from src.logger import setup_logger

logger = setup_logger(__name__)


class CompoundCollector:
    """Minimal Compound v2 collector for exchange rates and cToken mapping.

    Notes:
    - For safety in offline/no-RPC environments, defaults fall back to 1:1 rate.
    - When RPC is configured, attempts to read exchangeRateStored from cTokens.
    """

    # Known mainnet cToken addresses for major assets
    CTOKENS: Dict[str, str] = {
        # DAI, USDC only (stable and reliable)
        # DAI
        "0x6B175474E89094C44Da98b954EedeAC495271d0F": "0x5d3a536E4D6DbD6114cc1Ead35777bAB948E3643",
        # USDC
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "0x39AA39c021dfbaE8faC545936693aC917d5E7563",
    }

    def __init__(self, w3: Web3):
        self.w3 = w3
        self.ctoken_abi = [
            {
                "constant": True,
                "inputs": [],
                "name": "exchangeRateStored",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function",
            }
        ]

    def get_ctoken(self, underlying: str) -> Optional[str]:
        return self.CTOKENS.get(underlying.lower())

    def get_deposit_rate_underlying_to_ctoken(self, ctoken: str) -> float:
        """Return cToken per 1 underlying. Fallback to 1.0 if RPC not available.

        exchangeRateStored ~= underlying per 1 cToken scaled by 1e18;
        cToken per underlying ~= 1e18 / exchangeRateStored
        """
        try:
            if not self.w3 or not self.w3.is_connected():
                return 1.0
            c = self.w3.eth.contract(address=ctoken, abi=self.ctoken_abi)
            ex = c.functions.exchangeRateStored().call()
            ex = float(ex)
            if ex <= 0:
                return 1.0
            return float(1e18) / ex
        except Exception as e:
            logger.debug(f"Compound exchangeRateStored read failed for {ctoken[:6]}: {e}")
            return 1.0


class AaveV2Collector:
    """Minimal Aave v2 collector for aToken mapping.

    - aToken exchange acts 1:1 with underlying (balance accrues interest), so use 1.0.
    - Static mapping for DAI/USDC to avoid registry complexity.
    """

    ATOKENS: Dict[str, str] = {
        # DAI
        "0x6B175474E89094C44Da98b954EedeAC495271d0F": "0x028171bCA77440897B824Ca71D1c56caC55b68A3",
        # USDC
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "0xBcca60bB61934080951369a648Fb03DF4F96263C",
    }

    def __init__(self, w3: Web3):
        self.w3 = w3

    def get_atoken(self, underlying: str) -> Optional[str]:
        return self.ATOKENS.get(underlying.lower())

    def get_deposit_rate_underlying_to_atoken(self, atoken: str) -> float:
        # aToken is 1:1 mapping for deposit/withdraw
        return 1.0

