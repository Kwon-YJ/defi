import json
from typing import Dict, Optional, Tuple
from web3 import Web3
from src.logger import setup_logger
from config.config import config

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
        self.comptroller_address = getattr(config, 'compound_comptroller', '')
        self.ctoken_abi = [
            {
                "constant": True,
                "inputs": [],
                "name": "exchangeRateStored",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function",
            },
            {"constant": True, "inputs": [], "name": "borrowRatePerBlock", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "supplyRatePerBlock", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "reserveFactorMantissa", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
        ]
        self.comptroller_abi = [
            {
                "constant": True,
                "inputs": [{"name": "cTokenAddress", "type": "address"}],
                "name": "markets",
                "outputs": [
                    {"name": "isListed", "type": "bool"},
                    {"name": "collateralFactorMantissa", "type": "uint256"},
                    {"name": "isComped", "type": "bool"}
                ],
                "type": "function"
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

    def get_rates_per_block(self, ctoken: str) -> Dict:
        try:
            if not self.w3 or not self.w3.is_connected():
                return {}
            c = self.w3.eth.contract(address=ctoken, abi=self.ctoken_abi)
            br = float(c.functions.borrowRatePerBlock().call()) / 1e18
            sr = float(c.functions.supplyRatePerBlock().call()) / 1e18
            rf = float(c.functions.reserveFactorMantissa().call()) / 1e18
            return {"borrowRatePerBlock": br, "supplyRatePerBlock": sr, "reserveFactor": rf}
        except Exception as e:
            logger.debug(f"Compound rates read failed for {ctoken[:6]}: {e}")
            return {}

    def get_collateral_factor(self, ctoken: str) -> Optional[float]:
        try:
            if not self.w3 or not self.w3.is_connected() or not self.comptroller_address:
                return None
            comp = self.w3.eth.contract(address=self.comptroller_address, abi=self.comptroller_abi)
            _, cf, _ = comp.functions.markets(ctoken).call()
            return float(cf) / 1e18
        except Exception as e:
            logger.debug(f"Compound collateral factor read failed {ctoken[:6]}: {e}")
            return None

    def approx_interest_penalty(self, ctoken: str, hold_blocks: int) -> float:
        rates = self.get_rates_per_block(ctoken) or {}
        br = float(rates.get('borrowRatePerBlock', 0.0))
        try:
            blocks = max(0, int(hold_blocks))
        except Exception:
            blocks = 0
        # simple compounding approximation
        penalty = (1.0 + br) ** blocks - 1.0 if br > 0 and blocks > 0 else 0.0
        return max(0.0, penalty)


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
        # Aave v2 Protocol Data Provider
        self.data_provider_address = getattr(config, 'aave_v2_data_provider', '')
        self.data_provider_abi = [
            {
                "inputs": [{"internalType": "address", "name": "asset", "type": "address"}],
                "name": "getReserveTokensAddresses",
                "outputs": [
                    {"internalType": "address", "name": "aTokenAddress", "type": "address"},
                    {"internalType": "address", "name": "stableDebtTokenAddress", "type": "address"},
                    {"internalType": "address", "name": "variableDebtTokenAddress", "type": "address"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        self.data_provider_rates_abi = [
            {
                "inputs": [{"internalType": "address", "name": "asset", "type": "address"}],
                "name": "getReserveData",
                "outputs": [
                    {"internalType": "uint256", "name": "availableLiquidity", "type": "uint256"},
                    {"internalType": "uint256", "name": "totalStableDebt", "type": "uint256"},
                    {"internalType": "uint256", "name": "totalVariableDebt", "type": "uint256"},
                    {"internalType": "uint256", "name": "liquidityRate", "type": "uint256"},
                    {"internalType": "uint256", "name": "variableBorrowRate", "type": "uint256"},
                    {"internalType": "uint256", "name": "stableBorrowRate", "type": "uint256"},
                    {"internalType": "uint256", "name": "averageStableBorrowRate", "type": "uint256"},
                    {"internalType": "uint256", "name": "liquidityIndex", "type": "uint256"},
                    {"internalType": "uint256", "name": "variableBorrowIndex", "type": "uint256"},
                    {"internalType": "uint40",  "name": "lastUpdateTimestamp", "type": "uint40"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "address", "name": "asset", "type": "address"}],
                "name": "getReserveConfigurationData",
                "outputs": [
                    {"internalType": "uint256", "name": "decimals", "type": "uint256"},
                    {"internalType": "uint256", "name": "ltv", "type": "uint256"},
                    {"internalType": "uint256", "name": "liquidationThreshold", "type": "uint256"},
                    {"internalType": "uint256", "name": "liquidationBonus", "type": "uint256"},
                    {"internalType": "uint256", "name": "reserveFactor", "type": "uint256"},
                    {"internalType": "bool",    "name": "usageAsCollateralEnabled", "type": "bool"},
                    {"internalType": "bool",    "name": "borrowingEnabled", "type": "bool"},
                    {"internalType": "bool",    "name": "stableBorrowRateEnabled", "type": "bool"},
                    {"internalType": "bool",    "name": "isActive", "type": "bool"},
                    {"internalType": "bool",    "name": "isFrozen", "type": "bool"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        # Optional Aave v3 Data Provider for eMode
        self.v3_data_provider_address = getattr(config, 'aave_v3_data_provider', '')
        self.v3_data_provider_abi = [
            {
                "inputs": [{"internalType": "address", "name": "asset", "type": "address"}],
                "name": "getReserveEModeCategory",
                "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]

    def get_atoken(self, underlying: str) -> Optional[str]:
        # Try DataProvider first
        try:
            addrs = self.get_reserve_tokens(underlying)
            if addrs and addrs[0] and int(addrs[0], 16) != 0:
                return addrs[0]
        except Exception:
            pass
        return self.ATOKENS.get(underlying.lower())

    def get_deposit_rate_underlying_to_atoken(self, atoken: str) -> float:
        # aToken is 1:1 mapping for deposit/withdraw
        return 1.0

    def get_reserve_tokens(self, underlying: str) -> Optional[Tuple[str, str, str]]:
        """Return (aToken, stableDebt, variableDebt) using DataProvider; None on failure."""
        try:
            if not self.w3 or not self.w3.is_connected():
                return None
            if not self.data_provider_address:
                return None
            c = self.w3.eth.contract(address=self.data_provider_address, abi=self.data_provider_abi)
            a, sd, vd = c.functions.getReserveTokensAddresses(underlying).call()
            return a, sd, vd
        except Exception as e:
            logger.debug(f"Aave DataProvider read failed for {underlying[:6]}: {e}")
            return None

    def get_reserve_rates(self, underlying: str) -> Optional[Dict]:
        try:
            if not self.w3 or not self.w3.is_connected() or not self.data_provider_address:
                return None
            c = self.w3.eth.contract(address=self.data_provider_address, abi=self.data_provider_rates_abi)
            data = c.functions.getReserveData(underlying).call()
            if not isinstance(data, (list, tuple)) or len(data) < 10:
                return None
            liq_rate = float(data[3]) / 1e27
            var_rate = float(data[4]) / 1e27
            st_rate = float(data[5]) / 1e27
            return {"liquidityRate": liq_rate, "variableBorrowRate": var_rate, "stableBorrowRate": st_rate}
        except Exception as e:
            logger.debug(f"Aave getReserveData failed {underlying[:6]}: {e}")
            return None

    def get_reserve_configuration(self, underlying: str) -> Optional[Dict]:
        try:
            if not self.w3 or not self.w3.is_connected() or not self.data_provider_address:
                return None
            c = self.w3.eth.contract(address=self.data_provider_address, abi=self.data_provider_rates_abi)
            cfg = c.functions.getReserveConfigurationData(underlying).call()
            if not isinstance(cfg, (list, tuple)) or len(cfg) < 10:
                return None
            return {
                "decimals": int(cfg[0]),
                "ltv": float(cfg[1]) / 10000.0,
                "liquidationThreshold": float(cfg[2]) / 10000.0,
                "liquidationBonus": float(cfg[3]) / 10000.0,
                "reserveFactor": float(cfg[4]) / 10000.0,
                "usageAsCollateralEnabled": bool(cfg[5]),
                "borrowingEnabled": bool(cfg[6]),
                "stableBorrowRateEnabled": bool(cfg[7]),
                "isActive": bool(cfg[8]),
                "isFrozen": bool(cfg[9]),
            }
        except Exception as e:
            logger.debug(f"Aave getReserveConfigurationData failed {underlying[:6]}: {e}")
            return None

    def get_emode_category(self, underlying: str) -> Optional[int]:
        try:
            if not self.w3 or not self.w3.is_connected() or not self.v3_data_provider_address:
                return None
            c = self.w3.eth.contract(address=self.v3_data_provider_address, abi=self.v3_data_provider_abi)
            cat = c.functions.getReserveEModeCategory(underlying).call()
            try:
                return int(cat)
            except Exception:
                return None
        except Exception as e:
            logger.debug(f"Aave v3 eMode read failed {underlying[:6]}: {e}")
            return None
