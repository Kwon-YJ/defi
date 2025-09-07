import asyncio
from typing import Dict, List, Set
from src.logger import setup_logger

logger = setup_logger(__name__)

class TokenManager:
    def __init__(self):
        # Define the 25 assets as specified in the paper
        self.assets = self._define_paper_assets()
        self.supported_tokens: Set[str] = set()
        
    def _define_paper_assets(self) -> Dict[str, str]:
        """
        Define the 25 assets as specified in the paper.
        Returns a dictionary of token_symbol -> token_address
        """
        assets = {
            # Native and wrapped Ether
            'ETH': '0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE',
            'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            
            # Stablecoins (mentioned in current implementation)
            'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
            'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
            'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
            
            # Major tokens
            'WBTC': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',
            'UNI': '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984',
            'SUSHI': '0x6B3595068778DD592e39A122f4f5a5cF0d74de3C',
            'COMP': '0xc00e94Cb662C3520282E6f5717214004A7f26888',
            'AAVE': '0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9',
            
            # DeFi ecosystem tokens
            'CRV': '0xD533a949740bb3306d119CC777fa900bA034cd52',
            'BAL': '0xba100000625a3754423978a60c9317c58a424e3D',
            'YFI': '0x0bc529c00C6401aEF6D220BE8C6Ea1667F6Ad93e',
            'MKR': '0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2',
            
            # Lending protocol tokens
            'cETH': '0x4Ddc2D193948926D02f9B1fE9e1daa0718270ED5',  # Compound ETH
            'cUSDC': '0x39AA39c021dfbaE8faC545936693aC917d5E7563',  # Compound USDC
            'aETH': '0xE95A203B1a91a908F9B9CE46459d101078c2c3cb',  # Aave ETH
            'aUSDC': '0x9bA00D6856a4eDF4665BcA2C2309936572473B7E',  # Aave USDC
            
            # Additional tokens from the paper's Uniswap/Bancor lists
            'BNT': '0x1F573D6Fb3F13d689FF844B4cE37794d79a7FF1C',
            'BAT': '0x0D8775F648430679A709E98d2b0Cb6250d2887EF',
            'KNC': '0xdd974D5C2e2928deA02929219732bF4356A500c3',
            'MANA': '0x0F5D2fB29fb7d3CFeE444a200298f468908cC942',
            'GNO': '0x6810e776880C02933D47DB1b9fc05908e5386b96',
            'RLC': '0x607F4C5BB672230e8672085532f7e901544a7375',
            'UBT': '0x8400D94A5cb0fa0D041a3788e395285d61c9ee5e',
            
            # SAI (MakerDAO legacy token)
            'SAI': '0x89d24A6b4CcB1B6fAA2625fE562bDD9a23260359'
        }
        return assets
    
    def get_all_asset_symbols(self) -> List[str]:
        """Get all asset symbols."""
        return list(self.assets.keys())
    
    def get_asset_address(self, symbol: str) -> str:
        """Get the address for a given asset symbol."""
        return self.assets.get(symbol, None)
    
    def get_all_asset_addresses(self) -> Dict[str, str]:
        """Get all asset symbols and their addresses."""
        return self.assets.copy()
    
    def add_supported_token(self, token_symbol: str):
        """Add a token to the supported tokens set."""
        if token_symbol in self.assets:
            self.supported_tokens.add(token_symbol)
            logger.debug(f"Added {token_symbol} to supported tokens")
        else:
            logger.warning(f"Token {token_symbol} not in defined assets list")
    
    def get_supported_tokens(self) -> List[str]:
        """Get the list of currently supported tokens."""
        return list(self.supported_tokens)
    
    def is_token_supported(self, token_symbol: str) -> bool:
        """Check if a token is supported."""
        return token_symbol in self.supported_tokens
    
    def get_token_count(self) -> int:
        """Get the number of supported tokens."""
        return len(self.supported_tokens)

# Example usage:
# token_manager = TokenManager()
# print(f"Total assets defined: {len(token_manager.get_all_asset_symbols())}")
# print(f"Asset addresses: {token_manager.get_all_asset_addresses()}")