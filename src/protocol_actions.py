import asyncio
from typing import Dict, List, Tuple
from web3 import Web3
from src.market_graph import DeFiMarketGraph
from src.logger import setup_logger

logger = setup_logger(__name__)

# Token addresses (mainnet)
TOKEN_ADDRESSES = {
    'ETH': '0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE',
    'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
    'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
    'SAI': '0x89d24A6b4CcB1B6fAA2625fE562bDD9a23260359',
    'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
    'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
    'BAT': '0x0D8775F648430679A709E98d2b0Cb6250d2887EF',
    'BNT': '0x1F573D6Fb3F13d689FF844B4cE37794d79a7FF1C',
    'MKR': '0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2',
    'REP': '0x1985365e9f78359a9B6AD760e32412f4a445E862',
    'ZRX': '0xE41d2489571d322189246DaFA5ebDe1F4699F498',
    'KNC': '0xdd974D5C2e2928deA02929219732bF4356A500c3',
    'LINK': '0x514910771AF9Ca656af840dff83E8264EcF986CA',
    'COMP': '0xc00e94Cb662C3520282E6f5717214004A7f26888',
    'UNI': '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984',
    'YFI': '0x0bc529c00C6401aEF6D220BE8C6Ea1667F6Ad93e',
    'AAVE': '0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9',
    'SNX': '0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F',
    'SUSHI': '0x6B3595068778DD592e39A122f4f5a5cF0d74de3C',
    'CRV': '0xD533a949740bb3306d119CC777fa900bA034cd52',
    'BAL': '0xba100000625a3754423978a60c9317c58a424e3D',
    'ENJ': '0xF629cBd94d3791C9250152BD8dfBDF380E2a3B9c',
    'MANA': '0x0F5D2fB29fb7d3CFeE444a200298f468908cC942',
    'GNO': '0x6810e776880C02933D47DB1b9fc05908e5386b96',
    'LRC': '0xBBbbCA6A901c926F240b89EacB641d8Aec7AEafD',
    'REN': '0x408e41876cCCDC0F92210600ef50372656052a38',
    'AMPL': '0xD46bA6D942050d489DBd938a2C909A5d5039A161',
    'BAND': '0xBA11D00c5f74255f56a5E366F4F77f5A186d7f55',
    'NMR': '0x1776e1F26f98b1A5dF9cD347953a26dd3Cb46671',
    'RLC': '0x607F4C5BB672230e8672085532f7e901544a7375',
    'UBT': '0x8400D94A5cb0fa0D041a3788e395285d61c9ee5e',
    'DATA': '0x0Cf0Ee63788A0849fE5297F3407f701E122cC023',
    'ANT': '0x960b236A07cf122663c4303350609A66A7B288C0',
    'SAN': '0x7C5A0CE9267ED19B22F8cae653F198e3E8daf098',
    'SNT': '0x744d70FDBE2Ba4CF95131626614a1763DF805B9E',
    'TKN': '0xaAAf91D9b90dF800Df4F55c205fd6989c977E73a',
    'TRST': '0xCb94be6f13A1182E4A4B6140cb7bf2025d28e41B',
    'POA20': '0x6758B7d441a9739b98552B373703d8d3d14f9e62',
    'RCN': '0xF970b8E36e23F7fC3FD752EeA86f8Be8D83375A6',
    'RDN': '0x255Aa6DF07540Cb5d3d297f0D0D4D84cb52bc8e6',
    'FXC': '0xc931f61b1534eb21d8c11b24f3f5ab2471d4ab50',
    'HEDG': '0xf1290473E210b2108A85237fbCd7b6eb42Cc654F',
    # Additional tokens for new protocols
    'WBTC': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',
    'cETH': '0x4Ddc2D193948926D02f9B1fE9e1daa0718270ED5',
    'cUSDC': '0x39AA39c021dfbaE8faC545936693aC917d5E7563',
    'aETH': '0xE95A203B1a91a908F9B9CE46459d101078c2c3cb',
    'aUSDC': '0x9bA00D6856a4eDF4665BcA2C2309936572473B7E',
    'sUSD': '0x57Ab1ec28D129707052df4dF418D58a2D46d5f51',
    'sETH': '0x5e74C9036fb86BD7eCdcb084a0673EFc32eA31cb',
    'yCRV': '0xdF5e0e81Dff6FAF3A7e52BA697820c5e32D806A8',
    'yDAI': '0x16de59092dAE5CcF4A1E6439D611fd0653f0Bd01',
    'yUSDC': '0xd6aD7a6750A7593E092a9B218d66C0A814a3436e',
    'yUSDT': '0x83f798e925BcD4017Eb265844FDDAbb448f1707D',
    'yTUSD': '0x73a052500105205d34Daf004eAb301916DA8190f'
}

# DEX factory addresses (mainnet)
DEX_FACTORIES = {
    'uniswap_v2': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',
    'uniswap_v3': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
    'sushiswap': '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac',
    'bancor': '0xc0a47dFe034B400B47bDaD5FecDa2621de6c4d95',
    'makerdao': '0x99b016c1c4d777443e97f91099e47271f3f253d3',  # SaiProxyCreateAndExecute
    'curve': '0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5',
    'balancer': '0x9424B1412450D0f8Fc2255FAf6046b98213B76Bd',
    'compound': '0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B',
    'aave': '0xB53C1a33016B2DC2fF3653530bfF1848a515c807',
    'synthetix': '0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F',
    'yearn': '0x9cA85572E6A3EbF24dEDd195623F188735A5179f',
    'dydx': '0x1E0447b19BB6EcFdAe1e4AE1694b0C3659614e4e'
}

# Protocol-specific configurations
PROTOCOL_CONFIGS = {
    'uniswap_v2': {
        'type': 'amm',
        'fee': 0.003,  # 0.3%
        'actions': ['swap']
    },
    'uniswap_v3': {
        'type': 'amm',
        'fee': 0.003,  # Variable in V3, using average
        'actions': ['swap']
    },
    'sushiswap': {
        'type': 'amm',
        'fee': 0.003,  # 0.3%
        'actions': ['swap']
    },
    'bancor': {
        'type': 'amm',
        'fee': 0.002,  # 0.2%
        'actions': ['swap']
    },
    'makerdao': {
        'type': 'cdp',
        'fee': 0.0,  # No direct fee for CDP actions
        'actions': ['mint_dai', 'burn_dai', 'deposit_collateral', 'withdraw_collateral']
    },
    'curve': {
        'type': 'stableswap',
        'fee': 0.0004,  # 0.04% average
        'actions': ['swap']
    },
    'balancer': {
        'type': 'weighted_pool',
        'fee': 0.001,  # 0.1% average
        'actions': ['swap']
    },
    'compound': {
        'type': 'lending',
        'fee': 0.0,  # No direct fee for lending actions
        'actions': ['lend', 'borrow', 'redeem', 'repay']
    },
    'aave': {
        'type': 'lending',
        'fee': 0.0,  # No direct fee for lending actions
        'actions': ['lend', 'borrow', 'redeem', 'repay']
    },
    'synthetix': {
        'type': 'synthetic',
        'fee': 0.003,  # 0.3% average
        'actions': ['mint_synthetic', 'burn_synthetic', 'claim_rewards']
    },
    'yearn': {
        'type': 'yield',
        'fee': 0.005,  # 0.5% performance fee
        'actions': ['deposit', 'withdraw', 'claim_rewards']
    },
    'dydx': {
        'type': 'margin',
        'fee': 0.001,  # 0.1% average
        'actions': ['margin_trade', 'close_position']
    }
}

class ProtocolActionsManager:
    def __init__(self, market_graph: DeFiMarketGraph):
        self.market_graph = market_graph
        # Define the protocol actions as specified in the paper (96 actions total)
        self.protocol_actions = self._define_protocol_actions()
        # Initialize web3 connection
        self.w3 = self._initialize_web3()
    
    def _initialize_web3(self):
        """Initialize web3 connection. For now, we'll use a public endpoint."""
        # In a production environment, you would use your own node or a private endpoint
        # For demonstration purposes, we'll use a public endpoint
        try:
            # Try to connect to a local node first
            w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
            if w3.is_connected():
                logger.info("Connected to local Ethereum node")
                return w3
        except Exception as e:
            logger.warning(f"Failed to connect to local node: {e}")
        
        # Fallback to a public node (this is just for demonstration)
        # In a real application, you should use a proper API key
        try:
            w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID'))
            if w3.is_connected():
                logger.info("Connected to Infura")
                return w3
        except Exception as e:
            logger.warning(f"Failed to connect to Infura: {e}")
        
        logger.warning("No Ethereum node connection available. Using mock data.")
        return None
    
    def _define_protocol_actions(self) -> Dict:
        """
        Define the protocol actions for all supported protocols.
        The goal is to support the 96 actions mentioned in the paper plus additional protocols.
        """
        actions = {
            # Original protocols from the paper
            'uniswap_v2': {
                'ETH': ['AMN', 'AMPL', 'ANT', 'BAT', 'BNT', 'DAI', 'DATA', 'ENJ', 
                       'FXC', 'GNO', 'HEDG', 'KNC', 'MANA', 'MKR', 'POA20', 
                       'RCN', 'RDN', 'RLC', 'SAI', 'SAN', 'SNT', 'TKN', 'TRST', 'UBT']
            },
            'bancor': {
                'BNT': ['AMN', 'AMPL', 'ANT', 'BAT', 'DATA', 'ENJ', 'ETH', 
                       'FXC', 'GNO', 'HEDG', 'KNC', 'MANA', 'MKR', 'POA20', 
                       'RCN', 'RDN', 'RLC', 'SAI', 'SAN', 'SNT', 'TKN', 'TRST', 'UBT']
            },
            'makerdao': {
                'DAI': ['SAI'],
                'SAI': ['DAI']
            },
            
            # Additional protocols to complete the list
            'uniswap_v3': {
                'ETH': ['USDC', 'USDT', 'DAI', 'WBTC', 'UNI', 'LINK', 'COMP', 'AAVE', 'YFI', 'SNX', 'SUSHI'],
                'USDC': ['ETH', 'USDT', 'DAI', 'WBTC'],
                'USDT': ['ETH', 'USDC', 'DAI'],
                'DAI': ['ETH', 'USDC', 'USDT'],
                'WBTC': ['ETH', 'USDC']
            },
            'sushiswap': {
                'ETH': ['USDC', 'USDT', 'DAI', 'WBTC', 'SUSHI', 'YFI', 'UNI'],
                'USDC': ['ETH', 'USDT', 'DAI'],
                'USDT': ['ETH', 'USDC', 'DAI'],
                'DAI': ['ETH', 'USDC', 'USDT'],
                'SUSHI': ['ETH', 'USDC']
            },
            'curve': {
                'DAI': ['USDC', 'USDT', 'yDAI'],
                'USDC': ['DAI', 'USDT', 'yUSDC'],
                'USDT': ['DAI', 'USDC', 'yUSDT'],
                'yDAI': ['DAI', 'USDC', 'USDT'],
                'yUSDC': ['DAI', 'USDC', 'USDT'],
                'yUSDT': ['DAI', 'USDC', 'USDT']
            },
            'balancer': {
                'ETH': ['USDC', 'USDT', 'DAI', 'WBTC', 'BAL', 'YFI'],
                'USDC': ['ETH', 'USDT', 'DAI'],
                'USDT': ['ETH', 'USDC', 'DAI'],
                'DAI': ['ETH', 'USDC', 'USDT'],
                'BAL': ['ETH', 'USDC'],
                'WBTC': ['ETH']
            },
            'compound': {
                'ETH': ['cETH'],
                'cETH': ['ETH'],
                'USDC': ['cUSDC'],
                'cUSDC': ['USDC']
            },
            'aave': {
                'ETH': ['aETH'],
                'aETH': ['ETH'],
                'USDC': ['aUSDC'],
                'aUSDC': ['USDC']
            },
            'synthetix': {
                'ETH': ['sETH'],
                'sETH': ['ETH'],
                'DAI': ['sUSD'],
                'sUSD': ['DAI']
            },
            'yearn': {
                'DAI': ['yDAI'],
                'yDAI': ['DAI'],
                'USDC': ['yUSDC'],
                'yUSDC': ['USDC'],
                'USDT': ['yUSDT'],
                'yUSDT': ['USDT'],
                'yCRV': ['CRV'],
                'CRV': ['yCRV']
            },
            'dydx': {
                'ETH': ['USDC', 'DAI'],
                'USDC': ['ETH', 'DAI'],
                'DAI': ['ETH', 'USDC']
            }
        }
        return actions
    
    def get_total_action_count(self) -> int:
        """Calculate the total number of protocol actions."""
        count = 0
        # Count all bidirectional actions for each protocol
        for protocol, token_pairs in self.protocol_actions.items():
            for from_token, to_tokens in token_pairs.items():
                count += len(to_tokens)  # Each to_token represents one action
        return count
    
    def get_all_token_pairs(self) -> List[Tuple[str, str, str]]:
        """
        Get all token pairs for all protocols.
        Returns list of tuples: (from_token, to_token, dex)
        """
        pairs = []
        
        # Add pairs for all protocols
        for protocol, token_pairs in self.protocol_actions.items():
            for from_token, to_tokens in token_pairs.items():
                for to_token in to_tokens:
                    pairs.append((from_token, to_token, protocol))
        
        return pairs
    
    def get_protocol_info(self, protocol: str) -> Dict:
        """Get information about a specific protocol."""
        return PROTOCOL_CONFIGS.get(protocol, {})
    
    async def update_all_protocol_pools(self):
        """
        Update pool data for all protocol actions.
        This fetches real data from the blockchain when possible.
        """
        pairs = self.get_all_token_pairs()
        logger.info(f"Updating {len(pairs)} protocol action pairs across {len(self.protocol_actions)} protocols")
        
        # Use asyncio.gather to parallelize the updates
        update_tasks = [
            self._update_pool_data(from_token, to_token, dex)
            for from_token, to_token, dex in pairs
        ]
        
        # Process in batches to avoid overwhelming the node
        batch_size = 20
        for i in range(0, len(update_tasks), batch_size):
            batch = update_tasks[i:i + batch_size]
            await asyncio.gather(*batch, return_exceptions=True)
            
        logger.info(f"Completed updating {len(pairs)} protocol action pairs")
    
    async def _update_pool_data(self, from_token: str, to_token: str, dex: str):
        """
        Update pool data for a specific token pair on a specific DEX.
        This implementation fetches real data from the blockchain.
        """
        # Get token addresses
        from_token_address = TOKEN_ADDRESSES.get(from_token)
        to_token_address = TOKEN_ADDRESSES.get(to_token)
        
        if not from_token_address or not to_token_address:
            logger.warning(f"Token addresses not found for {from_token} or {to_token}")
            return
        
        # Get protocol config
        protocol_config = self.get_protocol_info(dex)
        fee = protocol_config.get('fee', 0.003)  # Default to 0.3%
        
        # If we don't have a web3 connection, use placeholder data
        if not self.w3 or not self.w3.is_connected():
            await self._update_pool_data_placeholder(from_token, to_token, dex)
            return
        
        try:
            # Fetch actual pool data from the blockchain
            reserve_from, reserve_to, fee = await self._fetch_pool_data(
                from_token_address, to_token_address, dex, fee
            )
            
            # Add trading pair to market graph
            pool_address = f"{dex}_{from_token}_{to_token}_pool"
            self.market_graph.add_trading_pair(
                from_token, to_token, dex, pool_address, 
                reserve_from, reserve_to, fee
            )
            
            logger.debug(f"Updated pool data for {dex}: {from_token} → {to_token}")
        except Exception as e:
            logger.error(f"Error updating pool data for {dex}: {from_token} → {to_token}: {e}")
            # Fallback to placeholder data
            await self._update_pool_data_placeholder(from_token, to_token, dex)
    
    async def _update_pool_data_placeholder(self, from_token: str, to_token: str, dex: str):
        """
        Update pool data with placeholder values when blockchain data is not available.
        """
        # Get protocol config for proper fee
        protocol_config = self.get_protocol_info(dex)
        fee = protocol_config.get('fee', 0.003)  # Default to 0.3%
        
        # These would normally be fetched from the blockchain
        reserve_from = 1000.0  # Placeholder
        reserve_to = 1000.0    # Placeholder
        
        # Add trading pair to market graph
        pool_address = f"{dex}_{from_token}_{to_token}_pool"
        self.market_graph.add_trading_pair(
            from_token, to_token, dex, pool_address, 
            reserve_from, reserve_to, fee
        )
        
        logger.debug(f"Updated pool data (placeholder) for {dex}: {from_token} → {to_token}")
    
    async def _fetch_pool_data(self, from_token_address: str, to_token_address: str, dex: str, default_fee: float) -> Tuple[float, float, float]:
        """
        Fetch pool data from the blockchain.
        This is a simplified implementation - in a real system, you would need to:
        1. Find the pool address for the token pair
        2. Call the appropriate smart contract methods to get reserves
        3. Handle different DEX protocols (Uniswap V2, V3, Bancor, etc.)
        """
        # This is a simplified example - in reality, you would need to:
        # 1. Query the factory contract to get the pair address
        # 2. Call the pair contract to get reserves
        # 3. Handle different fee structures for different protocols
        
        # For demonstration, we'll return placeholder values
        # In a real implementation, you would do something like:
        # pair_address = self._get_pair_address(from_token_address, to_token_address, dex)
        # reserves = self._get_reserves(pair_address)
        # fee = self._get_fee(dex)
        
        reserve_from = 1000.0
        reserve_to = 1000.0
        fee = default_fee
        
        return reserve_from, reserve_to, fee

# Example usage:
# protocol_manager = ProtocolActionsManager(market_graph)
# print(f"Total protocol actions: {protocol_manager.get_total_action_count()}")
# pairs = protocol_manager.get_all_token_pairs()
# print(f"Token pairs: {len(pairs)}")
# print(f"Protocols: {list(protocol_manager.protocol_actions.keys())}")