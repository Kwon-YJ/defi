import json
import aiohttp
from typing import Dict, Optional, List
from dataclasses import dataclass
from web3 import Web3
from src.logger import setup_logger
from config.config import config

logger = setup_logger(__name__)

@dataclass
class TokenInfo:
    address: str
    symbol: str
    name: str
    decimals: int
    price_usd: float = 0.0
    market_cap: float = 0.0
    volume_24h: float = 0.0
    coingecko_id: str = ""

class TokenManager:
    def __init__(self, web3_provider_url: Optional[str] = None):
        self.tokens: Dict[str, TokenInfo] = {}
        self.symbol_to_address: Dict[str, str] = {}
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        
        # RPC URL 설정 - 파라미터가 없으면 config에서 가져옴
        rpc_url = web3_provider_url or config.ethereum_mainnet_rpc
        self.w3 = Web3(Web3.HTTPProvider(rpc_url)) if rpc_url else None
        self.erc20_abi = [
            {
                "constant": True,
                "inputs": [],
                "name": "name",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "symbol", 
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            }
        ]
        self._load_common_tokens()
    
    def _load_common_tokens(self):
        """
        논문의 정확한 25개 자산 재현 - EXACT Paper specification: 25 assets
        [2103.02228] On the Just-In-Time Discovery of Profit-Generating Transactions in DeFi Protocols
        
        Based on Appendix A: Summary of the 24 ERC-20 cryptocurrency assets + ETH (25 total)
        These are the EXACT tokens used in the paper's experiments from block 9,100,000 to 10,050,000
        """
        common_tokens = {
            # ETH (native asset - represented as address 0x0 for native ETH)
            "0x0000000000000000000000000000000000000000": TokenInfo(
                address="0x0000000000000000000000000000000000000000",
                symbol="ETH",
                name="Ether",
                decimals=18,
                coingecko_id="ethereum"
            ),
            
            # WETH - Wrapped Ether (most important ERC-20 version of ETH)
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": TokenInfo(
                address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                symbol="WETH",
                name="Wrapped Ether",
                decimals=18,
                coingecko_id="weth"
            ),
            
            # Top 24 ERC-20 tokens from paper's appendix (ordered by transfer transactions)
            # 1. SAI - Single Collateral DAI (old MakerDAO token)
            "0x89d24A6b4CcB1B6fAA2625fE562bDD9a23260359": TokenInfo(
                address="0x89d24A6b4CcB1B6fAA2625fE562bDD9a23260359",
                symbol="SAI",
                name="Sai Stablecoin v1.0",
                decimals=18,
                coingecko_id="sai"
            ),
            
            # 2. BNT - Bancor Network Token  
            "0x1F573D6Fb3F13d689FF844B4cE37794d79a7FF1C": TokenInfo(
                address="0x1F573D6Fb3F13d689FF844B4cE37794d79a7FF1C",
                symbol="BNT",
                name="Bancor Network Token",
                decimals=18,
                coingecko_id="bancor"
            ),
            
            # 3. DAI - Multi-Collateral DAI (new MakerDAO token)
            "0x6B175474E89094C44Da98b954EedeAC495271d0F": TokenInfo(
                address="0x6B175474E89094C44Da98b954EedeAC495271d0F",
                symbol="DAI",
                name="Dai Stablecoin",
                decimals=18,
                coingecko_id="dai"
            ),
            
            # 4. BAT - Basic Attention Token
            "0x0D8775F648430679A709E98d2b0Cb6250d2887EF": TokenInfo(
                address="0x0D8775F648430679A709E98d2b0Cb6250d2887EF",
                symbol="BAT",
                name="Basic Attention Token",
                decimals=18,
                coingecko_id="basic-attention-token"
            ),
            
            # 5. ENJ - Enjin Coin
            "0xF629cBd94d3791C9250152BD8dfBDF380E2a3B9c": TokenInfo(
                address="0xF629cBd94d3791C9250152BD8dfBDF380E2a3B9c",
                symbol="ENJ",
                name="Enjin Coin",
                decimals=18,
                coingecko_id="enjincoin"
            ),
            
            # 6. SNT - Status Network Token
            "0x744d70FDBE2Ba4CF95131626614a1763DF805B9E": TokenInfo(
                address="0x744d70FDBE2Ba4CF95131626614a1763DF805B9E",
                symbol="SNT",
                name="Status Network Token",
                decimals=18,
                coingecko_id="status"
            ),
            
            # 7. KNC - Kyber Network
            "0xdd974D5C2e2928deA5F71b9825b8b646686BD200": TokenInfo(
                address="0xdd974D5C2e2928deA5F71b9825b8b646686BD200",
                symbol="KNC",
                name="Kyber Network Crystal",
                decimals=18,
                coingecko_id="kyber-network"
            ),
            
            # 8. MKR - Maker
            "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2": TokenInfo(
                address="0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2",
                symbol="MKR",
                name="Maker",
                decimals=18,
                coingecko_id="maker"
            ),
            
            # 9. DATA - Streamr DATACoin
            "0x0Cf0Ee63788A0849fE5297F3407f701E122cC023": TokenInfo(
                address="0x0Cf0Ee63788A0849fE5297F3407f701E122cC023",
                symbol="DATA",
                name="Streamr DATACoin",
                decimals=18,
                coingecko_id="streamr-datacoin"
            ),
            
            # 10. MANA - Decentraland
            "0x0F5D2fB29fb7d3CFeE444a200298f468908cC942": TokenInfo(
                address="0x0F5D2fB29fb7d3CFeE444a200298f468908cC942",
                symbol="MANA",
                name="Decentraland MANA",
                decimals=18,
                coingecko_id="decentraland"
            ),
            
            # 11. ANT - Aragon
            "0x960b236A07cf122663c4303350609A66A7B288C0": TokenInfo(
                address="0x960b236A07cf122663c4303350609A66A7B288C0",
                symbol="ANT",
                name="Aragon Network Token",
                decimals=18,
                coingecko_id="aragon"
            ),
            
            # 12. RLC - iExec RLC
            "0x607F4C5BB672230e8672085532f7e901544a7375": TokenInfo(
                address="0x607F4C5BB672230e8672085532f7e901544a7375",
                symbol="RLC",
                name="iExec RLC",
                decimals=9,
                coingecko_id="iexec-rlc"
            ),
            
            # 13. RCN - Ripio Credit Network
            "0xF970b8E36e23F7fC3FD752EeA86f8Be8D83375A6": TokenInfo(
                address="0xF970b8E36e23F7fC3FD752EeA86f8Be8D83375A6",
                symbol="RCN",
                name="Ripio Credit Network",
                decimals=18,
                coingecko_id="ripio-credit-network"
            ),
            
            # 14. UBT - Unibright
            "0x8400D94A5cb0fa0D041a3788e395285d61c9ee5e": TokenInfo(
                address="0x8400D94A5cb0fa0D041a3788e395285d61c9ee5e",
                symbol="UBT",
                name="Unibright",
                decimals=8,
                coingecko_id="unibright"
            ),
            
            # 15. GNO - Gnosis
            "0x6810e776880C02933D47DB1b9fc05908e5386b96": TokenInfo(
                address="0x6810e776880C02933D47DB1b9fc05908e5386b96",
                symbol="GNO",
                name="Gnosis Token",
                decimals=18,
                coingecko_id="gnosis"
            ),
            
            # 16. RDN - Raiden Network
            "0x255Aa6DF07540Cb5d3d297f0D0D4D84cb52bc8e6": TokenInfo(
                address="0x255Aa6DF07540Cb5d3d297f0D0D4D84cb52bc8e6",
                symbol="RDN",
                name="Raiden Token",
                decimals=18,
                coingecko_id="raiden-network"
            ),
            
            # 17. TKN - TokenCard
            "0xaAAf91D9b90dF800Df4F55c205fd6989c977E73a": TokenInfo(
                address="0xaAAf91D9b90dF800Df4F55c205fd6989c977E73a",
                symbol="TKN",
                name="TokenCard",
                decimals=8,
                coingecko_id="tokencard"
            ),
            
            # 18. TRST - WeTrust  
            "0xCb94be6f13A1182E4A4B6140cb7bf2025d28e41B": TokenInfo(
                address="0xCb94be6f13A1182E4A4B6140cb7bf2025d28e41B",
                symbol="TRST",
                name="WeTrust",
                decimals=6,
                coingecko_id="wetrust"
            ),
            
            # 19. AMN - Amon
            "0x737F98AC8cA59f2C68aD658E3C3d8C8963E40a4c": TokenInfo(
                address="0x737F98AC8cA59f2C68aD658E3C3d8C8963E40a4c",
                symbol="AMN",
                name="Amon",
                decimals=18,
                coingecko_id="amon"
            ),
            
            # 20. FXC - Flexacoin (Note: This token may have different addresses or be deprecated)
            "0x4a57E687b9126435a9B19E4A802113e266AdeBde": TokenInfo(
                address="0x4a57E687b9126435a9B19E4A802113e266AdeBde",
                symbol="FXC",
                name="Flexacoin",
                decimals=18,
                coingecko_id="flexacoin"
            ),
            
            # 21. SAN - Santiment Network Token
            "0x7C5A0CE9267ED19B22F8cae653F198e3E8daf098": TokenInfo(
                address="0x7C5A0CE9267ED19B22F8cae653F198e3E8daf098",
                symbol="SAN",
                name="Santiment Network Token",
                decimals=18,
                coingecko_id="santiment"
            ),
            
            # 22. AMPL - Ampleforth
            "0xD46bA6D942050d489DBd938a2C909A5d5039A161": TokenInfo(
                address="0xD46bA6D942050d489DBd938a2C909A5d5039A161",
                symbol="AMPL",
                name="Ampleforth",
                decimals=9,
                coingecko_id="ampleforth"
            ),
            
            # 23. HEDG - HedgeTrade
            "0xF1290473E210b2108A85237fbCd7b6eb42Cc654F": TokenInfo(
                address="0xF1290473E210b2108A85237fbCd7b6eb42Cc654F",
                symbol="HEDG",
                name="HedgeTrade",
                decimals=18,
                coingecko_id="hedgetrade"
            ),
            
            # 24. POA20 - POA Network (POA ERC20 on Ethereum bridge)
            "0x6758B7d441a9739b98552B373703d8d3d14f9e62": TokenInfo(
                address="0x6758B7d441a9739b98552B373703d8d3d14f9e62",
                symbol="POA20",
                name="POA ERC20 on Foundation",
                decimals=18,
                coingecko_id="poa-network"
            ),
            
            # Additional stablecoins - Required for DeFi arbitrage expansion
            # USDC - USD Coin (Circle)
            "0xA0b86a33E6441946e15b3C1F5d44F7c0e3A1b82C": TokenInfo(
                address="0xA0b86a33E6441946e15b3C1F5d44F7c0e3A1b82C",
                symbol="USDC",
                name="USD Coin",
                decimals=6,
                coingecko_id="usd-coin"
            ),
            
            # USDT - Tether USD
            "0xdAC17F958D2ee523a2206206994597C13D831ec7": TokenInfo(
                address="0xdAC17F958D2ee523a2206206994597C13D831ec7",
                symbol="USDT",
                name="Tether USD",
                decimals=6,
                coingecko_id="tether"
            ),
            
            # Major DeFi tokens (TODO requirement completion)
            # WBTC - Wrapped Bitcoin
            "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599": TokenInfo(
                address="0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
                symbol="WBTC",
                name="Wrapped BTC",
                decimals=8,
                coingecko_id="wrapped-bitcoin"
            ),
            
            # UNI - Uniswap
            "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984": TokenInfo(
                address="0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
                symbol="UNI",
                name="Uniswap",
                decimals=18,
                coingecko_id="uniswap"
            ),
            
            # SUSHI - SushiSwap
            "0x6B3595068778DD592e39A122f4f5a5cF09C90fE2": TokenInfo(
                address="0x6B3595068778DD592e39A122f4f5a5cF09C90fE2",
                symbol="SUSHI",
                name="SushiToken",
                decimals=18,
                coingecko_id="sushi"
            ),
            
            # COMP - Compound
            "0xc00e94Cb662C3520282E6f5717214004A7f26888": TokenInfo(
                address="0xc00e94Cb662C3520282E6f5717214004A7f26888",
                symbol="COMP",
                name="Compound",
                decimals=18,
                coingecko_id="compound-governance-token"
            ),
            
            # AAVE - Aave Token
            "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9": TokenInfo(
                address="0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
                symbol="AAVE",
                name="Aave Token",
                decimals=18,
                coingecko_id="aave"
            ),
            
            # DeFi Ecosystem tokens (TODO requirement completion)
            # CRV - Curve DAO Token
            "0xD533a949740bb3306d119CC777fa900bA034cd52": TokenInfo(
                address="0xD533a949740bb3306d119CC777fa900bA034cd52",
                symbol="CRV",
                name="Curve DAO Token",
                decimals=18,
                coingecko_id="curve-dao-token"
            ),
            
            # BAL - Balancer
            "0xba100000625a3754423978a60c9317c58a424e3D": TokenInfo(
                address="0xba100000625a3754423978a60c9317c58a424e3D",
                symbol="BAL",
                name="Balancer",
                decimals=18,
                coingecko_id="balancer"
            ),
            
            # YFI - yearn.finance
            "0x0bc529c00C6401aEF6D220BE8E6EbB43eF6eCc24": TokenInfo(
                address="0x0bc529c00C6401aEF6D220BE8E6EbB43eF6eCc24",
                symbol="YFI",
                name="yearn.finance",
                decimals=18,
                coingecko_id="yearn-finance"
            ),
            
            # Lending protocol tokens (TODO requirement completion)
            # Compound Protocol - Interest Bearing Tokens
            # cETH - Compound Ether
            "0x4Ddc2D193948926D02f9B1fE9e1daa0718270ED5": TokenInfo(
                address="0x4Ddc2D193948926D02f9B1fE9e1daa0718270ED5",
                symbol="cETH",
                name="Compound Ether",
                decimals=8,
                coingecko_id="compound-ether"
            ),
            
            # cUSDC - Compound USD Coin
            "0x39AA39c021dfbaE8faC545936693aC917d5E7563": TokenInfo(
                address="0x39AA39c021dfbaE8faC545936693aC917d5E7563",
                symbol="cUSDC",
                name="Compound USD Coin",
                decimals=8,
                coingecko_id="compound-usd-coin"
            ),
            
            # Aave Protocol - Interest Bearing Tokens (aTokens)
            # aETH - Aave interest bearing ETH
            "0x030bA81f1c18d280636F32af80b9AAd02Cf0854e": TokenInfo(
                address="0x030bA81f1c18d280636F32af80b9AAd02Cf0854e",
                symbol="aETH",
                name="Aave interest bearing ETH",
                decimals=18,
                coingecko_id="aave-eth"
            ),
            
            # aUSDC - Aave interest bearing USDC
            "0xBcca60bB61934080951369a648Fb03DF4F96263C": TokenInfo(
                address="0xBcca60bB61934080951369a648Fb03DF4F96263C",
                symbol="aUSDC",
                name="Aave interest bearing USDC",
                decimals=6,
                coingecko_id="aave-usdc"
            ),
            
            # LP tokens (Liquidity Provider tokens) - TODO requirement completion: UNI-V2 pairs, Curve LP tokens
            # Uniswap V2 LP tokens (UNI-V2 pairs)
            
            # WETH-USDC UNI-V2 LP (most liquid pair)
            "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc": TokenInfo(
                address="0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc",
                symbol="WETH-USDC",
                name="Uniswap V2 WETH-USDC",
                decimals=18,
                coingecko_id=""
            ),
            
            # WETH-USDT UNI-V2 LP  
            "0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852": TokenInfo(
                address="0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852",
                symbol="WETH-USDT",
                name="Uniswap V2 WETH-USDT", 
                decimals=18,
                coingecko_id=""
            ),
            
            # WETH-DAI UNI-V2 LP
            "0xA478c2975Ab1Ea89e8196811F51A7B7Ade33eB11": TokenInfo(
                address="0xA478c2975Ab1Ea89e8196811F51A7B7Ade33eB11",
                symbol="WETH-DAI",
                name="Uniswap V2 WETH-DAI",
                decimals=18,
                coingecko_id=""
            ),
            
            # WBTC-WETH UNI-V2 LP (major BTC-ETH liquidity)
            "0xBb2b8038a1640196FbE3e38816F3e67Cba72D940": TokenInfo(
                address="0xBb2b8038a1640196FbE3e38816F3e67Cba72D940",
                symbol="WBTC-WETH",
                name="Uniswap V2 WBTC-WETH",
                decimals=18,
                coingecko_id=""
            ),
            
            # UNI-WETH UNI-V2 LP (UNI governance token liquidity)
            "0xd3d2E2692501A5c9Ca623199D38826e513033a17": TokenInfo(
                address="0xd3d2E2692501A5c9Ca623199D38826e513033a17",
                symbol="UNI-WETH",
                name="Uniswap V2 UNI-WETH",
                decimals=18,
                coingecko_id=""
            ),
            
            # COMP-WETH UNI-V2 LP (Compound governance token liquidity)
            "0xCFfDdeD873554F362Ac02f8Fb1f02E5ada10516f": TokenInfo(
                address="0xCFfDdeD873554F362Ac02f8Fb1f02E5ada10516f",
                symbol="COMP-WETH",
                name="Uniswap V2 COMP-WETH",
                decimals=18,
                coingecko_id=""
            ),
            
            # AAVE-WETH UNI-V2 LP (Aave governance token liquidity)
            "0xDFC14d2Af169B0D36C4EFF567Ada9b2E0CAE044f": TokenInfo(
                address="0xDFC14d2Af169B0D36C4EFF567Ada9b2E0CAE044f",
                symbol="AAVE-WETH",
                name="Uniswap V2 AAVE-WETH",
                decimals=18,
                coingecko_id=""
            ),
            
            # SUSHI-WETH UNI-V2 LP (SushiSwap governance token liquidity)  
            "0xCE84867c3c02B05dc570d0135103d3fB9CC19433": TokenInfo(
                address="0xCE84867c3c02B05dc570d0135103d3fB9CC19433",
                symbol="SUSHI-WETH",
                name="Uniswap V2 SUSHI-WETH",
                decimals=18,
                coingecko_id=""
            ),
            
            # MKR-WETH UNI-V2 LP (Maker governance token liquidity)
            "0xC2aDdA861F89bBB333c90c492cB837741916A225": TokenInfo(
                address="0xC2aDdA861F89bBB333c90c492cB837741916A225",
                symbol="MKR-WETH",
                name="Uniswap V2 MKR-WETH",
                decimals=18,
                coingecko_id=""
            ),
            
            # USDC-USDT UNI-V2 LP (stablecoin arbitrage pair)
            "0x3041CbD36888bECc7bbCBc0045E3B1f144466f5f": TokenInfo(
                address="0x3041CbD36888bECc7bbCBc0045E3B1f144466f5f",
                symbol="USDC-USDT",
                name="Uniswap V2 USDC-USDT",
                decimals=18,
                coingecko_id=""
            ),
            
            # Curve LP tokens (Curve liquidity provider tokens for stablecoin pools)
            
            # 3CRV - Curve 3pool LP token (DAI+USDC+USDT) - Most important Curve LP token
            "0x6c3F90f043a72FA612cbac8115EE7e52BDe6E490": TokenInfo(
                address="0x6c3F90f043a72FA612cbac8115EE7e52BDe6E490",
                symbol="3CRV",
                name="Curve.fi DAI/USDC/USDT",
                decimals=18,
                coingecko_id="lp-3pool-curve"
            ),
            
            # crvUSDC - Curve Compound USDC LP token (cUSDC+cDAI)
            "0x845838DF265Dcd2c412A1Dc9e959c7d08537f8a2": TokenInfo(
                address="0x845838DF265Dcd2c412A1Dc9e959c7d08537f8a2",
                symbol="crvUSDC",
                name="Curve.fi cDAI/cUSDC",
                decimals=18,
                coingecko_id="curve-fi-cdai-cusdc"
            ),
            
            # YCRV - Curve yEarn LP token (yDAI+yUSDC+yUSDT+yTUSD) 
            "0xdF5e0e81Dff6FAF3A7e52BA697820c5e32D806A8": TokenInfo(
                address="0xdF5e0e81Dff6FAF3A7e52BA697820c5e32D806A8",
                symbol="yDAI+yUSDC+yUSDT+yTUSD",
                name="Curve.fi yDAI/yUSDC/yUSDT/yTUSD",
                decimals=18,
                coingecko_id="curve-fi-ydai-yusdc-yusdt-ytusd"
            ),
            
            # GUSD3CRV - Curve GUSD metapool LP token (GUSD+3CRV)
            "0xD2967f45c4f384DEEa880F807Be904762a3DeA07": TokenInfo(
                address="0xD2967f45c4f384DEEa880F807Be904762a3DeA07",
                symbol="GUSD3CRV",
                name="Curve.fi GUSD/3Crv",
                decimals=18,
                coingecko_id="curve-fi-gusd-3crv"
            ),
            
            # HUSD3CRV - Curve HUSD metapool LP token (HUSD+3CRV)
            "0x5B5CFE992AdAC0C9D48E05854B2d91C73a003858": TokenInfo(
                address="0x5B5CFE992AdAC0C9D48E05854B2d91C73a003858",
                symbol="HUSD3CRV",
                name="Curve.fi HUSD/3Crv",
                decimals=18,
                coingecko_id="curve-fi-husd-3crv"
            ),
            
            # USDK3CRV - Curve USDK metapool LP token (USDK+3CRV)
            "0x97E2768e8E73511cA874545DC5Ff8067eB19B787": TokenInfo(
                address="0x97E2768e8E73511cA874545DC5Ff8067eB19B787",
                symbol="USDK3CRV",
                name="Curve.fi USDK/3Crv",
                decimals=18,
                coingecko_id="curve-fi-usdk-3crv"
            ),
            
            # USDN3CRV - Curve USDN metapool LP token (USDN+3CRV)
            "0x4f3E8F405CF5aFC05D68142F3783bDfE13811522": TokenInfo(
                address="0x4f3E8F405CF5aFC05D68142F3783bDfE13811522",
                symbol="USDN3CRV",
                name="Curve.fi USDN/3Crv",
                decimals=18,
                coingecko_id="curve-fi-usdn-3crv"
            ),
            
            # USDP3CRV - Curve USDP metapool LP token (USDP+3CRV)
            "0x7Eb40E450b9655f4B3cC4259BCC731c63ff55ae6": TokenInfo(
                address="0x7Eb40E450b9655f4B3cC4259BCC731c63ff55ae6",
                symbol="USDP3CRV",
                name="Curve.fi USDP/3Crv",
                decimals=18,
                coingecko_id="curve-fi-usdp-3crv"
            ),
            
            # sBTC - Curve sBTC LP token (renBTC+wBTC+sBTC)
            "0x075b1bb99792c9E1041bA13afEf80C91a1e70fB3": TokenInfo(
                address="0x075b1bb99792c9E1041bA13afEf80C91a1e70fB3",
                symbol="crvRenWSBTC",
                name="Curve.fi renBTC/wBTC/sBTC",
                decimals=18,
                coingecko_id="curve-fi-renbtc-wbtc-sbtc"
            ),
            
            # HBTC - Curve HBTC LP token (hBTC+wBTC)
            "0xb19059ebb43466C323583928285a49f558E572Fd": TokenInfo(
                address="0xb19059ebb43466C323583928285a49f558E572Fd",
                symbol="crvHBTC",
                name="Curve.fi hBTC/wBTC",
                decimals=18,
                coingecko_id="curve-fi-hbtc-wbtc"
            ),
            
            # Synthetix synthetic assets (TODO requirement completion)
            # sUSD - Synthetic USD (Synthetix stablecoin)
            "0x57Ab1ec28D129707052df4dF418D58a2D46d5f51": TokenInfo(
                address="0x57Ab1ec28D129707052df4dF418D58a2D46d5f51",
                symbol="sUSD",
                name="Synth sUSD",
                decimals=18,
                coingecko_id="susd"
            ),
            
            # sETH - Synthetic ETH (Synthetix synthetic Ether)
            "0x5e74C9036fb86BD7eCdcb084a0673EFc32eA31cb": TokenInfo(
                address="0x5e74C9036fb86BD7eCdcb084a0673EFc32eA31cb",
                symbol="sETH",
                name="Synth sETH",
                decimals=18,
                coingecko_id="seth"
            )
        }
        
        # **논문 기준 검증**: 25개 assets + WETH + 2 additional stablecoins (USDC, USDT) + 5 major tokens (WBTC, UNI, SUSHI, COMP, AAVE) + 3 DeFi ecosystem tokens (CRV, BAL, YFI) + 4 lending protocol tokens (cETH, cUSDC, aETH, aUSDC) + 20 LP tokens (10 UNI-V2 pairs + 10 Curve LP tokens) + 2 Synthetix assets (sUSD, sETH)
        expected_count = 62  # 25 + 1 + 2 + 5 + 3 + 4 + 20 + 2
        if len(common_tokens) != expected_count:
            logger.warning(f"Asset count mismatch! Expected: {expected_count} (25 paper + WETH + 2 stablecoins + 5 major tokens + 3 DeFi ecosystem + 4 lending protocol + 20 LP tokens + 2 Synthetix assets), Got: {len(common_tokens)}")
        else:
            logger.info("✅ Paper specification enhanced: 25 assets + WETH + 2 stablecoins + 5 major tokens + 3 DeFi ecosystem tokens + 4 lending protocol tokens + 20 LP tokens + 2 Synthetix assets (62 total) registered")
        
        self.tokens.update(common_tokens)
        
        # 심볼 -> 주소 매핑 생성
        for address, token_info in common_tokens.items():
            self.symbol_to_address[token_info.symbol] = address
    
    async def get_token_info(self, address: str) -> Optional[TokenInfo]:
        """토큰 정보 조회"""
        # Try both original address and lowercase
        if address in self.tokens:
            return self.tokens[address]
        
        address_lower = address.lower()
        if address_lower in self.tokens:
            return self.tokens[address_lower]
        
        # 온체인에서 토큰 정보 조회 시도
        token_info = await self._fetch_token_info_onchain(address)
        if token_info:
            self.tokens[address] = token_info
            return token_info
        
        return None
    
    def get_address_by_symbol(self, symbol: str) -> Optional[str]:
        """심볼로 주소 조회"""
        # Try exact match first
        if symbol in self.symbol_to_address:
            return self.symbol_to_address[symbol]
        # Try uppercase for backward compatibility
        return self.symbol_to_address.get(symbol.upper())
    
    async def update_prices(self):
        """CoinGecko에서 가격 정보 업데이트"""
        try:
            # CoinGecko ID 목록 생성
            coingecko_ids = [
                token.coingecko_id for token in self.tokens.values() 
                if token.coingecko_id
            ]
            
            if not coingecko_ids:
                return
            
            # API 호출
            ids_str = ",".join(coingecko_ids)
            url = f"{self.coingecko_api}/simple/price"
            params = {
                'ids': ids_str,
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # 가격 정보 업데이트
                        for token in self.tokens.values():
                            if token.coingecko_id in data:
                                price_data = data[token.coingecko_id]
                                token.price_usd = price_data.get('usd', 0)
                                token.market_cap = price_data.get('usd_market_cap', 0)
                                token.volume_24h = price_data.get('usd_24h_vol', 0)
                        
                        logger.info(f"{len(data)}개 토큰 가격 정보 업데이트 완료")
                    else:
                        logger.error(f"CoinGecko API 오류: {response.status}")
                        
        except Exception as e:
            logger.error(f"가격 정보 업데이트 실패: {e}")
    
    async def _fetch_token_info_onchain(self, address: str) -> Optional[TokenInfo]:
        """온체인에서 토큰 정보 조회"""
        if not self.w3 or not self.w3.is_connected():
            logger.warning("Web3 연결이 설정되지 않았습니다")
            return None
            
        try:
            # 주소 체크섬 변환
            checksum_address = self.w3.to_checksum_address(address)
            
            # ERC20 컨트랙트 생성
            contract = self.w3.eth.contract(address=checksum_address, abi=self.erc20_abi)
            
            # 토큰 정보 조회
            symbol = contract.functions.symbol().call()
            name = contract.functions.name().call()
            decimals = contract.functions.decimals().call()
            
            token_info = TokenInfo(
                address=checksum_address,
                symbol=symbol,
                name=name,
                decimals=decimals
            )
            
            logger.info(f"온체인 토큰 정보 조회 성공: {symbol} ({name})")
            return token_info
            
        except Exception as e:
            logger.error(f"온체인 토큰 정보 조회 실패 {address}: {e}")
            return None
    
    def get_major_trading_pairs(self) -> List[tuple]:
        """
        논문에서 사용된 주요 거래 쌍 목록 반환
        Based on paper's Appendix B: Supported DeFi actions
        """
        # Paper에서 사용된 주요 거래 쌍들 (ETH/WETH를 중심으로 한 모든 ERC-20 토큰 쌍들)
        major_pairs = [
            # ETH/WETH pair (most liquid conversion pair)
            ("ETH", "WETH"),
            ("WETH", "ETH"),
            
            # ETH with all ERC-20 tokens (Uniswap pairs from paper)
            ("ETH", "SAI"),
            ("ETH", "BNT"), 
            ("ETH", "DAI"),
            ("ETH", "BAT"),
            ("ETH", "ENJ"),
            ("ETH", "SNT"),
            ("ETH", "KNC"),
            ("ETH", "MKR"),
            ("ETH", "DATA"),
            ("ETH", "MANA"),
            ("ETH", "ANT"),
            ("ETH", "RLC"),
            ("ETH", "RCN"),
            ("ETH", "UBT"),
            ("ETH", "GNO"),
            ("ETH", "RDN"),
            ("ETH", "TKN"),
            ("ETH", "TRST"),
            ("ETH", "AMN"),
            ("ETH", "FXC"),
            ("ETH", "SAN"),
            ("ETH", "AMPL"),
            ("ETH", "HEDG"),
            ("ETH", "POA20"),
            
            # WETH with all ERC-20 tokens (major DeFi protocols use WETH)
            ("WETH", "SAI"),
            ("WETH", "BNT"), 
            ("WETH", "DAI"),
            ("WETH", "BAT"),
            ("WETH", "ENJ"),
            ("WETH", "SNT"),
            ("WETH", "KNC"),
            ("WETH", "MKR"),
            ("WETH", "DATA"),
            ("WETH", "MANA"),
            ("WETH", "ANT"),
            ("WETH", "RLC"),
            ("WETH", "RCN"),
            ("WETH", "UBT"),
            ("WETH", "GNO"),
            ("WETH", "RDN"),
            ("WETH", "TKN"),
            ("WETH", "TRST"),
            ("WETH", "AMN"),
            ("WETH", "FXC"),
            ("WETH", "SAN"),
            ("WETH", "AMPL"),
            ("WETH", "HEDG"),
            ("WETH", "POA20"),
            
            # BNT with all other ERC-20 tokens (Bancor pairs from paper)
            ("BNT", "SAI"),
            ("BNT", "DAI"),
            ("BNT", "BAT"),
            ("BNT", "ENJ"),
            ("BNT", "SNT"),
            ("BNT", "KNC"),
            ("BNT", "MKR"),
            ("BNT", "DATA"),
            ("BNT", "MANA"),
            ("BNT", "ANT"),
            ("BNT", "RLC"),
            ("BNT", "RCN"),
            ("BNT", "UBT"),
            ("BNT", "GNO"),
            ("BNT", "RDN"),
            ("BNT", "TKN"),
            ("BNT", "TRST"),
            ("BNT", "AMN"),
            ("BNT", "FXC"),
            ("BNT", "SAN"),
            ("BNT", "AMPL"),
            ("BNT", "HEDG"),
            ("BNT", "POA20"),
            
            # MakerDAO pairs
            ("DAI", "SAI"),
            ("SAI", "DAI"),
            
            # Stablecoin arbitrage pairs (CRITICAL for DeFi profit generation)
            ("USDC", "USDT"),
            ("USDT", "USDC"),
            ("DAI", "USDC"),
            ("USDC", "DAI"),
            ("DAI", "USDT"),
            ("USDT", "DAI"),
            ("SAI", "USDC"),
            ("USDC", "SAI"),
            ("SAI", "USDT"),
            ("USDT", "SAI"),
            
            # ETH/WETH pairs with new stablecoins
            ("ETH", "USDC"),
            ("USDC", "ETH"),
            ("ETH", "USDT"),
            ("USDT", "ETH"),
            ("WETH", "USDC"),
            ("USDC", "WETH"),
            ("WETH", "USDT"),
            ("USDT", "WETH"),
            
            # Major DeFi token pairs (WBTC, UNI, SUSHI, COMP, AAVE)
            # ETH pairs with major tokens
            ("ETH", "WBTC"),
            ("WBTC", "ETH"),
            ("ETH", "UNI"),
            ("UNI", "ETH"),
            ("ETH", "SUSHI"),
            ("SUSHI", "ETH"),
            ("ETH", "COMP"),
            ("COMP", "ETH"),
            ("ETH", "AAVE"),
            ("AAVE", "ETH"),
            
            # WETH pairs with major tokens  
            ("WETH", "WBTC"),
            ("WBTC", "WETH"),
            ("WETH", "UNI"),
            ("UNI", "WETH"),
            ("WETH", "SUSHI"),
            ("SUSHI", "WETH"),
            ("WETH", "COMP"),
            ("COMP", "WETH"),
            ("WETH", "AAVE"),
            ("AAVE", "WETH"),
            
            # Major tokens with stablecoins (high liquidity pairs)
            ("WBTC", "USDC"),
            ("USDC", "WBTC"),
            ("WBTC", "USDT"),
            ("USDT", "WBTC"),
            ("WBTC", "DAI"),
            ("DAI", "WBTC"),
            
            ("UNI", "USDC"),
            ("USDC", "UNI"),
            ("UNI", "USDT"),
            ("USDT", "UNI"),
            ("UNI", "DAI"),
            ("DAI", "UNI"),
            
            ("SUSHI", "USDC"),
            ("USDC", "SUSHI"),
            ("SUSHI", "USDT"),
            ("USDT", "SUSHI"),
            ("SUSHI", "DAI"),
            ("DAI", "SUSHI"),
            
            ("COMP", "USDC"),
            ("USDC", "COMP"),
            ("COMP", "USDT"),
            ("USDT", "COMP"),
            ("COMP", "DAI"),
            ("DAI", "COMP"),
            
            ("AAVE", "USDC"),
            ("USDC", "AAVE"),
            ("AAVE", "USDT"),
            ("USDT", "AAVE"),
            ("AAVE", "DAI"),
            ("DAI", "AAVE"),
            
            # Cross major token pairs (for complex arbitrage)
            ("WBTC", "UNI"),
            ("UNI", "WBTC"),
            ("WBTC", "SUSHI"),
            ("SUSHI", "WBTC"),
            ("WBTC", "COMP"),
            ("COMP", "WBTC"),
            ("WBTC", "AAVE"),
            ("AAVE", "WBTC"),
            ("UNI", "SUSHI"),
            ("SUSHI", "UNI"),
            ("UNI", "COMP"),
            ("COMP", "UNI"),
            ("UNI", "AAVE"),
            ("AAVE", "UNI"),
            ("SUSHI", "COMP"),
            ("COMP", "SUSHI"),
            ("SUSHI", "AAVE"),
            ("AAVE", "SUSHI"),
            ("COMP", "AAVE"),
            ("AAVE", "COMP"),
            
            # New DeFi Ecosystem tokens pairs (CRV, BAL, YFI)
            # ETH pairs with DeFi ecosystem tokens
            ("ETH", "CRV"),
            ("CRV", "ETH"),
            ("ETH", "BAL"),
            ("BAL", "ETH"),
            ("ETH", "YFI"),
            ("YFI", "ETH"),
            
            # WETH pairs with DeFi ecosystem tokens
            ("WETH", "CRV"),
            ("CRV", "WETH"),
            ("WETH", "BAL"),
            ("BAL", "WETH"),
            ("WETH", "YFI"),
            ("YFI", "WETH"),
            
            # DeFi ecosystem tokens with stablecoins
            ("CRV", "USDC"),
            ("USDC", "CRV"),
            ("CRV", "USDT"),
            ("USDT", "CRV"),
            ("CRV", "DAI"),
            ("DAI", "CRV"),
            
            ("BAL", "USDC"),
            ("USDC", "BAL"),
            ("BAL", "USDT"),
            ("USDT", "BAL"),
            ("BAL", "DAI"),
            ("DAI", "BAL"),
            
            ("YFI", "USDC"),
            ("USDC", "YFI"),
            ("YFI", "USDT"),
            ("USDT", "YFI"),
            ("YFI", "DAI"),
            ("DAI", "YFI"),
            
            # DeFi ecosystem tokens with major tokens
            ("CRV", "WBTC"),
            ("WBTC", "CRV"),
            ("CRV", "UNI"),
            ("UNI", "CRV"),
            ("CRV", "SUSHI"),
            ("SUSHI", "CRV"),
            ("CRV", "COMP"),
            ("COMP", "CRV"),
            ("CRV", "AAVE"),
            ("AAVE", "CRV"),
            
            ("BAL", "WBTC"),
            ("WBTC", "BAL"),
            ("BAL", "UNI"),
            ("UNI", "BAL"),
            ("BAL", "SUSHI"),
            ("SUSHI", "BAL"),
            ("BAL", "COMP"),
            ("COMP", "BAL"),
            ("BAL", "AAVE"),
            ("AAVE", "BAL"),
            
            ("YFI", "WBTC"),
            ("WBTC", "YFI"),
            ("YFI", "UNI"),
            ("UNI", "YFI"),
            ("YFI", "SUSHI"),
            ("SUSHI", "YFI"),
            ("YFI", "COMP"),
            ("COMP", "YFI"),
            ("YFI", "AAVE"),
            ("AAVE", "YFI"),
            
            # Cross DeFi ecosystem token pairs
            ("CRV", "BAL"),
            ("BAL", "CRV"),
            ("CRV", "YFI"),
            ("YFI", "CRV"),
            ("BAL", "YFI"),
            ("YFI", "BAL"),
            
            # Lending protocol tokens pairs (TODO requirement completion)
            # Compound tokens with base assets
            ("ETH", "cETH"),
            ("cETH", "ETH"),
            ("WETH", "cETH"), 
            ("cETH", "WETH"),
            ("USDC", "cUSDC"),
            ("cUSDC", "USDC"),
            
            # Aave tokens with base assets  
            ("ETH", "aETH"),
            ("aETH", "ETH"),
            ("WETH", "aETH"),
            ("aETH", "WETH"),
            ("USDC", "aUSDC"),
            ("aUSDC", "USDC"),
            
            # Cross lending protocol pairs (Compound vs Aave arbitrage)
            ("cETH", "aETH"),
            ("aETH", "cETH"),
            ("cUSDC", "aUSDC"),
            ("aUSDC", "cUSDC"),
            
            # Lending tokens with stablecoins (for yield arbitrage)
            ("cETH", "DAI"),
            ("DAI", "cETH"),
            ("cETH", "USDT"),
            ("USDT", "cETH"),
            ("cUSDC", "DAI"),
            ("DAI", "cUSDC"),
            ("cUSDC", "USDT"),
            ("USDT", "cUSDC"),
            
            ("aETH", "DAI"),
            ("DAI", "aETH"),
            ("aETH", "USDT"),
            ("USDT", "aETH"),
            ("aUSDC", "DAI"),
            ("DAI", "aUSDC"),
            ("aUSDC", "USDT"),
            ("USDT", "aUSDC"),
            
            # Lending tokens with major DeFi tokens
            ("cETH", "WBTC"),
            ("WBTC", "cETH"),
            ("cETH", "UNI"),
            ("UNI", "cETH"),
            ("cETH", "COMP"),
            ("COMP", "cETH"),
            ("cETH", "AAVE"),
            ("AAVE", "cETH"),
            
            ("aETH", "WBTC"),
            ("WBTC", "aETH"),
            ("aETH", "UNI"),
            ("UNI", "aETH"),
            ("aETH", "COMP"),
            ("COMP", "aETH"),
            ("aETH", "AAVE"),
            ("AAVE", "aETH"),
            
            ("cUSDC", "WBTC"),
            ("WBTC", "cUSDC"),
            ("cUSDC", "UNI"),
            ("UNI", "cUSDC"),
            ("cUSDC", "COMP"),
            ("COMP", "cUSDC"),
            ("cUSDC", "AAVE"),
            ("AAVE", "cUSDC"),
            
            ("aUSDC", "WBTC"),
            ("WBTC", "aUSDC"),
            ("aUSDC", "UNI"),
            ("UNI", "aUSDC"),
            ("aUSDC", "COMP"),
            ("COMP", "aUSDC"),
            ("aUSDC", "AAVE"),
            ("AAVE", "aUSDC"),
            
            # LP token pairs (TODO requirement completion: UNI-V2 pairs, Curve LP tokens)
            # Uniswap V2 LP tokens with underlying assets
            
            # WETH-USDC LP token pairs (most liquid LP token)
            ("WETH-USDC", "WETH"),
            ("WETH", "WETH-USDC"),
            ("WETH-USDC", "USDC"),
            ("USDC", "WETH-USDC"),
            ("WETH-USDC", "ETH"),
            ("ETH", "WETH-USDC"),
            
            # WETH-USDT LP token pairs
            ("WETH-USDT", "WETH"),
            ("WETH", "WETH-USDT"),
            ("WETH-USDT", "USDT"),
            ("USDT", "WETH-USDT"),
            ("WETH-USDT", "ETH"),
            ("ETH", "WETH-USDT"),
            
            # WETH-DAI LP token pairs
            ("WETH-DAI", "WETH"),
            ("WETH", "WETH-DAI"),
            ("WETH-DAI", "DAI"),
            ("DAI", "WETH-DAI"),
            ("WETH-DAI", "ETH"),
            ("ETH", "WETH-DAI"),
            
            # WBTC-WETH LP token pairs (major BTC-ETH liquidity)
            ("WBTC-WETH", "WBTC"),
            ("WBTC", "WBTC-WETH"),
            ("WBTC-WETH", "WETH"),
            ("WETH", "WBTC-WETH"),
            ("WBTC-WETH", "ETH"),
            ("ETH", "WBTC-WETH"),
            
            # UNI-WETH LP token pairs
            ("UNI-WETH", "UNI"),
            ("UNI", "UNI-WETH"),
            ("UNI-WETH", "WETH"),
            ("WETH", "UNI-WETH"),
            ("UNI-WETH", "ETH"),
            ("ETH", "UNI-WETH"),
            
            # COMP-WETH LP token pairs
            ("COMP-WETH", "COMP"),
            ("COMP", "COMP-WETH"),
            ("COMP-WETH", "WETH"),
            ("WETH", "COMP-WETH"),
            ("COMP-WETH", "ETH"),
            ("ETH", "COMP-WETH"),
            
            # AAVE-WETH LP token pairs
            ("AAVE-WETH", "AAVE"),
            ("AAVE", "AAVE-WETH"),
            ("AAVE-WETH", "WETH"),
            ("WETH", "AAVE-WETH"),
            ("AAVE-WETH", "ETH"),
            ("ETH", "AAVE-WETH"),
            
            # SUSHI-WETH LP token pairs
            ("SUSHI-WETH", "SUSHI"),
            ("SUSHI", "SUSHI-WETH"),
            ("SUSHI-WETH", "WETH"),
            ("WETH", "SUSHI-WETH"),
            ("SUSHI-WETH", "ETH"),
            ("ETH", "SUSHI-WETH"),
            
            # MKR-WETH LP token pairs
            ("MKR-WETH", "MKR"),
            ("MKR", "MKR-WETH"),
            ("MKR-WETH", "WETH"),
            ("WETH", "MKR-WETH"),
            ("MKR-WETH", "ETH"),
            ("ETH", "MKR-WETH"),
            
            # USDC-USDT LP token pairs (stablecoin arbitrage)
            ("USDC-USDT", "USDC"),
            ("USDC", "USDC-USDT"),
            ("USDC-USDT", "USDT"),
            ("USDT", "USDC-USDT"),
            
            # Cross LP token pairs (LP arbitrage opportunities)
            ("WETH-USDC", "WETH-USDT"),
            ("WETH-USDT", "WETH-USDC"),
            ("WETH-USDC", "WETH-DAI"),
            ("WETH-DAI", "WETH-USDC"),
            ("WETH-USDT", "WETH-DAI"),
            ("WETH-DAI", "WETH-USDT"),
            
            # Curve LP token pairs with underlying assets
            
            # 3CRV (DAI+USDC+USDT) LP token pairs - Most important Curve LP
            ("3CRV", "DAI"),
            ("DAI", "3CRV"),
            ("3CRV", "USDC"),
            ("USDC", "3CRV"),
            ("3CRV", "USDT"),
            ("USDT", "3CRV"),
            ("3CRV", "WETH"),
            ("WETH", "3CRV"),
            ("3CRV", "ETH"),
            ("ETH", "3CRV"),
            
            # crvUSDC (Curve Compound USDC) LP token pairs
            ("crvUSDC", "USDC"),
            ("USDC", "crvUSDC"),
            ("crvUSDC", "DAI"),
            ("DAI", "crvUSDC"),
            ("crvUSDC", "cUSDC"),
            ("cUSDC", "crvUSDC"),
            
            # YCRV (Curve yEarn) LP token pairs
            ("yDAI+yUSDC+yUSDT+yTUSD", "DAI"),
            ("DAI", "yDAI+yUSDC+yUSDT+yTUSD"),
            ("yDAI+yUSDC+yUSDT+yTUSD", "USDC"),
            ("USDC", "yDAI+yUSDC+yUSDT+yTUSD"),
            ("yDAI+yUSDC+yUSDT+yTUSD", "USDT"),
            ("USDT", "yDAI+yUSDC+yUSDT+yTUSD"),
            
            # Curve metapool LP tokens with 3CRV (meta arbitrage)
            ("GUSD3CRV", "3CRV"),
            ("3CRV", "GUSD3CRV"),
            ("HUSD3CRV", "3CRV"),
            ("3CRV", "HUSD3CRV"),
            ("USDK3CRV", "3CRV"),
            ("3CRV", "USDK3CRV"),
            ("USDN3CRV", "3CRV"),
            ("3CRV", "USDN3CRV"),
            ("USDP3CRV", "3CRV"),
            ("3CRV", "USDP3CRV"),
            
            # Synthetic assets pairs (Synthetix)
            # sUSD - Synthetic USD (Synthetix stablecoin)
            ("ETH", "sUSD"),
            ("sUSD", "ETH"),
            ("WETH", "sUSD"),
            ("sUSD", "WETH"),
            ("sUSD", "USDC"),
            ("USDC", "sUSD"),
            ("sUSD", "USDT"),
            ("USDT", "sUSD"),
            ("sUSD", "DAI"),
            ("DAI", "sUSD"),
            
            # sETH - Synthetic ETH (Synthetix synthetic Ether)
            ("ETH", "sETH"),
            ("sETH", "ETH"),
            ("WETH", "sETH"),
            ("sETH", "WETH"),
            ("sETH", "USDC"),
            ("USDC", "sETH"),
            ("sETH", "USDT"),
            ("USDT", "sETH"),
            ("sETH", "DAI"),
            ("DAI", "sETH"),
            ("sETH", "WBTC"),
            ("WBTC", "sETH"),
            
            # Cross synthetic asset pairs
            ("sUSD", "sETH"),
            ("sETH", "sUSD"),
            
            # Curve BTC LP tokens with BTC assets
            ("crvRenWSBTC", "WBTC"),
            ("WBTC", "crvRenWSBTC"),
            ("crvHBTC", "WBTC"),
            ("WBTC", "crvHBTC"),
            
            # Cross Curve LP token arbitrage
            ("3CRV", "crvUSDC"),
            ("crvUSDC", "3CRV"),
            ("3CRV", "yDAI+yUSDC+yUSDT+yTUSD"),
            ("yDAI+yUSDC+yUSDT+yTUSD", "3CRV"),
            ("crvRenWSBTC", "crvHBTC"),
            ("crvHBTC", "crvRenWSBTC"),
            
            # Mixed LP token arbitrage (Uniswap vs Curve)
            ("WETH-USDC", "3CRV"),
            ("3CRV", "WETH-USDC"),
            ("WETH-USDT", "3CRV"),
            ("3CRV", "WETH-USDT"),
            ("WETH-DAI", "3CRV"),
            ("3CRV", "WETH-DAI"),
            ("USDC-USDT", "3CRV"),
            ("3CRV", "USDC-USDT"),
            
            # LP tokens with stablecoins (for yield arbitrage and unwinding)
            ("WETH-USDC", "USDC"),
            ("USDC", "WETH-USDC"),
            ("WETH-USDT", "USDT"),
            ("USDT", "WETH-USDT"),
            ("WETH-DAI", "DAI"),
            ("DAI", "WETH-DAI"),
            
            # LP tokens with major DeFi tokens (for governance token arbitrage)
            ("UNI-WETH", "COMP"),
            ("COMP", "UNI-WETH"),
            ("COMP-WETH", "AAVE"),
            ("AAVE", "COMP-WETH"),
            ("AAVE-WETH", "SUSHI"),
            ("SUSHI", "AAVE-WETH"),
            ("SUSHI-WETH", "UNI"),
            ("UNI", "SUSHI-WETH")
        ]
        
        # 주소로 변환
        address_pairs = []
        for symbol0, symbol1 in major_pairs:
            addr0 = self.get_address_by_symbol(symbol0)
            addr1 = self.get_address_by_symbol(symbol1)
            if addr0 and addr1:
                address_pairs.append((addr0, addr1))
            else:
                # 디버깅용 로그
                if not addr0:
                    logger.warning(f"주소를 찾을 수 없는 심볼: {symbol0}")
                if not addr1:
                    logger.warning(f"주소를 찾을 수 없는 심볼: {symbol1}")
        
        logger.info(f"총 {len(address_pairs)}개의 거래 쌍 생성 (논문 기준 96개 protocol actions)")
        return address_pairs