import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # RPC 설정
    alchemy_api_key: str = os.getenv('ALCHEMY_API_KEY', '')
    quicknode_endpoint: str = os.getenv('QUICKNODE_ENDPOINT', '')
    infura_project_id: str = os.getenv('INFURA_PROJECT_ID', '')
    
    # 네트워크 설정
    ethereum_mainnet_rpc: str = os.getenv('ETHEREUM_MAINNET_RPC', '')
    ethereum_mainnet_ws: str = os.getenv('ETHEREUM_MAINNET_WS', '')
    goerli_rpc: str = os.getenv('GOERLI_RPC', '')
    holesky_rpc: str = os.getenv('HOLESKY_RPC', 'https://ethereum-holesky-rpc.publicnode.com')
    sepolia_rpc: str = os.getenv('SEPOLIA_RPC', 'https://ethereum-sepolia-rpc.publicnode.com')
    
    # 거래 설정
    private_key: str = os.getenv('PRIVATE_KEY', '')
    max_gas_price: int = int(os.getenv('MAX_GAS_PRICE', '50000000000'))
    min_profit_threshold: float = float(os.getenv('MIN_PROFIT_THRESHOLD', '0.001'))
    
    # 모니터링
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    redis_url: str = os.getenv('REDIS_URL', 'redis://localhost:6379')
    # Slippage modeling
    slippage_trade_fraction: float = float(os.getenv('SLIPPAGE_TRADE_FRACTION', '0.01'))  # 1% reference size
    # Aave v2 Data Provider (mainnet default)
    aave_v2_data_provider: str = os.getenv('AAVE_V2_DATA_PROVIDER', '0x057835Ad21a177dbdd3090bB1CAE03EaCF78Fc6d')
    # V3 tier 선택 로직 가중치
    v3_score_weight_ema: float = float(os.getenv('V3_SCORE_W_EMA', '1.0'))
    v3_score_weight_liq: float = float(os.getenv('V3_SCORE_W_LIQ', '0.001'))
    # Aave v3 eMode LTV overrides mapping like "1:0.97,2:0.98"
    aave_emode_ltv_overrides: str = os.getenv('AAVE_EMODE_LTV_OVERRIDES', '')
    # Interest/constraint modeling
    interest_hold_blocks: int = int(os.getenv('INTEREST_HOLD_BLOCKS', '100'))
    # Compound Comptroller (mainnet default)
    compound_comptroller: str = os.getenv('COMPOUND_COMPTROLLER', '0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B')
    # Maker PSM settings (optional override)
    maker_psm_usdc: str = os.getenv('MAKER_PSM_USDC', '')
    maker_hold_seconds: int = int(os.getenv('MAKER_HOLD_SECONDS', '3600'))
    # Synthetix
    snx_system_settings: str = os.getenv('SNX_SYSTEM_SETTINGS', '')
    snx_debt_cache: str = os.getenv('SNX_DEBT_CACHE', '')
    
    def validate(self) -> bool:
        """설정 유효성 검사"""
        required_fields = [
            self.alchemy_api_key,
            self.ethereum_mainnet_rpc,
            self.private_key
        ]
        return all(field for field in required_fields)

# 전역 설정 인스턴스
config = Config()

# 개발 중에는 validation 비활성화
# if not config.validate():
#     raise ValueError("필수 환경 변수가 설정되지 않았습니다.")
