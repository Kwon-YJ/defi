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
    # dYdX risk/fee modeling (fallbacks; prefer official API if configured)
    dydx_api_endpoint: str = os.getenv('DYDX_API_ENDPOINT', '')
    dydx_default_taker_fee: float = float(os.getenv('DYDX_TAKER_FEE', '0.0005'))
    dydx_default_funding_per_hour: float = float(os.getenv('DYDX_FUNDING_PER_HOUR', '0.0001'))
    dydx_hold_hours: int = int(os.getenv('DYDX_HOLD_HOURS', '2'))
    dydx_initial_margin: float = float(os.getenv('DYDX_INITIAL_MARGIN', '0.1'))
    dydx_maintenance_margin: float = float(os.getenv('DYDX_MAINTENANCE_MARGIN', '0.05'))
    dydx_max_leverage: float = float(os.getenv('DYDX_MAX_LEVERAGE', '10'))
    dydx_desired_leverage: float = float(os.getenv('DYDX_DESIRED_LEVERAGE', '3'))
    
    # 논문(2103.02228) 25자산 사용 모드
    use_paper_25_assets: bool = os.getenv('USE_PAPER_25_ASSETS', '0') in ('1', 'true', 'True')
    # Curve 풀 화이트리스트 (쉼표 구분 주소). 빈 값이면 전체 허용.
    curve_pool_whitelist: str = os.getenv('CURVE_POOL_WHITELIST', '0xbEbc44782C7dB0a1A60Cb6fe97d0a2fEdcBcd44,0xA5407eAE9Ba41422680e2e00537571bcC53efBfD')
    # 주요 토큰/디파이 토큰 자동 포함 플래그
    include_major_tokens: bool = os.getenv('INCLUDE_MAJOR_TOKENS', '1') in ('1','true','True')
    include_defi_tokens: bool = os.getenv('INCLUDE_DEFI_TOKENS', '1') in ('1','true','True')
    include_synth_tokens: bool = os.getenv('INCLUDE_SYNTH_TOKENS', '1') in ('1','true','True')
    include_extra_tokens: bool = os.getenv('INCLUDE_EXTRA_TOKENS', '1') in ('1','true','True')
    # Curve liquidity scaling tuning
    curve_liq_scale_ref: float = float(os.getenv('CURVE_LIQ_SCALE_REF', '1000000'))  # reference LP totalSupply
    curve_liq_scale_min: float = float(os.getenv('CURVE_LIQ_SCALE_MIN', '0.25'))
    curve_liq_scale_max: float = float(os.getenv('CURVE_LIQ_SCALE_MAX', '3.0'))
    # Price feed validation / smoothing
    price_ema_alpha: float = float(os.getenv('PRICE_EMA_ALPHA', '0.2'))
    price_jump_max_pct: float = float(os.getenv('PRICE_JUMP_MAX_PCT', '0.2'))  # 20% per tick cap
    price_stable_max_dev: float = float(os.getenv('PRICE_STABLE_MAX_DEV', '0.03'))  # 3% around $1
    # Graph building concurrency
    graph_build_concurrency: int = int(os.getenv('GRAPH_BUILD_CONCURRENCY', '16'))
    
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
