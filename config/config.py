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
    # Uniswap V3 band tuning
    v3_band_hops: int = int(os.getenv('V3_BAND_HOPS', '0'))  # 0=active only, 1=±1 bands
    # Price feed validation / smoothing
    price_ema_alpha: float = float(os.getenv('PRICE_EMA_ALPHA', '0.2'))
    price_jump_max_pct: float = float(os.getenv('PRICE_JUMP_MAX_PCT', '0.2'))  # 20% per tick cap
    price_stable_max_dev: float = float(os.getenv('PRICE_STABLE_MAX_DEV', '0.03'))  # 3% around $1
    # Price TTL / history policy
    price_ttl_sec: int = int(os.getenv('PRICE_TTL_SEC', '60'))
    price_hist_ttl_sec: int = int(os.getenv('PRICE_HIST_TTL_SEC', str(30 * 24 * 3600)))
    price_hist_maxlen: int = int(os.getenv('PRICE_HIST_MAXLEN', '100000'))
    # Block alert settings
    enable_block_alerts: bool = os.getenv('ENABLE_BLOCK_ALERTS', '1') in ('1','true','True')
    block_alert_channel: str = os.getenv('BLOCK_ALERT_CHANNEL', 'blocks')
    blocks_recent_key: str = os.getenv('BLOCKS_RECENT_KEY', 'blocks_recent')
    blocks_recent_maxlen: int = int(os.getenv('BLOCKS_RECENT_MAXLEN', '1000'))
    # Graph building concurrency
    graph_build_concurrency: int = int(os.getenv('GRAPH_BUILD_CONCURRENCY', '16'))
    # Memory compaction
    memory_compact_meta: bool = os.getenv('MEMORY_COMPACT_META', '1') in ('1','true','True')
    memory_meta_keep_keys: str = os.getenv('MEMORY_META_KEEP_KEYS', 'contract,fee_tier,source,confidence,lp_token,risk_fot,risk_rebase')
    # Block processing budget (must be < 13.5s)
    block_time_budget_sec: float = float(os.getenv('BLOCK_TIME_BUDGET_SEC', '10.8'))
    
    # 수익 목표 (논문 기준)
    # 주간 평균 191.48 ETH, 76,592 USD
    weekly_profit_target_eth: float = float(os.getenv('WEEKLY_PROFIT_TARGET_ETH', '191.48'))
    weekly_profit_target_usd: float = float(os.getenv('WEEKLY_PROFIT_TARGET_USD', '76592'))
    # 단일 거래 최고 수익 목표 (논문 기준)
    single_trade_profit_target_eth: float = float(os.getenv('SINGLE_TRADE_PROFIT_TARGET_ETH', '81.31'))
    single_trade_profit_target_usd: float = float(os.getenv('SINGLE_TRADE_PROFIT_TARGET_USD', '32524'))

    # 알림/모니터링 설정
    alert_enable_stdout: bool = os.getenv('ALERT_ENABLE_STDOUT', '1') in ('1','true','True')
    alert_log_path: str = os.getenv('ALERT_LOG_PATH', 'reports/alerts.log')
    monitor_interval_sec: int = int(os.getenv('MONITOR_INTERVAL_SEC', '60'))
    dashboard_output_dir: str = os.getenv('DASHBOARD_OUTPUT_DIR', 'reports')
    dashboard_title: str = os.getenv('DASHBOARD_TITLE', 'DeFi Arbitrage Dashboard')
    # ROI 추적
    roi_initial_capital_eth: float = float(os.getenv('ROI_INITIAL_CAPITAL_ETH', '1.0'))
    roi_alert_max_drawdown_pct: float = float(os.getenv('ROI_ALERT_MAX_DRAWDOWN_PCT', '50'))

    # Flash arbitrage 설정
    enable_flash_arbitrage: bool = os.getenv('ENABLE_FLASH_ARB', '1') in ('1', 'true', 'True')
    flash_dry_run: bool = os.getenv('FLASH_DRY_RUN', '1') in ('1', 'true', 'True')
    flash_min_profit_eth: float = float(os.getenv('FLASH_MIN_PROFIT_ETH', '0.01'))
    flash_min_confidence: float = float(os.getenv('FLASH_MIN_CONFIDENCE', '0.7'))
    flash_deploy_on_start: bool = os.getenv('FLASH_DEPLOY_ON_START', '0') in ('1', 'true', 'True')
    flash_contract_address: str = os.getenv('FLASH_ARB_ADDRESS', '')
    flash_arb_aave_v3_address: str = os.getenv('FLASH_ARB_AAVE_V3_ADDRESS', '')
    # Flash provider 선택: 'aave' | 'dydx'
    flash_provider: str = os.getenv('FLASH_PROVIDER', 'aave')
    # dYdX SoloMargin 주소 (메인넷)
    dydx_solo_margin: str = os.getenv('DYDX_SOLO_MARGIN', '0x1e0447b19bb6ecfdae1e4ae1694b0c3659614e4e')
    
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
