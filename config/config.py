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
    max_gas_price: int = int(os.getenv('MAX_GAS_PRICE', '50000000000') or '50000000000')
    min_profit_threshold: float = float(os.getenv('MIN_PROFIT_THRESHOLD', '0.001') or '0.001')
    
    # 모니터링
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    redis_url: str = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
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
