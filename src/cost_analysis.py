from typing import Dict
from dataclasses import dataclass
from src.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class HardwareConfig:
    name: str
    cpu: str
    ram: str
    storage: str
    network: str
    monthly_cost: float
    initial_cost: float

@dataclass
class CloudConfig:
    name: str
    provider: str
    instance_type: str
    monthly_cost: float
    setup_cost: float

class CostAnalyzer:
    def __init__(self):
        self.hardware_options = self._load_hardware_options()
        self.cloud_options = self._load_cloud_options()
        self.rpc_services = self._load_rpc_services()
        
    def _load_hardware_options(self) -> Dict[str, HardwareConfig]:
        """하드웨어 옵션 로드"""
        return {
            'minimum': HardwareConfig(
                name="최소 사양",
                cpu="Intel i5-12400 (6코어)",
                ram="16GB DDR4",
                storage="1TB NVMe SSD",
                network="1Gbps",
                monthly_cost=50.0,  # 전기료 등
                initial_cost=800.0
            ),
            'recommended': HardwareConfig(
                name="권장 사양",
                cpu="Intel i7-12700K (12코어)",
                ram="32GB DDR4-3200",
                storage="2TB NVMe SSD",
                network="1Gbps",
                monthly_cost=80.0,
                initial_cost=1550.0
            ),
            'high_performance': HardwareConfig(
                name="고성능 사양",
                cpu="Intel i9-12900K (16코어)",
                ram="64GB DDR4-3600",
                storage="4TB NVMe SSD",
                network="10Gbps",
                monthly_cost=120.0,
                initial_cost=3000.0
            )
        }
    
    def _load_cloud_options(self) -> Dict[str, CloudConfig]:
        """클라우드 옵션 로드"""
        return {
            'aws_small': CloudConfig(
                name="AWS t3.large",
                provider="AWS",
                instance_type="t3.large (2 vCPU, 8GB RAM)",
                monthly_cost=60.0,
                setup_cost=0.0
            ),
            'aws_medium': CloudConfig(
                name="AWS c5.xlarge", 
                provider="AWS",
                instance_type="c5.xlarge (4 vCPU, 8GB RAM)",
                monthly_cost=120.0,
                setup_cost=0.0
            ),
            'gcp_small': CloudConfig(
                name="GCP e2-standard-2",
                provider="Google Cloud",
                instance_type="e2-standard-2 (2 vCPU, 8GB RAM)",
                monthly_cost=50.0,
                setup_cost=0.0
            ),
            'azure_medium': CloudConfig(
                name="Azure Standard_D2s_v3",
                provider="Microsoft Azure", 
                instance_type="Standard_D2s_v3 (2 vCPU, 8GB RAM)",
                monthly_cost=70.0,
                setup_cost=0.0
            )
        }
    
    def _load_rpc_services(self) -> Dict[str, Dict]:
        """RPC 서비스 옵션 로드"""
        return {
            'alchemy_growth': {
                'name': 'Alchemy Growth',
                'monthly_cost': 199.0,
                'requests_per_month': 40000000,
                'websocket': True,
                'archive_data': True
            },
            'quicknode_build': {
                'name': 'QuickNode Build',
                'monthly_cost': 49.0,
                'requests_per_month': 5000000,
                'websocket': True,
                'archive_data': False
            },
            'infura_developer': {
                'name': 'Infura Developer',
                'monthly_cost': 50.0,
                'requests_per_month': 3000000,
                'websocket': True,
                'archive_data': False
            }
        }
    
    def calculate_total_cost_comparison(self, months: int = 12) -> Dict:
        """총 비용 비교 분석"""
        results = {}
        
        # 하드웨어 + 프리미엄 RPC 조합
        for hw_key, hw_config in self.hardware_options.items():
            for rpc_key, rpc_config in self.rpc_services.items():
                key = f"hardware_{hw_key}_{rpc_key}"
                
                initial_cost = hw_config.initial_cost
                monthly_cost = hw_config.monthly_cost + rpc_config['monthly_cost']
                total_cost = initial_cost + (monthly_cost * months)
                
                results[key] = {
                    'type': 'hardware',
                    'config': hw_config.name,
                    'rpc_service': rpc_config['name'],
                    'initial_cost': initial_cost,
                    'monthly_cost': monthly_cost,
                    'total_cost_12m': total_cost,
                    'break_even_months': initial_cost / monthly_cost if monthly_cost > 0 else 0
                }
        
        # 클라우드 + 프리미엄 RPC 조합
        for cloud_key, cloud_config in self.cloud_options.items():
            for rpc_key, rpc_config in self.rpc_services.items():
                key = f"cloud_{cloud_key}_{rpc_key}"
                
                initial_cost = cloud_config.setup_cost
                monthly_cost = cloud_config.monthly_cost + rpc_config['monthly_cost']
                total_cost = initial_cost + (monthly_cost * months)
                
                results[key] = {
                    'type': 'cloud',
                    'config': cloud_config.name,
                    'rpc_service': rpc_config['name'],
                    'initial_cost': initial_cost,
                    'monthly_cost': monthly_cost,
                    'total_cost_12m': total_cost,
                    'break_even_months': 0  # 클라우드는 초기 비용이 거의 없음
                }
        
        return results
    
    def recommend_optimal_setup(self, expected_monthly_profit: float, 
                              risk_tolerance: str = 'medium') -> Dict:
        """최적 설정 추천"""
        cost_comparison = self.calculate_total_cost_comparison()
        
        recommendations = []
        
        for key, config in cost_comparison.items():
            # ROI 계산
            monthly_profit_after_cost = expected_monthly_profit - config['monthly_cost']
            annual_profit = monthly_profit_after_cost * 12
            roi = (annual_profit / config['total_cost_12m']) * 100 if config['total_cost_12m'] > 0 else 0
            
            # 위험도에 따른 필터링
            if risk_tolerance == 'low' and config['initial_cost'] > 1000:
                continue
            elif risk_tolerance == 'high' and roi < 50:
                continue
            
            recommendations.append({
                'setup': key,
                'config': config,
                'roi': roi,
                'monthly_net_profit': monthly_profit_after_cost,
                'payback_months': config['initial_cost'] / monthly_profit_after_cost if monthly_profit_after_cost > 0 else float('inf')
            })
        
        # ROI 기준으로 정렬
        recommendations.sort(key=lambda x: x['roi'], reverse=True)
        
        return {
            'best_roi': recommendations[0] if recommendations else None,
            'all_options': recommendations,
            'analysis_summary': self._generate_cost_summary(recommendations)
        }
    
    def _generate_cost_summary(self, recommendations: List[Dict]) -> Dict:
        """비용 분석 요약 생성"""
        if not recommendations:
            return {'message': '추천할 수 있는 설정이 없습니다'}
        
        hardware_options = [r for r in recommendations if r['config']['type'] == 'hardware']
        cloud_options = [r for r in recommendations if r['config']['type'] == 'cloud']
        
        return {
            'total_options': len(recommendations),
            'hardware_options': len(hardware_options),
            'cloud_options': len(cloud_options),
            'best_hardware_roi': max(hardware_options, key=lambda x: x['roi'])['roi'] if hardware_options else 0,
            'best_cloud_roi': max(cloud_options, key=lambda x: x['roi'])['roi'] if cloud_options else 0,
            'recommendation': self._get_final_recommendation(recommendations)
        }
    
    def _get_final_recommendation(self, recommendations: List[Dict]) -> str:
        """최종 추천 메시지"""
        if not recommendations:
            return "수익성이 확보되지 않아 투자를 권장하지 않습니다."
        
        best = recommendations[0]
        
        if best['roi'] > 100:
            return f"강력 추천: {best['setup']} (ROI: {best['roi']:.1f}%)"
        elif best['roi'] > 50:
            return f"추천: {best['setup']} (ROI: {best['roi']:.1f}%)"
        elif best['roi'] > 20:
            return f"조건부 추천: {best['setup']} (ROI: {best['roi']:.1f}%)"
        else:
            return "현재 시장 상황에서는 투자를 권장하지 않습니다."
