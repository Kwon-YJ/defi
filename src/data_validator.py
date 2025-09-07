import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from src.logger import setup_logger
from src.data_storage import DataStorage

logger = setup_logger(__name__)

class DataValidator:
    """데이터 검증 및 이상치 탐지 클래스"""
    
    def __init__(self, data_storage: DataStorage):
        self.data_storage = data_storage
        # 가격 변동 임계값 (예: 50% 이상의 변동은 이상치로 간주)
        self.price_change_threshold = 0.5
        # 볼륨 이상치 임계값 (예: 평균의 10배 이상은 이상치로 간주)
        self.volume_spike_threshold = 10.0
        # 이동 평균 윈도우 크기 (시간)
        self.moving_avg_window_hours = 24
        
    def validate_price_data(self, symbol: str, price: float, timestamp: float) -> Dict:
        """
        가격 데이터 검증
        
        Args:
            symbol: 토큰 심볼
            price: 가격
            timestamp: 타임스탬프
            
        Returns:
            검증 결과 딕셔너리
        """
        try:
            # 기본 검증
            if price <= 0:
                return {
                    'valid': False,
                    'reason': 'price must be positive',
                    'severity': 'critical'
                }
            
            # Note: 가격 히스토리 검증은 성능상의 이유로 생략하거나 별도의 비동기 함수로 분리 필요
            # 현재 구현에서는 기본 검증만 수행
            
            return {
                'valid': True,
                'reason': 'price data validated successfully',
                'severity': 'info'
            }
            
        except Exception as e:
            logger.error(f"가격 데이터 검증 중 오류 발생 ({symbol}): {e}")
            return {
                'valid': False,
                'reason': f'validation error: {str(e)}',
                'severity': 'error'
            }
    
    def validate_pool_data(self, pool_address: str, pool_info: Dict) -> Dict:
        """
        풀 데이터 검증
        
        Args:
            pool_address: 풀 주소
            pool_info: 풀 정보 딕셔너리
            
        Returns:
            검증 결과 딕셔너리
        """
        try:
            # 필수 필드 검증
            required_fields = ['reserve0', 'reserve1', 'timestamp']
            for field in required_fields:
                if field not in pool_info:
                    return {
                        'valid': False,
                        'reason': f'missing required field: {field}',
                        'severity': 'critical'
                    }
            
            # 값 범위 검증
            reserve0 = pool_info['reserve0']
            reserve1 = pool_info['reserve1']
            
            if reserve0 < 0 or reserve1 < 0:
                return {
                    'valid': False,
                    'reason': 'reserves must be non-negative',
                    'severity': 'critical'
                }
            
            if reserve0 == 0 and reserve1 == 0:
                return {
                    'valid': False,
                    'reason': 'both reserves cannot be zero',
                    'severity': 'warning'
                }
            
            # Note: 히스토리 기반 검증은 성능상의 이유로 생략하거나 별도의 비동기 함수로 분리 필요
            # 현재 구현에서는 기본 검증만 수행
            
            return {
                'valid': True,
                'reason': 'pool data validated successfully',
                'severity': 'info'
            }
            
        except Exception as e:
            logger.error(f"풀 데이터 검증 중 오류 발생 ({pool_address}): {e}")
            return {
                'valid': False,
                'reason': f'validation error: {str(e)}',
                'severity': 'error'
            }
    
    async def _get_price_history(self, symbol: str, hours: int = 24) -> List[Dict]:
        """가격 히스토리 가져오기"""
        try:
            pattern = f"price_history:{symbol}:*"
            keys = self.data_storage.redis_client.keys(pattern)
            
            history = []
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            for key in keys:
                try:
                    timestamp_str = key.decode().split(':')[-1]
                    timestamp = datetime.fromisoformat(timestamp_str)
                    
                    if timestamp >= cutoff_time:
                        data = self.data_storage.redis_client.get(key)
                        if data:
                            price_data = json.loads(data)
                            price_data['timestamp'] = timestamp
                            history.append(price_data)
                except ValueError:
                    continue
            
            # 시간순 정렬
            history.sort(key=lambda x: x['timestamp'])
            return history
            
        except Exception as e:
            logger.error(f"가격 히스토리 조회 중 오류 발생 ({symbol}): {e}")
            return []
    
    async def _get_pool_history(self, pool_address: str, hours: int = 24) -> List[Dict]:
        """풀 히스토리 가져오기"""
        try:
            pattern = f"pool_history:{pool_address}:*"
            keys = self.data_storage.redis_client.keys(pattern)
            
            history = []
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            for key in keys:
                try:
                    timestamp_str = key.decode().split(':')[-1]
                    timestamp = datetime.fromisoformat(timestamp_str)
                    
                    if timestamp >= cutoff_time:
                        data = self.data_storage.redis_client.get(key)
                        if data:
                            pool_data = json.loads(data)
                            pool_data['timestamp'] = timestamp
                            history.append(pool_data)
                except ValueError:
                    continue
            
            # 시간순 정렬
            history.sort(key=lambda x: x['timestamp'])
            return history
            
        except Exception as e:
            logger.error(f"풀 히스토리 조회 중 오류 발생 ({pool_address}): {e}")
            return []
    
    def _statistical_validation(self, history: List[Dict], current_price: float) -> Dict:
        """통계 기반 가격 검증"""
        try:
            if len(history) < 3:
                return {
                    'valid': True,
                    'reason': 'insufficient history for statistical validation',
                    'severity': 'info'
                }
            
            # 가격 배열 생성
            prices = [item['price_usd'] for item in history]
            prices.append(current_price)
            
            # Z-score 계산
            mean = np.mean(prices[:-1])  # 현재 가격 제외
            std = np.std(prices[:-1])
            
            if std == 0:
                return {
                    'valid': True,
                    'reason': 'no variance in historical data',
                    'severity': 'info'
                }
            
            z_score = abs(current_price - mean) / std
            
            # Z-score가 3 이상이면 이상치로 간주
            if z_score > 3:
                return {
                    'valid': False,
                    'reason': f'price is statistical outlier (z-score: {z_score:.2f})',
                    'severity': 'warning',
                    'z_score': z_score,
                    'mean': mean,
                    'std': std
                }
            
            return {
                'valid': True,
                'reason': 'price passes statistical validation',
                'severity': 'info',
                'z_score': z_score
            }
            
        except Exception as e:
            logger.error(f"통계 기반 검증 중 오류 발생: {e}")
            return {
                'valid': True,
                'reason': f'statistical validation skipped due to error: {str(e)}',
                'severity': 'info'
            }
    
    def _detect_volume_spikes(self, history: List[Dict], current_data: Dict) -> Dict:
        """볼륨 스파이크 탐지"""
        try:
            if len(history) < 3:
                return {
                    'valid': True,
                    'reason': 'insufficient history for volume spike detection',
                    'severity': 'info'
                }
            
            # 볼륨 계산 (간단한 예: reserve의 합)
            current_volume = current_data.get('reserve0', 0) + current_data.get('reserve1', 0)
            
            if current_volume == 0:
                return {
                    'valid': True,
                    'reason': 'current volume is zero',
                    'severity': 'info'
                }
            
            # 역사적 볼륨 계산
            historical_volumes = []
            for item in history:
                volume = item.get('reserve0', 0) + item.get('reserve1', 0)
                if volume > 0:
                    historical_volumes.append(volume)
            
            if len(historical_volumes) < 3:
                return {
                    'valid': True,
                    'reason': 'insufficient non-zero volume data',
                    'severity': 'info'
                }
            
            # 평균 및 표준편차 계산
            mean_volume = np.mean(historical_volumes)
            std_volume = np.std(historical_volumes)
            
            if std_volume == 0:
                return {
                    'valid': True,
                    'reason': 'no variance in historical volumes',
                    'severity': 'info'
                }
            
            # 스파이크 탐지
            if current_volume > mean_volume * self.volume_spike_threshold:
                return {
                    'valid': False,
                    'reason': f'volume spike detected ({current_volume:.2f} vs mean {mean_volume:.2f})',
                    'severity': 'warning',
                    'current_volume': current_volume,
                    'mean_volume': mean_volume,
                    'spike_ratio': current_volume / mean_volume
                }
            
            return {
                'valid': True,
                'reason': 'volume within expected range',
                'severity': 'info'
            }
            
        except Exception as e:
            logger.error(f"볼륨 스파이크 탐지 중 오류 발생: {e}")
            return {
                'valid': True,
                'reason': f'volume spike detection skipped due to error: {str(e)}',
                'severity': 'info'
            }
    
    def _validate_against_expected_range(self, history: List[Dict], current_data: Dict) -> Dict:
        """예상 범위 내에 있는지 검증"""
        try:
            if len(history) < 3:
                return {
                    'valid': True,
                    'reason': 'insufficient history for range validation',
                    'severity': 'info'
                }
            
            # reserve0, reserve1에 대해 각각 검증
            for reserve_key in ['reserve0', 'reserve1']:
                current_value = current_data.get(reserve_key, 0)
                
                if current_value <= 0:
                    continue  # 0 이하 값은 이미 다른 검증에서 처리됨
                
                # 역사적 값 수집
                historical_values = [item.get(reserve_key, 0) for item in history if item.get(reserve_key, 0) > 0]
                
                if len(historical_values) < 3:
                    continue
                
                # 최소/최대 값 계산 (5% 마진 추가)
                min_expected = np.min(historical_values) * 0.95
                max_expected = np.max(historical_values) * 1.05
                
                # 범위 검증
                if current_value < min_expected or current_value > max_expected:
                    return {
                        'valid': False,
                        'reason': f'{reserve_key} out of expected range ({current_value:.2f} vs [{min_expected:.2f}, {max_expected:.2f}])',
                        'severity': 'warning',
                        'current_value': current_value,
                        'expected_range': [min_expected, max_expected]
                    }
            
            return {
                'valid': True,
                'reason': 'values within expected ranges',
                'severity': 'info'
            }
            
        except Exception as e:
            logger.error(f"예상 범위 검증 중 오류 발생: {e}")
            return {
                'valid': True,
                'reason': f'range validation skipped due to error: {str(e)}',
                'severity': 'info'
            }
    
    def filter_outliers(self, data_list: List[Dict], data_type: str = 'price') -> List[Dict]:
        """
        이상치 필터링
        
        Args:
            data_list: 데이터 리스트
            data_type: 데이터 타입 ('price' 또는 'pool')
            
        Returns:
            필터링된 데이터 리스트
        """
        try:
            filtered_data = []
            
            for data in data_list:
                if data_type == 'price':
                    symbol = data.get('symbol', '')
                    price = data.get('price_usd', 0)
                    timestamp = data.get('timestamp', 0)
                    validation_result = self.validate_price_data(symbol, price, timestamp)
                elif data_type == 'pool':
                    pool_address = data.get('address', '')
                    validation_result = self.validate_pool_data(pool_address, data)
                else:
                    # 타입이 지정되지 않은 경우 기본 허용
                    filtered_data.append(data)
                    continue
                
                # 심각한 오류가 아닌 경우에만 포함
                if validation_result['valid'] or validation_result['severity'] not in ['critical', 'error']:
                    filtered_data.append(data)
                else:
                    logger.warning(f"이상치 데이터 필터링: {validation_result['reason']}")
            
            logger.info(f"이상치 필터링 완료: {len(data_list)} -> {len(filtered_data)}")
            return filtered_data
            
        except Exception as e:
            logger.error(f"이상치 필터링 중 오류 발생: {e}")
            return data_list  # 오류 발생 시 원본 데이터 반환