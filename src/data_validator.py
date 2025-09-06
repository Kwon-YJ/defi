"""
Data Validation and Outlier Detection Module
TODO requirement completion: Data validation 및 outlier detection

이 모듈은 논문 [2103.02228]의 DeFiPoser-ARB 시스템을 완전히 재현하기 위한
고급 데이터 검증 및 이상값 탐지 시스템을 구현합니다.

Features:
- Statistical outlier detection (IQR, Z-score, Modified Z-score)
- Time-series anomaly detection
- Cross-source data validation
- Price volatility analysis
- Data quality scoring
- Confidence interval estimation
"""

import asyncio
import statistics
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, deque
from decimal import Decimal
import time
import math

from src.logger import setup_logger
from src.real_time_price_feeds import PriceFeed, RealTimePriceFeeds

logger = setup_logger(__name__)

@dataclass
class ValidationResult:
    """데이터 검증 결과"""
    is_valid: bool
    confidence_score: float  # 0.0 ~ 1.0
    validation_errors: List[str]
    outlier_score: float  # 0.0 ~ 1.0 (높을수록 outlier 가능성 높음)
    timestamp: float
    
@dataclass 
class AnomalyAlert:
    """이상값 알림"""
    token_address: str
    symbol: str
    current_price: float
    expected_price_range: Tuple[float, float]
    anomaly_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: float
    sources: List[str]
    
class DataValidator:
    """
    고급 데이터 검증 및 이상값 탐지 시스템
    
    논문의 96개 protocol actions과 25개 assets 처리를 위한
    확장 가능한 데이터 품질 관리 시스템
    """
    
    def __init__(self):
        # 가격 히스토리 저장소 (토큰별 최근 1000개 데이터포인트)
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # 소스별 신뢰도 점수
        self.source_reliability: Dict[str, float] = defaultdict(lambda: 0.8)
        
        # 검증 통계
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'outliers_detected': 0,
            'anomalies_by_token': defaultdict(int),
            'anomalies_by_source': defaultdict(int)
        }
        
        # 설정 매개변수
        self.config = {
            # Outlier detection thresholds
            'iqr_multiplier': 2.0,           # IQR method의 배수
            'zscore_threshold': 3.0,         # Z-score 임계값
            'modified_zscore_threshold': 3.5, # Modified Z-score 임계값
            
            # Price volatility limits
            'max_price_change_1min': 0.20,   # 1분 내 최대 20% 변화
            'max_price_change_5min': 0.50,   # 5분 내 최대 50% 변화
            'max_price_change_1hour': 1.00,  # 1시간 내 최대 100% 변화
            
            # Volume analysis
            'min_volume_ratio': 0.1,         # 최소 거래량 비율
            'max_volume_spike': 10.0,        # 최대 거래량 스파이크 배수
            
            # Confidence scoring
            'min_sources_for_validation': 2,  # 검증을 위한 최소 소스 수
            'source_consensus_threshold': 0.05, # 소스 간 합의 임계값 (5%)
            
            # Time-series analysis
            'trend_analysis_window': 30,     # 트렌드 분석 윈도우 (분)
            'volatility_window': 60,         # 변동성 분석 윈도우 (분)
            
            # Alert thresholds
            'low_severity_threshold': 0.3,
            'medium_severity_threshold': 0.6,
            'high_severity_threshold': 0.8
        }
        
    async def validate_price_feed(self, price_feed: PriceFeed, 
                                cross_reference_feeds: Optional[List[PriceFeed]] = None) -> ValidationResult:
        """
        가격 피드 종합 검증
        
        Args:
            price_feed: 검증할 가격 피드
            cross_reference_feeds: 교차 검증용 다른 소스들의 피드
            
        Returns:
            ValidationResult: 검증 결과
        """
        try:
            self.validation_stats['total_validations'] += 1
            
            validation_errors = []
            confidence_scores = []
            outlier_scores = []
            
            # 1. 기본 데이터 무결성 검증
            basic_validation = self._validate_basic_data_integrity(price_feed)
            validation_errors.extend(basic_validation['errors'])
            confidence_scores.append(basic_validation['confidence'])
            
            # 2. Historical outlier detection
            historical_validation = await self._detect_historical_outliers(price_feed)
            validation_errors.extend(historical_validation['errors'])
            confidence_scores.append(historical_validation['confidence'])
            outlier_scores.append(historical_validation['outlier_score'])
            
            # 3. Cross-source validation (if available)
            if cross_reference_feeds:
                cross_validation = await self._validate_cross_source_consensus(
                    price_feed, cross_reference_feeds
                )
                validation_errors.extend(cross_validation['errors'])
                confidence_scores.append(cross_validation['confidence'])
                outlier_scores.append(cross_validation['outlier_score'])
                
            # 4. Time-series anomaly detection
            timeseries_validation = await self._detect_timeseries_anomalies(price_feed)
            validation_errors.extend(timeseries_validation['errors'])
            confidence_scores.append(timeseries_validation['confidence'])
            outlier_scores.append(timeseries_validation['outlier_score'])
            
            # 5. Volume consistency check
            volume_validation = self._validate_volume_consistency(price_feed)
            validation_errors.extend(volume_validation['errors'])
            confidence_scores.append(volume_validation['confidence'])
            
            # 종합 점수 계산
            final_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            final_outlier_score = np.mean(outlier_scores) if outlier_scores else 0.0
            
            # 검증 결과 결정
            is_valid = len(validation_errors) == 0 and final_confidence >= 0.5
            
            # 통계 업데이트
            if is_valid:
                self.validation_stats['passed_validations'] += 1
                # 소스 신뢰도 증가
                self.source_reliability[price_feed.source] = min(1.0, 
                    self.source_reliability[price_feed.source] + 0.01)
            else:
                self.validation_stats['failed_validations'] += 1
                # 소스 신뢰도 감소
                self.source_reliability[price_feed.source] = max(0.0,
                    self.source_reliability[price_feed.source] - 0.05)
                
            if final_outlier_score > 0.7:
                self.validation_stats['outliers_detected'] += 1
                self.validation_stats['anomalies_by_token'][price_feed.token_address] += 1
                self.validation_stats['anomalies_by_source'][price_feed.source] += 1
                
            # 가격 히스토리 업데이트 (검증된 데이터만)
            if is_valid:
                self._update_price_history(price_feed)
                
            return ValidationResult(
                is_valid=is_valid,
                confidence_score=final_confidence,
                validation_errors=validation_errors,
                outlier_score=final_outlier_score,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"가격 피드 검증 실패: {e}")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                validation_errors=[f"Validation error: {str(e)}"],
                outlier_score=1.0,
                timestamp=time.time()
            )
            
    def _validate_basic_data_integrity(self, price_feed: PriceFeed) -> Dict:
        """기본 데이터 무결성 검증"""
        errors = []
        confidence = 1.0
        
        try:
            # 가격 검증
            if price_feed.price_usd <= 0:
                errors.append("Invalid price: price must be positive")
                confidence -= 0.5
                
            if price_feed.price_usd > 1e12:  # 1조 달러 이상은 비현실적
                errors.append("Unrealistic price: exceeds reasonable maximum")
                confidence -= 0.3
                
            # 타임스탬프 검증
            current_time = time.time()
            if price_feed.timestamp > current_time + 300:  # 미래 5분 이상
                errors.append("Invalid timestamp: data from future")
                confidence -= 0.4
                
            if current_time - price_feed.timestamp > 3600:  # 1시간 이상 오래됨
                errors.append("Stale data: timestamp too old")
                confidence -= 0.2
                
            # 볼륨 검증
            if price_feed.volume_24h < 0:
                errors.append("Invalid volume: negative volume")
                confidence -= 0.1
                
            # 시가총액 검증
            if price_feed.market_cap < 0:
                errors.append("Invalid market cap: negative market cap")
                confidence -= 0.1
                
            # 신뢰도 검증
            if not 0 <= price_feed.confidence <= 1:
                errors.append("Invalid confidence: must be between 0 and 1")
                confidence -= 0.2
                
        except Exception as e:
            errors.append(f"Basic validation error: {str(e)}")
            confidence = 0.0
            
        return {
            'errors': errors,
            'confidence': max(0.0, confidence)
        }
        
    async def _detect_historical_outliers(self, price_feed: PriceFeed) -> Dict:
        """히스토리컬 데이터 기반 이상값 탐지"""
        errors = []
        confidence = 1.0
        outlier_score = 0.0
        
        try:
            history = self.price_history.get(price_feed.token_address.lower(), deque())
            
            if len(history) < 10:  # 충분한 히스토리가 없음
                return {
                    'errors': [],
                    'confidence': 0.7,  # 낮은 신뢰도
                    'outlier_score': 0.0
                }
                
            # 최근 가격들 추출
            recent_prices = [entry['price'] for entry in list(history)[-50:]]
            current_price = price_feed.price_usd
            
            # 1. IQR Method
            iqr_result = self._detect_outlier_iqr(current_price, recent_prices)
            if iqr_result['is_outlier']:
                errors.append(f"IQR outlier detected: {iqr_result['reason']}")
                outlier_score = max(outlier_score, 0.7)
                confidence -= 0.3
                
            # 2. Z-Score Method
            zscore_result = self._detect_outlier_zscore(current_price, recent_prices)
            if zscore_result['is_outlier']:
                errors.append(f"Z-score outlier detected: {zscore_result['reason']}")
                outlier_score = max(outlier_score, 0.6)
                confidence -= 0.2
                
            # 3. Modified Z-Score Method (more robust)
            modified_zscore_result = self._detect_outlier_modified_zscore(current_price, recent_prices)
            if modified_zscore_result['is_outlier']:
                errors.append(f"Modified Z-score outlier detected: {modified_zscore_result['reason']}")
                outlier_score = max(outlier_score, 0.8)
                confidence -= 0.4
                
            # 4. Volatility-based detection
            volatility_result = self._detect_excessive_volatility(price_feed, history)
            if volatility_result['is_anomaly']:
                errors.append(f"Excessive volatility detected: {volatility_result['reason']}")
                outlier_score = max(outlier_score, 0.5)
                confidence -= 0.2
                
        except Exception as e:
            errors.append(f"Historical outlier detection error: {str(e)}")
            confidence = 0.0
            outlier_score = 1.0
            
        return {
            'errors': errors,
            'confidence': max(0.0, confidence),
            'outlier_score': outlier_score
        }
        
    def _detect_outlier_iqr(self, current_price: float, prices: List[float]) -> Dict:
        """IQR 방법으로 이상값 탐지"""
        try:
            if len(prices) < 4:
                return {'is_outlier': False, 'reason': ''}
                
            sorted_prices = sorted(prices)
            q1 = sorted_prices[len(prices) // 4]
            q3 = sorted_prices[3 * len(prices) // 4]
            iqr = q3 - q1
            
            multiplier = self.config['iqr_multiplier']
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            
            is_outlier = current_price < lower_bound or current_price > upper_bound
            
            if is_outlier:
                reason = f"Price {current_price:.6f} outside IQR bounds [{lower_bound:.6f}, {upper_bound:.6f}]"
                return {'is_outlier': True, 'reason': reason}
            else:
                return {'is_outlier': False, 'reason': ''}
                
        except Exception as e:
            return {'is_outlier': True, 'reason': f'IQR calculation error: {str(e)}'}
            
    def _detect_outlier_zscore(self, current_price: float, prices: List[float]) -> Dict:
        """Z-score 방법으로 이상값 탐지"""
        try:
            if len(prices) < 3:
                return {'is_outlier': False, 'reason': ''}
                
            mean_price = statistics.mean(prices)
            stdev_price = statistics.stdev(prices)
            
            if stdev_price == 0:
                return {'is_outlier': False, 'reason': ''}
                
            z_score = abs((current_price - mean_price) / stdev_price)
            threshold = self.config['zscore_threshold']
            
            is_outlier = z_score > threshold
            
            if is_outlier:
                reason = f"Z-score {z_score:.2f} exceeds threshold {threshold}"
                return {'is_outlier': True, 'reason': reason}
            else:
                return {'is_outlier': False, 'reason': ''}
                
        except Exception as e:
            return {'is_outlier': True, 'reason': f'Z-score calculation error: {str(e)}'}
            
    def _detect_outlier_modified_zscore(self, current_price: float, prices: List[float]) -> Dict:
        """Modified Z-score 방법으로 이상값 탐지 (더 robust)"""
        try:
            if len(prices) < 3:
                return {'is_outlier': False, 'reason': ''}
                
            median_price = statistics.median(prices)
            
            # MAD (Median Absolute Deviation) 계산
            deviations = [abs(price - median_price) for price in prices]
            mad = statistics.median(deviations)
            
            if mad == 0:
                return {'is_outlier': False, 'reason': ''}
                
            # Modified Z-score 계산
            modified_zscore = 0.6745 * (current_price - median_price) / mad
            threshold = self.config['modified_zscore_threshold']
            
            is_outlier = abs(modified_zscore) > threshold
            
            if is_outlier:
                reason = f"Modified Z-score {abs(modified_zscore):.2f} exceeds threshold {threshold}"
                return {'is_outlier': True, 'reason': reason}
            else:
                return {'is_outlier': False, 'reason': ''}
                
        except Exception as e:
            return {'is_outlier': True, 'reason': f'Modified Z-score calculation error: {str(e)}'}
            
    def _detect_excessive_volatility(self, price_feed: PriceFeed, history: deque) -> Dict:
        """과도한 변동성 탐지"""
        try:
            if len(history) < 2:
                return {'is_anomaly': False, 'reason': ''}
                
            current_price = price_feed.price_usd
            current_time = price_feed.timestamp
            
            # 시간대별 변화율 확인
            time_checks = [
                (60, self.config['max_price_change_1min'], '1분'),
                (300, self.config['max_price_change_5min'], '5분'),
                (3600, self.config['max_price_change_1hour'], '1시간')
            ]
            
            for time_window, max_change, label in time_checks:
                cutoff_time = current_time - time_window
                recent_entries = [entry for entry in history if entry['timestamp'] >= cutoff_time]
                
                if recent_entries:
                    old_price = recent_entries[0]['price']
                    if old_price > 0:
                        change_ratio = abs(current_price - old_price) / old_price
                        
                        if change_ratio > max_change:
                            reason = f"{label} 내 {change_ratio*100:.1f}% 변화 (한도: {max_change*100:.1f}%)"
                            return {'is_anomaly': True, 'reason': reason}
                            
            return {'is_anomaly': False, 'reason': ''}
            
        except Exception as e:
            return {'is_anomaly': True, 'reason': f'Volatility detection error: {str(e)}'}
            
    async def _validate_cross_source_consensus(self, price_feed: PriceFeed, 
                                             cross_feeds: List[PriceFeed]) -> Dict:
        """교차 소스 합의 검증"""
        errors = []
        confidence = 1.0
        outlier_score = 0.0
        
        try:
            if len(cross_feeds) < self.config['min_sources_for_validation'] - 1:
                return {
                    'errors': [],
                    'confidence': 0.6,  # 교차 검증 불가능
                    'outlier_score': 0.0
                }
                
            # 다른 소스들의 가격 수집
            other_prices = [feed.price_usd for feed in cross_feeds if feed.price_usd > 0]
            current_price = price_feed.price_usd
            
            if not other_prices:
                return {
                    'errors': ['No valid cross-reference prices available'],
                    'confidence': 0.3,
                    'outlier_score': 0.5
                }
                
            # 평균 가격 계산
            avg_other_price = statistics.mean(other_prices)
            
            # 합의 임계값 확인
            consensus_threshold = self.config['source_consensus_threshold']
            if avg_other_price > 0:
                deviation = abs(current_price - avg_other_price) / avg_other_price
                
                if deviation > consensus_threshold:
                    errors.append(
                        f"Source consensus violation: {deviation*100:.1f}% deviation "
                        f"from other sources (threshold: {consensus_threshold*100:.1f}%)"
                    )
                    outlier_score = min(1.0, deviation * 2)
                    confidence -= min(0.5, deviation)
                    
            # 소스 신뢰도 가중 평균 계산
            weighted_prices = []
            total_weight = 0
            
            for feed in cross_feeds:
                source_reliability = self.source_reliability.get(feed.source, 0.5)
                weight = source_reliability * feed.confidence
                weighted_prices.append(feed.price_usd * weight)
                total_weight += weight
                
            if total_weight > 0:
                weighted_avg = sum(weighted_prices) / total_weight
                if weighted_avg > 0:
                    weighted_deviation = abs(current_price - weighted_avg) / weighted_avg
                    
                    if weighted_deviation > consensus_threshold * 1.5:
                        errors.append(
                            f"Weighted consensus violation: {weighted_deviation*100:.1f}% "
                            f"deviation from weighted average"
                        )
                        outlier_score = max(outlier_score, min(1.0, weighted_deviation * 1.5))
                        confidence -= min(0.3, weighted_deviation)
                        
        except Exception as e:
            errors.append(f"Cross-source validation error: {str(e)}")
            confidence = 0.0
            outlier_score = 1.0
            
        return {
            'errors': errors,
            'confidence': max(0.0, confidence),
            'outlier_score': outlier_score
        }
        
    async def _detect_timeseries_anomalies(self, price_feed: PriceFeed) -> Dict:
        """시계열 이상 패턴 탐지"""
        errors = []
        confidence = 1.0
        outlier_score = 0.0
        
        try:
            history = self.price_history.get(price_feed.token_address.lower(), deque())
            
            if len(history) < 20:
                return {
                    'errors': [],
                    'confidence': 0.7,
                    'outlier_score': 0.0
                }
                
            # 트렌드 분석
            trend_result = self._analyze_price_trend(price_feed, history)
            if trend_result['is_anomaly']:
                errors.append(f"Trend anomaly: {trend_result['reason']}")
                outlier_score = max(outlier_score, 0.6)
                confidence -= 0.2
                
            # 주기성 분석 (artificial pattern detection)
            pattern_result = self._detect_artificial_patterns(price_feed, history)
            if pattern_result['is_anomaly']:
                errors.append(f"Artificial pattern detected: {pattern_result['reason']}")
                outlier_score = max(outlier_score, 0.8)
                confidence -= 0.3
                
            # 급격한 방향 변화 탐지
            direction_result = self._detect_sudden_direction_change(price_feed, history)
            if direction_result['is_anomaly']:
                errors.append(f"Sudden direction change: {direction_result['reason']}")
                outlier_score = max(outlier_score, 0.5)
                confidence -= 0.1
                
        except Exception as e:
            errors.append(f"Time-series anomaly detection error: {str(e)}")
            confidence = 0.0
            outlier_score = 1.0
            
        return {
            'errors': errors,
            'confidence': max(0.0, confidence),
            'outlier_score': outlier_score
        }
        
    def _analyze_price_trend(self, price_feed: PriceFeed, history: deque) -> Dict:
        """가격 트렌드 분석"""
        try:
            current_time = price_feed.timestamp
            window_seconds = self.config['trend_analysis_window'] * 60
            cutoff_time = current_time - window_seconds
            
            recent_entries = [entry for entry in history if entry['timestamp'] >= cutoff_time]
            recent_entries.sort(key=lambda x: x['timestamp'])
            
            if len(recent_entries) < 5:
                return {'is_anomaly': False, 'reason': ''}
                
            # 선형 회귀를 통한 트렌드 분석
            times = [(entry['timestamp'] - cutoff_time) / 60 for entry in recent_entries]  # 분 단위
            prices = [entry['price'] for entry in recent_entries]
            
            # 단순 선형 회귀 계산
            n = len(times)
            sum_x = sum(times)
            sum_y = sum(prices)
            sum_xy = sum(t * p for t, p in zip(times, prices))
            sum_x2 = sum(t * t for t in times)
            
            if n * sum_x2 - sum_x * sum_x == 0:
                return {'is_anomaly': False, 'reason': ''}
                
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # 현재 가격이 트렌드에서 크게 벗어났는지 확인
            expected_price_change = slope * (current_time - cutoff_time) / 60
            last_price = recent_entries[-1]['price']
            expected_current_price = last_price + expected_price_change
            
            if expected_current_price > 0:
                deviation = abs(price_feed.price_usd - expected_current_price) / expected_current_price
                
                if deviation > 0.3:  # 30% 이상 편차
                    reason = f"Price deviates {deviation*100:.1f}% from trend expectation"
                    return {'is_anomaly': True, 'reason': reason}
                    
            return {'is_anomaly': False, 'reason': ''}
            
        except Exception as e:
            return {'is_anomaly': True, 'reason': f'Trend analysis error: {str(e)}'}
            
    def _detect_artificial_patterns(self, price_feed: PriceFeed, history: deque) -> Dict:
        """인위적 패턴 탐지 (조작 의심)"""
        try:
            if len(history) < 10:
                return {'is_anomaly': False, 'reason': ''}
                
            recent_prices = [entry['price'] for entry in list(history)[-10:]]
            
            # 1. 동일 가격의 연속적 출현 (비자연스러운 패턴)
            if len(set(recent_prices)) == 1:
                reason = "Identical prices in recent history (suspicious)"
                return {'is_anomaly': True, 'reason': reason}
                
            # 2. 규칙적인 패턴 탐지 (예: 반복되는 사이클)
            if len(recent_prices) >= 6:
                # 간단한 주기 패턴 확인
                for period in [2, 3]:
                    if len(recent_prices) >= period * 2:
                        pattern = recent_prices[:period]
                        is_repeating = True
                        
                        for i in range(period, len(recent_prices), period):
                            slice_end = min(i + period, len(recent_prices))
                            if recent_prices[i:slice_end] != pattern[:slice_end - i]:
                                is_repeating = False
                                break
                                
                        if is_repeating:
                            reason = f"Repeating pattern detected with period {period}"
                            return {'is_anomaly': True, 'reason': reason}
                            
            # 3. 너무 "완벽한" 소수점 패턴
            price_str = f"{price_feed.price_usd:.8f}"
            decimal_part = price_str.split('.')[1] if '.' in price_str else ''
            
            if len(decimal_part) >= 4:
                # 연속되는 동일 숫자나 너무 규칙적인 패턴
                if len(set(decimal_part[-4:])) == 1:  # 마지막 4자리가 모두 같음
                    reason = "Artificial precision pattern in decimal places"
                    return {'is_anomaly': True, 'reason': reason}
                    
            return {'is_anomaly': False, 'reason': ''}
            
        except Exception as e:
            return {'is_anomaly': True, 'reason': f'Pattern detection error: {str(e)}'}
            
    def _detect_sudden_direction_change(self, price_feed: PriceFeed, history: deque) -> Dict:
        """급격한 방향 전환 탐지"""
        try:
            if len(history) < 3:
                return {'is_anomaly': False, 'reason': ''}
                
            recent_entries = list(history)[-3:]
            recent_entries.sort(key=lambda x: x['timestamp'])
            
            # 연속된 세 점의 방향 변화 분석
            if len(recent_entries) == 3:
                p1, p2, p3 = [entry['price'] for entry in recent_entries]
                current_price = price_feed.price_usd
                
                # 이전 두 구간의 변화 방향
                change1 = p2 - p1
                change2 = p3 - p2  
                change3 = current_price - p3
                
                # 급격한 반전 탐지
                if abs(change1) > 0 and abs(change2) > 0:
                    # 방향이 반대이고 변화 크기가 급격히 증가
                    if (change1 > 0 > change2 and abs(change3) > abs(change2) * 3) or \
                       (change1 < 0 < change2 and abs(change3) > abs(change2) * 3):
                        reason = f"Sudden direction reversal with {abs(change3/change2):.1f}x magnitude increase"
                        return {'is_anomaly': True, 'reason': reason}
                        
            return {'is_anomaly': False, 'reason': ''}
            
        except Exception as e:
            return {'is_anomaly': True, 'reason': f'Direction change detection error: {str(e)}'}
            
    def _validate_volume_consistency(self, price_feed: PriceFeed) -> Dict:
        """거래량 일관성 검증"""
        errors = []
        confidence = 1.0
        
        try:
            volume = price_feed.volume_24h
            price = price_feed.price_usd
            
            # 기본 볼륨 검증
            if volume < 0:
                errors.append("Negative trading volume")
                confidence -= 0.3
                
            # 가격 대비 볼륨 합리성 검증
            if price > 0 and volume > 0:
                volume_usd = volume * price
                
                # 너무 큰 거래량 (시가총액 대비)
                if price_feed.market_cap > 0:
                    volume_to_mcap_ratio = volume_usd / price_feed.market_cap
                    
                    if volume_to_mcap_ratio > 10.0:  # 하루 거래량이 시총의 10배 초과
                        errors.append(f"Excessive volume: {volume_to_mcap_ratio:.1f}x market cap")
                        confidence -= 0.2
                        
                # 너무 작은 거래량 (유동성 부족)
                min_volume_threshold = self.config['min_volume_ratio']
                if volume_usd < min_volume_threshold and price_feed.market_cap > 1000000:  # $1M 이상 시총
                    errors.append("Insufficient trading volume for market cap")
                    confidence -= 0.1
                    
        except Exception as e:
            errors.append(f"Volume validation error: {str(e)}")
            confidence = 0.0
            
        return {
            'errors': errors,
            'confidence': max(0.0, confidence)
        }
        
    def _update_price_history(self, price_feed: PriceFeed):
        """가격 히스토리 업데이트"""
        try:
            token_key = price_feed.token_address.lower()
            
            entry = {
                'price': price_feed.price_usd,
                'timestamp': price_feed.timestamp,
                'source': price_feed.source,
                'volume': price_feed.volume_24h,
                'confidence': price_feed.confidence
            }
            
            self.price_history[token_key].append(entry)
            
        except Exception as e:
            logger.error(f"가격 히스토리 업데이트 실패: {e}")
            
    def create_anomaly_alert(self, price_feed: PriceFeed, validation_result: ValidationResult) -> Optional[AnomalyAlert]:
        """이상값 알림 생성"""
        try:
            if validation_result.outlier_score < self.config['low_severity_threshold']:
                return None
                
            # 예상 가격 범위 계산
            history = self.price_history.get(price_feed.token_address.lower(), deque())
            if history:
                recent_prices = [entry['price'] for entry in list(history)[-10:]]
                if recent_prices:
                    min_price = min(recent_prices)
                    max_price = max(recent_prices)
                    expected_range = (min_price, max_price)
                else:
                    expected_range = (price_feed.price_usd * 0.9, price_feed.price_usd * 1.1)
            else:
                expected_range = (price_feed.price_usd * 0.9, price_feed.price_usd * 1.1)
                
            # 심각도 결정
            if validation_result.outlier_score >= self.config['high_severity_threshold']:
                severity = 'critical'
            elif validation_result.outlier_score >= self.config['medium_severity_threshold']:
                severity = 'high'
            elif validation_result.outlier_score >= self.config['low_severity_threshold']:
                severity = 'medium'
            else:
                severity = 'low'
                
            # 이상 유형 결정
            anomaly_types = []
            for error in validation_result.validation_errors:
                if 'outlier' in error.lower():
                    anomaly_types.append('statistical_outlier')
                elif 'consensus' in error.lower():
                    anomaly_types.append('source_inconsistency')
                elif 'volatility' in error.lower():
                    anomaly_types.append('excessive_volatility')
                elif 'pattern' in error.lower():
                    anomaly_types.append('artificial_pattern')
                elif 'trend' in error.lower():
                    anomaly_types.append('trend_deviation')
                    
            anomaly_type = ', '.join(anomaly_types) if anomaly_types else 'unknown'
            
            return AnomalyAlert(
                token_address=price_feed.token_address,
                symbol=price_feed.symbol,
                current_price=price_feed.price_usd,
                expected_price_range=expected_range,
                anomaly_type=anomaly_type,
                severity=severity,
                timestamp=validation_result.timestamp,
                sources=[price_feed.source]
            )
            
        except Exception as e:
            logger.error(f"이상값 알림 생성 실패: {e}")
            return None
            
    def get_validation_statistics(self) -> Dict:
        """검증 통계 반환"""
        try:
            total = self.validation_stats['total_validations']
            passed = self.validation_stats['passed_validations']
            failed = self.validation_stats['failed_validations']
            
            success_rate = (passed / total * 100) if total > 0 else 0
            
            return {
                'total_validations': total,
                'passed_validations': passed,
                'failed_validations': failed,
                'success_rate_percent': round(success_rate, 2),
                'outliers_detected': self.validation_stats['outliers_detected'],
                'anomalies_by_token': dict(self.validation_stats['anomalies_by_token']),
                'anomalies_by_source': dict(self.validation_stats['anomalies_by_source']),
                'source_reliability_scores': dict(self.source_reliability),
                'tokens_tracked': len(self.price_history),
                'average_history_length': np.mean([len(hist) for hist in self.price_history.values()]) 
                                        if self.price_history else 0
            }
            
        except Exception as e:
            logger.error(f"검증 통계 계산 실패: {e}")
            return {}
            
    async def cleanup_old_data(self):
        """오래된 데이터 정리"""
        try:
            current_time = time.time()
            cutoff_time = current_time - 86400  # 24시간 이전
            
            # 각 토큰의 히스토리에서 오래된 데이터 제거
            for token_address, history in self.price_history.items():
                # deque를 리스트로 변환하여 필터링
                filtered_entries = [entry for entry in history if entry['timestamp'] > cutoff_time]
                
                # 새로운 deque로 교체
                self.price_history[token_address] = deque(filtered_entries, maxlen=1000)
                
            # 통계 정리 (매일 자정에)
            if int(current_time) % 86400 < 3600:  # 자정 1시간 내
                self.validation_stats['total_validations'] = 0
                self.validation_stats['passed_validations'] = 0
                self.validation_stats['failed_validations'] = 0
                self.validation_stats['outliers_detected'] = 0
                self.validation_stats['anomalies_by_token'].clear()
                self.validation_stats['anomalies_by_source'].clear()
                
            logger.info("오래된 데이터 정리 완료")
            
        except Exception as e:
            logger.error(f"데이터 정리 실패: {e}")

# 사용 예시
async def main():
    validator = DataValidator()
    
    # 테스트용 가격 피드들
    test_feeds = [
        PriceFeed(
            token_address="0xA0b86a33E6742E7FcAb4c85a2C096Bd5f53a7d5c",
            symbol="ETH",
            price_usd=3000.0,
            source="test_source1",
            timestamp=time.time(),
            volume_24h=1000000,
            market_cap=360000000000,
            confidence=0.95
        ),
        PriceFeed(
            token_address="0xA0b86a33E6742E7FcAb4c85a2C096Bd5f53a7d5c", 
            symbol="ETH",
            price_usd=4500.0,  # 이상값 (50% 증가)
            source="test_source2",
            timestamp=time.time(),
            volume_24h=2000000,
            market_cap=540000000000,
            confidence=0.90
        )
    ]
    
    # 첫 번째 피드로 히스토리 구축
    for i in range(20):
        historical_feed = PriceFeed(
            token_address="0xA0b86a33E6742E7FcAb4c85a2C096Bd5f53a7d5c",
            symbol="ETH", 
            price_usd=3000 + (i * 10),  # 점진적 증가
            source=f"historical_{i}",
            timestamp=time.time() - (20 - i) * 300,  # 5분 간격
            confidence=0.9
        )
        validator._update_price_history(historical_feed)
        
    # 검증 실행
    for feed in test_feeds:
        cross_feeds = [f for f in test_feeds if f != feed]
        result = await validator.validate_price_feed(feed, cross_feeds)
        
        print(f"\n=== {feed.symbol} 검증 결과 ===")
        print(f"유효성: {'✓' if result.is_valid else '✗'}")
        print(f"신뢰도: {result.confidence_score:.2f}")
        print(f"이상값 점수: {result.outlier_score:.2f}")
        
        if result.validation_errors:
            print("검증 오류:")
            for error in result.validation_errors:
                print(f"  - {error}")
                
        # 이상값 알림 생성
        if result.outlier_score > 0.3:
            alert = validator.create_anomaly_alert(feed, result)
            if alert:
                print(f"\n🚨 이상값 알림:")
                print(f"  심각도: {alert.severity}")
                print(f"  유형: {alert.anomaly_type}")
                print(f"  현재 가격: ${alert.current_price:.2f}")
                print(f"  예상 범위: ${alert.expected_price_range[0]:.2f} - ${alert.expected_price_range[1]:.2f}")
                
    # 통계 출력
    stats = validator.get_validation_statistics()
    print(f"\n=== 검증 통계 ===")
    print(f"총 검증 수: {stats['total_validations']}")
    print(f"성공률: {stats['success_rate_percent']:.1f}%")
    print(f"이상값 탐지: {stats['outliers_detected']}개")
    
    # 정리
    await validator.cleanup_old_data()

if __name__ == "__main__":
    asyncio.run(main())