import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
from src.data_storage import DataStorage
from src.logger import setup_logger

logger = setup_logger(__name__)

class PerformanceAnalyzer:
    def __init__(self):
        self.storage = DataStorage()
        # 주간 수익 목표 (config에서 로드, 실패 시 기본값)
        try:
            from config.config import config
            self._target_eth = float(getattr(config, 'weekly_profit_target_eth', 191.48))
            self._target_usd = float(getattr(config, 'weekly_profit_target_usd', 76592))
        except Exception:
            self._target_eth = 191.48
            self._target_usd = 76592.0
        
    async def generate_daily_report(self) -> Dict:
        """일일 성과 보고서 생성"""
        try:
            # 최근 24시간 데이터 조회
            opportunities = await self.storage.get_recent_opportunities(1000)
            
            # 오늘 날짜 필터링
            today = datetime.now().date()
            today_opportunities = [
                opp for opp in opportunities 
                if datetime.fromisoformat(opp['timestamp']).date() == today
            ]
            
            if not today_opportunities:
                return {
                    'date': today.isoformat(),
                    'total_opportunities': 0,
                    'avg_profit': 0,
                    'success_rate': 0,
                    'total_profit': 0
                }
            
            # 통계 계산
            total_opportunities = len(today_opportunities)
            profitable_opportunities = [
                opp for opp in today_opportunities 
                if opp.get('net_profit', 0) > 0
            ]
            
            avg_profit = sum(opp.get('net_profit', 0) for opp in today_opportunities) / total_opportunities
            success_rate = len(profitable_opportunities) / total_opportunities
            total_profit = sum(opp.get('net_profit', 0) for opp in profitable_opportunities)
            
            # 시간대별 분석
            hourly_stats = self._analyze_hourly_distribution(today_opportunities)
            
            # DEX별 분석
            dex_stats = self._analyze_dex_performance(today_opportunities)
            
            return {
                'date': today.isoformat(),
                'total_opportunities': total_opportunities,
                'profitable_opportunities': len(profitable_opportunities),
                'avg_profit': avg_profit,
                'success_rate': success_rate,
                'total_profit': total_profit,
                'hourly_distribution': hourly_stats,
                'dex_performance': dex_stats,
                'best_opportunity': max(today_opportunities, key=lambda x: x.get('net_profit', 0)) if today_opportunities else None
            }
            
        except Exception as e:
            logger.error(f"일일 보고서 생성 실패: {e}")
            return {'error': str(e)}
    
    async def generate_weekly_summary(self) -> Dict:
        """주간 요약 보고서 생성"""
        try:
            # 최근 7일 데이터 조회
            opportunities = await self.storage.get_recent_opportunities(10000)
            
            # 지난 7일 필터링
            week_ago = datetime.now() - timedelta(days=7)
            week_opportunities = [
                opp for opp in opportunities 
                if datetime.fromisoformat(opp['timestamp']) >= week_ago
            ]
            
            if not week_opportunities:
                return {
                    'week_start': week_ago.date().isoformat(),
                    'week_end': datetime.now().date().isoformat(),
                    'total_opportunities': 0,
                    'weekly_profit': 0,
                    'daily_average': 0
                }
            
            # 일별 그룹화
            daily_stats = {}
            for opp in week_opportunities:
                date = datetime.fromisoformat(opp['timestamp']).date()
                if date not in daily_stats:
                    daily_stats[date] = {
                        'opportunities': 0,
                        'profit': 0,
                        'avg_confidence': 0
                    }
                
                daily_stats[date]['opportunities'] += 1
                daily_stats[date]['profit'] += opp.get('net_profit', 0)
                daily_stats[date]['avg_confidence'] += opp.get('confidence', 0)
            
            # 평균 계산
            for date, stats in daily_stats.items():
                if stats['opportunities'] > 0:
                    stats['avg_confidence'] /= stats['opportunities']
            
            total_profit = sum(opp.get('net_profit', 0) for opp in week_opportunities)
            daily_average = total_profit / 7
            
            return {
                'week_start': week_ago.date().isoformat(),
                'week_end': datetime.now().date().isoformat(),
                'total_opportunities': len(week_opportunities),
                'weekly_profit': total_profit,
                'daily_average': daily_average,
                'daily_breakdown': {date.isoformat(): stats for date, stats in daily_stats.items()},
                'trend_analysis': self._analyze_weekly_trends(daily_stats)
            }
            
        except Exception as e:
            logger.error(f"주간 요약 생성 실패: {e}")
            return {'error': str(e)}

    async def evaluate_weekly_target(self) -> Dict:
        """주간 수익 목표(191.48 ETH) 달성 여부 평가"""
        try:
            summary = await self.generate_weekly_summary()
            if 'error' in summary:
                return {'error': summary['error']}

            achieved_eth = float(summary.get('weekly_profit', 0))
            target_eth = self._target_eth
            daily_target_eth = target_eth / 7.0
            avg_daily_eth = float(summary.get('daily_average', 0))

            gap_eth = target_eth - achieved_eth
            progress_pct = (achieved_eth / target_eth * 100) if target_eth > 0 else 0.0
            on_track = achieved_eth >= target_eth

            result = {
                'target_eth': target_eth,
                'target_usd': self._target_usd,
                'achieved_eth': achieved_eth,
                'gap_eth': gap_eth,
                'progress_pct': progress_pct,
                'avg_daily_eth': avg_daily_eth,
                'daily_target_eth': daily_target_eth,
                'on_track': on_track,
                'window': {
                    'start': summary.get('week_start'),
                    'end': summary.get('week_end')
                }
            }

            # 간단한 권고 메시지
            if on_track:
                result['recommendation'] = '목표 달성 중: 현 전략 유지 권장'
            else:
                # 부족분을 일일 기준으로 환산
                remaining_days = 7
                try:
                    # summary에는 일일 breakdown 키가 있음
                    remaining_days = max(0, 7 - len(summary.get('daily_breakdown', {})))
                except Exception:
                    pass

                per_day_needed = (max(0.0, gap_eth) / remaining_days) if remaining_days > 0 else gap_eth
                result['recommendation'] = (
                    f"목표 미달: 남은 기간 일일 {per_day_needed:.4f} ETH 추가 필요"
                )

            return result

        except Exception as e:
            logger.error(f"주간 목표 평가 실패: {e}")
            return {'error': str(e)}
    
    async def calculate_roi_projection(self, investment_amount: float) -> Dict:
        """ROI 예측 계산"""
        try:
            # 최근 30일 데이터 기반 예측
            opportunities = await self.storage.get_recent_opportunities(50000)
            
            month_ago = datetime.now() - timedelta(days=30)
            month_opportunities = [
                opp for opp in opportunities 
                if datetime.fromisoformat(opp['timestamp']) >= month_ago
            ]
            
            if not month_opportunities:
                return {'error': '충분한 데이터가 없습니다'}
            
            # 월간 통계
            monthly_profit = sum(opp.get('net_profit', 0) for opp in month_opportunities)
            monthly_opportunities = len(month_opportunities)
            
            # 연간 예측
            annual_profit = monthly_profit * 12
            annual_opportunities = monthly_opportunities * 12
            
            # ROI 계산
            roi_percentage = (annual_profit / investment_amount) * 100 if investment_amount > 0 else 0
            payback_months = investment_amount / monthly_profit if monthly_profit > 0 else float('inf')
            
            # 위험 조정 ROI (보수적 추정)
            conservative_roi = roi_percentage * 0.7  # 30% 할인
            
            return {
                'investment_amount': investment_amount,
                'monthly_profit': monthly_profit,
                'annual_profit_projection': annual_profit,
                'roi_percentage': roi_percentage,
                'conservative_roi': conservative_roi,
                'payback_months': payback_months,
                'monthly_opportunities': monthly_opportunities,
                'annual_opportunities_projection': annual_opportunities,
                'recommendation': self._generate_investment_recommendation(roi_percentage, payback_months)
            }
            
        except Exception as e:
            logger.error(f"ROI 예측 계산 실패: {e}")
            return {'error': str(e)}
    
    def _analyze_hourly_distribution(self, opportunities: List[Dict]) -> Dict:
        """시간대별 기회 분포 분석"""
        hourly_counts = {}
        
        for opp in opportunities:
            hour = datetime.fromisoformat(opp['timestamp']).hour
            if hour not in hourly_counts:
                hourly_counts[hour] = 0
            hourly_counts[hour] += 1
        
        # 가장 활발한 시간대 찾기
        peak_hour = max(hourly_counts.items(), key=lambda x: x[1]) if hourly_counts else (0, 0)
        
        return {
            'distribution': hourly_counts,
            'peak_hour': peak_hour[0],
            'peak_count': peak_hour[1]
        }
    
    def _analyze_dex_performance(self, opportunities: List[Dict]) -> Dict:
        """DEX별 성과 분석"""
        dex_stats = {}
        
        for opp in opportunities:
            dexes = opp.get('dexes', [])
            profit = opp.get('net_profit', 0)
            
            for dex in dexes:
                if dex not in dex_stats:
                    dex_stats[dex] = {
                        'count': 0,
                        'total_profit': 0,
                        'avg_profit': 0
                    }
                
                dex_stats[dex]['count'] += 1
                dex_stats[dex]['total_profit'] += profit
        
        # 평균 계산
        for dex, stats in dex_stats.items():
            if stats['count'] > 0:
                stats['avg_profit'] = stats['total_profit'] / stats['count']
        
        return dex_stats
    
    def _analyze_weekly_trends(self, daily_stats: Dict) -> Dict:
        """주간 트렌드 분석"""
        if len(daily_stats) < 2:
            return {'trend': 'insufficient_data'}
        
        # 날짜순 정렬
        sorted_days = sorted(daily_stats.items())
        
        # 수익 트렌드 계산
        profits = [stats['profit'] for _, stats in sorted_days]
        
        if len(profits) >= 3:
            # 선형 회귀로 트렌드 계산 (간단한 방법)
            x = list(range(len(profits)))
            n = len(profits)
            
            sum_x = sum(x)
            sum_y = sum(profits)
            sum_xy = sum(x[i] * profits[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) if (n * sum_x2 - sum_x ** 2) != 0 else 0
            
            if slope > 0.001:
                trend = 'increasing'
            elif slope < -0.001:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'slope': slope if 'slope' in locals() else 0,
            'best_day': max(sorted_days, key=lambda x: x[1]['profit'])[0].isoformat() if sorted_days else None,
            'worst_day': min(sorted_days, key=lambda x: x[1]['profit'])[0].isoformat() if sorted_days else None
        }
    
    def _generate_investment_recommendation(self, roi_percentage: float, payback_months: float) -> str:
        """투자 추천 생성"""
        if roi_percentage >= 100 and payback_months <= 6:
            return "강력 추천: 높은 수익률과 빠른 회수 기간"
        elif roi_percentage >= 50 and payback_months <= 12:
            return "추천: 양호한 수익률과 합리적인 회수 기간"
        elif roi_percentage >= 20 and payback_months <= 24:
            return "조건부 추천: 시장 상황을 더 지켜본 후 결정"
        else:
            return "비추천: 낮은 수익률 또는 긴 회수 기간"
