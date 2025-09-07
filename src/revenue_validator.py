#!/usr/bin/env python3
"""
DeFi Revenue Goal Validator
논문 목표 달성 검증: 주간 평균 191.48 ETH 수익 목표

This module validates and tracks progress toward achieving the paper's revenue goals:
- Weekly average: 191.48 ETH (≈$76,592 USD)
- Highest single transaction: 81.31 ETH (≈$32,524 USD) 
- 150-day backtest validation
- Capital efficiency: <1 ETH with flash loans, <150 ETH without
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import sqlite3
from dataclasses import dataclass
from src.data_storage import DataStorage
from src.performance_analyzer import PerformanceAnalyzer
from src.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class RevenueGoals:
    """논문에서 제시된 수익 목표들"""
    weekly_average_eth: float = 191.48  # 주간 평균 ETH
    weekly_average_usd: float = 76592   # 주간 평균 USD ($400/ETH 기준)
    highest_transaction_eth: float = 81.31  # 최고 단일 거래 ETH
    highest_transaction_usd: float = 32524  # 최고 단일 거래 USD
    backtest_days: int = 150  # 백테스트 기간
    max_capital_with_flash: float = 1.0  # Flash loan 사용시 최대 필요 자본 (ETH)
    max_capital_without_flash: float = 150.0  # Flash loan 미사용시 최대 필요 자본 (ETH)
    target_avg_execution_time: float = 6.43  # 목표 평균 실행시간 (초)

class RevenueValidator:
    def __init__(self):
        self.storage = DataStorage()
        self.performance_analyzer = PerformanceAnalyzer()
        self.goals = RevenueGoals()
        self.validation_db = "revenue_validation.db"
        self._init_validation_db()
        
    def _init_validation_db(self):
        """수익 검증용 데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.validation_db)
            cursor = conn.cursor()
            
            # 수익 검증 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS revenue_validation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    validation_type TEXT NOT NULL,
                    target_value REAL NOT NULL,
                    actual_value REAL NOT NULL,
                    achievement_rate REAL NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 주간 수익 추적 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weekly_revenue_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    week_start TEXT NOT NULL,
                    week_end TEXT NOT NULL,
                    total_revenue_eth REAL NOT NULL,
                    total_opportunities INTEGER NOT NULL,
                    avg_daily_revenue REAL NOT NULL,
                    achievement_rate REAL NOT NULL,
                    top_transaction_eth REAL NOT NULL,
                    capital_efficiency_score REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(week_start, week_end)
                )
            """)
            
            # 성과 지표 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    target_value REAL NOT NULL,
                    current_value REAL NOT NULL,
                    trend TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    UNIQUE(metric_name)
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("수익 검증 데이터베이스 초기화 완료")
            
        except Exception as e:
            logger.error(f"검증 DB 초기화 실패: {e}")

    async def validate_weekly_revenue_goal(self) -> Dict:
        """주간 수익 목표 달성 검증 (191.48 ETH)"""
        try:
            logger.info(f"주간 수익 목표 검증 시작: {self.goals.weekly_average_eth} ETH")
            
            # 최근 7일 수익 데이터 조회
            week_summary = await self.performance_analyzer.generate_weekly_summary()
            
            if 'error' in week_summary:
                return {
                    'status': 'error',
                    'message': f"주간 데이터 조회 실패: {week_summary['error']}"
                }
            
            actual_weekly_revenue = week_summary.get('weekly_profit', 0.0)
            achievement_rate = (actual_weekly_revenue / self.goals.weekly_average_eth) * 100
            
            # 상태 결정
            if achievement_rate >= 100:
                status = 'achieved'
            elif achievement_rate >= 80:
                status = 'near_target'
            elif achievement_rate >= 50:
                status = 'progressing'
            else:
                status = 'underperforming'
            
            # 결과 저장
            await self._save_validation_result({
                'validation_type': 'weekly_revenue',
                'target_value': self.goals.weekly_average_eth,
                'actual_value': actual_weekly_revenue,
                'achievement_rate': achievement_rate,
                'status': status,
                'details': json.dumps(week_summary)
            })
            
            # 주간 추적 데이터 저장
            await self._save_weekly_tracking(week_summary, actual_weekly_revenue, achievement_rate)
            
            result = {
                'validation_type': 'weekly_revenue',
                'target_eth': self.goals.weekly_average_eth,
                'actual_eth': actual_weekly_revenue,
                'achievement_rate': achievement_rate,
                'status': status,
                'gap_eth': self.goals.weekly_average_eth - actual_weekly_revenue,
                'daily_target': self.goals.weekly_average_eth / 7,
                'daily_actual': actual_weekly_revenue / 7,
                'recommendations': self._generate_revenue_recommendations(achievement_rate, actual_weekly_revenue)
            }
            
            logger.info(f"주간 수익 검증 완료: {achievement_rate:.1f}% 달성 ({status})")
            return result
            
        except Exception as e:
            logger.error(f"주간 수익 목표 검증 실패: {e}")
            return {'status': 'error', 'message': str(e)}

    async def validate_highest_transaction_goal(self) -> Dict:
        """최고 거래 수익 목표 검증 (81.31 ETH)"""
        try:
            logger.info(f"최고 거래 목표 검증 시작: {self.goals.highest_transaction_eth} ETH")
            
            # 최근 30일간 최고 수익 거래 조회
            opportunities = await self.storage.get_recent_opportunities(10000)
            
            if not opportunities:
                return {
                    'status': 'no_data',
                    'message': '거래 데이터가 없습니다.'
                }
            
            # 최고 수익 거래 찾기
            highest_profit = max((opp.get('net_profit', 0) for opp in opportunities), default=0)
            achievement_rate = (highest_profit / self.goals.highest_transaction_eth) * 100
            
            # 상위 10개 거래 분석
            top_10_transactions = sorted(
                opportunities, 
                key=lambda x: x.get('net_profit', 0), 
                reverse=True
            )[:10]
            
            status = 'achieved' if achievement_rate >= 100 else 'not_achieved'
            
            # 결과 저장
            await self._save_validation_result({
                'validation_type': 'highest_transaction',
                'target_value': self.goals.highest_transaction_eth,
                'actual_value': highest_profit,
                'achievement_rate': achievement_rate,
                'status': status,
                'details': json.dumps({'top_10': top_10_transactions[:3]})  # 상위 3개만 저장
            })
            
            result = {
                'validation_type': 'highest_transaction',
                'target_eth': self.goals.highest_transaction_eth,
                'actual_eth': highest_profit,
                'achievement_rate': achievement_rate,
                'status': status,
                'gap_eth': self.goals.highest_transaction_eth - highest_profit,
                'top_10_transactions': top_10_transactions,
                'recommendations': self._generate_transaction_recommendations(achievement_rate, highest_profit)
            }
            
            logger.info(f"최고 거래 검증 완료: {achievement_rate:.1f}% 달성 ({status})")
            return result
            
        except Exception as e:
            logger.error(f"최고 거래 목표 검증 실패: {e}")
            return {'status': 'error', 'message': str(e)}

    async def validate_capital_efficiency(self) -> Dict:
        """자본 효율성 검증 (<1 ETH with flash loans, <150 ETH without)"""
        try:
            logger.info("자본 효율성 검증 시작")
            
            # 최근 거래들의 필요 자본 분석
            opportunities = await self.storage.get_recent_opportunities(5000)
            
            if not opportunities:
                return {
                    'status': 'no_data',
                    'message': '거래 데이터가 없습니다.'
                }
            
            # Flash loan 사용 여부별 분석
            flash_loan_trades = []
            regular_trades = []
            
            for opp in opportunities:
                required_capital = opp.get('required_capital', 0)
                uses_flash = opp.get('uses_flash_loan', False)
                
                if uses_flash:
                    flash_loan_trades.append(required_capital)
                else:
                    regular_trades.append(required_capital)
            
            # 통계 계산
            flash_avg = sum(flash_loan_trades) / len(flash_loan_trades) if flash_loan_trades else 0
            regular_avg = sum(regular_trades) / len(regular_trades) if regular_trades else 0
            
            flash_max = max(flash_loan_trades) if flash_loan_trades else 0
            regular_max = max(regular_trades) if regular_trades else 0
            
            # 목표 달성 여부 확인
            flash_efficiency = flash_avg <= self.goals.max_capital_with_flash
            regular_efficiency = regular_avg <= self.goals.max_capital_without_flash
            
            overall_status = 'achieved' if flash_efficiency and regular_efficiency else 'partial'
            if not flash_efficiency and not regular_efficiency:
                overall_status = 'not_achieved'
            
            result = {
                'validation_type': 'capital_efficiency',
                'flash_loan_efficiency': {
                    'target_max_eth': self.goals.max_capital_with_flash,
                    'actual_avg_eth': flash_avg,
                    'actual_max_eth': flash_max,
                    'trades_count': len(flash_loan_trades),
                    'achieved': flash_efficiency
                },
                'regular_efficiency': {
                    'target_max_eth': self.goals.max_capital_without_flash,
                    'actual_avg_eth': regular_avg,
                    'actual_max_eth': regular_max,
                    'trades_count': len(regular_trades),
                    'achieved': regular_efficiency
                },
                'overall_status': overall_status,
                'recommendations': self._generate_capital_recommendations(flash_avg, regular_avg)
            }
            
            # 결과 저장
            await self._save_validation_result({
                'validation_type': 'capital_efficiency',
                'target_value': self.goals.max_capital_with_flash,
                'actual_value': flash_avg,
                'achievement_rate': 100 if flash_efficiency else 50,
                'status': overall_status,
                'details': json.dumps(result)
            })
            
            logger.info(f"자본 효율성 검증 완료: {overall_status}")
            return result
            
        except Exception as e:
            logger.error(f"자본 효율성 검증 실패: {e}")
            return {'status': 'error', 'message': str(e)}

    async def generate_comprehensive_revenue_report(self) -> Dict:
        """종합 수익 검증 보고서 생성"""
        try:
            logger.info("종합 수익 검증 보고서 생성 시작")
            
            # 모든 검증 수행
            weekly_validation = await self.validate_weekly_revenue_goal()
            transaction_validation = await self.validate_highest_transaction_goal()
            capital_validation = await self.validate_capital_efficiency()
            
            # 전체 성과 점수 계산
            scores = []
            if weekly_validation.get('achievement_rate', 0) > 0:
                scores.append(min(100, weekly_validation['achievement_rate']))
            
            if transaction_validation.get('achievement_rate', 0) > 0:
                scores.append(min(100, transaction_validation['achievement_rate']))
            
            if capital_validation.get('overall_status') == 'achieved':
                scores.append(100)
            elif capital_validation.get('overall_status') == 'partial':
                scores.append(70)
            else:
                scores.append(30)
            
            overall_score = sum(scores) / len(scores) if scores else 0
            
            # 전체 상태 결정
            if overall_score >= 90:
                overall_status = 'excellent'
            elif overall_score >= 75:
                overall_status = 'good'
            elif overall_score >= 50:
                overall_status = 'satisfactory'
            else:
                overall_status = 'needs_improvement'
            
            # 추가 분석
            monthly_projection = await self._calculate_monthly_projection()
            risk_analysis = await self._analyze_revenue_risks()
            
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'overall_score': overall_score,
                'overall_status': overall_status,
                'paper_goals': {
                    'weekly_target_eth': self.goals.weekly_average_eth,
                    'highest_transaction_target_eth': self.goals.highest_transaction_eth,
                    'backtest_period_days': self.goals.backtest_days
                },
                'validations': {
                    'weekly_revenue': weekly_validation,
                    'highest_transaction': transaction_validation,
                    'capital_efficiency': capital_validation
                },
                'projections': monthly_projection,
                'risk_analysis': risk_analysis,
                'recommendations': self._generate_comprehensive_recommendations(
                    weekly_validation, transaction_validation, capital_validation
                ),
                'next_milestones': self._define_next_milestones(overall_score)
            }
            
            # 보고서 저장
            await self._save_comprehensive_report(report)
            
            logger.info(f"종합 보고서 생성 완료: {overall_status} ({overall_score:.1f}/100)")
            return report
            
        except Exception as e:
            logger.error(f"종합 보고서 생성 실패: {e}")
            return {'status': 'error', 'message': str(e)}

    async def _calculate_monthly_projection(self) -> Dict:
        """월간 수익 예측"""
        try:
            # 최근 30일 데이터 기반 예측
            opportunities = await self.storage.get_recent_opportunities(20000)
            
            if not opportunities:
                return {'status': 'no_data'}
            
            # 월간 수익 계산
            month_ago = datetime.now() - timedelta(days=30)
            month_opportunities = [
                opp for opp in opportunities
                if datetime.fromisoformat(opp['timestamp']) >= month_ago
            ]
            
            monthly_revenue = sum(opp.get('net_profit', 0) for opp in month_opportunities)
            
            # 주간 평균으로 변환
            weekly_equivalent = monthly_revenue / 4.33  # 한 달 ≈ 4.33주
            
            # 연간 예측
            annual_projection = monthly_revenue * 12
            
            # 논문 목표와 비교
            target_monthly = self.goals.weekly_average_eth * 4.33
            monthly_achievement = (monthly_revenue / target_monthly) * 100 if target_monthly > 0 else 0
            
            return {
                'monthly_revenue_eth': monthly_revenue,
                'weekly_equivalent_eth': weekly_equivalent,
                'annual_projection_eth': annual_projection,
                'target_monthly_eth': target_monthly,
                'monthly_achievement_rate': monthly_achievement,
                'days_analyzed': 30,
                'opportunities_count': len(month_opportunities)
            }
            
        except Exception as e:
            logger.error(f"월간 예측 계산 실패: {e}")
            return {'status': 'error', 'message': str(e)}

    async def _analyze_revenue_risks(self) -> Dict:
        """수익 위험 분석"""
        try:
            opportunities = await self.storage.get_recent_opportunities(5000)
            
            if not opportunities:
                return {'status': 'no_data'}
            
            # 수익률 변동성 분석
            profits = [opp.get('net_profit', 0) for opp in opportunities if opp.get('net_profit', 0) > 0]
            
            if not profits:
                return {'status': 'no_profitable_trades'}
            
            avg_profit = sum(profits) / len(profits)
            variance = sum((p - avg_profit) ** 2 for p in profits) / len(profits)
            std_dev = variance ** 0.5
            
            # 위험 지표 계산
            volatility = (std_dev / avg_profit) * 100 if avg_profit > 0 else 0
            max_loss = min(profits)  # 음수일 수 있음
            
            # 위험 등급 결정
            if volatility < 20:
                risk_level = 'low'
            elif volatility < 50:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            return {
                'volatility_percent': volatility,
                'risk_level': risk_level,
                'average_profit_eth': avg_profit,
                'std_deviation_eth': std_dev,
                'max_loss_eth': max_loss,
                'profit_trades_count': len(profits),
                'recommendations': self._generate_risk_recommendations(risk_level, volatility)
            }
            
        except Exception as e:
            logger.error(f"위험 분석 실패: {e}")
            return {'status': 'error', 'message': str(e)}

    def _generate_revenue_recommendations(self, achievement_rate: float, actual_revenue: float) -> List[str]:
        """수익 목표 기반 추천사항 생성"""
        recommendations = []
        
        if achievement_rate < 50:
            recommendations.extend([
                "CRITICAL: 수익률이 목표의 50% 미달. 알고리즘 최적화 필요",
                "Local Search 병렬 처리 개선으로 더 많은 기회 발굴",
                "더 많은 Protocol Actions (현재 ~6개 → 96개) 구현 필요",
                "Flash Loan 활용으로 자본 효율성 극대화"
            ])
        elif achievement_rate < 80:
            recommendations.extend([
                "수익률 개선 필요. 현재 목표의 80% 미달",
                "Negative Cycle Detection 성능 최적화",
                "더 많은 자산 쌍 지원 (현재 4개 → 25개)",
                "실행 시간 단축으로 더 많은 기회 포착"
            ])
        elif achievement_rate < 100:
            recommendations.extend([
                "목표 근접. 미세 조정으로 목표 달성 가능",
                "그래프 업데이트 빈도 증가",
                "수수료 최적화로 순수익 증대"
            ])
        else:
            recommendations.extend([
                "목표 달성! 현재 성능 유지 및 확장성 개선",
                "더 큰 규모의 거래로 수익 극대화",
                "시스템 안정성 및 모니터링 강화"
            ])
        
        return recommendations

    def _generate_transaction_recommendations(self, achievement_rate: float, highest_profit: float) -> List[str]:
        """최고 거래 기반 추천사항 생성"""
        recommendations = []
        
        if achievement_rate < 25:
            recommendations.extend([
                "CRITICAL: 최고 거래 수익이 목표의 25% 미달",
                "복합 거래 전략 구현 (단순 차익거래 → 복잡한 전략)",
                "경제적 상태 악용 기능 개발 (bZx 공격 유형)",
                "더 큰 유동성 풀 활용으로 거래 규모 확대"
            ])
        elif achievement_rate < 50:
            recommendations.extend([
                "고수익 거래 기회 발굴 능력 개선 필요",
                "Multi-hop 차익거래 구현",
                "Lending/Borrowing 조합 거래 추가"
            ])
        elif achievement_rate < 75:
            recommendations.extend([
                "대형 거래 기회 포착 개선",
                "시장 변동성이 큰 시점 타겟팅",
                "더 정교한 수익 최적화 알고리즘"
            ])
        else:
            recommendations.extend([
                "우수한 고수익 거래 포착 능력",
                "현재 전략 확장 및 다양화",
                "위험 관리 강화로 지속 가능성 확보"
            ])
        
        return recommendations

    def _generate_capital_recommendations(self, flash_avg: float, regular_avg: float) -> List[str]:
        """자본 효율성 기반 추천사항"""
        recommendations = []
        
        if flash_avg > self.goals.max_capital_with_flash:
            recommendations.append(
                f"Flash Loan 자본 효율성 개선 필요 (현재 평균 {flash_avg:.2f} ETH > 목표 {self.goals.max_capital_with_flash} ETH)"
            )
        
        if regular_avg > self.goals.max_capital_without_flash:
            recommendations.append(
                f"일반 거래 자본 효율성 개선 필요 (현재 평균 {regular_avg:.2f} ETH > 목표 {self.goals.max_capital_without_flash} ETH)"
            )
        
        if flash_avg <= self.goals.max_capital_with_flash:
            recommendations.append("Flash Loan 자본 효율성 목표 달성")
        
        if regular_avg <= self.goals.max_capital_without_flash:
            recommendations.append("일반 거래 자본 효율성 목표 달성")
        
        return recommendations

    def _generate_comprehensive_recommendations(self, weekly_val: Dict, transaction_val: Dict, capital_val: Dict) -> List[str]:
        """종합 추천사항 생성"""
        recommendations = []
        
        # 우선순위 높은 개선사항
        critical_items = []
        important_items = []
        
        if weekly_val.get('achievement_rate', 0) < 50:
            critical_items.append("주간 수익 목표 50% 미달 - 즉시 개선 필요")
        
        if transaction_val.get('achievement_rate', 0) < 25:
            critical_items.append("최고 거래 수익 목표 25% 미달 - 전략 재검토 필요")
        
        if capital_val.get('overall_status') == 'not_achieved':
            important_items.append("자본 효율성 개선 필요")
        
        # 단계별 행동 계획
        if critical_items:
            recommendations.append("🚨 긴급 개선사항:")
            recommendations.extend([f"  - {item}" for item in critical_items])
        
        if important_items:
            recommendations.append("⚠️ 중요 개선사항:")
            recommendations.extend([f"  - {item}" for item in important_items])
        
        # 일반 추천사항
        recommendations.extend([
            "📈 성능 최적화:",
            "  - 96개 Protocol Actions 완전 구현",
            "  - 25개 자산 지원으로 확장",
            "  - 평균 6.43초 실행 시간 달성",
            "  - Local Search 병렬 처리 구현",
            "",
            "🎯 수익 극대화:",
            "  - Flash Loan 활용도 증대",
            "  - 복합 거래 전략 개발",
            "  - MEV 최적화 기능 추가"
        ])
        
        return recommendations

    def _define_next_milestones(self, overall_score: float) -> List[Dict]:
        """다음 목표 설정"""
        milestones = []
        
        if overall_score < 50:
            milestones = [
                {
                    "milestone": "기본 성능 달성",
                    "target": "주간 수익 50% 이상 달성",
                    "timeline": "2주 내",
                    "priority": "critical"
                },
                {
                    "milestone": "프로토콜 확장",
                    "target": "96개 Protocol Actions 구현",
                    "timeline": "4주 내",
                    "priority": "high"
                }
            ]
        elif overall_score < 75:
            milestones = [
                {
                    "milestone": "목표 근접",
                    "target": "주간 수익 80% 이상 달성",
                    "timeline": "3주 내",
                    "priority": "high"
                },
                {
                    "milestone": "고수익 거래 개발",
                    "target": "50 ETH 이상 단일 거래 달성",
                    "timeline": "6주 내",
                    "priority": "medium"
                }
            ]
        else:
            milestones = [
                {
                    "milestone": "완전한 목표 달성",
                    "target": "주간 191.48 ETH 달성",
                    "timeline": "2주 내",
                    "priority": "medium"
                },
                {
                    "milestone": "확장 및 최적화",
                    "target": "시스템 안정성 및 확장성 개선",
                    "timeline": "지속적",
                    "priority": "low"
                }
            ]
        
        return milestones

    async def _save_validation_result(self, result: Dict):
        """검증 결과 저장"""
        try:
            conn = sqlite3.connect(self.validation_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO revenue_validation 
                (timestamp, validation_type, target_value, actual_value, achievement_rate, status, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                result['validation_type'],
                result['target_value'],
                result['actual_value'],
                result['achievement_rate'],
                result['status'],
                result.get('details', '')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"검증 결과 저장 실패: {e}")

    async def _save_weekly_tracking(self, week_summary: Dict, revenue: float, achievement_rate: float):
        """주간 추적 데이터 저장"""
        try:
            conn = sqlite3.connect(self.validation_db)
            cursor = conn.cursor()
            
            # 기존 데이터 확인
            week_start = week_summary.get('week_start', '')
            week_end = week_summary.get('week_end', '')
            
            cursor.execute("""
                INSERT OR REPLACE INTO weekly_revenue_tracking 
                (week_start, week_end, total_revenue_eth, total_opportunities, 
                 avg_daily_revenue, achievement_rate, top_transaction_eth, capital_efficiency_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                week_start,
                week_end,
                revenue,
                week_summary.get('total_opportunities', 0),
                revenue / 7,
                achievement_rate,
                0,  # TODO: 최고 거래 계산
                100  # TODO: 자본 효율성 점수 계산
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"주간 추적 데이터 저장 실패: {e}")

    async def _save_comprehensive_report(self, report: Dict):
        """종합 보고서 저장"""
        try:
            # 파일로 저장
            report_file = f"revenue_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"종합 보고서 저장: {report_file}")
            
        except Exception as e:
            logger.error(f"보고서 저장 실패: {e}")

    def _generate_risk_recommendations(self, risk_level: str, volatility: float) -> List[str]:
        """위험 기반 추천사항"""
        if risk_level == 'high':
            return [
                "고위험: 수익 변동성 매우 높음",
                "포지션 크기 줄이고 분산 투자",
                "Stop-loss 메커니즘 구현",
                "더 안정적인 차익거래 기회 우선"
            ]
        elif risk_level == 'medium':
            return [
                "중간 위험: 적정 수준의 변동성",
                "위험 모니터링 지속",
                "수익/위험 비율 최적화"
            ]
        else:
            return [
                "저위험: 안정적인 수익 패턴",
                "현재 전략 유지",
                "수익 확대 기회 모색"
            ]

# CLI 인터페이스
async def main():
    """메인 실행 함수"""
    validator = RevenueValidator()
    
    print("🎯 DeFi Revenue Goal Validator")
    print("=" * 50)
    print(f"Target Weekly Revenue: {validator.goals.weekly_average_eth} ETH")
    print(f"Target Highest Transaction: {validator.goals.highest_transaction_eth} ETH")
    print("=" * 50)
    
    # 종합 검증 실행
    report = await validator.generate_comprehensive_revenue_report()
    
    if 'status' in report and report['status'] == 'error':
        print(f"❌ Error: {report['message']}")
        return
    
    # 결과 출력
    print(f"\n📊 Overall Score: {report['overall_score']:.1f}/100 ({report['overall_status']})")
    
    # 주간 수익 결과
    weekly = report['validations']['weekly_revenue']
    print(f"\n📈 Weekly Revenue:")
    print(f"  Target: {weekly['target_eth']:.2f} ETH")
    print(f"  Actual: {weekly['actual_eth']:.2f} ETH")
    print(f"  Achievement: {weekly['achievement_rate']:.1f}%")
    print(f"  Status: {weekly['status']}")
    
    # 최고 거래 결과
    highest = report['validations']['highest_transaction']
    print(f"\n🚀 Highest Transaction:")
    print(f"  Target: {highest['target_eth']:.2f} ETH")
    print(f"  Actual: {highest['actual_eth']:.2f} ETH")
    print(f"  Achievement: {highest['achievement_rate']:.1f}%")
    print(f"  Status: {highest['status']}")
    
    # 추천사항
    print(f"\n💡 Recommendations:")
    for rec in report['recommendations'][:10]:  # 상위 10개만 표시
        print(f"  {rec}")
    
    print(f"\n📄 Detailed report saved to file")

if __name__ == "__main__":
    asyncio.run(main())