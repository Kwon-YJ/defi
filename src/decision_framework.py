from typing import Dict, List, Tuple
from dataclasses import dataclass
from src.performance_analyzer import PerformanceAnalyzer
from src.cost_analysis import CostAnalyzer
from src.risk_calculator import RiskCalculator
from src.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class InvestmentCriteria:
    min_daily_opportunities: int = 3
    min_weekly_profit: float = 0.5  # ETH
    min_success_rate: float = 0.8
    max_payback_months: int = 12
    min_roi_percentage: float = 50.0
    max_var_loss: float = 0.1  # ETH
    min_sharpe_ratio: float = 1.0
    max_drawdown_percentage: float = 20.0

class InvestmentDecisionFramework:
    def __init__(self):
        self.performance_analyzer = PerformanceAnalyzer()
        self.cost_analyzer = CostAnalyzer()
        self.risk_calculator = RiskCalculator()
        
    async def evaluate_investment_decision(self, 
                                         investment_amount: float,
                                         criteria: InvestmentCriteria = None) -> Dict:
        """투자 결정 종합 평가"""
        if criteria is None:
            criteria = InvestmentCriteria()
        
        try:
            # 1. 성과 분석
            logger.info("성과 분석 실행 중...")
            daily_report = await self.performance_analyzer.generate_daily_report()
            weekly_summary = await self.performance_analyzer.generate_weekly_summary()
            roi_projection = await self.performance_analyzer.calculate_roi_projection(investment_amount)
            
            # 2. 비용 분석
            logger.info("비용 분석 실행 중...")
            cost_recommendation = self.cost_analyzer.recommend_optimal_setup(
                weekly_summary.get('daily_average', 0) * 7
            )
            
            # 3. 위험 분석
            logger.info("위험 분석 실행 중...")
            var_analysis = await self.risk_calculator.calculate_var()
            sharpe_analysis = await self.risk_calculator.calculate_sharpe_ratio()
            drawdown_analysis = await self.risk_calculator.calculate_maximum_drawdown()
            monte_carlo = await self.risk_calculator.monte_carlo_simulation(investment_amount)
            
            # 4. 기준 평가
            evaluation_results = self._evaluate_criteria(
                criteria, daily_report, weekly_summary, roi_projection,
                var_analysis, sharpe_analysis, drawdown_analysis
            )
            
            # 5. 최종 결정
            final_decision = self._make_final_decision(evaluation_results)
            
            return {
                'investment_amount': investment_amount,
                'evaluation_date': daily_report.get('date'),
                'performance_metrics': {
                    'daily_report': daily_report,
                    'weekly_summary': weekly_summary,
                    'roi_projection': roi_projection
                },
                'cost_analysis': cost_recommendation,
                'risk_analysis': {
                    'var': var_analysis,
                    'sharpe_ratio': sharpe_analysis,
                    'max_drawdown': drawdown_analysis,
                    'monte_carlo': monte_carlo
                },
                'criteria_evaluation': evaluation_results,
                'final_decision': final_decision,
                'recommendations': self._generate_recommendations(evaluation_results, final_decision)
            }
            
        except Exception as e:
            logger.error(f"투자 결정 평가 실패: {e}")
            return {
                'error': str(e),
                'final_decision': {
                    'decision': 'REJECT',
                    'reason': '평가 과정에서 오류가 발생했습니다.'
                }
            }
    
    def _evaluate_criteria(self, criteria: InvestmentCriteria, 
                          daily_report: Dict, weekly_summary: Dict, 
                          roi_projection: Dict, var_analysis: Dict,
                          sharpe_analysis: Dict, drawdown_analysis: Dict) -> Dict:
        """투자 기준 평가"""
        results = {}
        
        # 1. 일일 기회 수
        daily_opportunities = daily_report.get('total_opportunities', 0)
        results['daily_opportunities'] = {
            'value': daily_opportunities,
            'threshold': criteria.min_daily_opportunities,
            'passed': daily_opportunities >= criteria.min_daily_opportunities,
            'score': min(daily_opportunities / criteria.min_daily_opportunities, 2.0)
        }
        
        # 2. 주간 수익
        weekly_profit = weekly_summary.get('weekly_profit', 0)
        results['weekly_profit'] = {
            'value': weekly_profit,
            'threshold': criteria.min_weekly_profit,
            'passed': weekly_profit >= criteria.min_weekly_profit,
            'score': min(weekly_profit / criteria.min_weekly_profit, 2.0) if criteria.min_weekly_profit > 0 else 0
        }
        
        # 3. 성공률
        success_rate = daily_report.get('success_rate', 0)
        results['success_rate'] = {
            'value': success_rate,
            'threshold': criteria.min_success_rate,
            'passed': success_rate >= criteria.min_success_rate,
            'score': min(success_rate / criteria.min_success_rate, 1.5)
        }
        
        # 4. ROI
        roi_percentage = roi_projection.get('roi_percentage', 0)
        results['roi'] = {
            'value': roi_percentage,
            'threshold': criteria.min_roi_percentage,
            'passed': roi_percentage >= criteria.min_roi_percentage,
            'score': min(roi_percentage / criteria.min_roi_percentage, 2.0) if criteria.min_roi_percentage > 0 else 0
        }
        
        # 5. 회수 기간
        payback_months = roi_projection.get('payback_months', float('inf'))
        results['payback_period'] = {
            'value': payback_months,
            'threshold': criteria.max_payback_months,
            'passed': payback_months <= criteria.max_payback_months,
            'score': max(0, 2.0 - (payback_months / criteria.max_payback_months)) if payback_months != float('inf') else 0
        }
        
        # 6. VaR
        var_loss = abs(var_analysis.get('var_95', 0))
        results['var'] = {
            'value': var_loss,
            'threshold': criteria.max_var_loss,
            'passed': var_loss <= criteria.max_var_loss,
            'score': max(0, 2.0 - (var_loss / criteria.max_var_loss)) if criteria.max_var_loss > 0 else 1
        }
        
        # 7. 샤프 비율
        sharpe_ratio = sharpe_analysis.get('sharpe_ratio', 0)
        results['sharpe_ratio'] = {
            'value': sharpe_ratio,
            'threshold': criteria.min_sharpe_ratio,
            'passed': sharpe_ratio >= criteria.min_sharpe_ratio,
            'score': min(sharpe_ratio / criteria.min_sharpe_ratio, 2.0) if criteria.min_sharpe_ratio > 0 else 0
        }
        
        # 8. 최대 낙폭
        drawdown_pct = drawdown_analysis.get('drawdown_percentage', 0)
        results['max_drawdown'] = {
            'value': drawdown_pct,
            'threshold': criteria.max_drawdown_percentage,
            'passed': drawdown_pct <= criteria.max_drawdown_percentage,
            'score': max(0, 2.0 - (drawdown_pct / criteria.max_drawdown_percentage)) if criteria.max_drawdown_percentage > 0 else 1
        }
        
        # 종합 점수 계산
        weights = {
            'daily_opportunities': 0.15,
            'weekly_profit': 0.20,
            'success_rate': 0.15,
            'roi': 0.20,
            'payback_period': 0.10,
            'var': 0.10,
            'sharpe_ratio': 0.05,
            'max_drawdown': 0.05
        }
        
        total_score = sum(results[key]['score'] * weights[key] for key in weights.keys())
        passed_count = sum(1 for result in results.values() if result['passed'])
        
        results['summary'] = {
            'total_score': total_score,
            'max_possible_score': sum(weights.values()) * 2.0,  # 최대 점수는 2.0
            'normalized_score': total_score / (sum(weights.values()) * 2.0),
            'passed_criteria': passed_count,
            'total_criteria': len(results) - 1,  # summary 제외
            'pass_rate': passed_count / (len(results) - 1)
        }
        
        return results
    
    def _make_final_decision(self, evaluation_results: Dict) -> Dict:
        """최종 투자 결정"""
        summary = evaluation_results['summary']
        
        # 필수 조건 확인
        critical_criteria = ['weekly_profit', 'roi', 'success_rate']
        critical_passed = all(
            evaluation_results[criteria]['passed'] 
            for criteria in critical_criteria
        )
        
        # 위험 조건 확인
        risk_criteria = ['var', 'max_drawdown']
        risk_acceptable = all(
            evaluation_results[criteria]['passed']
            for criteria in risk_criteria
        )
        
        # 결정 로직
        if not critical_passed:
            decision = 'REJECT'
            reason = '핵심 수익성 기준을 충족하지 않습니다.'
            confidence = 0.9
        elif not risk_acceptable:
            decision = 'CONDITIONAL'
            reason = '수익성은 양호하나 위험 수준이 높습니다. 추가 검토가 필요합니다.'
            confidence = 0.6
        elif summary['normalized_score'] >= 0.8:
            decision = 'STRONG_APPROVE'
            reason = '모든 기준을 우수하게 충족합니다.'
            confidence = 0.95
        elif summary['normalized_score'] >= 0.6:
            decision = 'APPROVE'
            reason = '대부분의 기준을 충족합니다.'
            confidence = 0.8
        elif summary['pass_rate'] >= 0.6:
            decision = 'CONDITIONAL'
            reason = '일부 기준을 충족하나 추가 모니터링이 필요합니다.'
            confidence = 0.7
        else:
            decision = 'REJECT'
            reason = '투자 기준을 충족하지 않습니다.'
            confidence = 0.85
        
        return {
            'decision': decision,
            'reason': reason,
            'confidence': confidence,
            'score': summary['normalized_score'],
            'critical_criteria_passed': critical_passed,
            'risk_acceptable': risk_acceptable
        }
    
    def _generate_recommendations(self, evaluation_results: Dict, 
                                final_decision: Dict) -> List[str]:
        """추천사항 생성"""
        recommendations = []
        
        decision = final_decision['decision']
        
        if decision == 'STRONG_APPROVE':
            recommendations.append("투자를 강력히 권장합니다.")
            recommendations.append("권장 설정으로 시스템을 구축하세요.")
            recommendations.append("정기적인 성과 모니터링을 설정하세요.")
            
        elif decision == 'APPROVE':
            recommendations.append("투자를 권장합니다.")
            recommendations.append("소규모로 시작하여 점진적으로 확대하세요.")
            recommendations.append("위험 관리 시스템을 강화하세요.")
            
        elif decision == 'CONDITIONAL':
            recommendations.append("조건부 투자를 고려하세요.")
            recommendations.append("추가 데이터 수집 후 재평가하세요.")
            recommendations.append("더 보수적인 설정으로 시작하세요.")
            
            # 개선이 필요한 영역 식별
            for key, result in evaluation_results.items():
                if key != 'summary' and not result['passed']:
                    if key == 'weekly_profit':
                        recommendations.append("주간 수익성 개선이 필요합니다.")
                    elif key == 'success_rate':
                        recommendations.append("거래 성공률 향상이 필요합니다.")
                    elif key == 'var':
                        recommendations.append("위험 관리 강화가 필요합니다.")
                        
        else:  # REJECT
            recommendations.append("현재 상황에서는 투자를 권장하지 않습니다.")
            recommendations.append("시장 상황 개선을 기다리거나 전략을 재검토하세요.")
            recommendations.append("더 많은 데이터 수집 후 재평가하세요.")
        
        return recommendations
