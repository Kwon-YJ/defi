import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from src.data_storage import DataStorage
from src.logger import setup_logger

logger = setup_logger(__name__)

class RiskCalculator:
    def __init__(self):
        self.storage = DataStorage()
        
    async def calculate_var(self, confidence_level: float = 0.95, 
                          time_horizon_days: int = 1) -> Dict:
        """Value at Risk (VaR) 계산"""
        try:
            # 최근 30일 수익 데이터 조회
            opportunities = await self.storage.get_recent_opportunities(10000)
            
            month_ago = datetime.now() - timedelta(days=30)
            recent_opportunities = [
                opp for opp in opportunities 
                if datetime.fromisoformat(opp['timestamp']) >= month_ago
            ]
            
            if len(recent_opportunities) < 30:
                return {'error': 'VaR 계산을 위한 충분한 데이터가 없습니다'}
            
            # 일일 수익 계산
            daily_profits = self._calculate_daily_profits(recent_opportunities)
            
            if len(daily_profits) < 7:
                return {'error': 'VaR 계산을 위한 충분한 일일 데이터가 없습니다'}
            
            # VaR 계산 (Historical Simulation Method)
            profits_array = np.array(list(daily_profits.values()))
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(profits_array, var_percentile)
            
            # 조건부 VaR (Expected Shortfall) 계산
            cvar_value = np.mean(profits_array[profits_array <= var_value])
            
            # 통계 정보
            mean_profit = np.mean(profits_array)
            std_profit = np.std(profits_array)
            max_loss = np.min(profits_array)
            max_gain = np.max(profits_array)
            
            return {
                'var_95': var_value,
                'cvar_95': cvar_value,
                'confidence_level': confidence_level,
                'time_horizon_days': time_horizon_days,
                'mean_daily_profit': mean_profit,
                'daily_volatility': std_profit,
                'max_daily_loss': max_loss,
                'max_daily_gain': max_gain,
                'sample_size': len(profits_array),
                'interpretation': self._interpret_var(var_value, confidence_level)
            }
            
        except Exception as e:
            logger.error(f"VaR 계산 실패: {e}")
            return {'error': str(e)}
    
    async def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> Dict:
        """샤프 비율 계산"""
        try:
            # 최근 30일 데이터
            opportunities = await self.storage.get_recent_opportunities(10000)
            
            month_ago = datetime.now() - timedelta(days=30)
            recent_opportunities = [
                opp for opp in opportunities 
                if datetime.fromisoformat(opp['timestamp']) >= month_ago
            ]
            
            daily_profits = self._calculate_daily_profits(recent_opportunities)
            
            if len(daily_profits) < 7:
                return {'error': '샤프 비율 계산을 위한 충분한 데이터가 없습니다'}
            
            profits_array = np.array(list(daily_profits.values()))
            
            # 연간 수익률과 변동성 계산
            daily_return = np.mean(profits_array)
            daily_volatility = np.std(profits_array)
            
            annual_return = daily_return * 365
            annual_volatility = daily_volatility * np.sqrt(365)
            
            # 샤프 비율 계산
            excess_return = annual_return - risk_free_rate
            sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'risk_free_rate': risk_free_rate,
                'excess_return': excess_return,
                'interpretation': self._interpret_sharpe_ratio(sharpe_ratio)
            }
            
        except Exception as e:
            logger.error(f"샤프 비율 계산 실패: {e}")
            return {'error': str(e)}
    
    async def calculate_maximum_drawdown(self) -> Dict:
        """최대 낙폭(Maximum Drawdown) 계산"""
        try:
            # 최근 60일 데이터
            opportunities = await self.storage.get_recent_opportunities(20000)
            
            two_months_ago = datetime.now() - timedelta(days=60)
            recent_opportunities = [
                opp for opp in opportunities 
                if datetime.fromisoformat(opp['timestamp']) >= two_months_ago
            ]
            
            daily_profits = self._calculate_daily_profits(recent_opportunities)
            
            if len(daily_profits) < 14:
                return {'error': '최대 낙폭 계산을 위한 충분한 데이터가 없습니다'}
            
            # 누적 수익 계산
            sorted_dates = sorted(daily_profits.keys())
            cumulative_profits = []
            cumulative_sum = 0
            
            for date in sorted_dates:
                cumulative_sum += daily_profits[date]
                cumulative_profits.append(cumulative_sum)
            
            # 최대 낙폭 계산
            peak = cumulative_profits[0]
            max_drawdown = 0
            drawdown_start = sorted_dates[0]
            drawdown_end = sorted_dates[0]
            
            for i, profit in enumerate(cumulative_profits):
                if profit > peak:
                    peak = profit
                
                drawdown = peak - profit
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    drawdown_end = sorted_dates[i]
                    
                    # 낙폭 시작점 찾기
                    for j in range(i, -1, -1):
                        if cumulative_profits[j] == peak:
                            drawdown_start = sorted_dates[j]
                            break
            
            # 회복 기간 계산
            recovery_days = 0
            if max_drawdown > 0:
                recovery_start_idx = sorted_dates.index(drawdown_end)
                for i in range(recovery_start_idx, len(cumulative_profits)):
                    if cumulative_profits[i] >= peak:
                        recovery_days = i - recovery_start_idx
                        break
                else:
                    recovery_days = len(cumulative_profits) - recovery_start_idx  # 아직 회복 중
            
            return {
                'max_drawdown': max_drawdown,
                'drawdown_percentage': (max_drawdown / peak * 100) if peak > 0 else 0,
                'drawdown_start': drawdown_start.isoformat(),
                'drawdown_end': drawdown_end.isoformat(),
                'recovery_days': recovery_days,
                'current_cumulative_profit': cumulative_profits[-1],
                'peak_profit': peak,
                'interpretation': self._interpret_drawdown(max_drawdown, peak)
            }
            
        except Exception as e:
            logger.error(f"최대 낙폭 계산 실패: {e}")
            return {'error': str(e)}
    
    async def monte_carlo_simulation(self, initial_capital: float, 
                                   simulation_days: int = 365, 
                                   num_simulations: int = 1000) -> Dict:
        """몬테카르로 시뮬레이션"""
        try:
            # 과거 데이터에서 수익률 분포 추정
            opportunities = await self.storage.get_recent_opportunities(10000)
            
            month_ago = datetime.now() - timedelta(days=30)
            recent_opportunities = [
                opp for opp in opportunities 
                if datetime.fromisoformat(opp['timestamp']) >= month_ago
            ]
            
            daily_profits = self._calculate_daily_profits(recent_opportunities)
            
            if len(daily_profits) < 14:
                return {'error': '시뮬레이션을 위한 충분한 데이터가 없습니다'}
            
            profits_array = np.array(list(daily_profits.values()))
            mean_return = np.mean(profits_array)
            std_return = np.std(profits_array)
            
            # 몬테카르로 시뮬레이션 실행
            final_values = []
            
            for _ in range(num_simulations):
                capital = initial_capital
                
                for day in range(simulation_days):
                    # 정규분포에서 일일 수익률 샘플링
                    daily_return = np.random.normal(mean_return, std_return)
                    capital += daily_return
                    
                    # 자본이 0 이하로 떨어지면 시뮬레이션 종료
                    if capital <= 0:
                        capital = 0
                        break
                
                final_values.append(capital)
            
            final_values = np.array(final_values)
            
            # 통계 계산
            mean_final = np.mean(final_values)
            std_final = np.std(final_values)
            
            # 백분위수 계산
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            percentile_values = {p: np.percentile(final_values, p) for p in percentiles}
            
            # 손실 확률 계산
            loss_probability = np.sum(final_values < initial_capital) / num_simulations
            ruin_probability = np.sum(final_values <= 0) / num_simulations
            
            return {
                'initial_capital': initial_capital,
                'simulation_days': simulation_days,
                'num_simulations': num_simulations,
                'mean_final_value': mean_final,
                'std_final_value': std_final,
                'percentiles': percentile_values,
                'loss_probability': loss_probability,
                'ruin_probability': ruin_probability,
                'expected_return': mean_final - initial_capital,
                'return_percentage': ((mean_final - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0,
                'interpretation': self._interpret_monte_carlo(loss_probability, ruin_probability, mean_final, initial_capital)
            }
            
        except Exception as e:
            logger.error(f"몬테카르로 시뮬레이션 실패: {e}")
            return {'error': str(e)}
    
    def _calculate_daily_profits(self, opportunities: List[Dict]) -> Dict:
        """일일 수익 계산"""
        daily_profits = {}
        
        for opp in opportunities:
            date = datetime.fromisoformat(opp['timestamp']).date()
            profit = opp.get('net_profit', 0)
            
            if date not in daily_profits:
                daily_profits[date] = 0
            daily_profits[date] += profit
        
        return daily_profits
    
    def _interpret_var(self, var_value: float, confidence_level: float) -> str:
        """VaR 해석"""
        confidence_pct = confidence_level * 100
        
        if var_value >= 0:
            return f"{confidence_pct}% 신뢰도로 일일 손실이 발생하지 않을 것으로 예상됩니다."
        elif var_value > -0.01:
            return f"{confidence_pct}% 신뢰도로 일일 최대 손실이 {abs(var_value):.4f} ETH를 넘지 않을 것으로 예상됩니다. (낮은 위험)"
        elif var_value > -0.05:
            return f"{confidence_pct}% 신뢰도로 일일 최대 손실이 {abs(var_value):.4f} ETH를 넘지 않을 것으로 예상됩니다. (중간 위험)"
        else:
            return f"{confidence_pct}% 신뢰도로 일일 최대 손실이 {abs(var_value):.4f} ETH를 넘지 않을 것으로 예상됩니다. (높은 위험)"
    
    def _interpret_sharpe_ratio(self, sharpe_ratio: float) -> str:
        """샤프 비율 해석"""
        if sharpe_ratio > 2.0:
            return "매우 우수한 위험 대비 수익률"
        elif sharpe_ratio > 1.0:
            return "우수한 위험 대비 수익률"
        elif sharpe_ratio > 0.5:
            return "양호한 위험 대비 수익률"
        elif sharpe_ratio > 0:
            return "보통 수준의 위험 대비 수익률"
        else:
            return "위험 대비 수익률이 부족함"
    
    def _interpret_drawdown(self, max_drawdown: float, peak_profit: float) -> str:
        """최대 낙폭 해석"""
        if peak_profit <= 0:
            return "수익이 발생하지 않아 낙폭을 평가할 수 없습니다."
        
        drawdown_pct = (max_drawdown / peak_profit) * 100
        
        if drawdown_pct < 5:
            return f"최대 낙폭 {drawdown_pct:.1f}% - 매우 안정적"
        elif drawdown_pct < 15:
            return f"최대 낙폭 {drawdown_pct:.1f}% - 안정적"
        elif drawdown_pct < 30:
            return f"최대 낙폭 {drawdown_pct:.1f}% - 보통 수준의 변동성"
        else:
            return f"최대 낙폭 {drawdown_pct:.1f}% - 높은 변동성, 주의 필요"
    
    def _interpret_monte_carlo(self, loss_prob: float, ruin_prob: float, 
                             mean_final: float, initial_capital: float) -> str:
        """몬테카르로 시뮬레이션 해석"""
        interpretations = []
        
        if loss_prob < 0.2:
            interpretations.append("손실 확률이 낮아 안정적인 전략입니다.")
        elif loss_prob < 0.4:
            interpretations.append("적당한 수준의 손실 위험이 있습니다.")
        else:
            interpretations.append("높은 손실 위험이 있어 주의가 필요합니다.")
        
        if ruin_prob < 0.01:
            interpretations.append("파산 위험이 매우 낮습니다.")
        elif ruin_prob < 0.05:
            interpretations.append("파산 위험이 낮은 편입니다.")
        else:
            interpretations.append("파산 위험이 있어 자본 관리가 중요합니다.")
        
        expected_return_pct = ((mean_final - initial_capital) / initial_capital) * 100
        if expected_return_pct > 50:
            interpretations.append("높은 수익률이 기대됩니다.")
        elif expected_return_pct > 20:
            interpretations.append("양호한 수익률이 기대됩니다.")
        elif expected_return_pct > 0:
            interpretations.append("보통 수준의 수익률이 기대됩니다.")
        else:
            interpretations.append("손실이 예상되어 전략 재검토가 필요합니다.")
        
        return " ".join(interpretations)
