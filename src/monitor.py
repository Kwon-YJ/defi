import asyncio
import json
import os
from datetime import datetime
from typing import Dict

from src.logger import setup_logger
from src.performance_analyzer import PerformanceAnalyzer
from config.config import config

logger = setup_logger(__name__)


class RealtimeMonitor:
    """실시간 목표/성과 모니터.

    - 주기적으로 주간 수익 목표와 최고 거래 수익 목표를 평가
    - alerts.log에 경고 기록 및 stdout 알림
    - dashboard_state.json / dashboard.html 갱신
    """

    def __init__(self, interval_sec: int = None):
        self.interval = int(interval_sec or config.monitor_interval_sec)
        self.analyzer = PerformanceAnalyzer()
        self.out_dir = config.dashboard_output_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.alert_log = config.alert_log_path

    async def _emit_alert(self, msg: str) -> None:
        ts = datetime.now().isoformat()
        line = f"[{ts}] {msg}\n"
        try:
            os.makedirs(os.path.dirname(self.alert_log), exist_ok=True)
            with open(self.alert_log, 'a', encoding='utf-8') as f:
                f.write(line)
        except Exception as e:
            logger.debug(f"알림 파일 기록 실패: {e}")
        if config.alert_enable_stdout:
            logger.warning(msg)

    def _write_dashboard(self, state: Dict) -> None:
        try:
            # JSON 상태 저장
            js_path = os.path.join(self.out_dir, 'dashboard_state.json')
            with open(js_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            # HTML 대시보드 저장 (간단 템플릿)
            html_path = os.path.join(self.out_dir, 'dashboard.html')
            w = state.get('weekly_target', {})
            m = state.get('max_trade', {})
            status_week = 'OK' if w.get('on_track') else 'BELOW'
            status_max = 'OK' if m.get('met') else 'BELOW'
            html = f"""
<!doctype html>
<meta charset='utf-8' />
<title>{config.dashboard_title}</title>
<div style='font-family:sans-serif;max-width:960px;margin:20px auto'>
  <h2>{config.dashboard_title}</h2>
  <h3>Weekly Target</h3>
  <p>Status: <b>{status_week}</b><br/>
     Achieved: {w.get('achieved_eth', 0):.4f} / Target: {w.get('target_eth', 0):.4f} ETH ({w.get('progress_pct', 0):.2f}%)<br/>
     Avg daily: {w.get('avg_daily_eth', 0):.4f} vs daily target {w.get('daily_target_eth', 0):.4f}</p>
  <h3>Max Trade</h3>
  <p>Status: <b>{status_max}</b><br/>
     Best trade: {m.get('best_profit_eth', 0):.4f} / Target: {m.get('target_eth', 0):.2f} ETH<br/>
     Lookback: {m.get('lookback_days', 0)} days</p>
  <p style='color:#777'>Updated at {datetime.now().isoformat()}</p>
</div>
"""
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
        except Exception as e:
            logger.error(f"대시보드 쓰기 실패: {e}")

    async def tick(self) -> None:
        weekly = await self.analyzer.evaluate_weekly_target()
        max_trade = await self.analyzer.verify_max_trade_profit()
        state = {'weekly_target': weekly, 'max_trade': max_trade}
        self._write_dashboard(state)
        # 알림 조건
        if isinstance(weekly, dict) and not weekly.get('on_track', False):
            await self._emit_alert(
                f"주간 목표 미달: {weekly.get('achieved_eth', 0):.4f}/{weekly.get('target_eth', 0):.2f} ETH"
            )
        if isinstance(max_trade, dict) and not max_trade.get('met', False):
            await self._emit_alert(
                f"최고 거래 수익 미달: best={max_trade.get('best_profit_eth', 0):.4f} / target={max_trade.get('target_eth', 0):.2f} ETH"
            )

    async def run(self) -> None:
        logger.info(f"실시간 모니터 시작 (interval={self.interval}s)")
        while True:
            try:
                await self.tick()
            except Exception as e:
                logger.error(f"모니터 tick 실패: {e}")
            await asyncio.sleep(self.interval)

