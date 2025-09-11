from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, List, Any

from src.data_storage import DataStorage
from src.logger import setup_logger
from config.config import config

logger = setup_logger(__name__)


@dataclass
class ROIConfig:
    lookback_days: int = 150
    initial_capital_eth: float = float(getattr(config, 'roi_initial_capital_eth', 1.0))


class ROITracker:
    def __init__(self, cfg: ROIConfig | None = None):
        self.cfg = cfg or ROIConfig()
        self.storage = DataStorage()

    async def _daily_profits(self) -> Dict[date, float]:
        ops = await self.storage.get_recent_opportunities(100000)
        cutoff = datetime.now() - timedelta(days=self.cfg.lookback_days)
        days: Dict[date, float] = {}
        for o in ops:
            try:
                ts = datetime.fromisoformat(o['timestamp'])
            except Exception:
                continue
            if ts < cutoff:
                continue
            d = ts.date()
            days[d] = days.get(d, 0.0) + float(o.get('net_profit', 0) or 0.0)
        # fill missing
        start = (datetime.now() - timedelta(days=self.cfg.lookback_days)).date()
        for i in range(self.cfg.lookback_days):
            dd = start + timedelta(days=i)
            days.setdefault(dd, 0.0)
        return dict(sorted(days.items(), key=lambda kv: kv[0]))

    @staticmethod
    def _max_drawdown(series: List[float]) -> float:
        peak = -1e18
        max_dd = 0.0
        for v in series:
            if v > peak:
                peak = v
            dd = (peak - v)
            if dd > max_dd:
                max_dd = dd
        return max_dd

    async def generate_report(self) -> Dict[str, Any]:
        days = await self._daily_profits()
        ordered = list(days.items())
        cum = []
        acc = 0.0
        for _, p in ordered:
            acc += p
            cum.append(acc)
        total = acc
        avg_daily = (sum(days.values()) / len(days)) if days else 0.0
        std_daily = 0.0
        if len(days) > 1:
            m = avg_daily
            var = sum((p - m) ** 2 for p in days.values()) / (len(days) - 1)
            std_daily = math.sqrt(max(0.0, var))
        sharpe_like = (avg_daily / std_daily * math.sqrt(365)) if std_daily > 0 else 0.0
        mdd_eth = self._max_drawdown(cum)
        roi_pct = (total / max(1e-9, float(self.cfg.initial_capital_eth))) * 100.0
        return {
            'lookback_days': self.cfg.lookback_days,
            'initial_capital_eth': float(self.cfg.initial_capital_eth),
            'total_profit_eth': total,
            'avg_daily_profit_eth': avg_daily,
            'std_daily_profit_eth': std_daily,
            'sharpe_like': sharpe_like,
            'max_drawdown_eth': mdd_eth,
            'roi_percentage': roi_pct,
            'daily_series': [{ 'date': d.isoformat(), 'profit_eth': p } for d, p in ordered],
            'cumulative_series': [{ 'date': (ordered[i][0]).isoformat(), 'cum_eth': cum[i] } for i in range(len(ordered))]
        }

