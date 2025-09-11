import os
import json
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple

from src.logger import setup_logger

logger = setup_logger(__name__)


def _parse_timestamp(ts: str) -> datetime:
    """ISO8601 또는 일반 문자열 타임스탬프 파서."""
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        # 흔한 포맷들 추가 시도
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(ts, fmt)
            except Exception:
                continue
        raise


@dataclass
class BacktestConfig:
    input_path: Optional[str] = None
    days: int = 150
    synthesize: bool = False
    synthetic_daily_mean_eth: float = 3.0
    synthetic_daily_std_eth: float = 1.0
    out_dir: str = "reports"
    emit_html: bool = True


class Backtester:
    """150일 백테스트 러너.

    입력: 기회(opportunity) 레코드 파일(JSONL/JSON/CSV) 또는 합성 데이터.
      - 필수 필드: timestamp(ISO8601), net_profit(ETH 단위 float)
      - 선택 필드: dexes(list[str]), confidence(float), path(list[str]) 등
    출력: JSON 요약, 일별 CSV, (옵션) 간단 HTML 대시보드
    """

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        try:
            from config.config import config
            self._weekly_target = float(getattr(config, 'weekly_profit_target_eth', 191.48))
        except Exception:
            self._weekly_target = 191.48

    def _load_opportunities(self) -> List[Dict[str, Any]]:
        p = self.cfg.input_path
        if p and not os.path.exists(p):
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {p}")

        if p is None and not self.cfg.synthesize:
            # 입력 미지정 시 합성으로 대체
            logger.warning("입력 파일이 없어 합성 데이터로 대체합니다.")
            self.cfg.synthesize = True

        if self.cfg.synthesize:
            return self._synthesize_opportunities()

        ext = os.path.splitext(p)[1].lower()
        if ext in ('.jsonl', '.ndjson'):
            return self._load_jsonl(p)
        if ext == '.json':
            return self._load_json(p)
        if ext == '.csv':
            return self._load_csv(p)
        # 기본은 jsonl로 시도
        return self._load_jsonl(p)

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    out.append(obj)
                except Exception as e:
                    logger.debug(f"JSONL 파싱 실패: {e}")
                    continue
        return out

    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and 'opportunities' in obj:
            return list(obj['opportunities'])
        raise ValueError("JSON 포맷을 해석할 수 없습니다 (list 또는 {opportunities: []} 필요)")

    def _load_csv(self, path: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    obj: Dict[str, Any] = {
                        'timestamp': row.get('timestamp') or row.get('time') or row.get('date'),
                        'net_profit': float(row.get('net_profit', '0') or '0')
                    }
                    if 'dexes' in row and row['dexes']:
                        try:
                            obj['dexes'] = json.loads(row['dexes'])
                        except Exception:
                            obj['dexes'] = [x.strip() for x in row['dexes'].split('|') if x.strip()]
                    out.append(obj)
                except Exception as e:
                    logger.debug(f"CSV 행 파싱 실패: {e}")
                    continue
        return out

    def _synthesize_opportunities(self) -> List[Dict[str, Any]]:
        import random
        now = datetime.now()
        start = (now - timedelta(days=self.cfg.days)).replace(hour=0, minute=0, second=0, microsecond=0)
        out: List[Dict[str, Any]] = []
        # 하루당 40건 평균으로 임의 생성
        for i in range(self.cfg.days):
            d = start + timedelta(days=i)
            # 일일 목표치를 기준으로 분포를 만들 수도 있지만, 기본은 평균치 기반
            n = max(10, int(random.gauss(40, 10)))
            # net_profit 합이 대략 지정 평균 주변이 되도록 분배
            target_sum = max(0.0, random.gauss(self.cfg.synthetic_daily_mean_eth, self.cfg.synthetic_daily_std_eth))
            # n개의 양/음수 무작위 배분 후 스케일링
            raw = [random.gauss(0.1, 0.2) for _ in range(n)]
            s = sum(abs(x) for x in raw) or 1.0
            scale = target_sum / s
            for x in raw:
                # 일부는 손실도 포함
                val = (x if random.random() > 0.2 else -abs(x)) * scale
                t = d + timedelta(minutes=random.randint(0, 23 * 60 + 59))
                out.append({
                    'timestamp': t.isoformat(),
                    'net_profit': float(val),
                    'dexes': ['uniswap_v2']
                })
        return out

    @staticmethod
    def _group_daily(ops: List[Dict[str, Any]]) -> Tuple[Dict[date, Dict[str, Any]], int, int, float, float]:
        daily: Dict[date, Dict[str, Any]] = {}
        total_ops = 0
        profitable_ops = 0
        best_trade = float('-inf')
        worst_trade = float('inf')
        for opp in ops:
            try:
                ts = _parse_timestamp(str(opp['timestamp']))
            except Exception:
                continue
            d = ts.date()
            net = float(opp.get('net_profit', 0) or 0)
            total_ops += 1
            if net > 0:
                profitable_ops += 1
            best_trade = max(best_trade, net)
            worst_trade = min(worst_trade, net)
            slot = daily.setdefault(d, {'profit': 0.0, 'count': 0})
            slot['profit'] += net
            slot['count'] += 1
        return daily, total_ops, profitable_ops, best_trade, worst_trade

    @staticmethod
    def _weekly_buckets(daily: Dict[date, Dict[str, Any]], start: date, weeks: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        cursor = start
        for _ in range(weeks):
            end = cursor + timedelta(days=6)
            profit = 0.0
            cnt = 0
            for i in range(7):
                d = cursor + timedelta(days=i)
                if d in daily:
                    profit += float(daily[d]['profit'])
                    cnt += int(daily[d]['count'])
            out.append({'start': cursor.isoformat(), 'end': end.isoformat(), 'profit': profit, 'count': cnt})
            cursor = end + timedelta(days=1)
        return out

    def run(self) -> Dict[str, Any]:
        ops = self._load_opportunities()
        if not ops:
            raise ValueError("백테스트 입력 데이터가 없습니다.")

        # 기간 결정: 데이터 내 최신 시점 기준 과거 days일
        max_ts = max((_parse_timestamp(str(o['timestamp'])) for o in ops if 'timestamp' in o), default=datetime.now())
        start_ts = (max_ts - timedelta(days=self.cfg.days)).replace(hour=0, minute=0, second=0, microsecond=0)
        filtered = [o for o in ops if _parse_timestamp(str(o['timestamp'])) >= start_ts]

        daily, total_ops, profitable_ops, best_trade, worst_trade = self._group_daily(filtered)
        # 연속 days 일자 구성
        days_list: List[date] = [ (start_ts + timedelta(days=i)).date() for i in range(self.cfg.days) ]
        # 일자 누락은 0으로 채움
        for d in days_list:
            daily.setdefault(d, {'profit': 0.0, 'count': 0})

        avg_daily = sum(v['profit'] for v in daily.values()) / float(self.cfg.days)
        total_profit = sum(v['profit'] for v in daily.values())
        success_rate = (profitable_ops / total_ops * 100.0) if total_ops > 0 else 0.0

        # 주간 버킷
        full_weeks = self.cfg.days // 7
        weekly = self._weekly_buckets(daily, days_list[0], full_weeks)
        weeks_meeting_target = sum(1 for w in weekly if w['profit'] >= self._weekly_target)

        result: Dict[str, Any] = {
            'window': {'start': days_list[0].isoformat(), 'end': days_list[-1].isoformat()},
            'days': self.cfg.days,
            'total_opportunities': total_ops,
            'profitable_opportunities': profitable_ops,
            'success_rate_pct': success_rate,
            'total_profit_eth': total_profit,
            'avg_daily_profit_eth': avg_daily,
            'best_trade_eth': best_trade if best_trade != float('-inf') else 0.0,
            'worst_trade_eth': worst_trade if worst_trade != float('inf') else 0.0,
            'weekly_buckets': weekly,
            'weekly_target_eth': self._weekly_target,
            'weeks_meeting_target': weeks_meeting_target,
        }
        return result

    def save_reports(self, result: Dict[str, Any]) -> Dict[str, str]:
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = os.path.join(self.cfg.out_dir, f"backtest_{ts}")
        paths: Dict[str, str] = {}

        # JSON 요약 저장
        json_path = base + ".json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        paths['json'] = json_path

        # 일별 CSV 저장
        # daily 정보는 weekly만 있으므로 재구성
        start = datetime.fromisoformat(result['window']['start']).date()
        days = int(result.get('days', 0))
        daily_rows: List[Dict[str, Any]] = []
        # weekly에서 다시 합치기보다는 단순히 균등 분배가 아닌, run 시점에 계산한 값을 유지하는 편이 정확하지만
        # 여기서는 주 요약만 표준 출력하도록 하고, CSV는 주간 합계만 저장
        csv_path = base + "_weekly.csv"
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['start', 'end', 'profit_eth', 'count', 'target_eth', 'met_target'])
            writer.writeheader()
            for w in result['weekly_buckets']:
                writer.writerow({
                    'start': w['start'],
                    'end': w['end'],
                    'profit_eth': w['profit'],
                    'count': w['count'],
                    'target_eth': result['weekly_target_eth'],
                    'met_target': int(w['profit'] >= result['weekly_target_eth'])
                })
        paths['csv_weekly'] = csv_path

        # 간단 HTML 대시보드 (바 차트: 주간 수익 vs 목표)
        if self.cfg.emit_html:
            html_path = base + ".html"
            bars = []
            max_val = max(max((w['profit'] for w in result['weekly_buckets']), default=0.0), result['weekly_target_eth'])
            scale = 300.0 / max(1e-9, max_val)
            for idx, w in enumerate(result['weekly_buckets']):
                h = max(1.0, w['profit'] * scale)
                met = w['profit'] >= result['weekly_target_eth']
                color = '#4caf50' if met else '#f44336'
                label = f"W{idx+1}"
                bars.append(f'<div style="display:inline-block;width:20px;height:{h:.1f}px;background:{color};margin:0 3px" title="{label}: {w['+"'"+'profit'+"'"+']:.4f} ETH"></div>')
            target_line = int(result['weekly_target_eth'] * scale)
            html = f"""
<!doctype html>
<meta charset="utf-8" />
<title>Backtest Report</title>
<div style="font-family:sans-serif;max-width:960px;margin:20px auto">
  <h2>Backtest Report ({result['window']['start']} → {result['window']['end']}, {result['days']} days)</h2>
  <p>Total Profit: <b>{result['total_profit_eth']:.4f} ETH</b>, Avg Daily: {result['avg_daily_profit_eth']:.4f} ETH<br/>
     Success Rate: {result['success_rate_pct']:.2f}%<br/>
     Weeks meeting target {result['weekly_target_eth']} ETH: <b>{result['weeks_meeting_target']}/{len(result['weekly_buckets'])}</b></p>
  <div style="border-left:1px solid #999;border-bottom:1px solid #999;height:320px;padding:10px 0 0 10px;">
    <div style="position:relative;height:300px">
      <div style="position:absolute;left:0;bottom:{target_line}px;right:0;border-bottom:1px dashed #555;color:#555;font-size:12px;">
        <span style="position:absolute;left:0;bottom:4px;background:#fff">Target {result['weekly_target_eth']} ETH</span>
      </div>
      <div style="position:absolute;left:0;bottom:0;">
        {''.join(bars)}
      </div>
    </div>
  </div>
  <p style="color:#777">Green: met target, Red: below target</p>
  <p>Generated at {datetime.now().isoformat()}</p>
<div>
"""
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
            paths['html'] = html_path

        return paths

