#!/usr/bin/env python3
"""
Real-Time Monitoring System for DeFiPoser-ARB
실시간 모니터링 시스템 (TODO 109-112번 구현)

This system provides:
- 실행 시간 모니터링 (6.43초 목표 추적)
- 수익률 실시간 추적 시스템
- 시스템 리소스 모니터링
- Alert 및 notification 시스템

Based on DeFiPoser-ARB paper requirements for real-time operation
Target: Average 6.43 seconds execution time per block
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
import psutil
import statistics
from pathlib import Path
from collections import deque, defaultdict
import websockets
import requests
from concurrent.futures import ThreadPoolExecutor
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import telebot
from telebot import types

# Import existing performance components
try:
    from .performance_benchmarking import PerformanceBenchmarker, global_benchmarker
except ImportError:
    from performance_benchmarking import PerformanceBenchmarker, global_benchmarker

try:
    from roi_performance_tracker import ROIPerformanceTracker
except ImportError:
    from ..roi_performance_tracker import ROIPerformanceTracker

try:
    from .logger import setup_logger
except ImportError:
    def setup_logger(name):
        return logging.getLogger(name)

logger = setup_logger(__name__)

@dataclass
class MonitoringAlert:
    """모니터링 알림 데이터 클래스"""
    timestamp: datetime
    alert_type: str  # "performance", "revenue", "system", "error"
    severity: str   # "info", "warning", "critical"
    message: str
    data: Dict[str, Any]
    acknowledged: bool = False

@dataclass
class SystemMetrics:
    """시스템 메트릭 데이터 클래스"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int
    blockchain_sync_status: bool
    last_block_processed: int

class RealTimeMonitoringSystem:
    """
    DeFiPoser-ARB 실시간 모니터링 시스템
    
    Features:
    1. 실행 시간 모니터링 (6.43초 목표)
    2. 수익률 실시간 추적
    3. 시스템 리소스 모니터링  
    4. 알림 및 통지 시스템
    """
    
    def __init__(self, 
                 target_execution_time: float = 6.43,
                 monitoring_interval: int = 30,
                 alert_thresholds: Optional[Dict] = None):
        """
        Args:
            target_execution_time: 목표 실행 시간 (초)
            monitoring_interval: 모니터링 주기 (초)
            alert_thresholds: 알림 임계값 설정
        """
        self.target_execution_time = target_execution_time
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        
        # 기본 알림 임계값 설정
        self.alert_thresholds = alert_thresholds or {
            "execution_time_warning": target_execution_time * 0.8,  # 80%
            "execution_time_critical": target_execution_time,       # 100%
            "memory_usage_warning": 1024,    # MB
            "memory_usage_critical": 2048,   # MB
            "cpu_usage_warning": 80,         # %
            "cpu_usage_critical": 95,        # %
            "win_rate_warning": 70,          # %
            "win_rate_critical": 50,         # %
            "revenue_drop_warning": 20,      # % decrease
            "revenue_drop_critical": 40      # % decrease
        }
        
        # 컴포넌트 초기화
        self.performance_benchmarker = global_benchmarker
        self.roi_tracker = ROIPerformanceTracker()
        
        # 모니터링 데이터 저장소
        self.alerts_queue = deque(maxlen=1000)
        self.system_metrics_history = deque(maxlen=500)
        self.performance_trends = defaultdict(deque)
        
        # 알림 콜백 함수들
        self.alert_callbacks: List[Callable] = []
        
        # 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 데이터베이스 초기화
        self.db_path = "monitoring_system.db"
        self.init_database()
        
        # 알림 시스템 설정 로드
        self.load_notification_config()
        
        logger.info("Real-Time Monitoring System initialized")
        logger.info(f"Target execution time: {target_execution_time}s")
        logger.info(f"Monitoring interval: {monitoring_interval}s")

    def init_database(self):
        """모니터링 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 알림 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                data TEXT,
                acknowledged BOOLEAN DEFAULT FALSE
            )
        """)
        
        # 시스템 메트릭 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_usage REAL,
                memory_usage REAL,
                disk_usage REAL,
                network_io TEXT,
                active_connections INTEGER,
                blockchain_sync_status BOOLEAN,
                last_block_processed INTEGER
            )
        """)
        
        # 모니터링 이벤트 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS monitoring_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                description TEXT,
                data TEXT
            )
        """)
        
        conn.commit()
        conn.close()

    def load_notification_config(self):
        """알림 설정 로드"""
        config_path = Path("config/notification_config.json")
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.notification_config = json.load(f)
                logger.info("Notification config loaded")
            except Exception as e:
                logger.error(f"Failed to load notification config: {e}")
                self.notification_config = {}
        else:
            self.notification_config = {}
            logger.warning("No notification config found")

    def start_monitoring(self):
        """실시간 모니터링 시작"""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
            
        self.is_monitoring = True
        logger.info("🚀 Starting Real-Time Monitoring System")
        
        # 별도 스레드에서 모니터링 시작
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        # 시스템 메트릭 모니터링 시작
        system_thread = threading.Thread(target=self._system_monitoring_loop, daemon=True)
        system_thread.start()
        
        # 알림 처리 스레드 시작
        alert_thread = threading.Thread(target=self._alert_processing_loop, daemon=True)
        alert_thread.start()

    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        logger.info("🛑 Stopping Real-Time Monitoring System")

    def _monitoring_loop(self):
        """메인 모니터링 루프"""
        while self.is_monitoring:
            try:
                # 성능 메트릭 체크
                self._check_performance_metrics()
                
                # 수익률 메트릭 체크
                self._check_revenue_metrics()
                
                # 트렌드 분석
                self._analyze_trends()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self._create_alert("error", "critical", 
                                 f"Monitoring loop error: {str(e)}", 
                                 {"error": str(e)})
                time.sleep(self.monitoring_interval)

    def _system_monitoring_loop(self):
        """시스템 리소스 모니터링 루프"""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics_history.append(metrics)
                
                # 시스템 메트릭 체크
                self._check_system_metrics(metrics)
                
                # 데이터베이스에 저장
                self._save_system_metrics(metrics)
                
                time.sleep(15)  # 15초마다 시스템 메트릭 수집
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(15)

    def _alert_processing_loop(self):
        """알림 처리 루프"""
        while self.is_monitoring:
            try:
                if self.alerts_queue:
                    alert = self.alerts_queue.popleft()
                    self._process_alert(alert)
                    
                time.sleep(1)  # 1초마다 알림 큐 체크
                
            except Exception as e:
                logger.error(f"Error in alert processing: {e}")
                time.sleep(1)

    def _check_performance_metrics(self):
        """성능 메트릭 체크"""
        try:
            # 최근 성능 보고서 조회
            report = self.performance_benchmarker.get_performance_report(last_n_blocks=10)
            
            if "error" in report:
                return
                
            summary = report["summary"]
            
            # 실행 시간 체크
            avg_time = summary.get("average_time", 0)
            success_rate = summary.get("success_rate", 0)
            
            # 실행 시간 알림
            if avg_time > self.alert_thresholds["execution_time_critical"]:
                self._create_alert("performance", "critical",
                                 f"실행 시간이 목표를 크게 초과했습니다: {avg_time:.3f}초 (목표: {self.target_execution_time}초)",
                                 {"execution_time": avg_time, "target": self.target_execution_time})
            elif avg_time > self.alert_thresholds["execution_time_warning"]:
                self._create_alert("performance", "warning",
                                 f"실행 시간이 목표에 근접했습니다: {avg_time:.3f}초 (목표: {self.target_execution_time}초)",
                                 {"execution_time": avg_time, "target": self.target_execution_time})
            
            # 성공률 체크
            if success_rate < self.alert_thresholds["win_rate_critical"] / 100:
                self._create_alert("performance", "critical",
                                 f"성공률이 심각하게 낮습니다: {success_rate:.1%}",
                                 {"success_rate": success_rate})
            elif success_rate < self.alert_thresholds["win_rate_warning"] / 100:
                self._create_alert("performance", "warning",
                                 f"성공률이 낮습니다: {success_rate:.1%}",
                                 {"success_rate": success_rate})
                
            # 트렌드 데이터 저장
            self.performance_trends["execution_time"].append((datetime.now(), avg_time))
            self.performance_trends["success_rate"].append((datetime.now(), success_rate))
            
            # 오래된 트렌드 데이터 제거 (최대 1000개)
            for trend in self.performance_trends.values():
                if len(trend) > 1000:
                    trend.popleft()
                    
        except Exception as e:
            logger.error(f"Error checking performance metrics: {e}")

    def _check_revenue_metrics(self):
        """수익률 메트릭 체크"""
        try:
            # ROI 성과 보고서 조회
            report = self.roi_tracker.generate_performance_report(days=1)
            
            if "error" in report:
                return
                
            basic_metrics = report["basic_metrics"]
            target_comparison = report["target_comparison"]
            
            # 일일 수익률 체크
            daily_revenue = basic_metrics.get("total_revenue", 0)
            win_rate = basic_metrics.get("win_rate", 0)
            
            # 수익률 목표 달성 체크
            weekly_achievement = target_comparison.get("weekly_revenue_achievement", 0)
            
            if weekly_achievement < 50:  # 50% 미만
                self._create_alert("revenue", "critical",
                                 f"주간 수익 목표 달성률이 심각하게 낮습니다: {weekly_achievement:.1f}%",
                                 {"achievement": weekly_achievement})
            elif weekly_achievement < 80:  # 80% 미만
                self._create_alert("revenue", "warning",
                                 f"주간 수익 목표 달성률이 낮습니다: {weekly_achievement:.1f}%",
                                 {"achievement": weekly_achievement})
                
            # 승률 체크
            if win_rate < self.alert_thresholds["win_rate_critical"]:
                self._create_alert("revenue", "critical",
                                 f"거래 승률이 심각하게 낮습니다: {win_rate:.1f}%",
                                 {"win_rate": win_rate})
            elif win_rate < self.alert_thresholds["win_rate_warning"]:
                self._create_alert("revenue", "warning",
                                 f"거래 승률이 낮습니다: {win_rate:.1f}%",
                                 {"win_rate": win_rate})
            
            # 트렌드 데이터 저장
            self.performance_trends["daily_revenue"].append((datetime.now(), daily_revenue))
            self.performance_trends["win_rate"].append((datetime.now(), win_rate))
            
        except Exception as e:
            logger.error(f"Error checking revenue metrics: {e}")

    def _collect_system_metrics(self) -> SystemMetrics:
        """시스템 메트릭 수집"""
        # CPU 사용률
        cpu_usage = psutil.cpu_percent()
        
        # 메모리 사용률
        memory = psutil.virtual_memory()
        memory_usage = memory.used / 1024 / 1024  # MB
        
        # 디스크 사용률
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # 네트워크 I/O
        network_io = psutil.net_io_counters()
        network_data = {
            "bytes_sent": network_io.bytes_sent,
            "bytes_recv": network_io.bytes_recv,
            "packets_sent": network_io.packets_sent,
            "packets_recv": network_io.packets_recv
        }
        
        # 활성 연결 수
        active_connections = len(psutil.net_connections())
        
        # 블록체인 동기화 상태 (간단한 체크)
        blockchain_sync_status = self._check_blockchain_sync()
        
        # 마지막 처리된 블록 번호
        last_block_processed = self._get_last_processed_block()
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_data,
            active_connections=active_connections,
            blockchain_sync_status=blockchain_sync_status,
            last_block_processed=last_block_processed
        )

    def _check_system_metrics(self, metrics: SystemMetrics):
        """시스템 메트릭 알림 체크"""
        # CPU 사용률 체크
        if metrics.cpu_usage > self.alert_thresholds["cpu_usage_critical"]:
            self._create_alert("system", "critical",
                             f"CPU 사용률이 매우 높습니다: {metrics.cpu_usage:.1f}%",
                             {"cpu_usage": metrics.cpu_usage})
        elif metrics.cpu_usage > self.alert_thresholds["cpu_usage_warning"]:
            self._create_alert("system", "warning",
                             f"CPU 사용률이 높습니다: {metrics.cpu_usage:.1f}%",
                             {"cpu_usage": metrics.cpu_usage})
        
        # 메모리 사용률 체크
        if metrics.memory_usage > self.alert_thresholds["memory_usage_critical"]:
            self._create_alert("system", "critical",
                             f"메모리 사용량이 매우 높습니다: {metrics.memory_usage:.0f}MB",
                             {"memory_usage": metrics.memory_usage})
        elif metrics.memory_usage > self.alert_thresholds["memory_usage_warning"]:
            self._create_alert("system", "warning",
                             f"메모리 사용량이 높습니다: {metrics.memory_usage:.0f}MB",
                             {"memory_usage": metrics.memory_usage})
        
        # 블록체인 동기화 상태 체크
        if not metrics.blockchain_sync_status:
            self._create_alert("system", "critical",
                             "블록체인 동기화 상태에 문제가 있습니다",
                             {"sync_status": False, "last_block": metrics.last_block_processed})

    def _check_blockchain_sync(self) -> bool:
        """블록체인 동기화 상태 체크 (간단한 구현)"""
        try:
            # 여기서는 간단한 체크만 수행
            # 실제로는 Web3 연결을 통해 최신 블록과 로컬 블록을 비교해야 함
            return True  # 임시로 True 반환
        except:
            return False

    def _get_last_processed_block(self) -> int:
        """마지막 처리된 블록 번호 조회"""
        try:
            # 실제로는 시스템에서 마지막으로 처리한 블록 번호를 조회
            return 0  # 임시로 0 반환
        except:
            return 0

    def _analyze_trends(self):
        """트렌드 분석"""
        try:
            # 실행 시간 트렌드 분석
            if len(self.performance_trends["execution_time"]) >= 10:
                recent_times = list(self.performance_trends["execution_time"])[-10:]
                times_only = [t[1] for t in recent_times]
                
                # 증가 트렌드 체크
                if len(times_only) >= 5:
                    recent_avg = statistics.mean(times_only[-5:])
                    older_avg = statistics.mean(times_only[-10:-5])
                    
                    if recent_avg > older_avg * 1.2:  # 20% 증가
                        self._create_alert("performance", "warning",
                                         f"실행 시간이 증가하는 추세입니다 (평균 {recent_avg:.3f}초)",
                                         {"trend": "increasing", "recent_avg": recent_avg})
            
            # 수익률 트렌드 분석
            if len(self.performance_trends["daily_revenue"]) >= 7:
                recent_revenues = list(self.performance_trends["daily_revenue"])[-7:]
                revenues_only = [r[1] for r in recent_revenues]
                
                if len(revenues_only) >= 4:
                    recent_avg = statistics.mean(revenues_only[-3:])
                    older_avg = statistics.mean(revenues_only[-7:-3])
                    
                    if older_avg > 0 and (recent_avg / older_avg) < 0.8:  # 20% 감소
                        decrease_percent = ((older_avg - recent_avg) / older_avg) * 100
                        self._create_alert("revenue", "warning",
                                         f"수익률이 감소하는 추세입니다 ({decrease_percent:.1f}% 감소)",
                                         {"trend": "decreasing", "decrease_percent": decrease_percent})
                        
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")

    def _create_alert(self, alert_type: str, severity: str, message: str, data: Dict):
        """알림 생성"""
        alert = MonitoringAlert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            data=data
        )
        
        self.alerts_queue.append(alert)
        
        # 데이터베이스에 저장
        self._save_alert(alert)
        
        # 즉시 로깅
        if severity == "critical":
            logger.error(f"🚨 CRITICAL ALERT [{alert_type}]: {message}")
        elif severity == "warning":
            logger.warning(f"⚠️ WARNING [{alert_type}]: {message}")
        else:
            logger.info(f"ℹ️ INFO [{alert_type}]: {message}")

    def _process_alert(self, alert: MonitoringAlert):
        """알림 처리"""
        try:
            # 콜백 함수 실행
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
            
            # 알림 방식별 처리
            if alert.severity in ["critical", "warning"]:
                self._send_notifications(alert)
                
        except Exception as e:
            logger.error(f"Error processing alert: {e}")

    def _send_notifications(self, alert: MonitoringAlert):
        """알림 전송"""
        # 텔레그램 알림
        if "telegram" in self.notification_config:
            self._send_telegram_alert(alert)
        
        # 이메일 알림
        if "email" in self.notification_config:
            self._send_email_alert(alert)

    def _send_telegram_alert(self, alert: MonitoringAlert):
        """텔레그램 알림 전송"""
        try:
            config = self.notification_config.get("telegram", {})
            bot_token = config.get("bot_token")
            chat_id = config.get("chat_id")
            
            if not bot_token or not chat_id:
                return
                
            bot = telebot.TeleBot(bot_token)
            
            emoji = "🚨" if alert.severity == "critical" else "⚠️"
            message = f"{emoji} DeFiPoser-ARB Alert\n\n"
            message += f"Type: {alert.alert_type}\n"
            message += f"Severity: {alert.severity.upper()}\n"
            message += f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            message += f"Message: {alert.message}"
            
            bot.send_message(chat_id, message)
            logger.info("Telegram alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send telegram alert: {e}")

    def _send_email_alert(self, alert: MonitoringAlert):
        """이메일 알림 전송"""
        try:
            config = self.notification_config.get("email", {})
            smtp_server = config.get("smtp_server")
            smtp_port = config.get("smtp_port", 587)
            username = config.get("username")
            password = config.get("password")
            to_email = config.get("to_email")
            
            if not all([smtp_server, username, password, to_email]):
                return
                
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = to_email
            msg['Subject'] = f"DeFiPoser-ARB Alert: {alert.severity.upper()}"
            
            body = f"""
DeFiPoser-ARB Real-Time Monitoring Alert

Type: {alert.alert_type}
Severity: {alert.severity.upper()}
Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Message: {alert.message}

Data: {json.dumps(alert.data, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            text = msg.as_string()
            server.sendmail(username, to_email, text)
            server.quit()
            
            logger.info("Email alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _save_alert(self, alert: MonitoringAlert):
        """알림을 데이터베이스에 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO alerts 
                (timestamp, alert_type, severity, message, data, acknowledged)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                alert.timestamp.isoformat(),
                alert.alert_type,
                alert.severity,
                alert.message,
                json.dumps(alert.data),
                alert.acknowledged
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save alert to database: {e}")

    def _save_system_metrics(self, metrics: SystemMetrics):
        """시스템 메트릭을 데이터베이스에 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO system_metrics 
                (timestamp, cpu_usage, memory_usage, disk_usage, 
                 network_io, active_connections, blockchain_sync_status, last_block_processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(),
                metrics.cpu_usage,
                metrics.memory_usage,
                metrics.disk_usage,
                json.dumps(metrics.network_io),
                metrics.active_connections,
                metrics.blockchain_sync_status,
                metrics.last_block_processed
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save system metrics: {e}")

    def add_alert_callback(self, callback: Callable[[MonitoringAlert], None]):
        """알림 콜백 함수 추가"""
        self.alert_callbacks.append(callback)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """대시보드용 데이터 조회"""
        try:
            # 최근 성과 데이터
            performance_report = self.performance_benchmarker.get_performance_report(last_n_blocks=50)
            roi_report = self.roi_tracker.generate_performance_report(days=7)
            
            # 최근 알림들
            recent_alerts = list(self.alerts_queue)[-10:]
            
            # 최근 시스템 메트릭
            recent_metrics = list(self.system_metrics_history)[-10:]
            
            dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "performance": {
                    "target_execution_time": self.target_execution_time,
                    "current_performance": performance_report,
                    "roi_metrics": roi_report
                },
                "system": {
                    "current_metrics": recent_metrics[-1].__dict__ if recent_metrics else {},
                    "trend_data": {
                        key: list(deque_data)[-20:]  # 최근 20개
                        for key, deque_data in self.performance_trends.items()
                    }
                },
                "alerts": {
                    "recent_alerts": [asdict(alert) for alert in recent_alerts],
                    "alert_counts": {
                        "critical": sum(1 for a in recent_alerts if a.severity == "critical"),
                        "warning": sum(1 for a in recent_alerts if a.severity == "warning"),
                        "info": sum(1 for a in recent_alerts if a.severity == "info")
                    }
                },
                "status": {
                    "monitoring_active": self.is_monitoring,
                    "last_update": datetime.now().isoformat()
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
            return {"error": str(e)}

    def export_monitoring_report(self, filepath: str, days: int = 7):
        """모니터링 보고서 내보내기"""
        try:
            dashboard_data = self.get_dashboard_data()
            
            # 상세 보고서 데이터 추가
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 기간 내 알림들
            cursor.execute("""
                SELECT * FROM alerts 
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
            """, (start_date.isoformat(), end_date.isoformat()))
            
            alerts_data = cursor.fetchall()
            
            # 기간 내 시스템 메트릭
            cursor.execute("""
                SELECT * FROM system_metrics 
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
            """, (start_date.isoformat(), end_date.isoformat()))
            
            metrics_data = cursor.fetchall()
            conn.close()
            
            # 보고서 데이터 구성
            report = {
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days
                },
                "summary": dashboard_data,
                "detailed_alerts": alerts_data,
                "detailed_metrics": metrics_data,
                "generated_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Monitoring report exported to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export monitoring report: {e}")

# 전역 모니터링 시스템 인스턴스
global_monitoring_system = RealTimeMonitoringSystem()

# 편의 함수들
def start_monitoring():
    """전역 모니터링 시스템 시작"""
    global_monitoring_system.start_monitoring()

def stop_monitoring():
    """전역 모니터링 시스템 중지"""
    global_monitoring_system.stop_monitoring()

def get_monitoring_status():
    """모니터링 상태 조회"""
    return global_monitoring_system.is_monitoring

def add_monitoring_callback(callback: Callable):
    """모니터링 콜백 추가"""
    global_monitoring_system.add_alert_callback(callback)

def get_dashboard():
    """대시보드 데이터 조회"""
    return global_monitoring_system.get_dashboard_data()

if __name__ == "__main__":
    # 테스트 코드
    print("🔧 Real-Time Monitoring System Test")
    
    def test_callback(alert):
        print(f"📢 Alert Callback: {alert.severity} - {alert.message}")
    
    # 콜백 등록
    add_monitoring_callback(test_callback)
    
    # 모니터링 시작
    start_monitoring()
    
    print("✅ Monitoring system started")
    print("📊 Dashboard data:")
    
    # 대시보드 데이터 출력
    dashboard = get_dashboard()
    print(json.dumps(dashboard, indent=2, default=str))
    
    try:
        # 60초 동안 실행
        time.sleep(60)
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitoring...")
        stop_monitoring()
    
    print("✅ Test completed")