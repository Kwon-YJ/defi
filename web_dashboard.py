#!/usr/bin/env python3
"""
Web Dashboard for DeFiPoser-ARB Real-Time Monitoring
실시간 모니터링 웹 대시보드

Provides:
- Real-time performance metrics display
- System resource monitoring
- Alert management interface
- Revenue tracking charts
- Configuration management
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO, emit
import json
from datetime import datetime, timedelta
import threading
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.real_time_monitoring_system import global_monitoring_system, get_dashboard
from src.performance_benchmarking import get_performance_report
from roi_performance_tracker import ROIPerformanceTracker

app = Flask(__name__)
app.config['SECRET_KEY'] = 'defiposer-monitoring-dashboard'
socketio = SocketIO(app, cors_allowed_origins="*")

# 전역 변수
roi_tracker = ROIPerformanceTracker()

# HTML 템플릿 (간단한 인라인 템플릿)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeFiPoser-ARB Real-Time Monitoring Dashboard</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        .metric-card.warning {
            border-left-color: #f39c12;
        }
        .metric-card.critical {
            border-left-color: #e74c3c;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .alerts-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .alert-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        .alert-item.warning {
            border-left-color: #f39c12;
            background-color: #fef9e7;
        }
        .alert-item.critical {
            border-left-color: #e74c3c;
            background-color: #fdedec;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online {
            background-color: #2ecc71;
        }
        .status-offline {
            background-color: #e74c3c;
        }
        .refresh-button {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 10px;
        }
        .refresh-button:hover {
            background: #5a67d8;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 DeFiPoser-ARB Real-Time Monitoring Dashboard</h1>
        <p>Target Execution Time: 6.43 seconds | Real-time Performance Tracking</p>
        <div>
            <span class="status-indicator" id="connectionStatus"></span>
            <span id="connectionText">연결 중...</span>
            <button class="refresh-button" onclick="refreshData()">새로고침</button>
            <span id="lastUpdate"></span>
        </div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card" id="executionTimeCard">
            <div class="metric-value" id="executionTime">-</div>
            <div class="metric-label">평균 실행 시간 (초)</div>
        </div>
        <div class="metric-card" id="successRateCard">
            <div class="metric-value" id="successRate">-</div>
            <div class="metric-label">성공률 (%)</div>
        </div>
        <div class="metric-card" id="dailyRevenueCard">
            <div class="metric-value" id="dailyRevenue">-</div>
            <div class="metric-label">일일 수익 (ETH)</div>
        </div>
        <div class="metric-card" id="weeklyRevenueCard">
            <div class="metric-value" id="weeklyRevenue">-</div>
            <div class="metric-label">주간 수익 목표 달성률 (%)</div>
        </div>
        <div class="metric-card" id="memoryUsageCard">
            <div class="metric-value" id="memoryUsage">-</div>
            <div class="metric-label">메모리 사용량 (MB)</div>
        </div>
        <div class="metric-card" id="cpuUsageCard">
            <div class="metric-value" id="cpuUsage">-</div>
            <div class="metric-label">CPU 사용률 (%)</div>
        </div>
    </div>

    <div class="chart-container">
        <h3>📈 실행 시간 추이</h3>
        <canvas id="executionTimeChart" width="400" height="200"></canvas>
    </div>

    <div class="chart-container">
        <h3>💰 수익률 추이</h3>
        <canvas id="revenueChart" width="400" height="200"></canvas>
    </div>

    <div class="alerts-container">
        <h3>🚨 최근 알림</h3>
        <div id="alertsList"></div>
    </div>

    <script>
        const socket = io();
        let executionTimeChart, revenueChart;

        // Chart.js 설정
        const chartOptions = {
            responsive: true,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'minute'
                    }
                },
                y: {
                    beginAtZero: false
                }
            },
            plugins: {
                legend: {
                    display: true
                }
            }
        };

        // 차트 초기화
        function initCharts() {
            const executionCtx = document.getElementById('executionTimeChart').getContext('2d');
            executionTimeChart = new Chart(executionCtx, {
                type: 'line',
                data: {
                    datasets: [{
                        label: '실행 시간 (초)',
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        data: []
                    }, {
                        label: '목표 시간 (6.43초)',
                        borderColor: '#e74c3c',
                        backgroundColor: 'transparent',
                        borderDash: [5, 5],
                        data: []
                    }]
                },
                options: chartOptions
            });

            const revenueCtx = document.getElementById('revenueChart').getContext('2d');
            revenueChart = new Chart(revenueCtx, {
                type: 'line',
                data: {
                    datasets: [{
                        label: '일일 수익 (ETH)',
                        borderColor: '#2ecc71',
                        backgroundColor: 'rgba(46, 204, 113, 0.1)',
                        data: []
                    }]
                },
                options: chartOptions
            });
        }

        // 연결 상태 업데이트
        socket.on('connect', function() {
            document.getElementById('connectionStatus').className = 'status-indicator status-online';
            document.getElementById('connectionText').textContent = '연결됨';
        });

        socket.on('disconnect', function() {
            document.getElementById('connectionStatus').className = 'status-indicator status-offline';
            document.getElementById('connectionText').textContent = '연결 끊김';
        });

        // 실시간 데이터 수신
        socket.on('dashboard_update', function(data) {
            updateDashboard(data);
        });

        // 대시보드 업데이트
        function updateDashboard(data) {
            try {
                // 성능 메트릭 업데이트
                if (data.performance && data.performance.current_performance) {
                    const perf = data.performance.current_performance;
                    if (perf.summary) {
                        updateMetric('executionTime', perf.summary.average_time?.toFixed(3) || '-', 
                                   perf.summary.average_time > 6.43 ? 'critical' : 
                                   perf.summary.average_time > 5.14 ? 'warning' : '');
                        
                        updateMetric('successRate', (perf.summary.success_rate * 100)?.toFixed(1) || '-',
                                   perf.summary.success_rate < 0.5 ? 'critical' : 
                                   perf.summary.success_rate < 0.8 ? 'warning' : '');
                    }
                }

                // ROI 메트릭 업데이트
                if (data.performance && data.performance.roi_metrics) {
                    const roi = data.performance.roi_metrics;
                    if (roi.basic_metrics) {
                        updateMetric('dailyRevenue', roi.basic_metrics.total_revenue?.toFixed(4) || '-');
                    }
                    if (roi.target_comparison) {
                        updateMetric('weeklyRevenue', roi.target_comparison.weekly_revenue_achievement?.toFixed(1) || '-',
                                   roi.target_comparison.weekly_revenue_achievement < 50 ? 'critical' :
                                   roi.target_comparison.weekly_revenue_achievement < 80 ? 'warning' : '');
                    }
                }

                // 시스템 메트릭 업데이트
                if (data.system && data.system.current_metrics) {
                    const sys = data.system.current_metrics;
                    updateMetric('memoryUsage', sys.memory_usage?.toFixed(0) || '-',
                               sys.memory_usage > 2048 ? 'critical' : 
                               sys.memory_usage > 1024 ? 'warning' : '');
                    
                    updateMetric('cpuUsage', sys.cpu_usage?.toFixed(1) || '-',
                               sys.cpu_usage > 95 ? 'critical' : 
                               sys.cpu_usage > 80 ? 'warning' : '');
                }

                // 차트 업데이트
                updateCharts(data);

                // 알림 업데이트
                updateAlerts(data.alerts?.recent_alerts || []);

                // 마지막 업데이트 시간
                document.getElementById('lastUpdate').textContent = 
                    '마지막 업데이트: ' + new Date().toLocaleTimeString();

            } catch (error) {
                console.error('Dashboard update error:', error);
            }
        }

        function updateMetric(id, value, alertClass = '') {
            const element = document.getElementById(id);
            const card = document.getElementById(id + 'Card');
            
            if (element) element.textContent = value;
            if (card) {
                card.className = 'metric-card' + (alertClass ? ' ' + alertClass : '');
            }
        }

        function updateCharts(data) {
            // 실행 시간 차트 업데이트
            if (data.system && data.system.trend_data && data.system.trend_data.execution_time) {
                const execData = data.system.trend_data.execution_time.map(point => ({
                    x: new Date(point[0]),
                    y: point[1]
                }));
                
                executionTimeChart.data.datasets[0].data = execData;
                
                // 목표선 데이터
                if (execData.length > 0) {
                    const targetData = execData.map(point => ({
                        x: point.x,
                        y: 6.43
                    }));
                    executionTimeChart.data.datasets[1].data = targetData;
                }
                
                executionTimeChart.update();
            }

            // 수익률 차트 업데이트
            if (data.system && data.system.trend_data && data.system.trend_data.daily_revenue) {
                const revenueData = data.system.trend_data.daily_revenue.map(point => ({
                    x: new Date(point[0]),
                    y: point[1]
                }));
                
                revenueChart.data.datasets[0].data = revenueData;
                revenueChart.update();
            }
        }

        function updateAlerts(alerts) {
            const alertsList = document.getElementById('alertsList');
            alertsList.innerHTML = '';

            if (alerts.length === 0) {
                alertsList.innerHTML = '<p>📝 알림이 없습니다.</p>';
                return;
            }

            alerts.forEach(alert => {
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert-item ' + alert.severity;
                
                const timestamp = new Date(alert.timestamp).toLocaleString();
                const emoji = alert.severity === 'critical' ? '🚨' : 
                            alert.severity === 'warning' ? '⚠️' : 'ℹ️';
                
                alertDiv.innerHTML = `
                    <strong>${emoji} ${alert.alert_type.toUpperCase()}</strong> - ${alert.severity.toUpperCase()}
                    <br>
                    ${alert.message}
                    <br>
                    <small>${timestamp}</small>
                `;
                
                alertsList.appendChild(alertDiv);
            });
        }

        function refreshData() {
            socket.emit('request_dashboard_update');
        }

        // 초기화
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            refreshData();
            
            // 30초마다 자동 새로고침
            setInterval(refreshData, 30000);
        });
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """대시보드 메인 페이지"""
    return DASHBOARD_HTML

@app.route('/api/status')
def api_status():
    """API 상태 확인"""
    return jsonify({
        "status": "online",
        "monitoring_active": global_monitoring_system.is_monitoring,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/dashboard')
def api_dashboard():
    """대시보드 데이터 API"""
    try:
        dashboard_data = get_dashboard()
        return jsonify(dashboard_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance')
def api_performance():
    """성능 데이터 API"""
    try:
        blocks = request.args.get('blocks', 50, type=int)
        performance_data = get_performance_report(last_n_blocks=blocks)
        return jsonify(performance_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/roi')
def api_roi():
    """ROI 데이터 API"""
    try:
        days = request.args.get('days', 7, type=int)
        roi_data = roi_tracker.generate_performance_report(days=days)
        return jsonify(roi_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/alerts')
def api_alerts():
    """알림 데이터 API"""
    try:
        dashboard_data = get_dashboard()
        alerts = dashboard_data.get("alerts", {})
        return jsonify(alerts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@socketio.on('request_dashboard_update')
def handle_dashboard_update():
    """클라이언트가 대시보드 업데이트를 요청할 때"""
    try:
        dashboard_data = get_dashboard()
        emit('dashboard_update', dashboard_data)
    except Exception as e:
        emit('error', {'message': str(e)})

def broadcast_updates():
    """주기적으로 대시보드 업데이트를 브로드캐스트"""
    while True:
        try:
            if global_monitoring_system.is_monitoring:
                dashboard_data = get_dashboard()
                socketio.emit('dashboard_update', dashboard_data)
            time.sleep(30)  # 30초마다 업데이트
        except Exception as e:
            print(f"Broadcast error: {e}")
            time.sleep(30)

def setup_monitoring_callback():
    """모니터링 시스템에 콜백 등록"""
    def alert_callback(alert):
        """알림이 발생했을 때 웹소켓으로 즉시 전송"""
        alert_data = {
            "timestamp": alert.timestamp.isoformat(),
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "message": alert.message,
            "data": alert.data
        }
        socketio.emit('new_alert', alert_data)
    
    global_monitoring_system.add_alert_callback(alert_callback)

if __name__ == '__main__':
    print("🌐 Starting DeFiPoser-ARB Web Dashboard")
    print("=" * 50)
    
    # 모니터링 콜백 설정
    setup_monitoring_callback()
    
    # 모니터링 시스템 시작
    if not global_monitoring_system.is_monitoring:
        global_monitoring_system.start_monitoring()
        print("✅ Monitoring system started")
    
    # 백그라운드 업데이트 스레드 시작
    update_thread = threading.Thread(target=broadcast_updates, daemon=True)
    update_thread.start()
    
    print("🚀 Dashboard server starting...")
    print("📊 Dashboard URL: http://localhost:5000")
    print("🔌 API endpoints:")
    print("   - GET /api/status")
    print("   - GET /api/dashboard") 
    print("   - GET /api/performance?blocks=50")
    print("   - GET /api/roi?days=7")
    print("   - GET /api/alerts")
    
    try:
        # Flask-SocketIO 서버 시작
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down dashboard...")
        global_monitoring_system.stop_monitoring()
        print("✅ Dashboard stopped")