#!/usr/bin/env python3
"""
DEFIPOSER-ARB Performance Analyzer
실행 시간 분석 및 최적화 권장사항 제공

이 스크립트는 성능 로그를 분석하여 병목점을 식별하고 최적화 방안을 제시합니다.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import numpy as np
from typing import Dict, List, Tuple
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.performance_benchmarking import get_performance_report
from src.logger import setup_logger

logger = setup_logger(__name__)

class PerformanceAnalyzer:
    """성능 분석기"""
    
    def __init__(self, log_file_path: str = "logs/performance_benchmark.json"):
        self.log_file_path = Path(log_file_path)
        self.target_time = 6.43  # 논문 목표 시간
        
        # 플롯 스타일 설정
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 한글 폰트 설정 (matplotlib)
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    
    def load_performance_data(self) -> pd.DataFrame:
        """성능 로그 데이터 로드"""
        if not self.log_file_path.exists():
            logger.error(f"성능 로그 파일이 없습니다: {self.log_file_path}")
            return pd.DataFrame()
        
        data = []
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if 'metrics' in entry:
                            metrics = entry['metrics']
                            metrics['timestamp'] = pd.to_datetime(entry['timestamp'])
                            data.append(metrics)
            
            df = pd.DataFrame(data)
            logger.info(f"성능 데이터 로드 완료: {len(df)}개 기록")
            return df
            
        except Exception as e:
            logger.error(f"성능 데이터 로드 실패: {e}")
            return pd.DataFrame()
    
    def analyze_execution_times(self, df: pd.DataFrame) -> Dict:
        """실행 시간 분석"""
        if df.empty:
            return {"error": "데이터가 없습니다"}
        
        execution_times = df['total_execution_time']
        
        analysis = {
            "basic_stats": {
                "count": len(execution_times),
                "mean": execution_times.mean(),
                "median": execution_times.median(),
                "std": execution_times.std(),
                "min": execution_times.min(),
                "max": execution_times.max(),
                "target_achievement_rate": (execution_times <= self.target_time).mean()
            },
            "percentiles": {
                "p25": execution_times.quantile(0.25),
                "p50": execution_times.quantile(0.50),
                "p75": execution_times.quantile(0.75),
                "p90": execution_times.quantile(0.90),
                "p95": execution_times.quantile(0.95),
                "p99": execution_times.quantile(0.99)
            },
            "target_analysis": {
                "under_target": (execution_times <= self.target_time).sum(),
                "over_target": (execution_times > self.target_time).sum(),
                "worst_cases": execution_times.nlargest(5).tolist()
            }
        }
        
        return analysis
    
    def analyze_components(self, df: pd.DataFrame) -> Dict:
        """컴포넌트별 성능 분석"""
        if df.empty:
            return {"error": "데이터가 없습니다"}
        
        components = [
            'graph_building_time',
            'negative_cycle_detection_time', 
            'local_search_time',
            'parameter_optimization_time',
            'validation_time'
        ]
        
        component_analysis = {}
        
        for comp in components:
            if comp in df.columns:
                comp_data = df[comp]
                component_analysis[comp] = {
                    "mean": comp_data.mean(),
                    "median": comp_data.median(),
                    "max": comp_data.max(),
                    "std": comp_data.std(),
                    "percentage_of_target": (comp_data.mean() / self.target_time) * 100,
                    "bottleneck_frequency": (comp_data > self.target_time * 0.3).sum()  # 30% 이상인 경우
                }
        
        return component_analysis
    
    def identify_bottlenecks(self, df: pd.DataFrame) -> List[Dict]:
        """성능 병목점 식별"""
        if df.empty:
            return []
        
        bottlenecks = []
        
        # 컴포넌트별 병목점 분석
        components = [
            ('graph_building_time', '그래프 구축'),
            ('negative_cycle_detection_time', 'Negative Cycle 탐지'),
            ('local_search_time', 'Local Search'),
            ('parameter_optimization_time', '파라미터 최적화'),
            ('validation_time', '검증')
        ]
        
        for comp_col, comp_name in components:
            if comp_col in df.columns:
                comp_data = df[comp_col]
                mean_time = comp_data.mean()
                
                if mean_time > self.target_time * 0.4:  # 40% 이상
                    bottlenecks.append({
                        "component": comp_name,
                        "average_time": mean_time,
                        "percentage_of_target": (mean_time / self.target_time) * 100,
                        "severity": "높음" if mean_time > self.target_time * 0.6 else "중간",
                        "recommendation": self._get_component_recommendation(comp_name)
                    })
        
        # 실행 시간 초과 패턴 분석
        over_target_blocks = df[df['total_execution_time'] > self.target_time]
        if len(over_target_blocks) > 0:
            avg_over_time = over_target_blocks['total_execution_time'].mean()
            bottlenecks.append({
                "component": "전체 시스템",
                "average_time": avg_over_time,
                "percentage_of_target": (avg_over_time / self.target_time) * 100,
                "severity": "높음" if len(over_target_blocks) > len(df) * 0.3 else "중간",
                "frequency": len(over_target_blocks),
                "recommendation": "전체 시스템 최적화 필요"
            })
        
        return sorted(bottlenecks, key=lambda x: x['average_time'], reverse=True)
    
    def _get_component_recommendation(self, component_name: str) -> str:
        """컴포넌트별 최적화 권장사항"""
        recommendations = {
            "그래프 구축": "메모리 효율적인 그래프 표현, 캐싱 메커니즘 도입",
            "Negative Cycle 탐지": "Bellman-Ford 알고리즘 최적화, 병렬 처리 개선",
            "Local Search": "Hill climbing 알고리즘 최적화, 시작점 선택 개선",
            "파라미터 최적화": "이진 탐색 개선, 초기 추정값 정확도 향상",
            "검증": "검증 로직 간소화, 캐싱 활용"
        }
        return recommendations.get(component_name, "개별 최적화 방안 검토 필요")
    
    def generate_performance_plots(self, df: pd.DataFrame, output_dir: str = "reports"):
        """성능 분석 그래프 생성"""
        if df.empty:
            logger.error("그래프 생성할 데이터가 없습니다")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. 실행 시간 분포 히스토그램
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(df['total_execution_time'], bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(self.target_time, color='red', linestyle='--', 
                   label=f'Target: {self.target_time}s')
        plt.xlabel('Execution Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Execution Time Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 시간별 실행 시간 추이
        plt.subplot(1, 2, 2)
        plt.plot(df['timestamp'], df['total_execution_time'], alpha=0.7, linewidth=1)
        plt.axhline(self.target_time, color='red', linestyle='--', 
                   label=f'Target: {self.target_time}s')
        plt.xlabel('Time')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time Trend')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'execution_time_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 컴포넌트별 시간 분석
        component_columns = [
            'graph_building_time',
            'negative_cycle_detection_time',
            'local_search_time', 
            'parameter_optimization_time',
            'validation_time'
        ]
        
        available_components = [col for col in component_columns if col in df.columns]
        
        if available_components:
            plt.figure(figsize=(12, 8))
            
            # 컴포넌트별 박스 플롯
            component_data = df[available_components]
            component_data.columns = [col.replace('_time', '').replace('_', ' ').title() 
                                    for col in component_data.columns]
            
            plt.subplot(2, 1, 1)
            component_data.boxplot(ax=plt.gca())
            plt.title('Component Execution Times (Box Plot)')
            plt.ylabel('Time (seconds)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 컴포넌트별 평균 시간 막대 그래프
            plt.subplot(2, 1, 2)
            means = component_data.mean()
            bars = plt.bar(means.index, means.values, alpha=0.7)
            plt.axhline(self.target_time * 0.3, color='orange', linestyle='--', 
                       label='30% of Target', alpha=0.7)
            plt.axhline(self.target_time * 0.5, color='red', linestyle='--', 
                       label='50% of Target', alpha=0.7)
            plt.title('Average Component Execution Times')
            plt.ylabel('Time (seconds)')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 막대 위에 값 표시
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}s', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path / 'component_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. 성능 달성률 파이 차트
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        achievement_counts = [(df['total_execution_time'] <= self.target_time).sum(),
                             (df['total_execution_time'] > self.target_time).sum()]
        labels = ['Target Achieved', 'Target Missed']
        colors = ['green', 'red']
        plt.pie(achievement_counts, labels=labels, colors=colors, autopct='%1.1f%%', 
                startangle=90)
        plt.title('Target Achievement Rate')
        
        # 5. 기회 발견 대비 실행시간 산점도  
        plt.subplot(1, 2, 2)
        if 'opportunities_found' in df.columns:
            plt.scatter(df['opportunities_found'], df['total_execution_time'], 
                       alpha=0.6, s=30)
            plt.axhline(self.target_time, color='red', linestyle='--', alpha=0.7)
            plt.xlabel('Opportunities Found')
            plt.ylabel('Execution Time (seconds)')
            plt.title('Opportunities vs Execution Time')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No opportunity data available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"성능 분석 그래프가 {output_path}에 저장되었습니다")
    
    def generate_report(self, output_file: str = "reports/performance_analysis_report.txt"):
        """종합 성능 분석 보고서 생성"""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        # 데이터 로드
        df = self.load_performance_data()
        
        if df.empty:
            logger.error("분석할 데이터가 없습니다")
            return
        
        # 분석 수행
        execution_analysis = self.analyze_execution_times(df)
        component_analysis = self.analyze_components(df)
        bottlenecks = self.identify_bottlenecks(df)
        
        # 보고서 작성
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("DEFIPOSER-ARB 성능 분석 보고서\n")
            f.write("="*60 + "\n")
            f.write(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"분석 대상: {len(df)}개 블록 처리 기록\n")
            f.write(f"목표 시간: {self.target_time}초\n\n")
            
            # 실행 시간 분석
            f.write("1. 실행 시간 분석\n")
            f.write("-"*30 + "\n")
            stats = execution_analysis["basic_stats"]
            f.write(f"• 평균 실행 시간: {stats['mean']:.3f}초\n")
            f.write(f"• 중간값: {stats['median']:.3f}초\n")
            f.write(f"• 표준편차: {stats['std']:.3f}초\n")
            f.write(f"• 최소/최대: {stats['min']:.3f}초 / {stats['max']:.3f}초\n")
            f.write(f"• 목표 달성률: {stats['target_achievement_rate']:.1%}\n\n")
            
            # 백분위수 분석
            f.write("백분위수 분석:\n")
            percentiles = execution_analysis["percentiles"]
            for p, value in percentiles.items():
                f.write(f"• {p}: {value:.3f}초\n")
            f.write("\n")
            
            # 컴포넌트 분석
            if component_analysis and "error" not in component_analysis:
                f.write("2. 컴포넌트별 성능 분석\n")
                f.write("-"*30 + "\n")
                
                for comp, data in component_analysis.items():
                    comp_name = comp.replace('_time', '').replace('_', ' ').title()
                    f.write(f"• {comp_name}:\n")
                    f.write(f"  - 평균: {data['mean']:.3f}초 ({data['percentage_of_target']:.1f}% of target)\n")
                    f.write(f"  - 최대: {data['max']:.3f}초\n")
                    if data['bottleneck_frequency'] > 0:
                        f.write(f"  - 병목 빈도: {data['bottleneck_frequency']}회\n")
                    f.write("\n")
            
            # 병목점 분석
            if bottlenecks:
                f.write("3. 성능 병목점 및 권장사항\n")
                f.write("-"*30 + "\n")
                
                for i, bottleneck in enumerate(bottlenecks, 1):
                    f.write(f"{i}. {bottleneck['component']}\n")
                    f.write(f"   • 평균 시간: {bottleneck['average_time']:.3f}초\n")
                    f.write(f"   • 목표 대비: {bottleneck['percentage_of_target']:.1f}%\n")
                    f.write(f"   • 심각도: {bottleneck['severity']}\n")
                    if 'frequency' in bottleneck:
                        f.write(f"   • 발생 빈도: {bottleneck['frequency']}회\n")
                    f.write(f"   • 권장사항: {bottleneck['recommendation']}\n\n")
            
            # 종합 권장사항
            f.write("4. 종합 권장사항\n")
            f.write("-"*30 + "\n")
            
            if stats['target_achievement_rate'] >= 0.9:
                f.write("• 전체적으로 목표 성능을 만족하고 있습니다.\n")
                f.write("• 현재 최적화 수준을 유지하세요.\n")
            elif stats['target_achievement_rate'] >= 0.7:
                f.write("• 대부분의 경우 목표를 달성하고 있으나 개선 여지가 있습니다.\n")
                f.write("• 주요 병목점을 중심으로 최적화를 진행하세요.\n")
            else:
                f.write("• 목표 달성률이 낮습니다. 시급한 최적화가 필요합니다.\n")
                f.write("• 시스템 아키텍처 재검토를 고려하세요.\n")
            
            if stats['mean'] > self.target_time:
                f.write(f"• 평균 실행 시간({stats['mean']:.3f}초)이 목표를 초과합니다.\n")
                f.write("• 알고리즘 최적화 또는 하드웨어 업그레이드를 고려하세요.\n")
            
        logger.info(f"성능 분석 보고서가 {output_path}에 저장되었습니다")
        
        # 그래프도 생성
        self.generate_performance_plots(df)

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="DEFIPOSER-ARB 성능 분석 도구"
    )
    parser.add_argument(
        "--log-file", 
        default="logs/performance_benchmark.json",
        help="성능 로그 파일 경로"
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="결과 출력 디렉토리"
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="성능 그래프 생성"
    )
    
    args = parser.parse_args()
    
    # 성능 분석기 초기화
    analyzer = PerformanceAnalyzer(args.log_file)
    
    print("🔍 DEFIPOSER-ARB 성능 분석 시작")
    print(f"📊 목표: 평균 {analyzer.target_time}초 이하 실행")
    
    # 보고서 생성
    report_file = Path(args.output_dir) / "performance_analysis_report.txt"
    analyzer.generate_report(str(report_file))
    
    # 실시간 성능 보고서 출력
    try:
        current_report = get_performance_report(last_n_blocks=50)
        if "error" not in current_report:
            print(f"\n📈 최근 50블록 성능:")
            summary = current_report["summary"]
            print(f"   성공률: {summary['success_rate']:.1%}")
            print(f"   평균 시간: {summary['average_time']:.3f}초")
            print(f"   최고 기록: {summary['fastest_time']:.3f}초")
        else:
            print(f"⚠️ 실시간 데이터 없음: {current_report['error']}")
    except Exception as e:
        print(f"⚠️ 실시간 성능 조회 실패: {e}")
    
    print(f"\n✅ 분석 완료. 결과는 {args.output_dir}/ 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()