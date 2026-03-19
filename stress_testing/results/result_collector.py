"""Result collection and aggregation"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
from stress_testing.engine.stress_engine import StressScenarioResult
from stress_testing.evaluation import (
    PerformanceDegradationReport,
    RobustnessMetricsReport
)
class ResultCollector:
    """Collect and manage stress testing results."""
    def __init__(self, output_dir: str = "./stress_test_results"):
"""Initialize result collector. Args: output_dir: Directory to save results"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_metrics: Dict[str, float] = {}
        self.scenario_results: Dict[str, StressScenarioResult] = {}
        self.degradation_reports: List[PerformanceDegradationReport] = []
        self.robustness_report: RobustnessMetricsReport = None
        self.timestamp = datetime.now()
    def add_baseline_metrics(self, metrics: Dict[str, float]):
        """Store baseline metrics."""
        self.baseline_metrics = metrics
    def add_scenario_result(self, scenario_name: str,
    """add_scenario_result implementation"""
                           result: StressScenarioResult):
        """Add scenario result."""
        self.scenario_results[scenario_name] = result
    def add_degradation_report(self,
                              report: PerformanceDegradationReport):
        """Add degradation report."""
        self.degradation_reports.append(report)
    def add_robustness_report(self,
                             report: RobustnessMetricsReport):
        """Add robustness report."""
        self.robustness_report = report
    def save_results_json(self, filename: str = "stress_test_results.json") -> Path:
"""Save results to JSON file. Args: filename: Output filename Returns: Path to saved file"""
        output_path = self.output_dir / filename
        results_dict = {
            'timestamp': self.timestamp.isoformat(),
            'baseline_metrics': self.baseline_metrics,
            'scenario_results': {
                name: {
                    'scenario_name': res.scenario_name,
                    'scenario_type': res.scenario_type,
                    'scenario_params': res.scenario_params,
                    'stressed_metrics': res.stressed_metrics,
                    'performance_degradation': res.performance_degradation,
                }
                for name, res in self.scenario_results.items()
            },
            'robustness_report': {
                'num_scenarios': self.robustness_report.num_scenarios,
                'profitable_scenarios': self.robustness_report.profitable_scenarios,
                'profitable_percentage': self.robustness_report.profitable_percentage,
                'avg_return_drop': float(self.robustness_report.avg_return_drop),
                'worst_return_drop': float(self.robustness_report.worst_return_drop),
                'avg_sharpe_drop': float(self.robustness_report.avg_sharpe_drop),
                'worst_sharpe_drop': float(self.robustness_report.worst_sharpe_drop),
                'avg_max_dd_increase': float(self.robustness_report.avg_max_dd_increase),
                'worst_max_dd_increase': float(self.robustness_report.worst_max_dd_increase),
                'resilience_score': float(self.robustness_report.resilience_score),
                'stability_score': float(self.robustness_report.stability_score),
                'drawdown_resilience_score': float(self.robustness_report.drawdown_resilience_score),
                'overall_robustness_score': float(self.robustness_report.overall_robustness_score),
            }
        }
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        return output_path
    def save_comparison_csv(self, comparison_df: pd.DataFrame,
    """Save data or model"""
                           filename: str = "scenario_comparison.csv") -> Path:
"""Save scenario comparison to CSV. Args: comparison_df: Comparison DataFrame filename: Output filename Returns: Path to saved file"""
        output_path = self.output_dir / filename
        comparison_df.to_csv(output_path)
        return output_path
    def save_degradation_report_csv(self,
                                    filename: str = "degradation_report.csv") -> Path:
"""Save degradation reports to CSV. Args: filename: Output filename Returns: Path to saved file"""
        output_path = self.output_dir / filename
        data = {
            'Scenario': [r.scenario_name for r in self.degradation_reports],
            'Baseline Return (%)': [r.baseline_return for r in self.degradation_reports],
            'Stressed Return (%)': [r.stressed_return for r in self.degradation_reports],
            'Return Drop (%)': [r.return_drop for r in self.degradation_reports],
            'Return Drop %': [r.return_drop_pct * 100 for r in self.degradation_reports],
            'Baseline Sharpe': [r.baseline_sharpe for r in self.degradation_reports],
            'Stressed Sharpe': [r.stressed_sharpe for r in self.degradation_reports],
            'Sharpe Drop': [r.sharpe_drop for r in self.degradation_reports],
            'Max DD Increase (%)': [r.max_dd_increase for r in self.degradation_reports],
            'Is Profitable': [r.is_profitable for r in self.degradation_reports],
            'Is Acceptable': [r.is_acceptable for r in self.degradation_reports],
        }
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        return output_path
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of all results."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'num_scenarios': len(self.scenario_results),
            'baseline_metrics': self.baseline_metrics,
            'robustness_report': self.robustness_report,
        }

