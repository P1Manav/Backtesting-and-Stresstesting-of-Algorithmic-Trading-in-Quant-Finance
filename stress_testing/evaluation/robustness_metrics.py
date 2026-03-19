"""Robustness analysis metrics"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
@dataclass
class RobustnessMetricsReport:
    """Container for robustness metrics."""
    num_scenarios: int
    profitable_scenarios: int
    unprofitable_scenarios: int
    profitable_percentage: float
    avg_return_drop: float
    worst_return_drop: float
    best_return_drop: float
    std_return_drop: float
    avg_sharpe_drop: float
    worst_sharpe_drop: float
    best_sharpe_drop: float
    std_sharpe_drop: float
    avg_max_dd_increase: float
    worst_max_dd_increase: float
    best_max_dd_increase: float
    std_max_dd_increase: float
    resilience_score: float
    stability_score: float
    drawdown_resilience_score: float
    overall_robustness_score: float
class RobustnessMetricsCalculator:
    """Calculate robustness indicators from stress test results."""
    def __init__(self, resilience_threshold: float = 0.70):
"""Initialize robustness calculator. Args: resilience_threshold: Threshold for "resilient" (e.g., 0.70 = 70% profitable)"""
        self.resilience_threshold = resilience_threshold
    def calculate(self, scenario_results: List['PerformanceDegradationReport']
    """Perform calculations"""
                 ) -> RobustnessMetricsReport:
"""Calculate robustness metrics from scenario results. Args: scenario_results: List of PerformanceDegradationReport objects Returns: RobustnessMetricsReport with comprehensive metrics"""
        num_scenarios = len(scenario_results)
        return_drops = [r.return_drop for r in scenario_results]
        sharpe_drops = [r.sharpe_drop for r in scenario_results]
        dd_increases = [r.max_dd_increase for r in scenario_results]
        profitable_count = sum(1 for r in scenario_results if r.is_profitable)
        profitable_pct = profitable_count / num_scenarios * 100
        avg_return_drop = np.mean(return_drops)
        worst_return_drop = np.max(return_drops)
        best_return_drop = np.min(return_drops)
        std_return_drop = np.std(return_drops)
        avg_sharpe_drop = np.mean(sharpe_drops)
        worst_sharpe_drop = np.max(sharpe_drops)
        best_sharpe_drop = np.min(sharpe_drops)
        std_sharpe_drop = np.std(sharpe_drops)
        avg_max_dd_increase = np.mean(dd_increases)
        worst_max_dd_increase = np.max(dd_increases)
        best_max_dd_increase = np.min(dd_increases)
        std_max_dd_increase = np.std(dd_increases)
        resilience_score = min(100, (profitable_pct / 100) * 100) if self.resilience_threshold > 0 else 0
        stability_score = max(0, 100 - (std_return_drop * 10))
        drawdown_resilience = max(0, 100 - (worst_max_dd_increase * 100))
        drawdown_resilience_score = np.clip(drawdown_resilience, 0, 100)
        overall_robustness = (
            resilience_score * 0.4 +
            stability_score * 0.3 +
            drawdown_resilience_score * 0.3
        )
        overall_robustness = np.clip(overall_robustness, 0, 100)
        return RobustnessMetricsReport(
            num_scenarios=num_scenarios,
            profitable_scenarios=profitable_count,
            unprofitable_scenarios=num_scenarios - profitable_count,
            profitable_percentage=profitable_pct,
            avg_return_drop=avg_return_drop,
            worst_return_drop=worst_return_drop,
            best_return_drop=best_return_drop,
            std_return_drop=std_return_drop,
            avg_sharpe_drop=avg_sharpe_drop,
            worst_sharpe_drop=worst_sharpe_drop,
            best_sharpe_drop=best_sharpe_drop,
            std_sharpe_drop=std_sharpe_drop,
            avg_max_dd_increase=avg_max_dd_increase,
            worst_max_dd_increase=worst_max_dd_increase,
            best_max_dd_increase=best_max_dd_increase,
            std_max_dd_increase=std_max_dd_increase,
            resilience_score=resilience_score,
            stability_score=stability_score,
            drawdown_resilience_score=drawdown_resilience_score,
            overall_robustness_score=overall_robustness
        )

