"""Performance degradation analysis"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
@dataclass
class PerformanceDegradationReport:
    """Container for performance degradation analysis."""
    scenario_name: str
    baseline_return: float
    stressed_return: float
    return_drop: float
    return_drop_pct: float
    baseline_sharpe: float
    stressed_sharpe: float
    sharpe_drop: float
    baseline_max_dd: float
    stressed_max_dd: float
    max_dd_increase: float
    baseline_volatility: float
    stressed_volatility: float
    volatility_increase: float
    is_profitable: bool
    is_acceptable: bool
class PerformanceDegradationAnalyzer:
    """Analyze how strategy performance degrades under stress."""
    def __init__(self, acceptable_return_drop: float = 0.15,
    """Initialize instance"""
                 acceptable_sharpe_drop: float = 0.5,
                 acceptable_drawdown_increase: float = 0.10):
"""Initialize degradation analyzer. Args: acceptable_return_drop: Max tolerable return drop as fraction (0.15 = 15%) acceptable_sharpe_drop: Max tolerable Sharpe drop in ratio points acceptable_drawdown_increase: Max tolerable increase in max drawdown"""
        self.acceptable_return_drop = acceptable_return_drop
        self.acceptable_sharpe_drop = acceptable_sharpe_drop
        self.acceptable_drawdown_increase = acceptable_drawdown_increase
    def analyze_scenario(self, scenario_name: str,
    """analyze_scenario implementation"""
                        baseline_metrics: Dict[str, float],
                        stressed_metrics: Dict[str, float],
                        degradation: Dict[str, float]) -> PerformanceDegradationReport:
"""Analyze degradation for a single scenario. Args: scenario_name: Name of stress scenario baseline_metrics: Baseline performance metrics stressed_metrics: Stressed scenario metrics degradation: Pre-calculated degradation metrics Returns: PerformanceDegradationReport with analysis"""
        return_drop = degradation['total_return_drop']
        return_drop_pct = return_drop / abs(baseline_metrics.get('Total Return (%)', 0)) if baseline_metrics.get('Total Return (%)', 0) != 0 else 0
        sharpe_drop = degradation['sharpe_drop']
        max_dd_increase = degradation['max_drawdown_increase']
        is_profitable = stressed_metrics.get('Total Return (%)', 0) > 0
        is_acceptable = (
            return_drop_pct <= self.acceptable_return_drop and
            sharpe_drop <= self.acceptable_sharpe_drop and
            max_dd_increase <= self.acceptable_drawdown_increase
        )
        return PerformanceDegradationReport(
            scenario_name=scenario_name,
            baseline_return=baseline_metrics.get('Total Return (%)', 0),
            stressed_return=stressed_metrics.get('Total Return (%)', 0),
            return_drop=return_drop,
            return_drop_pct=return_drop_pct,
            baseline_sharpe=baseline_metrics.get('Sharpe Ratio', 0),
            stressed_sharpe=stressed_metrics.get('Sharpe Ratio', 0),
            sharpe_drop=sharpe_drop,
            baseline_max_dd=baseline_metrics.get('Max Drawdown (%)', 0),
            stressed_max_dd=stressed_metrics.get('Max Drawdown (%)', 0),
            max_dd_increase=max_dd_increase,
            baseline_volatility=baseline_metrics.get('Annualized Volatility (%)', 0),
            stressed_volatility=stressed_metrics.get('Annualized Volatility (%)', 0),
            volatility_increase=degradation['volatility_increase'],
            is_profitable=is_profitable,
            is_acceptable=is_acceptable
        )
    def analyze_multiple_scenarios(self,
                                   scenario_results: Dict[str, 'StressScenarioResult']
                                   ) -> List[PerformanceDegradationReport]:
"""Analyze degradation across multiple scenarios. Args: scenario_results: Dictionary of StressScenarioResult objects Returns: List of PerformanceDegradationReport objects"""
        reports = []
        for scenario_name, result in scenario_results.items():
            report = self.analyze_scenario(
                scenario_name,
                result.baseline_metrics,
                result.stressed_metrics,
                result.performance_degradation
            )
            reports.append(report)
        return reports

