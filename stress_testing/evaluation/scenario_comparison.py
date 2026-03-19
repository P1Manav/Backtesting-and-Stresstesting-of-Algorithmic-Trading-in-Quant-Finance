"""Scenario comparison utilities"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
@dataclass
class ScenarioComparison:
    """Container for scenario comparison."""
    scenario_name: str
    scenario_type: str
    baseline_return: float
    stressed_return: float
    baseline_sharpe: float
    stressed_sharpe: float
    return_retention_pct: float
    sharpe_retention_pct: float
    baseline_max_dd: float
    stressed_max_dd: float
    max_dd_ratio: float
class ScenarioComparator:
    """Compare performance across different stress scenarios."""
    def create_comparison_dataframe(self, scenario_results: Dict[str, 'StressScenarioResult']
    """create_comparison_dataframe implementation"""
                                   ) -> pd.DataFrame:
"""Create DataFrame comparing all scenarios. Args: scenario_results: Dictionary of StressScenarioResult objects Returns: DataFrame with scenario comparisons"""
        comparisons = []
        for scenario_name, result in scenario_results.items():
            baseline = result.baseline_metrics
            stressed = result.stressed_metrics
            return_retention = (stressed.get('Total Return (%)', 0) / baseline.get('Total Return (%)', 1) * 100) \
                if baseline.get('Total Return (%)', 0) != 0 else 0
            sharpe_retention = (stressed.get('Sharpe Ratio', 0) / baseline.get('Sharpe Ratio', 1) * 100) \
                if baseline.get('Sharpe Ratio', 0) != 0 else 0
            max_dd_ratio = stressed.get('Max Drawdown (%)', 0) / baseline.get('Max Drawdown (%)', 1) \
                if baseline.get('Max Drawdown (%)', 0) != 0 else 1.0
            comp = ScenarioComparison(
                scenario_name=scenario_name,
                scenario_type=result.scenario_type,
                baseline_return=baseline.get('Total Return (%)', 0),
                stressed_return=stressed.get('Total Return (%)', 0),
                baseline_sharpe=baseline.get('Sharpe Ratio', 0),
                stressed_sharpe=stressed.get('Sharpe Ratio', 0),
                return_retention_pct=return_retention,
                sharpe_retention_pct=sharpe_retention,
                baseline_max_dd=baseline.get('Max Drawdown (%)', 0),
                stressed_max_dd=stressed.get('Max Drawdown (%)', 0),
                max_dd_ratio=max_dd_ratio
            )
            comparisons.append(comp)
        data = {
            'Scenario': [c.scenario_name for c in comparisons],
            'Type': [c.scenario_type for c in comparisons],
            'Baseline Return (%)': [c.baseline_return for c in comparisons],
            'Stressed Return (%)': [c.stressed_return for c in comparisons],
            'Return Retention (%)': [c.return_retention_pct for c in comparisons],
            'Baseline Sharpe': [c.baseline_sharpe for c in comparisons],
            'Stressed Sharpe': [c.stressed_sharpe for c in comparisons],
            'Sharpe Retention (%)': [c.sharpe_retention_pct for c in comparisons],
            'Baseline Max DD (%)': [c.baseline_max_dd for c in comparisons],
            'Stressed Max DD (%)': [c.stressed_max_dd for c in comparisons],
            'Max DD Ratio': [c.max_dd_ratio for c in comparisons],
        }
        df = pd.DataFrame(data)
        return df.set_index('Scenario')
    def get_worst_scenarios(self, comparison_df: pd.DataFrame,
    """get_worst_scenarios implementation"""
                           metric: str = 'Return Retention (%)',
                           n: int = 5) -> pd.DataFrame:
"""Get the worst performing scenarios for a given metric. Args: comparison_df: Comparison DataFrame metric: Metric column name to sort by n: Number of worst scenarios to return Returns: DataFrame with worst scenarios"""
        if metric not in comparison_df.columns:
            raise ValueError(f"Metric '{metric}' not found in comparison data")
        return comparison_df.nsmallest(n, metric)
    def get_best_scenarios(self, comparison_df: pd.DataFrame,
    """get_best_scenarios implementation"""
                          metric: str = 'Return Retention (%)',
                          n: int = 5) -> pd.DataFrame:
"""Get the best performing scenarios for a given metric. Args: comparison_df: Comparison DataFrame metric: Metric column name to sort by n: Number of best scenarios to return Returns: DataFrame with best scenarios"""
        if metric not in comparison_df.columns:
            raise ValueError(f"Metric '{metric}' not found in comparison data")
        return comparison_df.nlargest(n, metric)
    def get_scenario_summary_by_type(self, comparison_df: pd.DataFrame) -> pd.DataFrame:
"""Summarize performance by scenario type. Args: comparison_df: Comparison DataFrame Returns: Summary DataFrame grouped by scenario type"""
        summary = comparison_df.groupby('Type').agg({
            'Return Retention (%)': ['min', 'mean', 'max', 'std'],
            'Sharpe Retention (%)': ['min', 'mean', 'max', 'std'],
            'Max DD Ratio': ['min', 'mean', 'max', 'std'],
        })
        return summary

