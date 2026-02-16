from typing import Dict, Any
from .performance_metrics import PerformanceMetrics


class MetricsCalculator:

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital

    def calculate(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate all performance metrics from backtest results."""
        pm = PerformanceMetrics(
            portfolio_values=results['portfolio_values'],
            initial_capital=self.initial_capital,
            num_trades=len(results['trades']),
        )
        pm.display()
        return pm.as_dict()
