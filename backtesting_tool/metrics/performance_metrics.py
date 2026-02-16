import numpy as np
from typing import Dict


def total_return(portfolio_values: list, initial_capital: float) -> float:
    """Total return as a percentage."""
    return (portfolio_values[-1] - initial_capital) / initial_capital * 100


def annualized_return(portfolio_values: list, initial_capital: float,
                      trading_days: int = 252) -> float:
    """Annualized return as a percentage."""
    n_years = len(portfolio_values) / trading_days
    if n_years <= 0:
        return 0.0
    return ((portfolio_values[-1] / initial_capital) ** (1 / n_years) - 1) * 100


def sharpe_ratio(portfolio_values: list, trading_days: int = 252) -> float:
    """Annualized Sharpe Ratio (assumes risk-free rate = 0)."""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    if np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(trading_days))


def max_drawdown(portfolio_values: list) -> float:
    """Maximum drawdown as a negative percentage."""
    pv = np.array(portfolio_values)
    cummax = np.maximum.accumulate(pv)
    drawdown = (pv - cummax) / cummax
    return float(np.min(drawdown) * 100)


class PerformanceMetrics:
    """Container for all computed metrics."""

    def __init__(self, portfolio_values: list, initial_capital: float,
                 num_trades: int):
        pv = list(portfolio_values)
        self.metrics: Dict[str, float] = {
            'Total Return (%)': total_return(pv, initial_capital),
            'Annualized Return (%)': annualized_return(pv, initial_capital),
            'Sharpe Ratio': sharpe_ratio(pv),
            'Max Drawdown (%)': max_drawdown(pv),
            'Total Trades': num_trades,
            'Final Portfolio ($)': pv[-1],
        }

    def display(self) -> None:
        """Print metrics to console."""
        print()
        for k, v in self.metrics.items():
            if isinstance(v, float):
                print(f"  {k:.<50} {v:>12.2f}")
            else:
                print(f"  {k:.<50} {v:>12}")

    def as_dict(self) -> Dict[str, float]:
        return dict(self.metrics)
