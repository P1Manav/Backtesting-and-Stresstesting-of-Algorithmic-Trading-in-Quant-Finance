"""Compute performance and risk metrics"""
import numpy as np
from typing import Dict, Optional

def total_return(portfolio_values: list, initial_capital: float) -> float:
    """Calculate total portfolio return percentage"""
    return (portfolio_values[-1] - initial_capital) / initial_capital * 100

def annualized_return(portfolio_values: list, initial_capital: float,
                      trading_days: int = 252) -> float:
    """Calculate annualized return percentage"""
    n_years = len(portfolio_values) / trading_days
    if n_years <= 0:
        return 0.0
    return ((portfolio_values[-1] / initial_capital) ** (1 / n_years) - 1) * 100

def sharpe_ratio(portfolio_values: list, trading_days: int = 252) -> float:
    """Calculate Sharpe ratio"""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    if np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(trading_days))

def sortino_ratio(portfolio_values: list, trading_days: int = 252) -> float:
    """Calculate Sortino ratio"""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    downside = returns[returns < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(downside) * np.sqrt(trading_days))

def calmar_ratio(portfolio_values: list, initial_capital: float,
                 trading_days: int = 252) -> float:
    """Calculate Calmar ratio"""
    ann = annualized_return(portfolio_values, initial_capital, trading_days)
    mdd = abs(max_drawdown(portfolio_values))
    return ann / mdd if mdd != 0 else 0.0

def max_drawdown(portfolio_values: list) -> float:
    """Calculate maximum drawdown percentage"""
    pv = np.array(portfolio_values)
    cummax = np.maximum.accumulate(pv)
    drawdown = (pv - cummax) / cummax
    return float(np.min(drawdown) * 100)

def annualized_volatility(portfolio_values: list,
                          trading_days: int = 252) -> float:
    """Calculate annualized volatility percentage"""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    return float(np.std(returns) * np.sqrt(trading_days) * 100)

def expected_shortfall(portfolio_values: list,
                       confidence_level: float = 0.95) -> float:
    """Calculate expected shortfall at confidence level"""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    if len(returns) == 0:
        return 0.0
    cutoff = np.percentile(returns, (1 - confidence_level) * 100)
    tail = returns[returns <= cutoff]
    return float(np.mean(tail) * 100) if len(tail) > 0 else 0.0

def marginal_expected_shortfall(portfolio_values: list,
                                asset_returns: Optional[np.ndarray] = None,
                                confidence_level: float = 0.95) -> float:
    """Calculate marginal expected shortfall"""
    port_ret = np.diff(portfolio_values) / portfolio_values[:-1]
    if asset_returns is None:
        asset_returns = port_ret.copy()
    if len(port_ret) == 0:
        return 0.0
    cutoff = np.percentile(port_ret, (1 - confidence_level) * 100)
    mask = port_ret <= cutoff
    if np.sum(mask) == 0:
        return 0.0
    ml = min(len(mask), len(asset_returns))
    return float(np.mean(asset_returns[:ml][mask[:ml]]) * 100)

def systemic_expected_shortfall(portfolio_values: list,
                                initial_capital: float,
                                capital_ratio: float = 0.08,
                                confidence_level: float = 0.95) -> float:
    """Calculate systemic expected shortfall"""
    port_ret = np.diff(portfolio_values) / portfolio_values[:-1]
    if len(port_ret) == 0:
        return 0.0
    cutoff = np.percentile(port_ret, (1 - confidence_level) * 100)
    tail = port_ret[port_ret <= cutoff]
    if len(tail) == 0:
        return 0.0
    mean_tail = float(np.mean(tail))
    expected_val = portfolio_values[-1] * (1 + mean_tail)
    req = initial_capital * capital_ratio
    return float(max(0.0, req - expected_val))

class PerformanceMetrics:
    """Calculate and store performance metrics"""

    def __init__(self, portfolio_values: list, initial_capital: float,
                 num_trades: int):
        """Initialize metrics calculator"""
        pv = list(portfolio_values)
        self.metrics: Dict[str, float] = {
            'Total Return (%)': total_return(pv, initial_capital),
            'Annualized Return (%)': annualized_return(pv, initial_capital),
            'Sharpe Ratio': sharpe_ratio(pv),
            'Sortino Ratio': sortino_ratio(pv),
            'Calmar Ratio': calmar_ratio(pv, initial_capital),
            'Max Drawdown (%)': max_drawdown(pv),
            'Annualized Volatility (%)': annualized_volatility(pv),
            'Expected Shortfall 95% (%)': expected_shortfall(pv, 0.95),
            'Expected Shortfall 99% (%)': expected_shortfall(pv, 0.99),
            'MES 95% (%)': marginal_expected_shortfall(pv, None, 0.95),
            'SES ($)': systemic_expected_shortfall(pv, initial_capital),
            'Total Trades': num_trades,
            'Final Portfolio ($)': pv[-1],
        }

    def display(self) -> None:
        """Print metrics to console"""
        print()
        for k, v in self.metrics.items():
            if isinstance(v, float):
                print(f"  {k:.<50} {v:>12.2f}")
            else:
                print(f"  {k:.<50} {v:>12}")

    def as_dict(self) -> Dict[str, float]:
        """Return metrics as dictionary"""
        return dict(self.metrics)

