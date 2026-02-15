"""
Metrics Calculator Module
Calculates all performance metrics from backtest results.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union

from .performance_metrics import PerformanceMetrics


class MetricsCalculator:
    """
    Calculator for backtesting performance metrics.
    
    Computes:
    - Equity curve
    - Cumulative return
    - Daily returns
    - Sharpe ratio
    - Maximum drawdown
    - Win rate
    - Turnover
    """
    
    def __init__(self,
                 risk_free_rate: float = 0.02,
                 trading_days_per_year: int = 252):
        """
        Initialize the Metrics Calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate
            trading_days_per_year: Number of trading days per year
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
    
    def calculate_all(self,
                      equity_curve: Union[List[float], np.ndarray],
                      trades: Optional[List[Dict]] = None) -> PerformanceMetrics:
        """
        Calculate all performance metrics.
        
        Args:
            equity_curve: Array of portfolio values over time
            trades: Optional list of trade records
            
        Returns:
            PerformanceMetrics object with all computed metrics
        """
        equity = np.array(equity_curve)
        
        if len(equity) < 2:
            return PerformanceMetrics()
        
        # Calculate returns
        returns = self.calculate_returns(equity)
        
        # Calculate all metrics
        metrics = PerformanceMetrics(
            total_return=self.calculate_total_return(equity),
            cumulative_return=self.calculate_cumulative_return(equity),
            annualized_return=self.calculate_annualized_return(equity),
            annualized_volatility=self.calculate_annualized_volatility(returns),
            max_drawdown=self.calculate_max_drawdown(equity),
            sharpe_ratio=self.calculate_sharpe_ratio(returns),
            win_rate=self.calculate_win_rate(returns),
            turnover=self.calculate_turnover(trades, equity) if trades else 0.0,
            num_trades=len(trades) if trades else 0,
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            equity_curve=equity.tolist(),
            daily_returns=returns.tolist()
        )
        
        return metrics
    
    def calculate_returns(self, equity: np.ndarray) -> np.ndarray:
        """
        Calculate period returns from equity curve.
        
        Args:
            equity: Array of portfolio values
            
        Returns:
            Array of period returns
        """
        returns = np.diff(equity) / equity[:-1]
        return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    def calculate_total_return(self, equity: np.ndarray) -> float:
        """
        Calculate total return.
        
        Args:
            equity: Array of portfolio values
            
        Returns:
            Total return as decimal (e.g., 0.15 for 15%)
        """
        if len(equity) < 2 or equity[0] == 0:
            return 0.0
        return (equity[-1] / equity[0]) - 1
    
    def calculate_cumulative_return(self, equity: np.ndarray) -> float:
        """
        Calculate cumulative return.
        
        Args:
            equity: Array of portfolio values
            
        Returns:
            Cumulative return as decimal
        """
        if len(equity) < 2 or equity[0] == 0:
            return 0.0
        return (equity[-1] - equity[0]) / equity[0]
    
    def calculate_annualized_return(self, equity: np.ndarray) -> float:
        """
        Calculate annualized return.
        
        Args:
            equity: Array of portfolio values
            
        Returns:
            Annualized return as decimal
        """
        if len(equity) < 2:
            return 0.0
        
        total_return = self.calculate_total_return(equity)
        num_periods = len(equity) - 1
        
        if num_periods == 0:
            return 0.0
        
        # Annualize based on trading days
        years = num_periods / self.trading_days_per_year
        if years == 0:
            return total_return
        
        annualized = (1 + total_return) ** (1 / years) - 1
        return float(annualized)
    
    def calculate_annualized_volatility(self, returns: np.ndarray) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            returns: Array of period returns
            
        Returns:
            Annualized volatility as decimal
        """
        if len(returns) < 2:
            return 0.0
        
        daily_vol = np.std(returns)
        annualized_vol = daily_vol * np.sqrt(self.trading_days_per_year)
        return float(annualized_vol)
    
    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Array of period returns
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        # Annualized metrics
        mean_return = np.mean(returns) * self.trading_days_per_year
        volatility = np.std(returns) * np.sqrt(self.trading_days_per_year)
        
        if volatility == 0:
            return 0.0
        
        excess_return = mean_return - self.risk_free_rate
        sharpe = excess_return / volatility
        return float(sharpe)
    
    def calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity: Array of portfolio values
            
        Returns:
            Maximum drawdown as negative decimal (e.g., -0.20 for 20% drawdown)
        """
        if len(equity) < 2:
            return 0.0
        
        # Calculate running maximum
        peak = np.maximum.accumulate(equity)
        
        # Calculate drawdown from peak
        drawdown = (equity - peak) / peak
        
        # Return maximum drawdown (most negative value)
        max_dd = np.min(drawdown)
        return float(max_dd)
    
    def calculate_drawdown_series(self, equity: np.ndarray) -> np.ndarray:
        """
        Calculate drawdown series.
        
        Args:
            equity: Array of portfolio values
            
        Returns:
            Array of drawdown values
        """
        if len(equity) < 2:
            return np.array([0.0])
        
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return drawdown
    
    def calculate_win_rate(self, returns: np.ndarray) -> float:
        """
        Calculate win rate (fraction of positive returns).
        
        Args:
            returns: Array of period returns
            
        Returns:
            Win rate as decimal (0-1)
        """
        if len(returns) == 0:
            return 0.0
        
        positive_returns = np.sum(returns > 0)
        total_periods = len(returns)
        
        return float(positive_returns / total_periods)
    
    def calculate_turnover(self,
                           trades: List[Dict],
                           equity: np.ndarray) -> float:
        """
        Calculate portfolio turnover.
        
        Args:
            trades: List of trade records
            equity: Array of portfolio values
            
        Returns:
            Turnover ratio
        """
        if not trades or len(equity) < 2:
            return 0.0
        
        # Calculate total traded volume
        total_traded = sum(
            abs(trade.get('quantity', 0) * trade.get('price', 0))
            for trade in trades
        )
        
        # Average portfolio value
        avg_value = np.mean(equity)
        
        if avg_value == 0:
            return 0.0
        
        # Turnover = total traded / (avg value * periods)
        periods = len(equity) - 1
        turnover = total_traded / (avg_value * periods)
        
        return float(turnover)
    
    def calculate_sortino_ratio(self,
                                 returns: np.ndarray,
                                 target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio (using downside deviation).
        
        Args:
            returns: Array of period returns
            target_return: Minimum acceptable return
            
        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        # Calculate downside returns
        downside_returns = np.minimum(returns - target_return, 0)
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_deviation == 0:
            return 0.0
        
        # Annualize
        mean_return = np.mean(returns) * self.trading_days_per_year
        downside_vol = downside_deviation * np.sqrt(self.trading_days_per_year)
        
        excess_return = mean_return - self.risk_free_rate
        sortino = excess_return / downside_vol
        
        return float(sortino)
    
    def calculate_calmar_ratio(self, equity: np.ndarray) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).
        
        Args:
            equity: Array of portfolio values
            
        Returns:
            Calmar ratio
        """
        annualized_return = self.calculate_annualized_return(equity)
        max_drawdown = abs(self.calculate_max_drawdown(equity))
        
        if max_drawdown == 0:
            return 0.0
        
        return float(annualized_return / max_drawdown)
    
    def calculate_profit_factor(self, returns: np.ndarray) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        
        Args:
            returns: Array of period returns
            
        Returns:
            Profit factor
        """
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return float('inf') if profits > 0 else 0.0
        
        return float(profits / losses)
    
    def generate_summary_table(self, metrics: PerformanceMetrics) -> pd.DataFrame:
        """
        Generate a summary table of metrics.
        
        Args:
            metrics: PerformanceMetrics object
            
        Returns:
            DataFrame with metrics summary
        """
        data = {
            'Metric': [
                'Total Return',
                'Annualized Return',
                'Annualized Volatility',
                'Sharpe Ratio',
                'Maximum Drawdown',
                'Win Rate',
                'Number of Trades',
                'Turnover'
            ],
            'Value': [
                f"{metrics.total_return_pct:.2f}%",
                f"{metrics.annualized_return_pct:.2f}%",
                f"{metrics.annualized_volatility*100:.2f}%",
                f"{metrics.sharpe_ratio:.2f}",
                f"{metrics.max_drawdown_pct:.2f}%",
                f"{metrics.win_rate_pct:.2f}%",
                f"{metrics.num_trades}",
                f"{metrics.turnover:.4f}"
            ]
        }
        
        return pd.DataFrame(data)
