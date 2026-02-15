"""
Performance Metrics Module
Data class for storing and accessing performance metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np
import json


@dataclass
class PerformanceMetrics:
    """
    Container for backtesting performance metrics.
    
    Stores all computed metrics from a backtest run.
    """
    
    # Return metrics
    total_return: float = 0.0
    cumulative_return: float = 0.0
    annualized_return: float = 0.0
    
    # Risk metrics
    annualized_volatility: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Trading metrics
    win_rate: float = 0.0
    turnover: float = 0.0
    num_trades: int = 0
    
    # Detailed statistics
    mean_return: float = 0.0
    std_return: float = 0.0
    
    # Equity curve (optional storage)
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    
    @property
    def total_return_pct(self) -> float:
        """Total return as percentage."""
        return self.total_return * 100
    
    @property
    def max_drawdown_pct(self) -> float:
        """Maximum drawdown as percentage."""
        return self.max_drawdown * 100
    
    @property
    def win_rate_pct(self) -> float:
        """Win rate as percentage."""
        return self.win_rate * 100
    
    @property
    def annualized_return_pct(self) -> float:
        """Annualized return as percentage."""
        return self.annualized_return * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_return': self.total_return,
            'total_return_pct': self.total_return_pct,
            'cumulative_return': self.cumulative_return,
            'annualized_return': self.annualized_return,
            'annualized_return_pct': self.annualized_return_pct,
            'annualized_volatility': self.annualized_volatility,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'win_rate_pct': self.win_rate_pct,
            'turnover': self.turnover,
            'num_trades': self.num_trades,
            'mean_return': self.mean_return,
            'std_return': self.std_return
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create metrics from dictionary."""
        return cls(
            total_return=data.get('total_return', 0.0),
            cumulative_return=data.get('cumulative_return', 0.0),
            annualized_return=data.get('annualized_return', 0.0),
            annualized_volatility=data.get('annualized_volatility', 0.0),
            max_drawdown=data.get('max_drawdown', 0.0),
            sharpe_ratio=data.get('sharpe_ratio', 0.0),
            win_rate=data.get('win_rate', 0.0),
            turnover=data.get('turnover', 0.0),
            num_trades=data.get('num_trades', 0),
            mean_return=data.get('mean_return', 0.0),
            std_return=data.get('std_return', 0.0),
            equity_curve=data.get('equity_curve', []),
            daily_returns=data.get('daily_returns', [])
        )
    
    def to_json(self) -> str:
        """Convert metrics to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PerformanceMetrics':
        """Create metrics from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def summary_table(self) -> str:
        """Generate a formatted summary table."""
        lines = [
            "=" * 50,
            "PERFORMANCE METRICS SUMMARY",
            "=" * 50,
            "",
            "Return Metrics:",
            f"  Total Return:          {self.total_return_pct:>12.2f}%",
            f"  Annualized Return:     {self.annualized_return_pct:>12.2f}%",
            f"  Cumulative Return:     {self.cumulative_return*100:>12.2f}%",
            "",
            "Risk Metrics:",
            f"  Annualized Volatility: {self.annualized_volatility*100:>12.2f}%",
            f"  Maximum Drawdown:      {self.max_drawdown_pct:>12.2f}%",
            f"  Sharpe Ratio:          {self.sharpe_ratio:>12.2f}",
            "",
            "Trading Metrics:",
            f"  Win Rate:              {self.win_rate_pct:>12.2f}%",
            f"  Turnover:              {self.turnover:>12.4f}",
            f"  Number of Trades:      {self.num_trades:>12}",
            "",
            "=" * 50
        ]
        return "\n".join(lines)
    
    def print_summary(self) -> None:
        """Print the summary table."""
        print(self.summary_table())
    
    def validate(self) -> bool:
        """Validate that metrics are within reasonable ranges."""
        # Check for NaN values
        numeric_fields = [
            self.total_return, self.cumulative_return, self.annualized_return,
            self.annualized_volatility, self.max_drawdown, self.sharpe_ratio,
            self.win_rate, self.turnover, self.mean_return, self.std_return
        ]
        
        for value in numeric_fields:
            if np.isnan(value) or np.isinf(value):
                return False
        
        # Check reasonable ranges
        if not -1 <= self.max_drawdown <= 0:
            return False
        
        if not 0 <= self.win_rate <= 1:
            return False
        
        return True
