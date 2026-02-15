"""
Backtesting Configuration Module
Configuration settings for the backtesting engine.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class BacktestingConfig:
    """
    Configuration for backtesting engine.
    
    Attributes:
        initial_capital: Starting capital amount
        commission_rate: Transaction commission as a percentage (e.g., 0.001 = 0.1%)
        slippage: Slippage percentage (e.g., 0.0005 = 0.05%)
        max_position_size: Maximum position size as fraction of portfolio (0-1)
        min_trade_size: Minimum trade size in units
        allow_short: Whether short selling is allowed
        margin_rate: Margin requirement for short positions
        risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        trading_days_per_year: Number of trading days per year
        window_size: Lookback window size for model predictions
        price_column: Column name to use for trade prices
        benchmark_column: Optional column for benchmark comparison
    """
    
    # Capital and position settings
    initial_capital: float = 100000.0
    max_position_size: float = 1.0
    min_trade_size: float = 0.0
    allow_short: bool = True
    
    # Transaction costs
    commission_rate: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    margin_rate: float = 0.5  # 50% margin for shorts
    
    # Calculation parameters
    risk_free_rate: float = 0.02  # 2% annual
    trading_days_per_year: int = 252
    
    # Model parameters
    window_size: int = 30
    
    # Data columns
    price_column: str = 'Close'
    benchmark_column: Optional[str] = None
    
    # Logging and output
    verbose: bool = True
    log_trades: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'initial_capital': self.initial_capital,
            'max_position_size': self.max_position_size,
            'min_trade_size': self.min_trade_size,
            'allow_short': self.allow_short,
            'commission_rate': self.commission_rate,
            'slippage': self.slippage,
            'margin_rate': self.margin_rate,
            'risk_free_rate': self.risk_free_rate,
            'trading_days_per_year': self.trading_days_per_year,
            'window_size': self.window_size,
            'price_column': self.price_column,
            'benchmark_column': self.benchmark_column,
            'verbose': self.verbose,
            'log_trades': self.log_trades
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BacktestingConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() 
                      if k in cls.__dataclass_fields__})
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        
        if not 0 <= self.max_position_size <= 1:
            raise ValueError("max_position_size must be between 0 and 1")
        
        if self.commission_rate < 0:
            raise ValueError("commission_rate cannot be negative")
        
        if self.slippage < 0:
            raise ValueError("slippage cannot be negative")
        
        if not 0 < self.margin_rate <= 1:
            raise ValueError("margin_rate must be between 0 and 1")
        
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1")
