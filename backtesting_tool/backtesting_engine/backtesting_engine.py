"""
Backtesting Engine Module
Core engine for historical simulation and backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import torch.nn as nn

from .portfolio import Portfolio
from .trade_executor import TradeExecutor
from .backtesting_config import BacktestingConfig

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from model_interface.prediction_controller import ModelInterface


class BacktestingEngine:
    """
    Main backtesting engine for historical simulation.
    
    Implements:
    - Rolling window time-step simulation
    - Trade execution logic
    - Transaction cost & commission handling
    - Trading constraints enforcement
    - Portfolio state tracking
    """
    
    def __init__(self, config: Optional[BacktestingConfig] = None):
        """
        Initialize the Backtesting Engine.
        
        Args:
            config: Backtesting configuration (uses defaults if None)
        """
        self.config = config or BacktestingConfig()
        self.config.validate()
        
        # Core components
        self.portfolio: Optional[Portfolio] = None
        self.executor: Optional[TradeExecutor] = None
        self.model_interface: Optional[ModelInterface] = None
        
        # State tracking
        self._is_initialized = False
        self._backtest_complete = False
        
        # Results
        self.results: Dict[str, Any] = {}
        self.timestep_log: List[Dict[str, Any]] = []
    
    def initialize(self,
                   model: nn.Module,
                   data: pd.DataFrame,
                   model_config: Optional[Dict[str, Any]] = None) -> 'BacktestingEngine':
        """
        Initialize the engine with model and data.
        
        Args:
            model: PyTorch model for predictions
            data: Market data DataFrame
            model_config: Optional configuration for model interface
            
        Returns:
            Self for chaining
        """
        # Create model interface
        model_config = model_config or {}
        model_config['window_size'] = self.config.window_size
        
        self.model_interface = ModelInterface(model, model_config)
        
        # Fit normalizer on data
        self.model_interface.fit(data)
        
        # Create portfolio and executor
        self.portfolio = Portfolio(
            initial_capital=self.config.initial_capital,
            allow_short=self.config.allow_short
        )
        
        self.executor = TradeExecutor(self.config)
        
        self._is_initialized = True
        self._backtest_complete = False
        
        if self.config.verbose:
            print(f"Backtesting engine initialized")
            print(f"  Initial capital: ${self.config.initial_capital:,.2f}")
            print(f"  Window size: {self.config.window_size}")
            print(f"  Commission rate: {self.config.commission_rate*100:.2f}%")
        
        return self
    
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the backtest simulation.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary with backtesting results
        """
        if not self._is_initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        if self.config.verbose:
            print(f"\nStarting backtest on {len(data)} time periods...")
        
        # Reset state
        self.portfolio.reset()
        self.executor.reset_statistics()
        self.timestep_log = []
        
        # Get price column
        price_col = self.config.price_column
        if price_col not in data.columns:
            raise ValueError(f"Price column '{price_col}' not found in data")
        
        # Main simulation loop
        start_idx = self.config.window_size - 1  # Start after we have enough history
        
        for t in range(start_idx, len(data)):
            self._simulate_timestep(data, t, price_col)
        
        # Generate results
        self.results = self._generate_results(data)
        self._backtest_complete = True
        
        if self.config.verbose:
            self._print_summary()
        
        return self.results
    
    def _simulate_timestep(self, 
                           data: pd.DataFrame, 
                           timestep: int,
                           price_col: str) -> None:
        """
        Simulate a single timestep.
        
        Args:
            data: Full market data
            timestep: Current timestep index
            price_col: Column name for prices
        """
        # Get current price and timestamp
        current_price = data[price_col].iloc[timestep]
        timestamp = data['Date'].iloc[timestep] if 'Date' in data.columns else timestep
        
        # Get model prediction
        action, position_size = self.model_interface.get_action(data, timestep)
        
        # Execute trade
        execution_result = self.executor.execute_signal(
            portfolio=self.portfolio,
            action=action,
            position_size=position_size,
            current_price=current_price,
            timestamp=timestamp
        )
        
        # Update portfolio value
        portfolio_value = self.portfolio.update(current_price, timestamp)
        
        # Log timestep
        if self.config.log_trades:
            self.timestep_log.append({
                'timestep': timestep,
                'timestamp': timestamp,
                'price': current_price,
                'action': action,
                'position_size': position_size,
                'executed': execution_result.executed,
                'portfolio_value': portfolio_value,
                'cash': self.portfolio.cash,
                'position': self.portfolio.position.quantity,
                'commission': execution_result.commission,
                'slippage': execution_result.slippage_cost
            })
    
    def _generate_results(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive backtest results.
        
        Args:
            data: Market data used for backtest
            
        Returns:
            Dictionary with all results
        """
        # Calculate returns
        returns = self.portfolio.get_returns()
        
        # Get portfolio summary
        portfolio_summary = self.portfolio.get_summary()
        
        # Get execution statistics
        execution_stats = self.executor.get_statistics()
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(returns)
        
        # Compile results
        results = {
            'config': self.config.to_dict(),
            'portfolio': portfolio_summary,
            'execution': execution_stats,
            'metrics': metrics,
            'equity_curve': self.portfolio.equity_curve,
            'returns': returns.tolist(),
            'trade_log': self.portfolio.get_trade_history().to_dict('records'),
            'timestep_log': self.timestep_log,
            'data_info': {
                'start_date': str(data['Date'].iloc[0]) if 'Date' in data.columns else None,
                'end_date': str(data['Date'].iloc[-1]) if 'Date' in data.columns else None,
                'num_periods': len(data),
                'simulated_periods': len(data) - self.config.window_size + 1
            }
        }
        
        return results
    
    def _calculate_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            returns: Array of period returns
            
        Returns:
            Dictionary of metrics
        """
        equity = np.array(self.portfolio.equity_curve)
        
        # Total return
        total_return = (equity[-1] / equity[0]) - 1
        
        # Cumulative return
        cumulative_return = (equity[-1] - equity[0]) / equity[0]
        
        # Daily returns statistics
        mean_return = np.mean(returns) if len(returns) > 0 else 0
        std_return = np.std(returns) if len(returns) > 0 else 0
        
        # Annualized returns
        trading_days = self.config.trading_days_per_year
        annualized_return = (1 + total_return) ** (trading_days / max(1, len(returns))) - 1
        annualized_volatility = std_return * np.sqrt(trading_days)
        
        # Sharpe ratio
        excess_return = annualized_return - self.config.risk_free_rate
        sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        winning_trades = sum(1 for r in returns if r > 0)
        total_periods = len(returns)
        win_rate = winning_trades / total_periods if total_periods > 0 else 0
        
        # Turnover (average daily trading volume relative to portfolio value)
        trades = self.portfolio.trades
        if trades and len(equity) > 1:
            total_traded = sum(abs(t.quantity * t.price) for t in trades)
            avg_portfolio_value = np.mean(equity[1:])
            turnover = total_traded / (avg_portfolio_value * len(equity))
        else:
            turnover = 0
        
        return {
            'total_return': float(total_return),
            'total_return_pct': float(total_return * 100),
            'cumulative_return': float(cumulative_return),
            'annualized_return': float(annualized_return),
            'annualized_volatility': float(annualized_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'max_drawdown_pct': float(max_drawdown * 100),
            'win_rate': float(win_rate),
            'win_rate_pct': float(win_rate * 100),
            'turnover': float(turnover),
            'mean_return': float(mean_return),
            'std_return': float(std_return),
            'num_trades': len(trades) if trades else 0
        }
    
    def _print_summary(self) -> None:
        """Print a summary of backtest results."""
        if not self.results:
            return
        
        print("\n" + "="*60)
        print("BACKTESTING RESULTS")
        print("="*60)
        
        portfolio = self.results['portfolio']
        metrics = self.results['metrics']
        
        print(f"\nPortfolio Summary:")
        print(f"  Initial Capital:    ${portfolio['initial_capital']:>15,.2f}")
        print(f"  Final Value:        ${portfolio['final_value']:>15,.2f}")
        print(f"  Total Return:       {metrics['total_return_pct']:>15.2f}%")
        
        print(f"\nPerformance Metrics:")
        print(f"  Annualized Return:  {metrics['annualized_return']*100:>15.2f}%")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>15.2f}")
        print(f"  Max Drawdown:       {metrics['max_drawdown_pct']:>15.2f}%")
        print(f"  Win Rate:           {metrics['win_rate_pct']:>15.2f}%")
        
        execution = self.results['execution']
        print(f"\nExecution Statistics:")
        print(f"  Total Trades:       {execution['total_trades']:>15}")
        print(f"  Rejected Trades:    {execution['rejected_trades']:>15}")
        print(f"  Total Commission:   ${execution['total_commission']:>15,.2f}")
        print(f"  Total Slippage:     ${execution['total_slippage']:>15,.2f}")
        
        print("="*60 + "\n")
    
    def get_results(self) -> Dict[str, Any]:
        """Get backtest results."""
        if not self._backtest_complete:
            raise RuntimeError("Backtest not complete. Call run() first.")
        return self.results
    
    def get_equity_curve(self) -> List[float]:
        """Get the equity curve."""
        if self.portfolio:
            return self.portfolio.equity_curve
        return []
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if self.portfolio:
            return self.portfolio.get_trade_history()
        return pd.DataFrame()
    
    def get_timestep_log(self) -> pd.DataFrame:
        """Get timestep log as DataFrame."""
        return pd.DataFrame(self.timestep_log)
    
    def reset(self) -> None:
        """Reset the engine for a new backtest."""
        if self.portfolio:
            self.portfolio.reset()
        if self.executor:
            self.executor.reset_statistics()
        self.timestep_log = []
        self.results = {}
        self._backtest_complete = False
