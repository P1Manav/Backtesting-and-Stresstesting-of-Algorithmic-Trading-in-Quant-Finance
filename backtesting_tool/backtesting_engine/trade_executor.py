"""
Trade Executor Module
Handles trade execution with constraints and cost modeling.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .portfolio import Portfolio
from .backtesting_config import BacktestingConfig


@dataclass
class ExecutionResult:
    """Result of a trade execution attempt."""
    executed: bool
    target_quantity: float
    actual_quantity: float
    price: float
    commission: float
    slippage_cost: float
    rejection_reason: Optional[str] = None
    
    @property
    def total_cost(self) -> float:
        """Total execution cost."""
        return abs(self.actual_quantity * self.price) + self.commission + self.slippage_cost


class TradeExecutor:
    """
    Executes trades with constraints and cost modeling.
    
    Handles:
    - Transaction cost calculation (commission + slippage)
    - Trading constraints enforcement
    - Position size limits
    - Short selling restrictions
    """
    
    def __init__(self, config: BacktestingConfig):
        """
        Initialize the Trade Executor.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        
        # Execution statistics
        self._total_trades = 0
        self._rejected_trades = 0
        self._total_volume = 0.0
        self._total_commission = 0.0
        self._total_slippage = 0.0
    
    def execute_signal(self,
                       portfolio: Portfolio,
                       action: int,
                       position_size: float,
                       current_price: float,
                       timestamp: Optional[Any] = None) -> ExecutionResult:
        """
        Execute a trading signal.
        
        Args:
            portfolio: Portfolio to execute on
            action: Trading action (-1=short, 0=flat, 1=long)
            position_size: Target position size (-1 to 1)
            current_price: Current market price
            timestamp: Optional timestamp
            
        Returns:
            ExecutionResult with execution details
        """
        # Calculate target position
        portfolio_value = portfolio.get_portfolio_value(current_price)
        max_position_value = portfolio_value * self.config.max_position_size
        
        # Target quantity based on position size
        target_quantity = (position_size * max_position_value) / current_price
        
        # Apply constraints
        target_quantity, rejection_reason = self._apply_constraints(
            portfolio, target_quantity, current_price
        )
        
        if rejection_reason:
            self._rejected_trades += 1
            return ExecutionResult(
                executed=False,
                target_quantity=target_quantity,
                actual_quantity=0,
                price=current_price,
                commission=0,
                slippage_cost=0,
                rejection_reason=rejection_reason
            )
        
        # Calculate execution price with slippage
        execution_price = self._apply_slippage(current_price, target_quantity)
        
        # Calculate trade details
        current_quantity = portfolio.position.quantity
        delta_quantity = target_quantity - current_quantity
        
        if abs(delta_quantity) < self.config.min_trade_size:
            return ExecutionResult(
                executed=False,
                target_quantity=target_quantity,
                actual_quantity=current_quantity,
                price=execution_price,
                commission=0,
                slippage_cost=0,
                rejection_reason="Trade size below minimum"
            )
        
        # Calculate costs
        trade_value = abs(delta_quantity * execution_price)
        commission = trade_value * self.config.commission_rate
        slippage_cost = trade_value * self.config.slippage
        
        # Execute the trade
        success = portfolio.set_position(
            target_quantity=target_quantity,
            price=execution_price,
            commission_rate=self.config.commission_rate,
            slippage_rate=self.config.slippage,
            timestamp=timestamp
        )
        
        if success:
            self._total_trades += 1
            self._total_volume += trade_value
            self._total_commission += commission
            self._total_slippage += slippage_cost
        
        return ExecutionResult(
            executed=success,
            target_quantity=target_quantity,
            actual_quantity=portfolio.position.quantity,
            price=execution_price,
            commission=commission,
            slippage_cost=slippage_cost
        )
    
    def _apply_constraints(self,
                           portfolio: Portfolio,
                           target_quantity: float,
                           current_price: float) -> Tuple[float, Optional[str]]:
        """
        Apply trading constraints to target quantity.
        
        Args:
            portfolio: Current portfolio
            target_quantity: Desired quantity
            current_price: Current price
            
        Returns:
            Tuple of (adjusted_quantity, rejection_reason)
        """
        # Check short selling constraint
        if target_quantity < 0 and not self.config.allow_short:
            return 0, "Short selling not allowed"
        
        # Check position size constraint
        portfolio_value = portfolio.get_portfolio_value(current_price)
        max_position_value = portfolio_value * self.config.max_position_size
        max_quantity = max_position_value / current_price
        
        if abs(target_quantity) > max_quantity:
            # Scale down to max allowed
            target_quantity = np.sign(target_quantity) * max_quantity
        
        # Check cash availability for long positions
        current_quantity = portfolio.position.quantity
        delta = target_quantity - current_quantity
        
        if delta > 0:  # Buying
            required_cash = delta * current_price * (1 + self.config.commission_rate + self.config.slippage)
            if required_cash > portfolio.cash:
                # Calculate max affordable quantity
                affordable_delta = portfolio.cash / (current_price * (1 + self.config.commission_rate + self.config.slippage))
                target_quantity = current_quantity + affordable_delta
                
                if affordable_delta < self.config.min_trade_size:
                    return current_quantity, "Insufficient cash"
        
        return target_quantity, None
    
    def _apply_slippage(self, price: float, quantity: float) -> float:
        """
        Apply slippage to execution price.
        
        Args:
            price: Market price
            quantity: Trade quantity
            
        Returns:
            Adjusted execution price
        """
        # Positive quantity (buy) -> price goes up
        # Negative quantity (sell) -> price goes down
        slippage_multiplier = 1 + np.sign(quantity) * self.config.slippage
        return price * slippage_multiplier
    
    def calculate_transaction_cost(self,
                                    quantity: float,
                                    price: float) -> Tuple[float, float]:
        """
        Calculate transaction costs for a trade.
        
        Args:
            quantity: Trade quantity
            price: Trade price
            
        Returns:
            Tuple of (commission, slippage_cost)
        """
        trade_value = abs(quantity * price)
        commission = trade_value * self.config.commission_rate
        slippage_cost = trade_value * self.config.slippage
        return commission, slippage_cost
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'total_trades': self._total_trades,
            'rejected_trades': self._rejected_trades,
            'total_volume': self._total_volume,
            'total_commission': self._total_commission,
            'total_slippage': self._total_slippage,
            'average_commission': self._total_commission / max(1, self._total_trades),
            'average_slippage': self._total_slippage / max(1, self._total_trades)
        }
    
    def reset_statistics(self) -> None:
        """Reset execution statistics."""
        self._total_trades = 0
        self._rejected_trades = 0
        self._total_volume = 0.0
        self._total_commission = 0.0
        self._total_slippage = 0.0
