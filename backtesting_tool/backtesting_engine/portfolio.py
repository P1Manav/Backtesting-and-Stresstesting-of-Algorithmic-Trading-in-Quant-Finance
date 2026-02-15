"""
Portfolio Module
Tracks portfolio state including cash, positions, and value.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Position:
    """Represents a trading position."""
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Calculate market value of position."""
        return self.quantity * self.current_price
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0
    
    def update_price(self, price: float) -> None:
        """Update current price and unrealized PnL."""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: Any
    action: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    commission: float
    slippage_cost: float
    portfolio_value: float
    cash_after: float
    position_after: float
    
    @property
    def total_cost(self) -> float:
        """Total cost including commission and slippage."""
        return abs(self.quantity * self.price) + self.commission + self.slippage_cost


class Portfolio:
    """
    Manages portfolio state and tracking.
    
    Tracks:
    - Cash balance
    - Positions
    - Portfolio value
    - Trade history
    - Equity curve
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 allow_short: bool = True):
        """
        Initialize the Portfolio.
        
        Args:
            initial_capital: Starting cash amount
            allow_short: Whether short selling is allowed
        """
        self.initial_capital = initial_capital
        self.allow_short = allow_short
        
        # Current state
        self.cash = initial_capital
        self.position = Position()
        
        # History tracking
        self.equity_curve: List[float] = [initial_capital]
        self.cash_history: List[float] = [initial_capital]
        self.position_history: List[float] = [0.0]
        self.trades: List[Trade] = []
        self.timestamps: List[Any] = []
        
        # Statistics
        self._total_commission = 0.0
        self._total_slippage = 0.0
    
    def update(self, 
               current_price: float, 
               timestamp: Optional[Any] = None) -> float:
        """
        Update portfolio value with current price.
        
        Args:
            current_price: Current market price
            timestamp: Optional timestamp
            
        Returns:
            Current portfolio value
        """
        self.position.update_price(current_price)
        portfolio_value = self.get_portfolio_value(current_price)
        
        self.equity_curve.append(portfolio_value)
        self.cash_history.append(self.cash)
        self.position_history.append(self.position.quantity)
        
        if timestamp is not None:
            self.timestamps.append(timestamp)
        
        return portfolio_value
    
    def get_portfolio_value(self, current_price: Optional[float] = None) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            current_price: Current market price (uses position's current_price if None)
            
        Returns:
            Total portfolio value (cash + positions)
        """
        if current_price is not None:
            position_value = self.position.quantity * current_price
        else:
            position_value = self.position.market_value
        
        return self.cash + position_value
    
    def open_position(self,
                      quantity: float,
                      price: float,
                      commission: float = 0.0,
                      slippage_cost: float = 0.0,
                      timestamp: Optional[Any] = None) -> bool:
        """
        Open or add to a position.
        
        Args:
            quantity: Quantity to buy/sell (positive = buy, negative = short)
            price: Execution price
            commission: Transaction commission
            slippage_cost: Slippage cost
            timestamp: Trade timestamp
            
        Returns:
            True if trade was executed
        """
        if quantity == 0:
            return False
        
        # Check if short selling is allowed
        if quantity < 0 and not self.allow_short:
            return False
        
        # Calculate trade cost
        trade_value = quantity * price
        total_cost = trade_value + commission + slippage_cost
        
        # Check if we have enough cash for long positions
        if quantity > 0 and self.cash < total_cost:
            return False
        
        # Execute trade
        self.cash -= total_cost
        
        # Update position
        if self.position.quantity == 0:
            # New position
            self.position.quantity = quantity
            self.position.entry_price = price
        else:
            # Add to existing position
            total_quantity = self.position.quantity + quantity
            if total_quantity != 0:
                # Weighted average entry price
                total_cost_basis = (self.position.entry_price * self.position.quantity + 
                                    price * quantity)
                self.position.entry_price = total_cost_basis / total_quantity
            self.position.quantity = total_quantity
        
        self.position.current_price = price
        
        # Record trade
        trade = Trade(
            timestamp=timestamp,
            action='BUY' if quantity > 0 else 'SELL',
            quantity=quantity,
            price=price,
            commission=commission,
            slippage_cost=slippage_cost,
            portfolio_value=self.get_portfolio_value(price),
            cash_after=self.cash,
            position_after=self.position.quantity
        )
        self.trades.append(trade)
        
        # Update totals
        self._total_commission += commission
        self._total_slippage += slippage_cost
        
        return True
    
    def close_position(self,
                       price: float,
                       commission: float = 0.0,
                       slippage_cost: float = 0.0,
                       timestamp: Optional[Any] = None) -> float:
        """
        Close the entire position.
        
        Args:
            price: Execution price
            commission: Transaction commission
            slippage_cost: Slippage cost
            timestamp: Trade timestamp
            
        Returns:
            Realized PnL from closing the position
        """
        if self.position.quantity == 0:
            return 0.0
        
        # Calculate realized PnL
        trade_value = self.position.quantity * price
        pnl = (price - self.position.entry_price) * self.position.quantity
        
        # Update cash
        self.cash += trade_value - commission - slippage_cost
        
        # Record realized PnL
        self.position.realized_pnl += pnl
        
        # Record trade
        trade = Trade(
            timestamp=timestamp,
            action='SELL' if self.position.quantity > 0 else 'BUY',
            quantity=-self.position.quantity,
            price=price,
            commission=commission,
            slippage_cost=slippage_cost,
            portfolio_value=self.get_portfolio_value(price),
            cash_after=self.cash,
            position_after=0
        )
        self.trades.append(trade)
        
        # Update totals
        self._total_commission += commission
        self._total_slippage += slippage_cost
        
        # Reset position
        realized_pnl = self.position.realized_pnl
        self.position = Position(realized_pnl=realized_pnl)
        
        return pnl
    
    def set_position(self,
                     target_quantity: float,
                     price: float,
                     commission_rate: float = 0.0,
                     slippage_rate: float = 0.0,
                     timestamp: Optional[Any] = None) -> bool:
        """
        Set position to a target quantity.
        
        Args:
            target_quantity: Desired position quantity
            price: Current market price
            commission_rate: Commission as percentage
            slippage_rate: Slippage as percentage
            timestamp: Trade timestamp
            
        Returns:
            True if position was adjusted
        """
        current_quantity = self.position.quantity
        delta = target_quantity - current_quantity
        
        if abs(delta) < 1e-8:
            return False
        
        # Calculate costs
        trade_value = abs(delta * price)
        commission = trade_value * commission_rate
        slippage_cost = trade_value * slippage_rate
        
        if delta > 0:
            # Buying
            return self.open_position(delta, price, commission, slippage_cost, timestamp)
        else:
            # Selling
            if target_quantity == 0:
                self.close_position(price, commission, slippage_cost, timestamp)
                return True
            else:
                return self.open_position(delta, price, commission, slippage_cost, timestamp)
    
    def get_returns(self) -> np.ndarray:
        """Calculate return series from equity curve."""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    def get_equity_dataframe(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        data = {
            'portfolio_value': self.equity_curve,
            'cash': self.cash_history,
            'position': self.position_history
        }
        
        if self.timestamps:
            data['timestamp'] = [None] + self.timestamps
        
        return pd.DataFrame(data)
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'timestamp': trade.timestamp,
                'action': trade.action,
                'quantity': trade.quantity,
                'price': trade.price,
                'commission': trade.commission,
                'slippage_cost': trade.slippage_cost,
                'total_cost': trade.total_cost,
                'portfolio_value': trade.portfolio_value,
                'cash_after': trade.cash_after,
                'position_after': trade.position_after
            })
        
        return pd.DataFrame(trade_data)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get portfolio summary statistics."""
        final_value = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'cash': self.cash,
            'position_quantity': self.position.quantity,
            'position_value': self.position.market_value,
            'total_trades': len(self.trades),
            'total_commission': self._total_commission,
            'total_slippage': self._total_slippage,
            'realized_pnl': self.position.realized_pnl,
            'unrealized_pnl': self.position.unrealized_pnl
        }
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.position = Position()
        self.equity_curve = [self.initial_capital]
        self.cash_history = [self.initial_capital]
        self.position_history = [0.0]
        self.trades = []
        self.timestamps = []
        self._total_commission = 0.0
        self._total_slippage = 0.0
