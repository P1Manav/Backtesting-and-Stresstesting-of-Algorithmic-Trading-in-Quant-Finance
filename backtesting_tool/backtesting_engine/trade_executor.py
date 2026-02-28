from typing import List, Optional, Dict, Any
from .portfolio import Portfolio

class TradeExecutor:

    def __init__(self, portfolio: Portfolio, commission_rate: float):
        self.portfolio = portfolio
        self.commission_rate = commission_rate
        self.trade_log: List[Dict[str, Any]] = []

    # Execute a BUY or SELL action for a specific stock.
    def execute(self, action: str, ticker: str, price: float,
                date=None, budget: Optional[float] = None) -> None:
        if action == 'BUY':
            msg = self.portfolio.buy(ticker, price, self.commission_rate, budget)
            if msg:
                self.trade_log.append({
                    'date': date, 'ticker': ticker, 'action': 'BUY',
                    'price': price, 'description': msg,
                })
        elif action == 'SELL':
            msg = self.portfolio.sell(ticker, price, self.commission_rate)
            if msg:
                self.trade_log.append({
                    'date': date, 'ticker': ticker, 'action': 'SELL',
                    'price': price, 'description': msg,
                })
