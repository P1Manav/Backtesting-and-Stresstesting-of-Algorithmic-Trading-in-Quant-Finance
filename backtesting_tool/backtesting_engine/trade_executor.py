from typing import List
from .portfolio import Portfolio


class TradeExecutor:

    def __init__(self, portfolio: Portfolio, commission_rate: float):
        self.portfolio = portfolio
        self.commission_rate = commission_rate
        self.trade_log: List[str] = []

    def execute(self, action: str, price: float) -> None:
        """Execute a BUY or SELL action at the given price."""
        if action == 'BUY':
            msg = self.portfolio.buy(price, self.commission_rate)
            if msg:
                self.trade_log.append(msg)
        elif action == 'SELL':
            msg = self.portfolio.sell(price, self.commission_rate)
            if msg:
                self.trade_log.append(msg)
