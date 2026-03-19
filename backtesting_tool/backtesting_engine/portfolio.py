"""Portfolio state and holdings management"""
from typing import Dict, List, Optional

class Portfolio:
    """Portfolio for managing holdings and cash"""

    def __init__(self, initial_capital: float, tickers: List[str]):
        """Initialize portfolio"""
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.tickers = list(tickers)
        self.n_stocks = len(self.tickers)
        self.shares: Dict[str, int] = {t: 0 for t in self.tickers}
        self.positions: Dict[str, int] = {t: 0 for t in self.tickers}

    def buy(self, ticker: str, price: float, commission_rate: float,
            budget: Optional[float] = None) -> str:
        """Execute buy order and return description"""
        if price <= 0:
            return ''
        available = min(self.cash, budget) if budget is not None else self.cash
        max_shares = int(available / (price * (1 + commission_rate)))
        if max_shares <= 0:
            return ''
        cost = max_shares * price * (1 + commission_rate)
        self.cash -= cost
        self.shares[ticker] += max_shares
        self.positions[ticker] = 1
        return f"BUY {ticker} {max_shares} @ ${price:.2f}"

    def sell(self, ticker: str, price: float, commission_rate: float) -> str:
        """Execute sell order and return description"""
        if self.shares.get(ticker, 0) <= 0:
            return ''
        n = self.shares[ticker]
        revenue = n * price * (1 - commission_rate)
        desc = f"SELL {ticker} {n} @ ${price:.2f}"
        self.cash += revenue
        self.shares[ticker] = 0
        self.positions[ticker] = 0
        return desc

    def value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        holdings = sum(self.shares[t] * current_prices.get(t, 0.0)
                       for t in self.tickers)
        return self.cash + holdings

    def stock_value(self, ticker: str, price: float) -> float:
        """Calculate valuefor specific stock"""
        return self.shares.get(ticker, 0) * price

