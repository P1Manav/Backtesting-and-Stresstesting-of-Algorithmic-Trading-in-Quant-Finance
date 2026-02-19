from typing import Dict, List, Optional


class Portfolio:
    """Multi-stock portfolio tracker with a shared cash pool."""

    def __init__(self, initial_capital: float, tickers: List[str]):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.tickers = list(tickers)
        self.n_stocks = len(self.tickers)
        self.shares: Dict[str, int] = {t: 0 for t in self.tickers}
        self.positions: Dict[str, int] = {t: 0 for t in self.tickers}

    def buy(self, ticker: str, price: float, commission_rate: float,
            budget: Optional[float] = None) -> str:
        """Buy shares of *ticker* within an optional budget.
        Returns trade description or empty string."""
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
        """Sell all shares of *ticker*. Returns trade description or empty string."""
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
        """Total portfolio value (cash + all holdings)."""
        holdings = sum(self.shares[t] * current_prices.get(t, 0.0)
                       for t in self.tickers)
        return self.cash + holdings

    def stock_value(self, ticker: str, price: float) -> float:
        """Value of holdings in a single stock."""
        return self.shares.get(ticker, 0) * price
