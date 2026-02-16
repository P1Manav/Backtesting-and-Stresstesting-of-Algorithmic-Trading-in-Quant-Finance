class Portfolio:

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.shares = 0
        self.position = 0

    def buy(self, price: float, commission_rate: float) -> str:
        """Buy as many shares as possible at `price`. Returns trade description."""
        max_shares = int(self.cash / (price * (1 + commission_rate)))
        if max_shares <= 0:
            return ''
        cost = max_shares * price * (1 + commission_rate)
        self.cash -= cost
        self.shares += max_shares
        self.position = 1
        return f"BUY {max_shares} @ ${price:.2f}"

    def sell(self, price: float, commission_rate: float) -> str:
        """Sell all shares at `price`. Returns trade description."""
        if self.shares <= 0:
            return ''
        revenue = self.shares * price * (1 - commission_rate)
        desc = f"SELL {self.shares} @ ${price:.2f}"
        self.cash += revenue
        self.shares = 0
        self.position = 0
        return desc

    def value(self, current_price: float) -> float:
        """Current total portfolio value."""
        return self.cash + self.shares * current_price
