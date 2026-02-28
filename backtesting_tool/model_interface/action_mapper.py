class ActionMapper:

    def __init__(self, strategy: str = 'simple', threshold_pct: float = 0.0):
        self.strategy = strategy
        self.threshold = threshold_pct / 100.0

    # Determine action (BUY/SELL/HOLD) given prediction and current position.
    def get_action(self, predicted_price: float, current_price: float,
                   current_position: int) -> str:
        if self.strategy == 'simple':
            if predicted_price > current_price and current_position == 0:
                return 'BUY'
            elif predicted_price < current_price and current_position == 1:
                return 'SELL'
            return 'HOLD'

        else:
            pct_change = (predicted_price - current_price) / current_price
            if pct_change > self.threshold and current_position == 0:
                return 'BUY'
            elif pct_change < -self.threshold and current_position == 1:
                return 'SELL'
            return 'HOLD'
