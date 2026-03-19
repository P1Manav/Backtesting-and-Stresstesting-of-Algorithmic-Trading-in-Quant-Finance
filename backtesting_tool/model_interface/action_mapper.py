"""Map predictions to trading actions"""

class ActionMapper:
    """Convert price predictions to trading actions"""

    def __init__(self, strategy: str = 'simple', threshold_pct: float = 0.0):
        """Initialize action mapper"""
        self.strategy = strategy
        self.threshold = threshold_pct / 100.0

    def get_action(self, predicted_price: float, current_price: float,
                   current_position: int) -> str:
        """Determine trading action based on prediction"""
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

