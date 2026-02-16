from dataclasses import dataclass


@dataclass
class BacktestConfig:
    """All parameters needed for a backtesting run."""
    initial_capital: float = 100_000.0
    commission_pct: float = 0.1
    strategy: str = 'simple'
    threshold_pct: float = 0.0
    sequence_length: int = 60

    @property
    def commission_rate(self) -> float:
        """Commission as a decimal fraction."""
        return self.commission_pct / 100.0
