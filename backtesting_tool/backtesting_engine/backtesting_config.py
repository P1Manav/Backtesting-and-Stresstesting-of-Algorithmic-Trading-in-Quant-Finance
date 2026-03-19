"""Backtesting configuration settings"""
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    """Backtesting configuration dataclass"""
    initial_capital: float = 100_000.0
    commission_pct: float = 0.1
    strategy: str = 'simple'
    threshold_pct: float = 0.0
    sequence_length: int = 60

    @property
    def commission_rate(self) -> float:
        """Calculate commission rate from percentage"""
        return self.commission_pct / 100.0

