from dataclasses import dataclass

@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_pct: float = 0.1
    strategy: str = 'simple'
    threshold_pct: float = 0.0
    sequence_length: int = 60

    # Commission as a decimal fraction.
    @property
    def commission_rate(self) -> float:
        return self.commission_pct / 100.0
