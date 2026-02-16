import pandas as pd
from typing import List


class DataValidator:

    REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
    PRICE_COLUMNS = ['Open', 'High', 'Low', 'Close']

    def __init__(self):
        self.warnings: List[str] = []

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame has required OHLCV columns and reasonable values."""
        self.warnings = []

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Dataset must have: {self.REQUIRED_COLUMNS}"
            )

        for col in self.REQUIRED_COLUMNS:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' must be numeric, got {df[col].dtype}")

        for col in self.PRICE_COLUMNS:
            if (df[col] < 0).any():
                self.warnings.append(f"Column '{col}' has negative values")

        if (df['Volume'] == 0).sum() > len(df) * 0.5:
            self.warnings.append("More than 50% of Volume values are zero")

        if (df['High'] < df['Low']).any():
            self.warnings.append("Some rows have High < Low")

        for w in self.warnings:
            print(f"  [WARNING] {w}")

        return True
