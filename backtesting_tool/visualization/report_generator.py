import pandas as pd
from pathlib import Path
from typing import Dict, Any


class ReportGenerator:

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, results: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """Save portfolio history, trades, and metrics as CSVs."""
        df = pd.DataFrame({
            'Date': results['dates'],
            'Actual_Price': results['actual_prices'],
            'Predicted_Price': results['predicted_prices'],
            'Position': results['positions'],
            'Portfolio_Value': results['portfolio_values'],
            'Cash': results['cash'],
            'Holdings': results['holdings'],
        })
        df.to_csv(self.save_dir / 'portfolio_history.csv', index=False)

        if results['trades']:
            pd.DataFrame({'Trade': results['trades']}).to_csv(
                self.save_dir / 'trades.csv', index=False
            )

        pd.DataFrame([metrics]).to_csv(self.save_dir / 'metrics.csv', index=False)

        print(f"  [OK] CSV reports saved to: {self.save_dir}")
