import pandas as pd
from pathlib import Path
from typing import Dict, Any


class ReportGenerator:

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, results: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Save portfolio history, trades, and metrics as CSVs."""
        tickers = results.get('tickers', ['STOCK'])

        # ---- Aggregate portfolio history --------------------------------
        agg = {
            'Date': results['dates'],
            'Portfolio_Value': results['portfolio_values'],
            'Cash': results['cash'],
        }
        # Add per-stock columns
        for t in tickers:
            ps = results['per_stock'][t]
            agg[f'{t}_Price'] = ps['actual_prices']
            agg[f'{t}_Predicted'] = ps['predicted_prices']
            agg[f'{t}_Position'] = ps['positions']
            agg[f'{t}_Shares'] = ps['shares']

        pd.DataFrame(agg).to_csv(
            self.save_dir / 'portfolio_history.csv', index=False)

        # ---- Trades log -------------------------------------------------
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            trades_df.to_csv(self.save_dir / 'trades.csv', index=False)

        # ---- Metrics ----------------------------------------------------
        if 'aggregate' in metrics:
            # Save aggregate metrics
            pd.DataFrame([metrics['aggregate']]).to_csv(
                self.save_dir / 'metrics.csv', index=False)

            # Save per-stock metrics (if any)
            per_stock = {k: v for k, v in metrics.items() if k != 'aggregate'}
            if per_stock:
                ps_df = pd.DataFrame(per_stock).T
                ps_df.index.name = 'Ticker'
                ps_df.to_csv(self.save_dir / 'per_stock_metrics.csv')
        else:
            pd.DataFrame([metrics]).to_csv(
                self.save_dir / 'metrics.csv', index=False)

        print(f"  [OK] CSV reports saved to: {self.save_dir}")
