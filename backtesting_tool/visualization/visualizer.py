import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class BacktestVisualizer:

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            try:
                plt.style.use('seaborn-whitegrid')
            except Exception:
                pass

    def generate_all(self, results: Dict[str, Any]) -> None:
        """Generate and save all charts."""
        self._plot_equity_curve(results)
        self._plot_predictions(results)
        self._plot_drawdown(results)
        print(f"\n  [OK] Charts saved to: {self.save_dir}")

    def _plot_equity_curve(self, results: Dict[str, Any]):
        dates = pd.to_datetime(results['dates'])
        values = results['portfolio_values']

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(dates, values, color='#2196F3', linewidth=2)
        ax.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.save_dir / 'equity_curve.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_predictions(self, results: Dict[str, Any]):
        dates = pd.to_datetime(results['dates'])
        actual = results['actual_prices']
        predicted = results['predicted_prices']

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(dates, actual, label='Actual', color='blue', linewidth=1.5)
        ax.plot(dates, predicted, label='Predicted', color='red', linewidth=1, alpha=0.7)
        ax.set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.save_dir / 'predictions.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_drawdown(self, results: Dict[str, Any]):
        dates = pd.to_datetime(results['dates'])
        pv = np.array(results['portfolio_values'])
        cummax = np.maximum.accumulate(pv)
        dd = (pv - cummax) / cummax * 100

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.fill_between(dates, dd, 0, color='red', alpha=0.3)
        ax.plot(dates, dd, color='red', linewidth=1.5)
        ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.save_dir / 'drawdown.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
