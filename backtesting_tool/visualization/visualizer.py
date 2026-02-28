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

    # Generate and save all charts.
    def generate_all(self, results: Dict[str, Any]) -> None:
        self._plot_equity_curve(results)
        self._plot_predictions(results)
        self._plot_drawdown(results)

        tickers = results.get('tickers', [])
        if len(tickers) > 1:
            self._plot_allocation(results)
            self._plot_normalized_prices(results)

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
        tickers = results.get('tickers', [])
        dates = pd.to_datetime(results['dates'])
        n = len(tickers)

        if n <= 1:
            t = tickers[0]
            actual = results['per_stock'][t]['actual_prices']
            predicted = results['per_stock'][t]['predicted_prices']
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(dates, actual, label=f'{t} Actual', color='blue', linewidth=1.5)
            ax.plot(dates, predicted, label=f'{t} Predicted', color='red',
                    linewidth=1, alpha=0.7)
            ax.set_title(f'{t}: Actual vs Predicted Prices', fontsize=14,
                         fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(self.save_dir / 'predictions.png', dpi=300,
                        bbox_inches='tight')
            plt.close(fig)
        else:
            cols = min(2, n)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows),
                                     squeeze=False)
            for idx, t in enumerate(tickers):
                r, c = idx // cols, idx % cols
                ax = axes[r][c]
                actual = results['per_stock'][t]['actual_prices']
                predicted = results['per_stock'][t]['predicted_prices']
                ax.plot(dates, actual, label='Actual', color='blue', linewidth=1)
                ax.plot(dates, predicted, label='Predicted', color='red',
                        linewidth=0.8, alpha=0.7)
                ax.set_title(f'{t}', fontsize=11, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            for idx in range(n, rows * cols):
                r, c = idx // cols, idx % cols
                axes[r][c].set_visible(False)

            fig.suptitle('Actual vs Predicted Prices (Per Stock)',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            fig.savefig(self.save_dir / 'predictions.png', dpi=300,
                        bbox_inches='tight')
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

    def _plot_allocation(self, results: Dict[str, Any]):
        tickers = results.get('tickers', [])
        dates = pd.to_datetime(results['dates'])
        n = len(dates)

        alloc_data = {t: np.zeros(n) for t in tickers}
        cash_alloc = np.zeros(n)

        for i in range(n):
            port_val = results['portfolio_values'][i]
            if port_val <= 0:
                continue
            cash_alloc[i] = results['cash'][i] / port_val * 100
            for t in tickers:
                shares = results['per_stock'][t]['shares'][i]
                price = results['per_stock'][t]['actual_prices'][i]
                alloc_data[t][i] = shares * price / port_val * 100

        fig, ax = plt.subplots(figsize=(14, 5))
        colors = plt.cm.Set3(np.linspace(0, 1, len(tickers) + 1))

        bottom = np.zeros(n)
        for idx, t in enumerate(tickers):
            ax.fill_between(dates, bottom, bottom + alloc_data[t],
                            label=t, alpha=0.8, color=colors[idx])
            bottom = bottom + alloc_data[t]

        ax.fill_between(dates, bottom, bottom + cash_alloc,
                        label='Cash', alpha=0.6, color='gray')

        ax.set_title('Portfolio Allocation Over Time', fontsize=14,
                     fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Allocation (%)')
        ax.set_ylim(0, 105)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.save_dir / 'allocation.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

    def _plot_normalized_prices(self, results: Dict[str, Any]):
        tickers = results.get('tickers', [])
        dates = pd.to_datetime(results['dates'])

        fig, ax = plt.subplots(figsize=(14, 5))
        for t in tickers:
            prices = np.array(results['per_stock'][t]['actual_prices'])
            normalized = prices / prices[0] * 100
            ax.plot(dates, normalized, label=t, linewidth=1.5)

        ax.axhline(y=100, color='black', linestyle='--', alpha=0.3)
        ax.set_title('Normalized Stock Price Performance (base = 100)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.save_dir / 'price_performance.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)
