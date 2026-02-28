import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

_CL_COLORS = {
    0.90: ('#FF9800', '#E65100'),
    0.95: ('#F44336', '#B71C1C'),
    0.99: ('#9C27B0', '#4A148C'),
}
_CL_MARKERS = {0.90: 'v', 0.95: 'x', 0.99: 's'}

class StatisticalVisualizer:

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

    # Generate all statistical backtesting charts.
    def generate_all(self, backtest_results: Dict[str, Any],
                     stat_results: Dict[str, Any]) -> None:
        self._plot_var_breaches(backtest_results, stat_results)
        self._plot_violation_timeline(backtest_results, stat_results)
        self._plot_benchmark_comparison(backtest_results, stat_results)
        self._plot_test_summary(stat_results)
        print(f"  [OK] Statistical charts saved to: {self.save_dir}")

    def _plot_var_breaches(self, backtest_results: Dict[str, Any],
                           stat_results: Dict[str, Any]):
        returns = stat_results['_returns']
        confidence_levels = stat_results['confidence_levels']

        dates = pd.to_datetime(backtest_results['dates'][1:])
        dates = dates[:len(returns)]

        fig, ax = plt.subplots(figsize=(14, 7))

        ax.bar(dates, returns * 100, width=1, color='steelblue',
               alpha=0.5, label='Daily Returns')

        for cl in confidence_levels:
            level = stat_results['levels'][cl]
            var_series = level['_var_series']
            hits = level['_hit_sequence']
            valid = level['_valid_mask']

            line_color, dark_color = _CL_COLORS.get(cl, ('#F44336', '#B71C1C'))
            marker = _CL_MARKERS.get(cl, 'x')
            pct_label = f"{cl * 100:.0f}%"

            valid_idx = valid[:len(dates)]
            ax.plot(dates[valid_idx], -var_series[valid_idx] * 100,
                    color=line_color, linewidth=1.8, alpha=0.9,
                    label=f'VaR ({pct_label})')

            breach_mask = (hits == 1) & valid
            breach_mask = breach_mask[:len(dates)]
            n_breaches = int(np.sum(breach_mask))
            if n_breaches > 0:
                ax.scatter(dates[breach_mask], returns[breach_mask] * 100,
                           color=dark_color, marker=marker, s=60, zorder=5,
                           alpha=0.85,
                           label=f'Violations {pct_label} ({n_breaches})')

        ax.set_title('Daily Returns vs Value-at-Risk (VaR) — Multi-Level',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Return / VaR (%)')
        ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(self.save_dir / 'var_breaches.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

    def _plot_violation_timeline(self, backtest_results: Dict[str, Any],
                                 stat_results: Dict[str, Any]):
        confidence_levels = stat_results['confidence_levels']
        dates = pd.to_datetime(backtest_results['dates'][1:])

        n_levels = len(confidence_levels)
        fig, axes = plt.subplots(n_levels * 2, 1,
                                 figsize=(14, 4 * n_levels),
                                 height_ratios=[2, 1] * n_levels)
        if n_levels == 1:
            axes = [axes] if not hasattr(axes, '__len__') else list(axes)

        for idx, cl in enumerate(confidence_levels):
            level = stat_results['levels'][cl]
            hits = level['_hit_sequence']
            valid = level['_valid_mask']
            pct_label = f"{cl * 100:.0f}%"

            valid_dates = dates[valid[:len(dates)]]
            valid_hits = hits[valid]

            line_color, dark_color = _CL_COLORS.get(cl, ('#F44336', '#B71C1C'))

            ax_top = axes[idx * 2]
            violation_idx = np.where(valid_hits == 1)[0]

            ax_top.vlines(valid_dates[violation_idx], 0, 1, colors=dark_color,
                          linewidth=1.5, alpha=0.8, label=f'Violation ({pct_label})')
            ax_top.set_yticks([])
            ax_top.set_title(f'VaR {pct_label} — Violation Timeline (Clustering Analysis)',
                             fontsize=13, fontweight='bold')
            ax_top.legend(loc='upper right', fontsize=9)
            ax_top.grid(True, alpha=0.3, axis='x')

            ax_bot = axes[idx * 2 + 1]
            window = min(60, max(10, len(valid_hits) // 5))
            if window > 1 and len(valid_hits) > window:
                rolling_rate = pd.Series(valid_hits).rolling(
                    window=window, min_periods=1).mean().values
                ax_bot.plot(valid_dates, rolling_rate, color=dark_color,
                            linewidth=1.5,
                            label=f'Rolling Violation Rate ({window}-day)')
                expected_rate = 1 - cl
                ax_bot.axhline(y=expected_rate, color='green', linestyle='--',
                               alpha=0.8, linewidth=1.5,
                               label=f'Expected Rate ({expected_rate:.2%})')
                ax_bot.set_ylabel('Violation Rate')
                ax_bot.set_xlabel('Date')
                ax_bot.legend(loc='upper right', fontsize=9)
                ax_bot.grid(True, alpha=0.3)
                ax_bot.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(self.save_dir / 'violation_timeline.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

    def _plot_benchmark_comparison(self, backtest_results: Dict[str, Any],
                                    stat_results: Dict[str, Any]):
        bench = stat_results['benchmark_comparison']
        benchmark_values = bench['_benchmark_values']
        portfolio_values = np.array(backtest_results['portfolio_values'])
        dates = pd.to_datetime(backtest_results['dates'])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                        height_ratios=[2, 1])

        ax1.plot(dates, portfolio_values, color='#2196F3', linewidth=1.8,
                 label=f'Model Portfolio (${portfolio_values[-1]:,.0f})')
        ax1.plot(dates, benchmark_values, color='#FF9800', linewidth=1.8,
                 linestyle='--',
                 label=f'Buy-and-Hold Benchmark (${benchmark_values[-1]:,.0f})')
        ax1.set_title('Model Portfolio vs Buy-and-Hold Benchmark',
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)

        port_norm = portfolio_values / portfolio_values[0]
        bench_norm = benchmark_values / benchmark_values[0]
        excess_cum = (port_norm - bench_norm) * 100

        ax2.fill_between(dates, excess_cum, 0,
                         where=excess_cum >= 0, color='#4CAF50', alpha=0.3)
        ax2.fill_between(dates, excess_cum, 0,
                         where=excess_cum < 0, color='#F44336', alpha=0.3)
        ax2.plot(dates, excess_cum, color='#333333', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Cumulative Excess Return (Model - Benchmark)',
                      fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Excess Return (pp)')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(self.save_dir / 'benchmark_comparison.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

    def _plot_test_summary(self, stat_results: Dict[str, Any]):
        confidence_levels = stat_results['confidence_levels']
        n_levels = len(confidence_levels)

        n_cols = 3
        fig, axes = plt.subplots(n_levels, n_cols,
                                 figsize=(15, 5.5 * n_levels),
                                 squeeze=False)

        for row, cl in enumerate(confidence_levels):
            level = stat_results['levels'][cl]
            kup = level['kupiec_test']
            chris = level['christoffersen_test']
            ind = chris['independence']
            cc = chris['conditional_coverage']
            pct_label = f"{cl * 100:.0f}%"

            tests = [
                (f'Kupiec POF\n(Coverage {pct_label})', kup['lr_statistic'],
                 kup['critical_value'], kup['p_value'], kup['result']),
                (f'Christoffersen\n(Independence {pct_label})', ind['lr_statistic'],
                 ind['critical_value'], ind['p_value'], ind['result']),
                (f'Conditional\nCoverage ({pct_label})', cc['lr_statistic'],
                 cc['critical_value'], cc['p_value'], cc['result']),
            ]

            for col, (name, lr, crit, pval, result) in enumerate(tests):
                ax = axes[row][col]
                color = '#4CAF50' if result == 'PASS' else '#F44336'
                bg_color = '#E8F5E9' if result == 'PASS' else '#FFEBEE'

                ax.set_xlim(0, 10)
                ax.set_ylim(0, 10)
                ax.set_facecolor(bg_color)

                ax.text(5, 8.5, name, ha='center', va='center', fontsize=12,
                        fontweight='bold', color='#333333')

                ax.text(5, 6.5, result, ha='center', va='center', fontsize=20,
                        fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                                  alpha=0.15, edgecolor=color))

                ax.text(5, 4.5, f'LR Statistic: {lr:.4f}', ha='center',
                        va='center', fontsize=10, color='#555555')
                ax.text(5, 3.5, f'Critical Value: {crit:.4f}', ha='center',
                        va='center', fontsize=10, color='#555555')
                ax.text(5, 2.5, f'p-value: {pval:.6f}', ha='center',
                        va='center', fontsize=10, color='#555555')

                verdict = 'Do not reject H0' if result == 'PASS' else 'Reject H0'
                ax.text(5, 1.2, verdict, ha='center', va='center', fontsize=9,
                        style='italic', color='#777777')

                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)

        fig.suptitle('Statistical Backtesting \u2014 Test Results Summary '
                     f'({", ".join(f"{cl*100:.0f}%" for cl in confidence_levels)})',
                     fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(self.save_dir / 'statistical_test_summary.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)
