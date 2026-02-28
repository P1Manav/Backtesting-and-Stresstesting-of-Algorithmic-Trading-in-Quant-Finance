import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class RobustnessVisualizer:

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

    # Generate all robustness validation charts.
    def generate_all(self, robustness_results: Dict[str, Any]) -> None:
        wf = robustness_results.get('walk_forward', {})
        bs = robustness_results.get('bootstrap', {})
        mc = robustness_results.get('monte_carlo', {})

        if wf.get('fold_results'):
            self._plot_walk_forward(wf)
        if 'sharpe_ratio' in bs:
            self._plot_bootstrap(bs)
        if 'null_sharpe' in mc:
            self._plot_monte_carlo(mc)
        self._plot_robustness_summary(robustness_results)

        print(f"  [OK] Robustness charts saved to: {self.save_dir}")

    # Bar chart of OOS Sharpe, Return, and Max Drawdown per fold.
    def _plot_walk_forward(self, wf: Dict[str, Any]):
        folds = wf['fold_results']
        agg = wf['aggregate']
        n = len(folds)

        fold_labels = [f"Fold {f['fold']}" for f in folds]
        sharpe_vals = [f['sharpe_ratio'] for f in folds]
        return_vals = [f['total_return_pct'] for f in folds]
        mdd_vals = [f['max_drawdown_pct'] for f in folds]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        ax = axes[0]
        colors_s = ['#4CAF50' if v > 0 else '#F44336' for v in sharpe_vals]
        bars = ax.bar(fold_labels, sharpe_vals, color=colors_s, alpha=0.85,
                      edgecolor='white', linewidth=0.8)
        ax.axhline(y=agg['mean_sharpe'], color='#2196F3', linestyle='--',
                   linewidth=2, label=f"Mean = {agg['mean_sharpe']:.3f}")
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        ax.set_title('OOS Sharpe Ratio per Fold', fontsize=13, fontweight='bold')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, sharpe_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center',
                    va='bottom' if val >= 0 else 'top', fontsize=9)

        ax = axes[1]
        colors_r = ['#4CAF50' if v > 0 else '#F44336' for v in return_vals]
        bars = ax.bar(fold_labels, return_vals, color=colors_r, alpha=0.85,
                      edgecolor='white', linewidth=0.8)
        ax.axhline(y=agg['mean_return_pct'], color='#2196F3', linestyle='--',
                   linewidth=2, label=f"Mean = {agg['mean_return_pct']:.2f}%")
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        ax.set_title('OOS Total Return (%) per Fold', fontsize=13, fontweight='bold')
        ax.set_ylabel('Return (%)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, return_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.2f}%', ha='center',
                    va='bottom' if val >= 0 else 'top', fontsize=9)

        ax = axes[2]
        bars = ax.bar(fold_labels, mdd_vals, color='#FF5722', alpha=0.75,
                      edgecolor='white', linewidth=0.8)
        ax.axhline(y=agg['mean_max_drawdown_pct'], color='#2196F3',
                   linestyle='--', linewidth=2,
                   label=f"Mean = {agg['mean_max_drawdown_pct']:.2f}%")
        ax.set_title('OOS Max Drawdown (%) per Fold', fontsize=13, fontweight='bold')
        ax.set_ylabel('Max Drawdown (%)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, mdd_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.2f}%', ha='center', va='top', fontsize=9)

        consistency = wf.get('consistency_ratio', 0)
        result = wf.get('result', 'N/A')
        badge_color = '#4CAF50' if result == 'CONSISTENT' else '#F44336'
        fig.suptitle(
            f'Walk-Forward Analysis  —  Consistency: {consistency:.0%}  [{result}]',
            fontsize=15, fontweight='bold', y=1.02, color=badge_color)

        plt.tight_layout()
        fig.savefig(self.save_dir / 'walk_forward_analysis.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

    # Horizontal CI bars with observed values for bootstrap metrics.
    def _plot_bootstrap(self, bs: Dict[str, Any]):
        metrics = [
            ('Sharpe Ratio', bs['sharpe_ratio']),
            ('Total Return (%)', bs['total_return_pct']),
            ('Max Drawdown (%)', bs['max_drawdown_pct']),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        cl_pct = f"{bs['confidence_level']:.0%}"

        for idx, (label, data) in enumerate(metrics):
            ax = axes[idx]
            obs = data['observed']
            mean = data['mean']
            ci_lo = data['ci_lower']
            ci_hi = data['ci_upper']

            ax.barh(0, ci_hi - ci_lo, left=ci_lo, height=0.4,
                    color='#E3F2FD', edgecolor='#1565C0', linewidth=1.5,
                    alpha=0.8, label=f'{cl_pct} CI')

            ax.axvline(x=mean, color='#1565C0', linestyle='--', linewidth=2,
                       label=f'Bootstrap Mean = {mean:.4f}')

            ax.plot(obs, 0, marker='D', markersize=14, color='#F44336',
                    markeredgecolor='white', markeredgewidth=2, zorder=5,
                    label=f'Observed = {obs:.4f}')

            if label != 'Max Drawdown (%)':
                ax.axvline(x=0, color='gray', linestyle=':', linewidth=1,
                           alpha=0.5)

            ax.set_title(label, fontsize=13, fontweight='bold')
            ax.set_yticks([])
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3, axis='x')

            ax.text(ci_lo, -0.35, f'{ci_lo:.4f}', ha='center', fontsize=8,
                    color='#1565C0')
            ax.text(ci_hi, -0.35, f'{ci_hi:.4f}', ha='center', fontsize=8,
                    color='#1565C0')

        result = bs.get('result', 'N/A')
        badge_color = '#4CAF50' if result == 'ROBUST' else '#F44336'
        fig.suptitle(
            f'Bootstrap Resampling ({bs["n_bootstrap"]} replications, {cl_pct} CI)  [{result}]',
            fontsize=15, fontweight='bold', y=1.02, color=badge_color)

        plt.tight_layout()
        fig.savefig(self.save_dir / 'bootstrap_resampling.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

    # Histogram of null Sharpe distribution with observed marker.
    def _plot_monte_carlo(self, mc: Dict[str, Any]):
        ns = mc['null_sharpe']
        nr = mc['null_return_pct']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        n_sim = mc['n_simulations']
        null_sharpe_samples = np.random.normal(ns['mean'], max(ns['std'], 1e-6),
                                               n_sim)
        ax1.hist(null_sharpe_samples, bins=40, color='#90CAF9', edgecolor='white',
                 alpha=0.8, density=True, label='Null Distribution')

        ax1.axvline(x=mc['observed_sharpe'], color='#F44336', linewidth=2.5,
                    linestyle='-', label=f"Observed = {mc['observed_sharpe']:.4f}")

        ax1.axvline(x=ns['percentile_5'], color='#FF9800', linewidth=1.5,
                    linestyle='--', alpha=0.7, label=f"5th %ile = {ns['percentile_5']:.4f}")
        ax1.axvline(x=ns['percentile_95'], color='#FF9800', linewidth=1.5,
                    linestyle='--', alpha=0.7, label=f"95th %ile = {ns['percentile_95']:.4f}")

        ax1.set_title('Sharpe Ratio — Null Distribution', fontsize=13,
                      fontweight='bold')
        ax1.set_xlabel('Sharpe Ratio')
        ax1.set_ylabel('Density')
        ax1.legend(fontsize=9, loc='upper right')
        ax1.grid(True, alpha=0.3)

        p_s = mc['p_value_sharpe']
        sig_s = 'Significant' if p_s < 0.05 else 'Not Significant'
        color_s = '#4CAF50' if p_s < 0.05 else '#F44336'
        ax1.text(0.02, 0.95, f'p-value = {p_s:.4f}\n({sig_s})',
                 transform=ax1.transAxes, fontsize=10, fontweight='bold',
                 color=color_s, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor=color_s, alpha=0.9))

        null_return_samples = np.random.normal(nr['mean'], max(nr['std'], 1e-6),
                                               n_sim)
        ax2.hist(null_return_samples, bins=40, color='#A5D6A7', edgecolor='white',
                 alpha=0.8, density=True, label='Null Distribution')

        observed_ret = mc['observed_return_pct']
        ax2.axvline(x=observed_ret, color='#F44336', linewidth=2.5,
                    linestyle='-', label=f'Observed = {observed_ret:.2f}%')

        ax2.axvline(x=nr['percentile_5'], color='#FF9800', linewidth=1.5,
                    linestyle='--', alpha=0.7, label=f"5th %ile = {nr['percentile_5']:.2f}%")
        ax2.axvline(x=nr['percentile_95'], color='#FF9800', linewidth=1.5,
                    linestyle='--', alpha=0.7, label=f"95th %ile = {nr['percentile_95']:.2f}%")

        ax2.set_title('Total Return (%) — Null Distribution', fontsize=13,
                      fontweight='bold')
        ax2.set_xlabel('Total Return (%)')
        ax2.set_ylabel('Density')
        ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(True, alpha=0.3)

        p_r = mc['p_value_return']
        sig_r = 'Significant' if p_r < 0.05 else 'Not Significant'
        color_r = '#4CAF50' if p_r < 0.05 else '#F44336'
        ax2.text(0.02, 0.95, f'p-value = {p_r:.4f}\n({sig_r})',
                 transform=ax2.transAxes, fontsize=10, fontweight='bold',
                 color=color_r, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor=color_r, alpha=0.9))

        result = mc.get('result', 'N/A')
        badge_color = '#4CAF50' if result == 'SIGNIFICANT' else '#F44336'
        fig.suptitle(
            f'Monte Carlo Permutation Test ({mc["n_simulations"]} simulations)  [{result}]',
            fontsize=15, fontweight='bold', y=1.02, color=badge_color)

        plt.tight_layout()
        fig.savefig(self.save_dir / 'monte_carlo_test.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

    # Single-page dashboard summarising all three robustness tests.
    def _plot_robustness_summary(self, rob: Dict[str, Any]):
        wf = rob.get('walk_forward', {})
        bs = rob.get('bootstrap', {})
        mc = rob.get('monte_carlo', {})
        overall = rob.get('overall_verdict', 'N/A')

        tests = [
            ('Walk-Forward\nAnalysis',
             wf.get('result', 'N/A'),
             self._wf_details(wf)),
            ('Bootstrap\nResampling',
             bs.get('result', 'N/A'),
             self._bs_details(bs)),
            ('Monte Carlo\nPermutation',
             mc.get('result', 'N/A'),
             self._mc_details(mc)),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 7))

        for idx, (name, result, details) in enumerate(tests):
            ax = axes[idx]
            is_pass = result in ('CONSISTENT', 'ROBUST', 'SIGNIFICANT')
            color = '#4CAF50' if is_pass else '#F44336'
            bg_color = '#E8F5E9' if is_pass else '#FFEBEE'

            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_facecolor(bg_color)

            ax.text(5, 9, name, ha='center', va='center', fontsize=14,
                    fontweight='bold', color='#333333')

            ax.text(5, 7.2, result, ha='center', va='center', fontsize=18,
                    fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=color,
                              alpha=0.15, edgecolor=color, linewidth=2))

            y_pos = 5.5
            for line in details:
                ax.text(5, y_pos, line, ha='center', va='center',
                        fontsize=9.5, color='#444444')
                y_pos -= 0.85

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)

        n_pass = sum(1 for _, r, _ in tests
                     if r in ('CONSISTENT', 'ROBUST', 'SIGNIFICANT'))
        overall_color = ('#4CAF50' if n_pass == 3
                         else '#FF9800' if n_pass >= 2
                         else '#F44336')
        fig.suptitle(
            f'Robustness Validation Summary  —  {overall}',
            fontsize=14, fontweight='bold', y=1.03, color=overall_color)

        plt.tight_layout()
        fig.savefig(self.save_dir / 'robustness_summary.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def _wf_details(wf: Dict[str, Any]) -> list:
        if not wf.get('aggregate'):
            return ['Insufficient data']
        a = wf['aggregate']
        return [
            f"Folds: {a.get('n_folds', '?')}",
            f"Mean OOS Sharpe: {a.get('mean_sharpe', 0):.4f} ± {a.get('std_sharpe', 0):.4f}",
            f"Mean OOS Return: {a.get('mean_return_pct', 0):.2f}%",
            f"Mean Max DD: {a.get('mean_max_drawdown_pct', 0):.2f}%",
            f"Consistency: {wf.get('consistency_ratio', 0):.0%}",
        ]

    @staticmethod
    def _bs_details(bs: Dict[str, Any]) -> list:
        if 'sharpe_ratio' not in bs:
            return ['Insufficient data']
        sr = bs['sharpe_ratio']
        tr = bs['total_return_pct']
        return [
            f"Replications: {bs.get('n_bootstrap', '?')}",
            f"Sharpe CI: [{sr['ci_lower']:.4f}, {sr['ci_upper']:.4f}]",
            f"Return CI: [{tr['ci_lower']:.2f}%, {tr['ci_upper']:.2f}%]",
            f"Observed Sharpe: {sr['observed']:.4f}",
            f"CI Level: {bs.get('confidence_level', 0.95):.0%}",
        ]

    @staticmethod
    def _mc_details(mc: Dict[str, Any]) -> list:
        if 'null_sharpe' not in mc:
            return ['Insufficient data']
        return [
            f"Simulations: {mc.get('n_simulations', '?')}",
            f"Observed Sharpe: {mc.get('observed_sharpe', 0):.4f}",
            f"p-value (Sharpe): {mc.get('p_value_sharpe', 1):.4f}",
            f"p-value (Return): {mc.get('p_value_return', 1):.4f}",
            f"Null Sharpe μ: {mc['null_sharpe'].get('mean', 0):.4f}",
        ]
