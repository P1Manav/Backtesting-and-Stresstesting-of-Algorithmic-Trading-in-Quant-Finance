"""Stress testing visualization"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
class StressTestingVisualizer:
    """Generate visualizations for stress testing results."""
    def __init__(self, output_dir: str = "./stress_test_results"):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')
    def plot_return_degradation(self, degradation_reports: List,
    """Generate plots"""
                                filename: str = "return_degradation.png"):
        """Plot return degradation across scenarios."""
        scenarios = [r.scenario_name for r in degradation_reports]
        return_drops = [r.return_drop for r in degradation_reports]
        colors = ['green' if not drop > 0 else 'red' for drop in return_drops]
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(range(len(scenarios)), return_drops, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Scenario', fontsize=12)
        ax.set_ylabel('Return Drop (%)', fontsize=12)
        ax.set_title('Performance Degradation Across Stress Scenarios', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    def plot_sharpe_comparison(self, degradation_reports: List,
    """Generate plots"""
                               filename: str = "sharpe_comparison.png"):
        """Plot Sharpe ratio comparison."""
        scenarios = [r.scenario_name for r in degradation_reports]
        baseline_sharpe = [r.baseline_sharpe for r in degradation_reports]
        stressed_sharpe = [r.stressed_sharpe for r in degradation_reports]
        x = np.arange(len(scenarios))
        width = 0.35
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width/2, baseline_sharpe, width, label='Baseline', alpha=0.8)
        ax.bar(x + width/2, stressed_sharpe, width, label='Stressed', alpha=0.8)
        ax.set_xlabel('Scenario', fontsize=12)
        ax.set_ylabel('Sharpe Ratio', fontsize=12)
        ax.set_title('Sharpe Ratio: Baseline vs Stressed', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    def plot_drawdown_comparison(self, degradation_reports: List,
    """Generate plots"""
                                 filename: str = "drawdown_comparison.png"):
        """Plot maximum drawdown comparison."""
        scenarios = [r.scenario_name for r in degradation_reports]
        baseline_dd = [r.baseline_max_dd for r in degradation_reports]
        stressed_dd = [r.stressed_max_dd for r in degradation_reports]
        x = np.arange(len(scenarios))
        width = 0.35
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width/2, baseline_dd, width, label='Baseline', alpha=0.8, color='blue')
        ax.bar(x + width/2, stressed_dd, width, label='Stressed', alpha=0.8, color='red')
        ax.set_xlabel('Scenario', fontsize=12)
        ax.set_ylabel('Max Drawdown (%)', fontsize=12)
        ax.set_title('Maximum Drawdown: Baseline vs Stressed', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    def plot_robustness_heatmap(self, degradation_reports: List,
    """Generate plots"""
                                filename: str = "robustness_heatmap.png"):
        """Plot robustness heatmap."""
        scenarios = [r.scenario_name for r in degradation_reports]
        metrics = {
            'Return\nRetention': [r.return_drop / r.baseline_return for r in degradation_reports],
            'Sharpe\nRetention': [r.sharpe_drop / r.baseline_sharpe if r.baseline_sharpe != 0 else 0 
                                for r in degradation_reports],
            'DD\nIncrease': [r.max_dd_increase / abs(r.baseline_max_dd) if r.baseline_max_dd != 0 else 0 
                           for r in degradation_reports],
        }
        data = np.array([metrics['Return\nRetention'],
                        metrics['Sharpe\nRetention'],
                        metrics['DD\nIncrease']])
        fig, ax = plt.subplots(figsize=(14, 4))
        im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.set_yticklabels(metrics.keys())
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Degradation Level', rotation=270, labelpad=20)
        ax.set_title('Robustness Heatmap (Red=Worse)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    def plot_scenario_type_summary(self, comparison_df: pd.DataFrame,
    """Generate plots"""
                                   filename: str = "scenario_type_summary.png"):
        """Plot summary by scenario type."""
        summary = comparison_df.groupby('Type').agg({
            'Return Retention (%)': 'mean',
            'Sharpe Retention (%)': 'mean',
            'Max DD Ratio': 'mean'
        })
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        summary['Return Retention (%)'].plot(kind='bar', ax=axes[0], color='steelblue', alpha=0.7)
        axes[0].set_title('Avg Return Retention by Type', fontweight='bold')
        axes[0].set_ylabel('Return Retention (%)')
        axes[0].set_xlabel('')
        axes[0].axhline(y=100, color='red', linestyle='--', alpha=0.5)
        axes[0].tick_params(axis='x', rotation=45)
        summary['Sharpe Retention (%)'].plot(kind='bar', ax=axes[1], color='green', alpha=0.7)
        axes[1].set_title('Avg Sharpe Retention by Type', fontweight='bold')
        axes[1].set_ylabel('Sharpe Retention (%)')
        axes[1].set_xlabel('')
        axes[1].axhline(y=100, color='red', linestyle='--', alpha=0.5)
        axes[1].tick_params(axis='x', rotation=45)
        summary['Max DD Ratio'].plot(kind='bar', ax=axes[2], color='orange', alpha=0.7)
        axes[2].set_title('Avg Max DD Ratio by Type', fontweight='bold')
        axes[2].set_ylabel('Max DD Ratio')
        axes[2].set_xlabel('')
        axes[2].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        axes[2].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    def plot_robustness_scores(self, robustness_report,
                               filename: str = "robustness_scores.png"):
        """Plot robustness indicator scores."""
        scores = {
            'Resilience': robustness_report.resilience_score,
            'Stability': robustness_report.stability_score,
            'Drawdown\nResilience': robustness_report.drawdown_resilience_score,
            'Overall': robustness_report.overall_robustness_score
        }
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if v >= 70 else 'orange' if v >= 50 else 'red' 
                 for v in scores.values()]
        bars = ax.bar(scores.keys(), scores.values(), color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Excellent (70)')
        ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Good (50)')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontweight='bold')
        ax.set_ylabel('Score (0-100)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_title('Robustness Indicator Scores', fontsize=14, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path

