import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


class BacktestVisualizer:
    
    def __init__(self,
                 style: str = 'seaborn-v0_8-whitegrid',
                 figsize: Tuple[int, int] = (12, 6),
                 save_path: Optional[str] = None):
        self.figsize = figsize
        self.save_path = Path(save_path) if save_path else None
        
        try:
            plt.style.use(style)
        except:
            try:
                plt.style.use('seaborn-whitegrid')
            except:
                pass
        
        self.colors = {
            'equity': '#2196F3',
            'drawdown': '#F44336',
            'positive': '#4CAF50',
            'negative': '#F44336',
            'fill': '#90CAF9',
            'benchmark': '#9E9E9E'
        }
    
    def plot_equity_curve(self,
                          equity_curve: List[float],
                          dates: Optional[List] = None,
                          benchmark: Optional[List[float]] = None,
                          title: str = "Portfolio Equity Curve",
                          save: bool = False,
                          filename: str = "equity_curve.png") -> plt.Figure:
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if dates is not None and len(dates) == len(equity_curve):
            x = pd.to_datetime(dates)
        else:
            x = np.arange(len(equity_curve))
        
        ax.plot(x, equity_curve, color=self.colors['equity'], 
                linewidth=2, label='Portfolio')
        
        if benchmark is not None and len(benchmark) == len(equity_curve):
            ax.plot(x, benchmark, color=self.colors['benchmark'],
                    linewidth=1.5, linestyle='--', label='Benchmark')
            ax.legend()
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date' if dates is not None else 'Time Period')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis with dollar signs
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Format x-axis for dates
        if dates is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_drawdown(self,
                      equity_curve: List[float],
                      dates: Optional[List] = None,
                      title: str = "Portfolio Drawdown",
                      save: bool = False,
                      filename: str = "drawdown.png") -> plt.Figure:
        fig, ax = plt.subplots(figsize=self.figsize)
        
        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        
        if dates is not None and len(dates) == len(equity_curve):
            x = pd.to_datetime(dates)
        else:
            x = np.arange(len(drawdown))
        
        ax.fill_between(x, drawdown, 0, color=self.colors['drawdown'], 
                        alpha=0.3, label='Drawdown')
        ax.plot(x, drawdown, color=self.colors['drawdown'], linewidth=1)
        
        max_dd = np.min(drawdown)
        ax.axhline(y=max_dd, color=self.colors['drawdown'], 
                   linestyle='--', linewidth=1, alpha=0.7)
        ax.text(x[len(x)//10], max_dd - 1, f'Max DD: {max_dd:.2f}%', 
                color=self.colors['drawdown'], fontsize=10)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date' if dates is not None else 'Time Period')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(top=0)
        
        if dates is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_returns_distribution(self,
                                   returns: List[float],
                                   title: str = "Returns Distribution",
                                   save: bool = False,
                                   filename: str = "returns_dist.png") -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1]))
        
        returns = np.array(returns) * 100
        
        # Histogram
        ax1 = axes[0]
        n, bins, patches = ax1.hist(returns, bins=50, density=True, 
                                     alpha=0.7, color=self.colors['equity'])
        
        for i, patch in enumerate(patches):
            if bins[i] >= 0:
                patch.set_facecolor(self.colors['positive'])
            else:
                patch.set_facecolor(self.colors['negative'])
        
        from scipy import stats
        x = np.linspace(returns.min(), returns.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()),
                 color='black', linewidth=2, linestyle='--', label='Normal')
        
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_title('Returns Histogram')
        ax1.set_xlabel('Return (%)')
        ax1.set_ylabel('Density')
        ax1.legend()
        
        ax2 = axes[1]
        bp = ax2.boxplot(returns, patch_artist=True)
        bp['boxes'][0].set_facecolor(self.colors['equity'])
        ax2.set_title('Returns Box Plot')
        ax2.set_ylabel('Return (%)')
        
        # Add statistics text
        stats_text = (
            f"Mean: {returns.mean():.3f}%\n"
            f"Std: {returns.std():.3f}%\n"
            f"Skew: {stats.skew(returns):.3f}\n"
            f"Kurt: {stats.kurtosis(returns):.3f}"
        )
        ax2.text(1.3, np.median(returns), stats_text, fontsize=9,
                 verticalalignment='center')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_trade_analysis(self,
                             trades: pd.DataFrame,
                             title: str = "Trade Analysis",
                             save: bool = False,
                             filename: str = "trade_analysis.png") -> plt.Figure:
        """
        Plot trade statistics and analysis.
        
        Args:
            trades: DataFrame with trade records
            title: Plot title
            save: Whether to save the plot
            filename: Filename for saving
            
        Returns:
            Matplotlib Figure
        """
        if trades.empty:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No trades to display', ha='center', va='center')
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0], self.figsize[1] * 1.5))
        
        # Trade volume over time
        ax1 = axes[0, 0]
        if 'timestamp' in trades.columns:
            trades['trade_value'] = trades['quantity'].abs() * trades['price']
            ax1.bar(range(len(trades)), trades['trade_value'], 
                    color=self.colors['equity'], alpha=0.7)
            ax1.set_xlabel('Trade #')
            ax1.set_ylabel('Trade Value ($)')
            ax1.set_title('Trade Values Over Time')
        
        # Buy vs Sell distribution
        ax2 = axes[0, 1]
        if 'action' in trades.columns:
            action_counts = trades['action'].value_counts()
            colors = [self.colors['positive'] if a == 'BUY' else self.colors['negative'] 
                      for a in action_counts.index]
            ax2.pie(action_counts, labels=action_counts.index, autopct='%1.1f%%',
                    colors=colors, startangle=90)
            ax2.set_title('Buy vs Sell Distribution')
        
        # Commission costs
        ax3 = axes[1, 0]
        if 'commission' in trades.columns:
            cumulative_comm = trades['commission'].cumsum()
            ax3.plot(range(len(trades)), cumulative_comm, 
                     color=self.colors['drawdown'], linewidth=2)
            ax3.fill_between(range(len(trades)), cumulative_comm, 
                             alpha=0.3, color=self.colors['drawdown'])
            ax3.set_xlabel('Trade #')
            ax3.set_ylabel('Cumulative Commission ($)')
            ax3.set_title('Cumulative Transaction Costs')
        
        # Position over time
        ax4 = axes[1, 1]
        if 'position_after' in trades.columns:
            positions = trades['position_after'].values
            colors = [self.colors['positive'] if p > 0 else 
                      (self.colors['negative'] if p < 0 else 'gray') 
                      for p in positions]
            ax4.bar(range(len(trades)), positions, color=colors, alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax4.set_xlabel('Trade #')
            ax4.set_ylabel('Position')
            ax4.set_title('Position After Each Trade')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_summary_dashboard(self,
                                results: Dict[str, Any],
                                save: bool = False,
                                filename: str = "dashboard.png") -> plt.Figure:
        fig = plt.figure(figsize=(self.figsize[0] * 1.5, self.figsize[1] * 2))
        
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :])
        equity_curve = results.get('equity_curve', [])
        ax1.plot(equity_curve, color=self.colors['equity'], linewidth=2)
        ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[1, 0])
        equity = np.array(equity_curve)
        if len(equity) > 0:
            peak = np.maximum.accumulate(equity)
            drawdown = (equity - peak) / peak * 100
            ax2.fill_between(range(len(drawdown)), drawdown, 0, 
                             color=self.colors['drawdown'], alpha=0.3)
            ax2.plot(drawdown, color=self.colors['drawdown'], linewidth=1)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_ylim(top=0)
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[1, 1])
        returns = results.get('returns', [])
        if len(returns) > 0:
            returns_pct = np.array(returns) * 100
            ax3.hist(returns_pct, bins=30, color=self.colors['equity'], 
                     alpha=0.7, edgecolor='white')
        ax3.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.axis('off')
        metrics = results.get('metrics', {})
        table_data = [
            ['Total Return', f"{metrics.get('total_return_pct', 0):.2f}%"],
            ['Annualized Return', f"{metrics.get('annualized_return', 0)*100:.2f}%"],
            ['Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"],
            ['Max Drawdown', f"{metrics.get('max_drawdown_pct', 0):.2f}%"],
            ['Win Rate', f"{metrics.get('win_rate_pct', 0):.2f}%"],
            ['# Trades', f"{metrics.get('num_trades', 0)}"]
        ]
        table = ax4.table(cellText=table_data, 
                          colLabels=['Metric', 'Value'],
                          loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Performance Summary', fontsize=12, fontweight='bold', y=0.95)
        
        # Execution stats (bottom right)
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        execution = results.get('execution', {})
        portfolio = results.get('portfolio', {})
        exec_data = [
            ['Initial Capital', f"${portfolio.get('initial_capital', 0):,.2f}"],
            ['Final Value', f"${portfolio.get('final_value', 0):,.2f}"],
            ['Total Commission', f"${execution.get('total_commission', 0):,.2f}"],
            ['Total Slippage', f"${execution.get('total_slippage', 0):,.2f}"],
            ['Total Trades', f"{execution.get('total_trades', 0)}"],
            ['Rejected Trades', f"{execution.get('rejected_trades', 0)}"]
        ]
        table2 = ax5.table(cellText=exec_data,
                           colLabels=['Metric', 'Value'],
                           loc='center', cellLoc='left')
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1.2, 1.5)
        ax5.set_title('Execution Summary', fontsize=12, fontweight='bold', y=0.95)
        
        fig.suptitle('Backtesting Results Dashboard', fontsize=16, fontweight='bold', y=1.02)
        
        if save:
            self._save_figure(fig, filename)
        
        return fig
    
    def _save_figure(self, fig: plt.Figure, filename: str) -> str:
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            filepath = self.save_path / filename
        else:
            filepath = Path(filename)
        
        fig.savefig(filepath, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Plot saved to: {filepath}")
        return str(filepath)
    
    def save_all_plots(self, results: Dict[str, Any]) -> List[str]:
        saved_files = []
        
        fig = self.plot_equity_curve(results.get('equity_curve', []), save=True)
        saved_files.append('equity_curve.png')
        plt.close(fig)
        
        fig = self.plot_drawdown(results.get('equity_curve', []), save=True)
        saved_files.append('drawdown.png')
        plt.close(fig)
        
        returns = results.get('returns', [])
        if len(returns) > 0:
            fig = self.plot_returns_distribution(returns, save=True)
            saved_files.append('returns_dist.png')
            plt.close(fig)
        
        fig = self.plot_summary_dashboard(results, save=True)
        saved_files.append('dashboard.png')
        plt.close(fig)
        
        return saved_files
