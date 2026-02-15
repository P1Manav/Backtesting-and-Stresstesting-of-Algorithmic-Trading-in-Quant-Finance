"""
Report Generator Module
Generates comprehensive reports from backtesting results.
"""

import json
import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime


class ReportGenerator:
    """
    Generates reports and exports for backtesting results.
    
    Output formats:
    - Console summary
    - CSV result files
    - JSON report
    - Text report
    """
    
    def __init__(self, output_path: Optional[str] = None):
        """
        Initialize the Report Generator.
        
        Args:
            output_path: Path to save reports
        """
        if output_path is None:
            self.output_path = Path(__file__).parent.parent / "results"
        else:
            self.output_path = Path(output_path)
        
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def generate_full_report(self, 
                              results: Dict[str, Any],
                              save_files: bool = True) -> Dict[str, Any]:
        """
        Generate a complete evaluation report.
        
        Args:
            results: Backtesting results dictionary
            save_files: Whether to save report files
            
        Returns:
            Compiled report dictionary
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': self._generate_summary(results),
            'metrics': results.get('metrics', {}),
            'portfolio': results.get('portfolio', {}),
            'execution': results.get('execution', {}),
            'config': results.get('config', {}),
            'data_info': results.get('data_info', {})
        }
        
        if save_files:
            self._save_all_files(results, report)
        
        return report
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a human-readable summary."""
        metrics = results.get('metrics', {})
        portfolio = results.get('portfolio', {})
        
        return {
            'performance': {
                'total_return': f"{metrics.get('total_return_pct', 0):.2f}%",
                'annualized_return': f"{metrics.get('annualized_return', 0)*100:.2f}%",
                'sharpe_ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'max_drawdown': f"{metrics.get('max_drawdown_pct', 0):.2f}%"
            },
            'portfolio': {
                'initial_capital': f"${portfolio.get('initial_capital', 0):,.2f}",
                'final_value': f"${portfolio.get('final_value', 0):,.2f}",
                'profit_loss': f"${portfolio.get('final_value', 0) - portfolio.get('initial_capital', 0):,.2f}"
            },
            'trading': {
                'total_trades': metrics.get('num_trades', 0),
                'win_rate': f"{metrics.get('win_rate_pct', 0):.2f}%",
                'turnover': f"{metrics.get('turnover', 0):.4f}"
            }
        }
    
    def _save_all_files(self, results: Dict[str, Any], report: Dict[str, Any]) -> None:
        """Save all report files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_path = self.output_path / f"report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"JSON report saved: {json_path}")
        
        # Save equity curve CSV
        equity_df = pd.DataFrame({
            'period': range(len(results.get('equity_curve', []))),
            'portfolio_value': results.get('equity_curve', [])
        })
        equity_path = self.output_path / f"equity_curve_{timestamp}.csv"
        equity_df.to_csv(equity_path, index=False)
        print(f"Equity curve saved: {equity_path}")
        
        # Save trade log CSV
        trades = results.get('trade_log', [])
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_path = self.output_path / f"trades_{timestamp}.csv"
            trades_df.to_csv(trades_path, index=False)
            print(f"Trade log saved: {trades_path}")
        
        # Save timestep log CSV
        timestep_log = results.get('timestep_log', [])
        if timestep_log:
            log_df = pd.DataFrame(timestep_log)
            log_path = self.output_path / f"timestep_log_{timestamp}.csv"
            log_df.to_csv(log_path, index=False)
            print(f"Timestep log saved: {log_path}")
        
        # Save text summary
        text_report = self.generate_text_report(results)
        text_path = self.output_path / f"summary_{timestamp}.txt"
        with open(text_path, 'w') as f:
            f.write(text_report)
        print(f"Text summary saved: {text_path}")
    
    def generate_text_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a formatted text report.
        
        Args:
            results: Backtesting results dictionary
            
        Returns:
            Formatted text report string
        """
        metrics = results.get('metrics', {})
        portfolio = results.get('portfolio', {})
        execution = results.get('execution', {})
        config = results.get('config', {})
        data_info = results.get('data_info', {})
        
        lines = [
            "=" * 70,
            "                    BACKTESTING EVALUATION REPORT",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "-" * 70,
            "CONFIGURATION",
            "-" * 70,
            f"  Initial Capital:        ${config.get('initial_capital', 0):>20,.2f}",
            f"  Commission Rate:        {config.get('commission_rate', 0)*100:>20.3f}%",
            f"  Slippage:               {config.get('slippage', 0)*100:>20.3f}%",
            f"  Window Size:            {config.get('window_size', 0):>20}",
            f"  Allow Short:            {str(config.get('allow_short', False)):>20}",
            "",
            "-" * 70,
            "DATA INFORMATION",
            "-" * 70,
            f"  Start Date:             {str(data_info.get('start_date', 'N/A')):>20}",
            f"  End Date:               {str(data_info.get('end_date', 'N/A')):>20}",
            f"  Total Periods:          {data_info.get('num_periods', 0):>20}",
            f"  Simulated Periods:      {data_info.get('simulated_periods', 0):>20}",
            "",
            "-" * 70,
            "PORTFOLIO SUMMARY",
            "-" * 70,
            f"  Initial Capital:        ${portfolio.get('initial_capital', 0):>20,.2f}",
            f"  Final Value:            ${portfolio.get('final_value', 0):>20,.2f}",
            f"  Total Return:           {portfolio.get('total_return_pct', 0):>20.2f}%",
            f"  Cash Balance:           ${portfolio.get('cash', 0):>20,.2f}",
            f"  Position Value:         ${portfolio.get('position_value', 0):>20,.2f}",
            f"  Realized P&L:           ${portfolio.get('realized_pnl', 0):>20,.2f}",
            f"  Unrealized P&L:         ${portfolio.get('unrealized_pnl', 0):>20,.2f}",
            "",
            "-" * 70,
            "PERFORMANCE METRICS",
            "-" * 70,
            f"  Total Return:           {metrics.get('total_return_pct', 0):>20.2f}%",
            f"  Annualized Return:      {metrics.get('annualized_return', 0)*100:>20.2f}%",
            f"  Annualized Volatility:  {metrics.get('annualized_volatility', 0)*100:>20.2f}%",
            f"  Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):>20.2f}",
            f"  Maximum Drawdown:       {metrics.get('max_drawdown_pct', 0):>20.2f}%",
            f"  Win Rate:               {metrics.get('win_rate_pct', 0):>20.2f}%",
            f"  Turnover:               {metrics.get('turnover', 0):>20.4f}",
            "",
            "-" * 70,
            "EXECUTION STATISTICS",
            "-" * 70,
            f"  Total Trades:           {execution.get('total_trades', 0):>20}",
            f"  Rejected Trades:        {execution.get('rejected_trades', 0):>20}",
            f"  Total Volume:           ${execution.get('total_volume', 0):>20,.2f}",
            f"  Total Commission:       ${execution.get('total_commission', 0):>20,.2f}",
            f"  Total Slippage:         ${execution.get('total_slippage', 0):>20,.2f}",
            f"  Avg Commission/Trade:   ${execution.get('average_commission', 0):>20,.2f}",
            "",
            "=" * 70,
            "                          END OF REPORT",
            "=" * 70
        ]
        
        return "\n".join(lines)
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a summary to console.
        
        Args:
            results: Backtesting results dictionary
        """
        print(self.generate_text_report(results))
    
    def generate_metrics_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate a metrics table as DataFrame.
        
        Args:
            results: Backtesting results dictionary
            
        Returns:
            DataFrame with metrics
        """
        metrics = results.get('metrics', {})
        
        data = [
            ('Total Return', f"{metrics.get('total_return_pct', 0):.2f}%"),
            ('Annualized Return', f"{metrics.get('annualized_return', 0)*100:.2f}%"),
            ('Annualized Volatility', f"{metrics.get('annualized_volatility', 0)*100:.2f}%"),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
            ('Maximum Drawdown', f"{metrics.get('max_drawdown_pct', 0):.2f}%"),
            ('Win Rate', f"{metrics.get('win_rate_pct', 0):.2f}%"),
            ('Number of Trades', f"{metrics.get('num_trades', 0)}"),
            ('Turnover', f"{metrics.get('turnover', 0):.4f}")
        ]
        
        return pd.DataFrame(data, columns=['Metric', 'Value'])
    
    def generate_trade_statistics(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate trade statistics table.
        
        Args:
            results: Backtesting results dictionary
            
        Returns:
            DataFrame with trade statistics
        """
        trades = results.get('trade_log', [])
        
        if not trades:
            return pd.DataFrame()
        
        trades_df = pd.DataFrame(trades)
        
        # Calculate statistics
        buy_trades = trades_df[trades_df['action'] == 'BUY'] if 'action' in trades_df.columns else pd.DataFrame()
        sell_trades = trades_df[trades_df['action'] == 'SELL'] if 'action' in trades_df.columns else pd.DataFrame()
        
        stats = {
            'Total Trades': len(trades_df),
            'Buy Trades': len(buy_trades),
            'Sell Trades': len(sell_trades),
            'Avg Trade Value': trades_df['quantity'].abs().mean() * trades_df['price'].mean() if 'quantity' in trades_df.columns else 0,
            'Total Commission': trades_df['commission'].sum() if 'commission' in trades_df.columns else 0,
            'Total Slippage': trades_df['slippage_cost'].sum() if 'slippage_cost' in trades_df.columns else 0
        }
        
        return pd.DataFrame(list(stats.items()), columns=['Statistic', 'Value'])
    
    def export_to_excel(self, 
                        results: Dict[str, Any],
                        filename: Optional[str] = None) -> str:
        """
        Export results to Excel file.
        
        Args:
            results: Backtesting results dictionary
            filename: Optional filename
            
        Returns:
            Path to saved Excel file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{timestamp}.xlsx"
        
        filepath = self.output_path / filename
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Metrics sheet
            metrics_df = self.generate_metrics_table(results)
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Trade statistics sheet
            trade_stats = self.generate_trade_statistics(results)
            if not trade_stats.empty:
                trade_stats.to_excel(writer, sheet_name='Trade Statistics', index=False)
            
            # Equity curve sheet
            equity_df = pd.DataFrame({
                'Period': range(len(results.get('equity_curve', []))),
                'Portfolio Value': results.get('equity_curve', [])
            })
            equity_df.to_excel(writer, sheet_name='Equity Curve', index=False)
            
            # Trade log sheet
            trades = results.get('trade_log', [])
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_df.to_excel(writer, sheet_name='Trade Log', index=False)
            
            # Configuration sheet
            config = results.get('config', {})
            config_df = pd.DataFrame(list(config.items()), columns=['Parameter', 'Value'])
            config_df.to_excel(writer, sheet_name='Configuration', index=False)
        
        print(f"Excel report saved: {filepath}")
        return str(filepath)
