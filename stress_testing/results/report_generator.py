"""HTML report generation"""

from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from stress_testing.evaluation import RobustnessMetricsReport, PerformanceDegradationReport
class ReportGenerator:
    """Generate HTML and text reports from stress test results."""
    def __init__(self, output_dir: str = "./stress_test_results"):
        """Initialize report generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    def generate_html_report(self,
                            baseline_metrics: Dict[str, float],
                            robustness_report: RobustnessMetricsReport,
                            degradation_reports: list[PerformanceDegradationReport],
                            comparison_df,
                            filename: str = "stress_test_report.html") -> Path:
"""Generate comprehensive HTML report. Args: baseline_metrics: Baseline performance metrics robustness_report: Robustness analysis report degradation_reports: List of degradation reports comparison_df: Comparison DataFrame filename: Output filename Returns: Path to generated report"""
        html_content = self._generate_html_content(
            baseline_metrics,
            robustness_report,
            degradation_reports,
            comparison_df
        )
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(html_content)
        return output_path
    def _generate_html_content(self, baseline_metrics, robustness_report,
                              degradation_reports, comparison_df) -> str:
        """Generate HTML content."""
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
"""<!DOCTYPE html> <html> <head> <title>Stress Testing Report</title> <style> body {{ font-family: Arial, sans-serif; margin: 20px; background-color: }} .header {{ background-color: color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }} .section {{ background-color: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }} .subsection {{ margin-top: 15px; padding: 15px; background-color: border-left: 4px solid }} h1 {{ margin: 0; color: white; }} h2 {{ color: border-bottom: 2px solid padding-bottom: 10px; }} h3 {{ color: }} table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }} th, td {{ border: 1px solid padding: 12px; text-align: left; }} th {{ background-color: color: white; }} tr:nth-child(even) {{ background-color: }} .metric {{ display: inline-block; width: 23%; margin: 1%; padding: 15px; background-color: border-radius: 5px; text-align: center; }} .metric-value {{ font-size: 24px; font-weight: bold; color: }} .metric-label {{ font-size: 12px; color: margin-top: 5px; }} .score-excellent {{ color: font-weight: bold; }} .score-good {{ color: font-weight: bold; }} .score-poor {{ color: font-weight: bold; }} .footer {{ text-align: center; color: margin-top: 30px; padding-top: 20px; border-top: 1px solid }} </style> </head> <body> <div class="header"> <h1>Stress Testing Report</h1> <p>Generated: {timestamp}</p> </div>"""
        html += '<div class="section">'
        html += '<h2>Executive Summary</h2>'
        html += f'<p>Total Scenarios Tested: <strong>{robustness_report.num_scenarios}</strong></p>'
        html += f'<p>Profitable Scenarios: <strong>{robustness_report.profitable_scenarios}/{robustness_report.num_scenarios} '
        html += f'({robustness_report.profitable_percentage:.1f}%)</strong></p>'
        html += '</div>'
        html += '<div class="section">'
        html += '<h2>Robustness Indicators</h2>'
        html += '<div style="display: flex; flex-wrap: wrap;">'
        score_class = self._get_score_class(robustness_report.resilience_score)
        html += f'<div class="metric"><div class="metric-value {score_class}">{robustness_report.resilience_score:.1f}</div>'
        html += '<div class="metric-label">Resilience Score</div></div>'
        score_class = self._get_score_class(robustness_report.stability_score)
        html += f'<div class="metric"><div class="metric-value {score_class}">{robustness_report.stability_score:.1f}</div>'
        html += '<div class="metric-label">Stability Score</div></div>'
        score_class = self._get_score_class(robustness_report.drawdown_resilience_score)
        html += f'<div class="metric"><div class="metric-value {score_class}">{robustness_report.drawdown_resilience_score:.1f}</div>'
        html += '<div class="metric-label">Drawdown Resilience</div></div>'
        score_class = self._get_score_class(robustness_report.overall_robustness_score)
        html += f'<div class="metric"><div class="metric-value {score_class}">{robustness_report.overall_robustness_score:.1f}</div>'
        html += '<div class="metric-label">Overall Robustness</div></div>'
        html += '</div></div>'
        html += '<div class="section">'
        html += '<h2>Baseline Performance</h2>'
        html += '<div style="display: flex; flex-wrap: wrap;">'
        html += '<div class="metric"><div class="metric-value">{:.2f}%</div><div class="metric-label">Total Return</div></div>'.format(baseline_metrics.get('Total Return (%)', 0))
        html += '<div class="metric"><div class="metric-value">{:.4f}</div><div class="metric-label">Sharpe Ratio</div></div>'.format(baseline_metrics.get('Sharpe Ratio', 0))
        html += '<div class="metric"><div class="metric-value">{:.2f}%</div><div class="metric-label">Max Drawdown</div></div>'.format(baseline_metrics.get('Max Drawdown (%)', 0))
        html += '<div class="metric"><div class="metric-value">{:.2f}%</div><div class="metric-label">Volatility</div></div>'.format(baseline_metrics.get('Annualized Volatility (%)', 0))
        html += '</div></div>'
        html += '<div class="section">'
        html += '<h2>Performance Degradation Statistics</h2>'
        html += '<div class="subsection">'
        html += '<h3>Return Degradation</h3>'
        html += f'<p>Average Drop: <strong>{robustness_report.avg_return_drop:.2f}%</strong></p>'
        html += f'<p>Worst Case: <strong>{robustness_report.worst_return_drop:.2f}%</strong></p>'
        html += f'<p>Best Case: <strong>{robustness_report.best_return_drop:.2f}%</strong></p>'
        html += f'<p>Std Dev: <strong>{robustness_report.std_return_drop:.2f}%</strong></p>'
        html += '</div>'
        html += '<div class="subsection">'
        html += '<h3>Sharpe Ratio Degradation</h3>'
        html += f'<p>Average Drop: <strong>{robustness_report.avg_sharpe_drop:.4f}</strong></p>'
        html += f'<p>Worst Case: <strong>{robustness_report.worst_sharpe_drop:.4f}</strong></p>'
        html += f'<p>Best Case: <strong>{robustness_report.best_sharpe_drop:.4f}</strong></p>'
        html += '</div>'
        html += '<div class="subsection">'
        html += '<h3>Max Drawdown Increase</h3>'
        html += f'<p>Average Increase: <strong>{robustness_report.avg_max_dd_increase:.2f}%</strong></p>'
        html += f'<p>Worst Case: <strong>{robustness_report.worst_max_dd_increase:.2f}%</strong></p>'
        html += f'<p>Best Case: <strong>{robustness_report.best_max_dd_increase:.2f}%</strong></p>'
        html += '</div>'
        html += '</div>'
        html += '<div class="section">'
        html += '<h2>Scenario Comparison</h2>'
        html += comparison_df.to_html(classes='scenario-table')
        html += '</div>'
        html += '<div class="section">'
        html += '<h2>Worst Performing Scenarios</h2>'
        html += '<p>Scenarios with largest return degradation:</p>'
        worst_scenarios = sorted(degradation_reports,
                               key=lambda x: x.return_drop, reverse=True)[:5]
        html += '<table>'
        html += '<tr><th>Scenario</th><th>Return Drop (%)</th><th>Sharpe Drop</th><th>Profitable</th></tr>'
        for report in worst_scenarios:
            html += f'<tr><td>{report.scenario_name}</td>'
            html += f'<td>{report.return_drop:.2f}</td>'
            html += f'<td>{report.sharpe_drop:.4f}</td>'
            html += f'<td>{"Yes" if report.is_profitable else "No"}</td></tr>'
        html += '</table>'
        html += '</div>'
        html += '<div class="footer">'
        html += '<p>© 2026 Algorithmic Trading Research Platform</p>'
        html += '</div>'
        html += '</body></html>'
        return html.format(timestamp=timestamp_str)
    @staticmethod
    def _get_score_class(score: float) -> str:
        """Get CSS class based on score."""
        if score >= 70:
            return "score-excellent"
        elif score >= 50:
            return "score-good"
        else:
            return "score-poor"
    def generate_text_report(self,
                            baseline_metrics: Dict[str, float],
                            robustness_report: RobustnessMetricsReport,
                            degradation_reports: list[PerformanceDegradationReport],
                            filename: str = "stress_test_report.txt") -> Path:
        """Generate text report."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STRESS TESTING REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Scenarios Tested: {robustness_report.num_scenarios}\n")
            f.write(f"Profitable Scenarios: {robustness_report.profitable_scenarios}/"
                   f"{robustness_report.num_scenarios} "
                   f"({robustness_report.profitable_percentage:.1f}%)\n\n")
            f.write("ROBUSTNESS INDICATORS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Resilience Score:              {robustness_report.resilience_score:.1f}\n")
            f.write(f"Stability Score:               {robustness_report.stability_score:.1f}\n")
            f.write(f"Drawdown Resilience Score:     {robustness_report.drawdown_resilience_score:.1f}\n")
            f.write(f"Overall Robustness Score:      {robustness_report.overall_robustness_score:.1f}\n\n")
            f.write("BASELINE PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Return:       {baseline_metrics.get('Total Return (%)', 0):>10.2f}%\n")
            f.write(f"Sharpe Ratio:       {baseline_metrics.get('Sharpe Ratio', 0):>10.4f}\n")
            f.write(f"Max Drawdown:       {baseline_metrics.get('Max Drawdown (%)', 0):>10.2f}%\n")
            f.write(f"Volatility:         {baseline_metrics.get('Annualized Volatility (%)', 0):>10.2f}%\n\n")
            f.write("PERFORMANCE DEGRADATION STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write("Return Degradation:\n")
            f.write(f"  Average Drop:     {robustness_report.avg_return_drop:>10.2f}%\n")
            f.write(f"  Worst Case:       {robustness_report.worst_return_drop:>10.2f}%\n")
            f.write(f"  Best Case:        {robustness_report.best_return_drop:>10.2f}%\n")
            f.write(f"  Std Dev:          {robustness_report.std_return_drop:>10.2f}%\n\n")
            f.write("Sharpe Ratio Degradation:\n")
            f.write(f"  Average Drop:     {robustness_report.avg_sharpe_drop:>10.4f}\n")
            f.write(f"  Worst Case:       {robustness_report.worst_sharpe_drop:>10.4f}\n")
            f.write(f"  Best Case:        {robustness_report.best_sharpe_drop:>10.4f}\n\n")
            f.write("Max Drawdown Increase:\n")
            f.write(f"  Average Increase: {robustness_report.avg_max_dd_increase:>10.2f}%\n")
            f.write(f"  Worst Case:       {robustness_report.worst_max_dd_increase:>10.2f}%\n")
            f.write(f"  Best Case:        {robustness_report.best_max_dd_increase:>10.2f}%\n\n")
            f.write("WORST PERFORMING SCENARIOS\n")
            f.write("-" * 80 + "\n")
            worst_scenarios = sorted(degradation_reports,
                                   key=lambda x: x.return_drop, reverse=True)[:10]
            for i, report in enumerate(worst_scenarios, 1):
                f.write(f"\n{i}. {report.scenario_name}\n")
                f.write(f"   Return Drop:     {report.return_drop:>10.2f}%\n")
                f.write(f"   Sharpe Drop:     {report.sharpe_drop:>10.4f}\n")
                f.write(f"   Max DD Increase: {report.max_dd_increase:>10.2f}%\n")
                f.write(f"   Profitable:      {'Yes' if report.is_profitable else 'No'}\n")
        return output_path

