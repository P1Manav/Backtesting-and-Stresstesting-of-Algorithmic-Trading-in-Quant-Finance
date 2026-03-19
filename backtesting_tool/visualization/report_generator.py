"""Generate HTML reports"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

class ReportGenerator:
    """Generate report"""

    def __init__(self, save_dir: str):
    """Initialize instance"""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, results: Dict[str, Any], metrics: Dict[str, Any],
    """Save data or model"""
             stat_results: Optional[Dict[str, Any]] = None,
             robustness_results: Optional[Dict[str, Any]] = None) -> None:
        tickers = results.get('tickers', ['STOCK'])

        agg = {
            'Date': results['dates'],
            'Portfolio_Value': results['portfolio_values'],
            'Cash': results['cash'],
        }
        for t in tickers:
            ps = results['per_stock'][t]
            agg[f'{t}_Price'] = ps['actual_prices']
            agg[f'{t}_Predicted'] = ps['predicted_prices']
            agg[f'{t}_Position'] = ps['positions']
            agg[f'{t}_Shares'] = ps['shares']

        pd.DataFrame(agg).to_csv(
            self.save_dir / 'portfolio_history.csv', index=False)

        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            trades_df.to_csv(self.save_dir / 'trades.csv', index=False)

        if 'aggregate' in metrics:
            pd.DataFrame([metrics['aggregate']]).to_csv(
                self.save_dir / 'metrics.csv', index=False)

            per_stock = {k: v for k, v in metrics.items() if k != 'aggregate'}
            if per_stock:
                ps_df = pd.DataFrame(per_stock).T
                ps_df.index.name = 'Ticker'
                ps_df.to_csv(self.save_dir / 'per_stock_metrics.csv')
        else:
            pd.DataFrame([metrics]).to_csv(
                self.save_dir / 'metrics.csv', index=False)

        if stat_results is not None:
            self._save_statistical_results(stat_results)

        if robustness_results is not None:
            self._save_robustness_results(robustness_results)

        print(f"  [OK] CSV reports saved to: {self.save_dir}")

    def _save_statistical_results(self, stat_results: Dict[str, Any]) -> None:
    """Save data or model"""
        confidence_levels = stat_results.get('confidence_levels', [])
        levels = stat_results.get('levels', {})

        stat_rows = []
        for cl in confidence_levels:
            level = levels[cl]
            summary = level['summary']
            kup = level['kupiec_test']
            chris = level['christoffersen_test']
            ind = chris['independence']
            cc = chris['conditional_coverage']

            stat_rows.append({
                'VaR_Confidence_Level': summary['confidence_level'],
                'VaR_Method': summary['var_method'],
                'VaR_Window': summary['var_window'],
                'Total_Observations': summary['total_observations'],
                'Total_Violations': summary['total_violations'],
                'Expected_Violations': summary['expected_violations'],
                'Expected_Violation_Rate': summary['expected_violation_rate'],
                'Observed_Violation_Rate': summary['observed_violation_rate'],
                'Kupiec_LR_Statistic': kup['lr_statistic'],
                'Kupiec_p_value': kup['p_value'],
                'Kupiec_Critical_Value': kup['critical_value'],
                'Kupiec_Result': kup['result'],
                'Independence_LR_Statistic': ind['lr_statistic'],
                'Independence_p_value': ind['p_value'],
                'Independence_Critical_Value': ind['critical_value'],
                'Independence_Result': ind['result'],
                'Transition_n00': ind['n00'],
                'Transition_n01': ind['n01'],
                'Transition_n10': ind['n10'],
                'Transition_n11': ind['n11'],
                'CC_LR_Statistic': cc['lr_statistic'],
                'CC_p_value': cc['p_value'],
                'CC_Critical_Value': cc['critical_value'],
                'CC_Result': cc['result'],
                'Reliability_Decision': level['reliability_decision'],
            })

        pd.DataFrame(stat_rows).to_csv(
            self.save_dir / 'statistical_test_results.csv', index=False)

        bench = stat_results.get('benchmark_comparison', {})
        if bench and 'portfolio' in bench:
            port = bench['portfolio']
            bm = bench['benchmark']
            comp = bench['comparison']
            bench_row = {}
            bench_row.update({f'Model_{k}': v for k, v in port.items()})
            bench_row.update({f'Benchmark_{k}': v for k, v in bm.items()})
            bench_row.update({f'Comparison_{k}': v for k, v in comp.items()})
            pd.DataFrame([bench_row]).to_csv(
                self.save_dir / 'benchmark_comparison.csv', index=False)

        returns = stat_results.get('_returns')
        if returns is not None:
            var_df = pd.DataFrame({'Daily_Return': returns})

            for cl in confidence_levels:
                level = levels[cl]
                pct = f"{cl * 100:.0f}"
                var_df[f'VaR_{pct}'] = level['_var_series']
                var_df[f'Violation_{pct}'] = level['_hit_sequence']

            var_df.to_csv(self.save_dir / 'var_violations.csv', index=False)

    def _save_robustness_results(self, rob: Dict[str, Any]) -> None:
    """Save data or model"""
        rows = []

        wf = rob.get('walk_forward', {})
        if wf.get('fold_results'):
            agg = wf['aggregate']
            rows.append({
                'Test': 'Walk-Forward',
                'Result': wf.get('result', ''),
                'Detail_1': f"Mean_OOS_Sharpe={agg.get('mean_sharpe', '')}",
                'Detail_2': f"Consistency={wf.get('consistency_ratio', '')}",
                'Detail_3': f"Folds={agg.get('n_folds', '')}",
            })

        bs = rob.get('bootstrap', {})
        if 'sharpe_ratio' in bs:
            sr = bs['sharpe_ratio']
            rows.append({
                'Test': 'Bootstrap',
                'Result': bs.get('result', ''),
                'Detail_1': f"Sharpe_CI=[{sr['ci_lower']},{sr['ci_upper']}]",
                'Detail_2': f"N={bs.get('n_bootstrap', '')}",
                'Detail_3': f"CL={bs.get('confidence_level', '')}",
            })

        mc = rob.get('monte_carlo', {})
        if 'p_value_sharpe' in mc:
            rows.append({
                'Test': 'Monte-Carlo',
                'Result': mc.get('result', ''),
                'Detail_1': f"p_Sharpe={mc['p_value_sharpe']}",
                'Detail_2': f"p_Return={mc['p_value_return']}",
                'Detail_3': f"N={mc.get('n_simulations', '')}",
            })

        rows.append({
            'Test': 'Overall',
            'Result': rob.get('overall_verdict', ''),
            'Detail_1': '', 'Detail_2': '', 'Detail_3': '',
        })

        pd.DataFrame(rows).to_csv(
            self.save_dir / 'robustness_results.csv', index=False)

