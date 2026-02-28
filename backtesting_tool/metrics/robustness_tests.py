import numpy as np
from typing import Dict, Any, List, Optional

class WalkForwardAnalyzer:

    def __init__(self, n_folds: int = 5, train_ratio: float = 0.7):
        self.n_folds = max(2, n_folds)
        self.train_ratio = train_ratio

    # Run walk-forward analysis.
    def analyze(self, portfolio_values: np.ndarray,
                initial_capital: float) -> Dict[str, Any]:
        pv = np.array(portfolio_values, dtype=float)
        n = len(pv)
        step = n // self.n_folds
        if step < 20:
            return self._insufficient()

        fold_results: List[Dict[str, Any]] = []

        for fold in range(self.n_folds):
            start = fold * step
            end = min(start + step, n) if fold < self.n_folds - 1 else n
            fold_pv = pv[start:end]
            if len(fold_pv) < 10:
                continue
            split = int(len(fold_pv) * self.train_ratio)
            if split < 5 or len(fold_pv) - split < 5:
                continue

            oos_pv = fold_pv[split:]
            oos_ret = np.diff(oos_pv) / oos_pv[:-1]
            oos_ret = oos_ret[np.isfinite(oos_ret)]
            if len(oos_ret) < 2:
                continue

            fold_results.append({
                'fold': fold + 1,
                'start_idx': start + split,
                'end_idx': end,
                'oos_days': len(oos_ret),
                'sharpe_ratio': round(self._sharpe(oos_ret), 4),
                'total_return_pct': round(
                    (oos_pv[-1] - oos_pv[0]) / oos_pv[0] * 100, 4),
                'max_drawdown_pct': round(self._mdd(oos_pv), 4),
            })

        if not fold_results:
            return self._insufficient()

        sharpe_arr = [f['sharpe_ratio'] for f in fold_results]
        ret_arr = [f['total_return_pct'] for f in fold_results]
        mdd_arr = [f['max_drawdown_pct'] for f in fold_results]

        consistency = sum(1 for s in sharpe_arr if s > 0) / len(sharpe_arr)

        return {
            'fold_results': fold_results,
            'aggregate': {
                'mean_sharpe': round(float(np.mean(sharpe_arr)), 4),
                'std_sharpe': round(float(np.std(sharpe_arr)), 4),
                'mean_return_pct': round(float(np.mean(ret_arr)), 4),
                'std_return_pct': round(float(np.std(ret_arr)), 4),
                'mean_max_drawdown_pct': round(float(np.mean(mdd_arr)), 4),
                'n_folds': len(fold_results),
            },
            'consistency_ratio': round(consistency, 4),
            'result': 'CONSISTENT' if consistency >= 0.6 else 'INCONSISTENT',
        }

    @staticmethod
    def _sharpe(r, td=252):
        s = float(np.std(r))
        return float(np.mean(r) / s * np.sqrt(td)) if s > 0 else 0.0

    @staticmethod
    def _mdd(pv):
        cm = np.maximum.accumulate(pv)
        return float(np.min((pv - cm) / cm) * 100)

    @staticmethod
    def _insufficient():
        return {'fold_results': [], 'aggregate': {},
                'consistency_ratio': 0.0, 'result': 'INSUFFICIENT DATA'}

class BootstrapResampler:

    def __init__(self, n_bootstrap: int = 1000,
                 confidence_level: float = 0.95,
                 random_seed: Optional[int] = 42):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.rng = np.random.default_rng(random_seed)

    # Run bootstrap resampling analysis.
    def resample(self, portfolio_values: np.ndarray,
                 initial_capital: float) -> Dict[str, Any]:
        pv = np.array(portfolio_values, dtype=float)
        returns = np.diff(pv) / pv[:-1]
        returns = returns[np.isfinite(returns)]
        if len(returns) < 10:
            return {'result': 'INSUFFICIENT DATA'}

        n = len(returns)
        alpha = 1 - self.confidence_level
        lo_p, hi_p = alpha / 2 * 100, (1 - alpha / 2) * 100

        s_dist = np.zeros(self.n_bootstrap)
        r_dist = np.zeros(self.n_bootstrap)
        d_dist = np.zeros(self.n_bootstrap)

        for i in range(self.n_bootstrap):
            br = self.rng.choice(returns, size=n, replace=True)
            bpv = initial_capital * np.cumprod(1 + br)
            bpv = np.concatenate(([initial_capital], bpv))

            sd = float(np.std(br))
            s_dist[i] = float(np.mean(br) / sd * np.sqrt(252)) if sd > 0 else 0.0
            r_dist[i] = (bpv[-1] - initial_capital) / initial_capital * 100
            cm = np.maximum.accumulate(bpv)
            d_dist[i] = float(np.min((bpv - cm) / cm) * 100)

        obs_std = float(np.std(returns))
        obs_s = float(np.mean(returns) / obs_std * np.sqrt(252)) if obs_std > 0 else 0.0
        obs_r = (pv[-1] - initial_capital) / initial_capital * 100
        cm = np.maximum.accumulate(pv)
        obs_d = float(np.min((pv - cm) / cm) * 100)

        def _ci(dist, obs):
            return {
                'observed': round(obs, 4),
                'mean': round(float(np.mean(dist)), 4),
                'std': round(float(np.std(dist)), 4),
                'ci_lower': round(float(np.percentile(dist, lo_p)), 4),
                'ci_upper': round(float(np.percentile(dist, hi_p)), 4),
            }

        return {
            'n_bootstrap': self.n_bootstrap,
            'confidence_level': self.confidence_level,
            'sharpe_ratio': _ci(s_dist, obs_s),
            'total_return_pct': _ci(r_dist, obs_r),
            'max_drawdown_pct': _ci(d_dist, obs_d),
            'result': 'ROBUST' if float(np.percentile(s_dist, lo_p)) > 0 else 'NOT ROBUST',
        }

class MonteCarloSimulator:

    def __init__(self, n_simulations: int = 1000,
                 random_seed: Optional[int] = 42):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)

    # Run Monte Carlo permutation test.
    def simulate(self, portfolio_values: np.ndarray,
                 initial_capital: float) -> Dict[str, Any]:
        pv = np.array(portfolio_values, dtype=float)
        returns = np.diff(pv) / pv[:-1]
        returns = returns[np.isfinite(returns)]
        if len(returns) < 10:
            return {'result': 'INSUFFICIENT DATA'}

        obs_std = float(np.std(returns))
        obs_s = float(np.mean(returns) / obs_std * np.sqrt(252)) if obs_std > 0 else 0.0
        obs_r = (pv[-1] - initial_capital) / initial_capital * 100

        ns = np.zeros(self.n_simulations)
        nr = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            sh = self.rng.permutation(returns)
            spv = initial_capital * np.cumprod(1 + sh)
            spv = np.concatenate(([initial_capital], spv))
            sd = float(np.std(sh))
            ns[i] = float(np.mean(sh) / sd * np.sqrt(252)) if sd > 0 else 0.0
            nr[i] = (spv[-1] - initial_capital) / initial_capital * 100

        p_s = float(np.mean(ns >= obs_s))
        p_r = float(np.mean(nr >= obs_r))

        return {
            'n_simulations': self.n_simulations,
            'observed_sharpe': round(obs_s, 4),
            'observed_return_pct': round(obs_r, 4),
            'null_sharpe': {
                'mean': round(float(np.mean(ns)), 4),
                'std': round(float(np.std(ns)), 4),
                'percentile_5': round(float(np.percentile(ns, 5)), 4),
                'percentile_95': round(float(np.percentile(ns, 95)), 4),
            },
            'null_return_pct': {
                'mean': round(float(np.mean(nr)), 4),
                'std': round(float(np.std(nr)), 4),
                'percentile_5': round(float(np.percentile(nr, 5)), 4),
                'percentile_95': round(float(np.percentile(nr, 95)), 4),
            },
            'p_value_sharpe': round(p_s, 4),
            'p_value_return': round(p_r, 4),
            'result': 'SIGNIFICANT' if p_s < 0.05 else 'NOT SIGNIFICANT',
        }

class RobustnessAnalyzer:

    def __init__(self, n_folds: int = 5, n_bootstrap: int = 1000,
                 n_simulations: int = 1000,
                 random_seed: Optional[int] = 42):
        self.wf = WalkForwardAnalyzer(n_folds=n_folds)
        self.bs = BootstrapResampler(n_bootstrap=n_bootstrap,
                                     random_seed=random_seed)
        self.mc = MonteCarloSimulator(n_simulations=n_simulations,
                                      random_seed=random_seed)

    # Execute all robustness tests on backtest results.
    def run(self, results: Dict[str, Any],
            initial_capital: float) -> Dict[str, Any]:
        pv = np.array(results['portfolio_values'], dtype=float)

        wf_result = self.wf.analyze(pv, initial_capital)
        bs_result = self.bs.resample(pv, initial_capital)
        mc_result = self.mc.simulate(pv, initial_capital)

        verdicts = [
            wf_result.get('result', 'N/A'),
            bs_result.get('result', 'N/A'),
            mc_result.get('result', 'N/A'),
        ]
        passes = sum(1 for v in verdicts
                     if v in ('CONSISTENT', 'ROBUST', 'SIGNIFICANT'))

        if passes == 3:
            overall = 'ROBUST — Strategy passes all robustness checks'
        elif passes >= 2:
            overall = 'PARTIALLY ROBUST — Strategy passes majority of checks'
        elif passes == 1:
            overall = 'WEAK — Strategy passes only one robustness check'
        else:
            overall = 'NOT ROBUST — Strategy fails all robustness checks'

        return {
            'walk_forward': wf_result,
            'bootstrap': bs_result,
            'monte_carlo': mc_result,
            'overall_verdict': overall,
        }

# Pretty-print robustness analysis results (W1).
def print_robustness_results(rob: Dict[str, Any]) -> None:
    print(f"\n{'=' * 70}")
    print("  ROBUSTNESS VALIDATION (Walk-Forward / Bootstrap / Monte Carlo)")
    print(f"{'=' * 70}")

    wf = rob['walk_forward']
    print(f"\n{'─' * 70}")
    print(f"  1. WALK-FORWARD ANALYSIS  (Pardo, 2008)")
    print(f"{'─' * 70}")
    if wf.get('fold_results'):
        a = wf['aggregate']
        print(f"  {'Folds':<35} {a['n_folds']}")
        print(f"  {'Mean OOS Sharpe':<35} {a['mean_sharpe']:.4f} +/- {a['std_sharpe']:.4f}")
        print(f"  {'Mean OOS Return (%)':<35} {a['mean_return_pct']:.4f} +/- {a['std_return_pct']:.4f}")
        print(f"  {'Mean OOS Max Drawdown (%)':<35} {a['mean_max_drawdown_pct']:.4f}")
        print(f"  {'Consistency Ratio':<35} {wf['consistency_ratio']:.2%}")
        print(f"\n  {'Fold':<6} {'OOS Days':<10} {'Sharpe':<10} {'Return%':<12} {'MaxDD%':<10}")
        print(f"  {'-' * 48}")
        for f in wf['fold_results']:
            print(f"  {f['fold']:<6} {f['oos_days']:<10} {f['sharpe_ratio']:<10.4f} "
                  f"{f['total_return_pct']:<12.4f} {f['max_drawdown_pct']:<10.4f}")
    print(f"\n  >> Walk-Forward Result: {wf['result']}")

    bs = rob['bootstrap']
    print(f"\n{'─' * 70}")
    print(f"  2. BOOTSTRAP RESAMPLING  (Efron & Tibshirani, 1993)")
    print(f"{'─' * 70}")
    if 'sharpe_ratio' in bs:
        cl = bs['confidence_level']
        sr = bs['sharpe_ratio']
        tr = bs['total_return_pct']
        md = bs['max_drawdown_pct']
        print(f"  {'Replications':<35} {bs['n_bootstrap']}")
        print(f"  {'Confidence Level':<35} {cl:.0%}")
        print(f"\n  {'Metric':<25} {'Observed':>10} {'Mean':>10} {'CI Low':>10} {'CI High':>10}")
        print(f"  {'-' * 65}")
        for label, d in [('Sharpe Ratio', sr), ('Total Return (%)', tr),
                         ('Max Drawdown (%)', md)]:
            print(f"  {label:<25} {d['observed']:>10.4f} {d['mean']:>10.4f} "
                  f"{d['ci_lower']:>10.4f} {d['ci_upper']:>10.4f}")
    print(f"\n  >> Bootstrap Result: {bs['result']}")

    mc = rob['monte_carlo']
    print(f"\n{'─' * 70}")
    print(f"  3. MONTE CARLO PERMUTATION TEST  (White, 2000)")
    print(f"{'─' * 70}")
    if 'null_sharpe' in mc:
        ns = mc['null_sharpe']
        print(f"  {'Simulations':<35} {mc['n_simulations']}")
        print(f"  {'Observed Sharpe':<35} {mc['observed_sharpe']:.4f}")
        print(f"  {'Null Sharpe (mean +/- std)':<35} {ns['mean']:.4f} +/- {ns['std']:.4f}")
        print(f"  {'Null Sharpe [5th, 95th %ile]':<35} [{ns['percentile_5']:.4f}, {ns['percentile_95']:.4f}]")
        print(f"  {'p-value (Sharpe)':<35} {mc['p_value_sharpe']:.4f}")
        print(f"  {'p-value (Return)':<35} {mc['p_value_return']:.4f}")
    print(f"\n  >> Monte Carlo Result: {mc['result']}")

    print(f"\n{'=' * 70}")
    print(f"  OVERALL ROBUSTNESS VERDICT")
    print(f"{'=' * 70}")
    print(f"  {rob['overall_verdict']}")
    print(f"{'=' * 70}")
