import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List

# Compute Value-at-Risk for a returns series.
def compute_var(returns: np.ndarray,
                confidence_level: float = 0.95,
                method: str = 'historical',
                window: Optional[int] = None) -> np.ndarray:
    n = len(returns)
    alpha = 1 - confidence_level
    var_series = np.full(n, np.nan)

    min_obs = max(30, window or 30)

    for i in range(min_obs, n):
        if window is not None:
            start = max(0, i - window)
        else:
            start = 0
        hist = returns[start:i]

        if method == 'parametric':
            mu = np.mean(hist)
            sigma = np.std(hist, ddof=1)
            var_series[i] = -(mu + stats.norm.ppf(alpha) * sigma)
        else:
            var_series[i] = -np.quantile(hist, alpha)

    return var_series

# Create a binary hit sequence: 1 if actual loss > VaR, else 0.
def identify_violations(returns: np.ndarray,
                        var_series: np.ndarray) -> np.ndarray:
    hits = np.zeros(len(returns), dtype=int)
    valid = ~np.isnan(var_series)
    hits[valid] = (returns[valid] < -var_series[valid]).astype(int)
    return hits

class KupiecTest:

    # Initialize the instance.
    def __init__(self, confidence_level: float = 0.95,
                 significance: float = 0.05):
        self.confidence_level = confidence_level
        self.p = 1 - confidence_level
        self.significance = significance

    # Run the Kupiec POF test on a binary hit sequence.
    def test(self, hit_sequence: np.ndarray) -> Dict[str, Any]:
        T = len(hit_sequence)
        N = int(np.sum(hit_sequence))
        p = self.p

        observed_rate = N / T if T > 0 else 0.0

        if N == 0:
            lr_stat = -2 * (T * np.log(1 - p))
        elif N == T:
            lr_stat = -2 * (T * np.log(p))
        else:
            log_L0 = (T - N) * np.log(1 - p) + N * np.log(p)
            p_hat = N / T
            log_L1 = (T - N) * np.log(1 - p_hat) + N * np.log(p_hat)
            lr_stat = -2 * (log_L0 - log_L1)

        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        critical_value = stats.chi2.ppf(1 - self.significance, df=1)
        reject = lr_stat > critical_value

        return {
            'test_name': 'Kupiec POF (Coverage) Test',
            'T': T,
            'N': N,
            'expected_rate': round(p, 6),
            'observed_rate': round(observed_rate, 6),
            'lr_statistic': round(float(lr_stat), 4),
            'p_value': round(float(p_value), 6),
            'critical_value': round(float(critical_value), 4),
            'significance': self.significance,
            'reject_H0': bool(reject),
            'result': 'FAIL' if reject else 'PASS',
        }

class ChristoffersenTest:

    def __init__(self, confidence_level: float = 0.95,
                 significance: float = 0.05):
        self.confidence_level = confidence_level
        self.p = 1 - confidence_level
        self.significance = significance

    # Count transition pairs in the hit sequence.
    @staticmethod
    def _transition_counts(hit_sequence: np.ndarray) -> Tuple[int, int, int, int]:
        n00 = n01 = n10 = n11 = 0
        for i in range(1, len(hit_sequence)):
            prev, curr = hit_sequence[i - 1], hit_sequence[i]
            if prev == 0 and curr == 0:
                n00 += 1
            elif prev == 0 and curr == 1:
                n01 += 1
            elif prev == 1 and curr == 0:
                n10 += 1
            else:
                n11 += 1
        return n00, n01, n10, n11

    # Independence component of the Christoffersen test.
    def _independence_test(self, hit_sequence: np.ndarray) -> Dict[str, Any]:
        n00, n01, n10, n11 = self._transition_counts(hit_sequence)

        pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0

        pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0

        eps = 1e-15

        log_L1 = 0.0
        if n00 > 0:
            log_L1 += n00 * np.log(max(1 - pi_01, eps))
        if n01 > 0:
            log_L1 += n01 * np.log(max(pi_01, eps))
        if n10 > 0:
            log_L1 += n10 * np.log(max(1 - pi_11, eps))
        if n11 > 0:
            log_L1 += n11 * np.log(max(pi_11, eps))

        log_L0 = 0.0
        if (n00 + n10) > 0:
            log_L0 += (n00 + n10) * np.log(max(1 - pi, eps))
        if (n01 + n11) > 0:
            log_L0 += (n01 + n11) * np.log(max(pi, eps))

        lr_ind = -2 * (log_L0 - log_L1)
        lr_ind = max(lr_ind, 0.0)

        p_value = 1 - stats.chi2.cdf(lr_ind, df=1)
        critical_value = stats.chi2.ppf(1 - self.significance, df=1)
        reject = lr_ind > critical_value

        return {
            'test_name': 'Christoffersen Independence Test',
            'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11,
            'pi_unconditional': round(pi, 6),
            'pi_01': round(pi_01, 6),
            'pi_11': round(pi_11, 6),
            'lr_statistic': round(float(lr_ind), 4),
            'p_value': round(float(p_value), 6),
            'critical_value': round(float(critical_value), 4),
            'significance': self.significance,
            'reject_H0': bool(reject),
            'result': 'FAIL' if reject else 'PASS',
        }

    # Run the full Christoffersen Conditional Coverage test.
    def test(self, hit_sequence: np.ndarray) -> Dict[str, Any]:
        kupiec = KupiecTest(self.confidence_level, self.significance)
        coverage = kupiec.test(hit_sequence)

        independence = self._independence_test(hit_sequence)

        lr_cc = coverage['lr_statistic'] + independence['lr_statistic']
        p_value_cc = 1 - stats.chi2.cdf(lr_cc, df=2)
        critical_cc = stats.chi2.ppf(1 - self.significance, df=2)
        reject_cc = lr_cc > critical_cc

        conditional_coverage = {
            'test_name': 'Christoffersen Conditional Coverage Test',
            'lr_statistic': round(float(lr_cc), 4),
            'p_value': round(float(p_value_cc), 6),
            'critical_value': round(float(critical_cc), 4),
            'significance': self.significance,
            'reject_H0': bool(reject_cc),
            'result': 'FAIL' if reject_cc else 'PASS',
        }

        return {
            'coverage': coverage,
            'independence': independence,
            'conditional_coverage': conditional_coverage,
        }

class BenchmarkComparison:

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital

    # Compute buy-and-hold benchmark portfolio values.
    def compute_benchmark(self, results: Dict[str, Any]) -> np.ndarray:
        tickers = results.get('tickers', [])
        n_stocks = len(tickers)
        portfolio_values = np.array(results['portfolio_values'], dtype=float)

        if n_stocks == 0:
            return portfolio_values.copy()

        all_prices = {}
        for t in tickers:
            all_prices[t] = np.array(results['per_stock'][t]['actual_prices'], dtype=float)

        n_days = len(all_prices[tickers[0]])

        capital_per_stock = self.initial_capital / n_stocks
        benchmark_daily = np.zeros(n_days)

        for t in tickers:
            prices = all_prices[t]
            shares = capital_per_stock / prices[0]
            benchmark_daily += shares * prices

        benchmark_values = np.empty(len(portfolio_values))
        benchmark_values[0] = self.initial_capital
        benchmark_values[1:] = benchmark_daily[:len(portfolio_values) - 1]

        return benchmark_values

    # Run benchmark comparison analysis.
    def compare(self, results: Dict[str, Any]) -> Dict[str, Any]:
        portfolio_values = np.array(results['portfolio_values'], dtype=float)
        benchmark_values = self.compute_benchmark(results)

        port_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        bench_returns = np.diff(benchmark_values) / benchmark_values[:-1]

        port_total_ret = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
        bench_total_ret = (benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0] * 100
        excess_return = port_total_ret - bench_total_ret

        n_years = len(port_returns) / 252
        port_ann_ret = ((portfolio_values[-1] / portfolio_values[0]) ** (1 / max(n_years, 0.01)) - 1) * 100
        bench_ann_ret = ((benchmark_values[-1] / benchmark_values[0]) ** (1 / max(n_years, 0.01)) - 1) * 100

        excess_daily = port_returns - bench_returns
        tracking_error = float(np.std(excess_daily, ddof=1) * np.sqrt(252) * 100)

        info_ratio = float(np.mean(excess_daily) / np.std(excess_daily, ddof=1) * np.sqrt(252)) if np.std(excess_daily) > 0 else 0.0

        port_sharpe = float(np.mean(port_returns) / np.std(port_returns, ddof=1) * np.sqrt(252)) if np.std(port_returns) > 0 else 0.0
        bench_sharpe = float(np.mean(bench_returns) / np.std(bench_returns, ddof=1) * np.sqrt(252)) if np.std(bench_returns) > 0 else 0.0

        def _max_dd(values):
            cummax = np.maximum.accumulate(values)
            dd = (values - cummax) / cummax
            return float(np.min(dd) * 100)

        port_max_dd = _max_dd(portfolio_values)
        bench_max_dd = _max_dd(benchmark_values)

        port_vol = float(np.std(port_returns, ddof=1) * np.sqrt(252) * 100)
        bench_vol = float(np.std(bench_returns, ddof=1) * np.sqrt(252) * 100)

        win_days = int(np.sum(port_returns > bench_returns))
        total_days = len(port_returns)
        win_rate = win_days / total_days * 100

        if excess_return > 5:
            assessment = 'STRONG OUTPERFORMANCE — Model significantly beats benchmark'
        elif excess_return > 0:
            assessment = 'OUTPERFORMANCE — Model beats buy-and-hold benchmark'
        elif excess_return > -5:
            assessment = 'COMPARABLE — Model performs similarly to benchmark'
        else:
            assessment = 'UNDERPERFORMANCE — Buy-and-hold beats the model'

        return {
            'portfolio': {
                'total_return_pct': round(port_total_ret, 2),
                'annualized_return_pct': round(port_ann_ret, 2),
                'sharpe_ratio': round(port_sharpe, 4),
                'max_drawdown_pct': round(port_max_dd, 2),
                'volatility_pct': round(port_vol, 2),
                'final_value': round(float(portfolio_values[-1]), 2),
            },
            'benchmark': {
                'total_return_pct': round(bench_total_ret, 2),
                'annualized_return_pct': round(bench_ann_ret, 2),
                'sharpe_ratio': round(bench_sharpe, 4),
                'max_drawdown_pct': round(bench_max_dd, 2),
                'volatility_pct': round(bench_vol, 2),
                'final_value': round(float(benchmark_values[-1]), 2),
            },
            'comparison': {
                'excess_return_pct': round(excess_return, 2),
                'tracking_error_pct': round(tracking_error, 2),
                'information_ratio': round(info_ratio, 4),
                'win_rate_pct': round(win_rate, 2),
                'win_days': win_days,
                'total_days': total_days,
                'assessment': assessment,
            },
            '_benchmark_values': benchmark_values,
            '_port_returns': port_returns,
            '_bench_returns': bench_returns,
        }

class StatisticalBacktester:

    # Initialize the instance.
    def __init__(self,
                 confidence_levels: Optional[List[float]] = None,
                 significance: float = 0.05,
                 var_method: str = 'historical',
                 var_window: Optional[int] = 250,
                 initial_capital: float = 100_000.0):
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
        self.confidence_levels = sorted(confidence_levels)
        self.significance = significance
        self.var_method = var_method
        self.var_window = var_window
        self.initial_capital = initial_capital

    # Extract clean daily returns from portfolio value series.
    @staticmethod
    def _clean_returns(portfolio_values: np.ndarray) -> np.ndarray:
        pv = np.array(portfolio_values, dtype=float)

        valid = np.isfinite(pv) & (pv > 0)
        if not np.all(valid):
            first_valid = np.argmax(valid)
            last_valid = len(pv) - np.argmax(valid[::-1])
            pv = pv[first_valid:last_valid]

        returns = np.diff(pv) / pv[:-1]

        returns = np.where(np.isfinite(returns), returns, 0.0)

        return returns

    # Run all statistical tests for a single confidence level.
    def _run_single_level(self, returns: np.ndarray,
                          confidence_level: float) -> Dict[str, Any]:
        var_series = compute_var(
            returns,
            confidence_level=confidence_level,
            method=self.var_method,
            window=self.var_window,
        )

        hit_sequence = identify_violations(returns, var_series)

        valid_mask = ~np.isnan(var_series)
        valid_hits = hit_sequence[valid_mask]

        total_valid = len(valid_hits)
        total_violations = int(np.sum(valid_hits))
        expected_violations = round((1 - confidence_level) * total_valid, 2)

        kupiec = KupiecTest(confidence_level, self.significance)
        kupiec_result = kupiec.test(valid_hits)

        christoffersen = ChristoffersenTest(confidence_level, self.significance)
        christoffersen_result = christoffersen.test(valid_hits)

        kup_pass = kupiec_result['result'] == 'PASS'
        ind_pass = christoffersen_result['independence']['result'] == 'PASS'
        cc_pass = christoffersen_result['conditional_coverage']['result'] == 'PASS'

        if kup_pass and ind_pass and cc_pass:
            reliability = 'RELIABLE — Risk forecasts correctly calibrated and violations independent'
        elif kup_pass and ind_pass:
            reliability = 'PARTIALLY RELIABLE — Coverage and independence OK, combined test marginal'
        elif kup_pass:
            reliability = 'UNRELIABLE — Coverage OK but violations are clustered (not independent)'
        elif ind_pass:
            reliability = 'UNRELIABLE — Violations independent but risk forecasts miscalibrated'
        else:
            reliability = 'UNRELIABLE — Risk forecasts miscalibrated and violations clustered'

        return {
            'summary': {
                'confidence_level': confidence_level,
                'var_method': self.var_method,
                'var_window': self.var_window,
                'total_observations': total_valid,
                'total_violations': total_violations,
                'expected_violations': expected_violations,
                'expected_violation_rate': round(1 - confidence_level, 4),
                'observed_violation_rate': round(
                    total_violations / max(1, total_valid), 6),
            },
            'kupiec_test': kupiec_result,
            'christoffersen_test': christoffersen_result,
            'reliability_decision': reliability,
            '_var_series': var_series,
            '_hit_sequence': hit_sequence,
            '_valid_mask': valid_mask,
        }

    # Execute all statistical backtesting tests at every confidence level.
    def run(self, results: Dict[str, Any]) -> Dict[str, Any]:
        portfolio_values = np.array(results['portfolio_values'], dtype=float)

        returns = self._clean_returns(portfolio_values)

        level_results = {}
        for cl in self.confidence_levels:
            level_results[cl] = self._run_single_level(returns, cl)

        benchmark = BenchmarkComparison(self.initial_capital)
        benchmark_result = benchmark.compare(results)

        stat_results = {
            'confidence_levels': self.confidence_levels,
            'var_method': self.var_method,
            'var_window': self.var_window,
            'levels': level_results,
            'benchmark_comparison': benchmark_result,
            '_returns': returns,
        }

        return stat_results

# Pretty-print the multi-level statistical backtesting report.
def print_statistical_results(stat_results: Dict[str, Any]) -> None:

    confidence_levels = stat_results['confidence_levels']
    var_method = stat_results['var_method']
    var_window = stat_results['var_window']

    print(f"\n{'=' * 70}")
    print("  STATISTICAL BACKTESTING RESULTS")
    print(f"{'=' * 70}")

    print(f"\n  VaR Configuration")
    print(f"  {'Method':<30} {var_method}")
    print(f"  {'Rolling Window':<30} {var_window or 'Expanding'}")
    print(f"  {'Confidence Levels':<30} {', '.join(f'{cl*100:.0f}%' for cl in confidence_levels)}")

    all_pass_global = True

    for cl in confidence_levels:
        level = stat_results['levels'][cl]
        summary = level['summary']
        kup = level['kupiec_test']
        chris = level['christoffersen_test']
        ind = chris['independence']
        cc = chris['conditional_coverage']

        pct_label = f"{cl * 100:.0f}%"

        print(f"\n{'─' * 70}")
        print(f"  VaR {pct_label} CONFIDENCE LEVEL")
        print(f"{'─' * 70}")

        print(f"\n  {'Total Observations':<35} {summary['total_observations']}")
        print(f"  {'Total Violations':<35} {summary['total_violations']}")
        print(f"  {'Expected Violations':<35} {summary['expected_violations']}")
        print(f"  {'Expected Violation Rate':<35} {summary['expected_violation_rate']:.4f}")
        print(f"  {'Observed Violation Rate':<35} {summary['observed_violation_rate']:.6f}")

        print(f"\n  ┌─ Kupiec POF Test (Unconditional Coverage) ─────────────────┐")
        print(f"  │  {'LR Statistic':<30} {kup['lr_statistic']:>10.4f}            │")
        print(f"  │  {'p-value':<30} {kup['p_value']:>10.6f}          │")
        print(f"  │  {'Critical Value (χ²₁)':<30} {kup['critical_value']:>10.4f}            │")
        _box_pass_fail(kup['result'])
        print(f"  │  Interpretation: {'observed violation rate ≈ expected → well calibrated' if kup['result'] == 'PASS' else 'violation rate deviates from expected → miscalibrated'}  │")
        print(f"  └────────────────────────────────────────────────────────────┘")

        print(f"\n  ┌─ Christoffersen Independence Test ──────────────────────────┐")
        print(f"  │  {'Transition Matrix:':<50}          │")
        print(f"  │    n00={ind['n00']:>5}  n01={ind['n01']:>5}  (from state 0)              │")
        print(f"  │    n10={ind['n10']:>5}  n11={ind['n11']:>5}  (from state 1)              │")
        print(f"  │  {'π (unconditional)':<30} {ind['pi_unconditional']:>10.6f}          │")
        print(f"  │  {'π₀₁ (0→1 transition)':<30} {ind['pi_01']:>10.6f}          │")
        print(f"  │  {'π₁₁ (1→1 transition)':<30} {ind['pi_11']:>10.6f}          │")
        print(f"  │  {'LR Statistic':<30} {ind['lr_statistic']:>10.4f}            │")
        print(f"  │  {'p-value':<30} {ind['p_value']:>10.6f}          │")
        _box_pass_fail(ind['result'])
        print(f"  │  Interpretation: {'violations are independent → stable model' if ind['result'] == 'PASS' else 'violations are clustered → unstable risk model'}  │")
        print(f"  └────────────────────────────────────────────────────────────┘")

        print(f"\n  ┌─ Conditional Coverage Test (Kupiec + Independence) ────────┐")
        print(f"  │  {'LR_cc = LR_uc + LR_ind':<30} {cc['lr_statistic']:>10.4f}            │")
        print(f"  │  {'p-value':<30} {cc['p_value']:>10.6f}          │")
        print(f"  │  {'Critical Value (χ²₂)':<30} {cc['critical_value']:>10.4f}            │")
        _box_pass_fail(cc['result'])
        print(f"  │  This is the FINAL statistical reliability score.           │")
        print(f"  └────────────────────────────────────────────────────────────┘")

        print(f"\n  ► Reliability Decision ({pct_label}): {level['reliability_decision']}")

        if kup['result'] == 'FAIL' or ind['result'] == 'FAIL' or cc['result'] == 'FAIL':
            all_pass_global = False

    bench = stat_results['benchmark_comparison']
    port = bench['portfolio']
    bm = bench['benchmark']
    comp = bench['comparison']

    print(f"\n{'─' * 70}")
    print(f"  BENCHMARK COMPARISON (Model vs Buy-and-Hold)")
    print(f"{'─' * 70}")
    print(f"\n  {'Metric':<30} {'Model':>12} {'Benchmark':>12}")
    print(f"  {'-' * 54}")
    print(f"  {'Total Return (%)':<30} {port['total_return_pct']:>12.2f} {bm['total_return_pct']:>12.2f}")
    print(f"  {'Annualized Return (%)':<30} {port['annualized_return_pct']:>12.2f} {bm['annualized_return_pct']:>12.2f}")
    print(f"  {'Sharpe Ratio':<30} {port['sharpe_ratio']:>12.4f} {bm['sharpe_ratio']:>12.4f}")
    print(f"  {'Max Drawdown (%)':<30} {port['max_drawdown_pct']:>12.2f} {bm['max_drawdown_pct']:>12.2f}")
    print(f"  {'Volatility (%)':<30} {port['volatility_pct']:>12.2f} {bm['volatility_pct']:>12.2f}")
    print(f"  {'Final Value ($)':<30} {port['final_value']:>12,.2f} {bm['final_value']:>12,.2f}")
    print(f"\n  {'Excess Return (%)':<30} {comp['excess_return_pct']:>+.2f}")
    print(f"  {'Tracking Error (%)':<30} {comp['tracking_error_pct']:.2f}")
    print(f"  {'Information Ratio':<30} {comp['information_ratio']:.4f}")
    print(f"  {'Win Rate (%)':<30} {comp['win_rate_pct']:.2f}  ({comp['win_days']}/{comp['total_days']} days)")
    print(f"\n  Assessment: {comp['assessment']}")

    print(f"\n{'=' * 70}")
    print("  SUMMARY STATISTICAL TABLE")
    print(f"{'=' * 70}")
    header = f"  {'Confidence':<12} {'Violations':>10} {'Expected':>10} {'Kupiec':>10} {'Indep.':>10} {'Cond.Cov':>10} {'Decision':>12}"
    print(header)
    print(f"  {'-' * 68}")
    for cl in confidence_levels:
        level = stat_results['levels'][cl]
        s = level['summary']
        k_res = level['kupiec_test']['result']
        i_res = level['christoffersen_test']['independence']['result']
        c_res = level['christoffersen_test']['conditional_coverage']['result']
        decision = 'RELIABLE' if (k_res == 'PASS' and i_res == 'PASS' and c_res == 'PASS') else 'UNRELIABLE'
        print(f"  {cl*100:>5.0f}%       {s['total_violations']:>10} {s['expected_violations']:>10.1f} {k_res:>10} {i_res:>10} {c_res:>10} {decision:>12}")

    print(f"\n{'=' * 70}")
    print("  OVERALL STATISTICAL VERDICT")
    print(f"{'=' * 70}")
    for cl in confidence_levels:
        level = stat_results['levels'][cl]
        kup = level['kupiec_test']
        ind = level['christoffersen_test']['independence']
        cc = level['christoffersen_test']['conditional_coverage']
        pct_label = f"{cl * 100:.0f}%"

        tests = [
            (f'Kupiec Coverage ({pct_label})', kup['result']),
            (f'Independence ({pct_label})', ind['result']),
            (f'Conditional Coverage ({pct_label})', cc['result']),
        ]
        for name, result in tests:
            marker = '[✓]' if result == 'PASS' else '[✗]'
            status = 'PASS' if result == 'PASS' else 'FAIL'
            print(f"  {marker} {name:<40} {status}")

    bench_marker = '[✓]' if comp['excess_return_pct'] >= 0 else '[✗]'
    bench_status = 'OUTPERFORMS' if comp['excess_return_pct'] >= 0 else 'UNDERPERFORMS'
    print(f"  {bench_marker} {'Benchmark Comparison':<40} {bench_status}")

    if all_pass_global:
        print(f"\n  >>> MODEL PASSES all statistical backtests at all confidence levels <<<")
    else:
        print(f"\n  >>> MODEL FAILS one or more statistical backtests <<<")
    print("=" * 70)

# Print pass/fail inside a box-drawing frame.
def _box_pass_fail(result: str) -> None:
    if result == 'PASS':
        print(f"  │  {'Result':<30} ✓ PASS (Do not reject H₀)     │")
    else:
        print(f"  │  {'Result':<30} ✗ FAIL (Reject H₀)            │")

def _pass_fail(result: str) -> None:
    if result == 'PASS':
        print(f"  {'Result':<30} ✓ PASS (Do not reject H₀)")
    else:
        print(f"  {'Result':<30} ✗ FAIL (Reject H₀)")
