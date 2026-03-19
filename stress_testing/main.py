"""Unified Stress Testing Pipeline Stress Testing Module for Algorithmic Trading Research Platform This produces a complete stress testing analysis in a single execution: 1. Load model and dataset (with interactive selection) 2. Configure stress testing parameters (with interactive configuration) 3. Run baseline backtest 4. Generate stressed market scenarios 5. Run backtests on stressed data 6. Analyze performance degradation 7. Calculate robustness indicators 8. Generate reports and visualizations Usage: python main_unified.py python main_unified.py --quick python main_unified.py --full"""
import sys
from pathlib import Path
import warnings
import argparse
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from stress_testing.engine import StressTestingEngine
from stress_testing.scenarios import (MarketCrashScenario, VolatilityShockScenario,
                      RegimeShiftScenario, SyntheticStressScenario)
from stress_testing.evaluation import (
    PerformanceDegradationAnalyzer,
    RobustnessMetricsCalculator,
    ScenarioComparator
)
from stress_testing.results import ResultCollector, ReportGenerator
from stress_testing.visualization import StressTestingVisualizer
from stress_testing.utils import setup_logger, log_section, ConfigLoader
from backtesting_engine import BacktestConfig
from model_loader import UniversalModelLoader
from model_interface import PredictionController, ActionMapper
from data_loader import DatasetManager
def print_banner():
    """Print welcome banner."""
"""================================================================================ UNIFIED STRESS TESTING PIPELINE Algorithmic Trading Research Platform Evaluate model robustness under extreme market conditions: * Market Crashes      * Volatility Shocks * Regime Shifts       * Synthetic Scenarios ================================================================================"""
def select_dataset() -> str:
    """Let user select a dataset category and file."""
    dm = DatasetManager()
    print("\n" + "=" * 80)
    print("SELECT DATASET CATEGORY")
    print("=" * 80)
    categories = dm.list_categories()
    for i, cat in enumerate(categories, 1):
        print(f"  {i}. {cat}")
    print("-" * 80)
    while True:
        try:
            idx = int(input("Enter choice: ").strip())
            if 1 <= idx <= len(categories):
                break
        except ValueError:
            pass
        print(f"  Enter a number between 1 and {len(categories)}")
    category = categories[idx - 1]
    datasets = dm.list_datasets(category)
    print(f"\n{'=' * 80}")
    print(f"AVAILABLE {category.upper()} DATASETS")
    print("=" * 80)
    for ds in datasets:
        print(f"  {ds['id']}. {ds['name']}  ({ds['size_mb']} MB)")
    print("-" * 80)
    while True:
        try:
            idx = int(input(f"Select dataset (1-{len(datasets)}): ").strip())
            if 1 <= idx <= len(datasets):
                return datasets[idx - 1]['path']
        except ValueError:
            pass
        print(f"  Enter a number between 1 and {len(datasets)}")
def select_model() -> str:
    """Let user select a .pt model."""
    models_dir = Path(__file__).parent.parent / 'backtesting_tool' / 'models'
    models = sorted(models_dir.glob("*.pt")) if models_dir.exists() else []
    print(f"\n{'=' * 80}")
    print("MODEL SELECTION (TorchScript .pt files only)")
    print("=" * 80)
    if models:
        for i, m in enumerate(models, 1):
            size_mb = m.stat().st_size / (1024 * 1024)
            print(f"  {i}. {m.name}  ({size_mb:.2f} MB)")
        print(f"  {len(models) + 1}. Enter custom path")
        print("-" * 80)
        while True:
            try:
                idx = int(input(f"Select model (1-{len(models) + 1}): ").strip())
                if 1 <= idx <= len(models):
                    return str(models[idx - 1])
                elif idx == len(models) + 1:
                    path = input("  Model path (.pt): ").strip()
                    if not path.endswith('.pt'):
                        print("  [WARNING] Only .pt (TorchScript) files are supported.")
                    return path
            except ValueError:
                pass
            print(f"  Enter a number between 1 and {len(models) + 1}")
    else:
        print("  No .pt files found in models/")
        path = input("  Enter full model path: ").strip()
        return path
def configure_backtest() -> BacktestConfig:
    """Ask for backtesting parameters."""
    print(f"\n{'=' * 80}")
    print("BACKTESTING PARAMETERS")
    print("=" * 80)
    capital = float(input("  Initial capital (default $100,000): ").strip() or "100000")
    comm = float(input("  Commission % per trade (default 0.1): ").strip() or "0.1")
    print("\n  Trading Strategy:")
    print("    1. Simple  (buy if predicted > current, sell otherwise)")
    print("    2. Threshold  (trade only if change > threshold %)")
    choice = input("  Select (1 or 2, default 1): ").strip() or "1"
    threshold = 0.0
    strategy = 'simple'
    if choice == '2':
        strategy = 'threshold'
        threshold = float(input("  Threshold % (default 0.5): ").strip() or "0.5")
    seq_len = int(input("  Sequence length / lookback days (default 60): ").strip() or "60")
    return BacktestConfig(
        initial_capital=capital,
        commission_pct=comm,
        strategy=strategy,
        threshold_pct=threshold,
        sequence_length=seq_len,
    )
def configure_stress_scenarios() -> dict:
    """Ask for stress testing scenario configuration."""
    print(f"\n{'=' * 80}")
    print("STRESS TESTING SCENARIO CONFIGURATION")
    print("=" * 80)
    print("\n  Scope of Testing:")
    print("    1. Quick   (2 scenarios per type = 8 total) - Fast testing")
    print("    2. Standard (4 levels per scenario type = ~16 total) - Balanced")
    print("    3. Full    (All 4 levels per type = 20+ scenarios) - Comprehensive")
    scope = input("  Select scope (1, 2, or 3, default 2): ").strip() or "2"
    print("\n  Scenario Types to Run:")
    print("    1. Market Crashes     (simulate sudden price drops)")
    print("    2. Volatility Shocks  (amplify market volatility)")
    print("    3. Regime Shifts      (change market conditions)")
    print("    4. Synthetic Tests    (bootstrap and synthetic data)")
    print("\n  Which scenarios to include? (comma-separated: 1,3,4 or 'all')")
    scenarios_input = input("  Select (default: all): ").strip().lower() or "all"
    if scenarios_input == "all":
        enabled_scenarios = {'market_crash', 'volatility_shock', 'regime_shift', 'synthetic_stress'}
    else:
        enabled_scenarios = set()
        scenario_map = {
            '1': 'market_crash',
            '2': 'volatility_shock',
            '3': 'regime_shift',
            '4': 'synthetic_stress'
        }
        for idx in scenarios_input.split(','):
            idx = idx.strip()
            if idx in scenario_map:
                enabled_scenarios.add(scenario_map[idx])
    num_runs = 1
    if 'synthetic_stress' in enabled_scenarios:
        num_runs = int(input("  Number of bootstrap runs for synthetic scenarios (default 5): ").strip() or "5")
    return {
        'scope': scope,
        'enabled_scenarios': enabled_scenarios,
        'num_runs': num_runs,
    }
def main():
    """Main pipeline."""
    print_banner()
    parser = argparse.ArgumentParser(description='Unified Stress Testing Pipeline')
    parser.add_argument('--quick', action='store_true', help='Quick mode (limited scenarios)')
    parser.add_argument('--full', action='store_true', help='Full mode (all scenarios)')
    parser.add_argument('--model', type=str, help='Model path (skip selection)')
    parser.add_argument('--data', type=str, help='Dataset path (skip selection)')
    args = parser.parse_args()
    if args.data:
        dataset_path = args.data
        print(f"\n[INFO] Using dataset: {dataset_path}")
    else:
        dataset_path = select_dataset()
    if args.model:
        model_path = args.model
        print(f"[INFO] Using model: {model_path}")
    else:
        model_path = select_model()
    if not Path(model_path).exists():
        print(f"\n[ERROR] File not found: {model_path}")
        return
    if args.quick or args.full:
        config = BacktestConfig()
        print(f"\n[INFO] Using default backtesting config: capital={config.initial_capital}, commission={config.commission_pct}%")
    else:
        config = configure_backtest()
    if args.quick:
        stress_config = {'scope': '1', 'enabled_scenarios': {'market_crash', 'volatility_shock', 'regime_shift', 'synthetic_stress'}, 'num_runs': 1}
        print(f"[INFO] Using quick mode (8 scenarios)")
    elif args.full:
        stress_config = {'scope': '3', 'enabled_scenarios': {'market_crash', 'volatility_shock', 'regime_shift', 'synthetic_stress'}, 'num_runs': 5}
        print(f"[INFO] Using full mode (all scenarios)")
    else:
        stress_config = configure_stress_scenarios()
    print(f"\n{'=' * 80}")
    print("DATA LOADING")
    print("=" * 80)
    dm = DatasetManager()
    stock_data = dm.load_dataset(dataset_path)
    tickers = list(stock_data.keys())
    print(f"  Portfolio stocks: {', '.join(tickers)} ({len(tickers)} total)")
    print(f"\n{'=' * 80}")
    print("MODEL LOADING")
    print("=" * 80)
    loader = UniversalModelLoader()
    model, model_info = loader.load(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  Model: {model_info['architecture']}")
    print(f"  Parameters: {model_info['total_parameters']:,}")
    predictor = PredictionController(model, device)
    print(f"\n{'=' * 80}")
    print("STRESS TESTING ENGINE")
    print("=" * 80)
    engine = StressTestingEngine(
        model_path=model_path,
        dataset_path=dataset_path,
        predictor=predictor,
        action_mapper=ActionMapper(),
        config=config,
        feature_columns=model_info['feature_columns'],
        logger_name="StressTestingEngine"
    )
    data, ticker = engine.load_dataset()
    baseline_metrics = engine.run_baseline_backtest(data, ticker)
    if stress_config['scope'] == '1':
        crash_levels = [0.20, 0.50]
        vol_factors = [3.0]
        num_synthetic = 1
    elif stress_config['scope'] == '3':
        crash_levels = [0.10, 0.20, 0.30, 0.50]
        vol_factors = [2.0, 3.0, 5.0]
        num_synthetic = stress_config.get('num_runs', 5)
    else:
        crash_levels = [0.15, 0.30, 0.50]
        vol_factors = [2.0, 3.0]
        num_synthetic = 2
    all_results = {}
    enabled = stress_config['enabled_scenarios']
    if 'market_crash' in enabled:
        results = engine.run_market_crash_scenarios(data, ticker, crash_levels=crash_levels)
        all_results.update(results)
    if 'volatility_shock' in enabled:
        results = engine.run_volatility_shock_scenarios(data, ticker, vol_factors=vol_factors)
        all_results.update(results)
    if 'regime_shift' in enabled:
        results = engine.run_regime_shift_scenarios(data, ticker)
        all_results.update(results)
    if 'synthetic_stress' in enabled:
        results = engine.run_synthetic_scenarios(
            data, ticker,
            methods=["gbm", "bootstrap"],
            num_per_method=num_synthetic
        )
        all_results.update(results)
    print(f"\n{'=' * 80}")
    print("ANALYSIS & REPORTING")
    print("=" * 80)
    degradation_analyzer = PerformanceDegradationAnalyzer()
    degradation_reports = degradation_analyzer.analyze_multiple_scenarios(all_results)
    robustness_calculator = RobustnessMetricsCalculator()
    robustness_report = robustness_calculator.calculate(degradation_reports)
    comparator = ScenarioComparator()
    comparison_df = comparator.create_comparison_dataframe(all_results)
    print(f"\n  Overall Robustness Score: {robustness_report.overall_robustness_score:.1f}/100")
    print(f"  Profitable Scenarios: {robustness_report.profitable_scenarios}/{robustness_report.num_scenarios}")
    print(f"\n{'=' * 80}")
    print("GENERATING REPORTS")
    print("=" * 80)
    results_dir = Path(__file__).parent / "results" / "outputs"
    results_dir.mkdir(parents=True, exist_ok=True)
    collector = ResultCollector(str(results_dir))
    collector.add_baseline_metrics(baseline_metrics)
    for scenario_name, result in all_results.items():
        collector.add_scenario_result(scenario_name, result)
    for report in degradation_reports:
        collector.add_degradation_report(report)
    collector.add_robustness_report(robustness_report)
    report_gen = ReportGenerator(str(results_dir))
    html_path = report_gen.generate_html_report(
        baseline_metrics, robustness_report, degradation_reports, comparison_df
    )
    print(f"  HTML Report: {html_path.name}")
    visualizer = StressTestingVisualizer(str(results_dir))
    visualizer.plot_return_degradation(degradation_reports)
    visualizer.plot_sharpe_comparison(degradation_reports)
    visualizer.plot_drawdown_comparison(degradation_reports)
    visualizer.plot_robustness_scores(robustness_report)
    print(f"  Visualizations generated")
    collector.save_results_json()
    collector.save_comparison_csv(comparison_df)
    collector.save_degradation_report_csv()
    print(f"  Data exports complete")
    print(f"\n{'=' * 80}")
    print("STRESS TESTING COMPLETE")
    print("=" * 80)
    print(f"\n[OK] Stress testing completed successfully!")
    print(f"\nResults Summary:")
    print(f"  * Total Scenarios: {robustness_report.num_scenarios}")
    print(f"  * Profitable: {robustness_report.profitable_scenarios}")
    print(f"    ({robustness_report.profitable_percentage:.1f}%)")
    print(f"\nRobustness Scores (0-100):")
    print(f"  * Resilience: {robustness_report.resilience_score:.1f}")
    print(f"  * Stability: {robustness_report.stability_score:.1f}")
    print(f"  * Drawdown Resilience: {robustness_report.drawdown_resilience_score:.1f}")
    print(f"  * Overall: {robustness_report.overall_robustness_score:.1f}")
    if robustness_report.overall_robustness_score >= 70:
        recommendation = "[OK] EXCELLENT - Strategy is robust"
    elif robustness_report.overall_robustness_score >= 50:
        recommendation = "[WARN] GOOD - Strategy is reasonably robust"
    else:
        recommendation = "[WARN] FAIR - Strategy has weaknesses, needs improvement"
    print(f"\nRecommendation:\n  {recommendation}")
    print(f"\nResults saved to: results/outputs")
    print(f"Open 'stress_test_report.html' in a web browser for detailed analysis.")
if __name__ == '__main__':
    main()

