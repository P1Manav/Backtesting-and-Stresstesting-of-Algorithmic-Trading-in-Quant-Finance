"""Backtesting tool main - loads model and dataset, runs backtest with statistical analysis"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from data_loader import DatasetManager
from model_loader import UniversalModelLoader
from model_interface import PredictionController, ActionMapper
from backtesting_engine import BacktestConfig, BacktestingEngine
from metrics import (MetricsCalculator, StatisticalBacktester, print_statistical_results,
                    RobustnessAnalyzer, print_robustness_results)
from visualization import (BacktestVisualizer, ReportGenerator,
                          StatisticalVisualizer, RobustnessVisualizer)

def print_banner():
    """Display welcome banner"""
    print("""
==================================================================

         UNIVERSAL STOCK PRICE BACKTESTING TOOL

       Loads TorchScript (.pt) models directly
       Architecture & weights embedded in the file
       Works with LSTM, GRU, RNN, BiLSTM, CNN …

==================================================================
    """)

def select_dataset() -> str:
    """Select a dataset category and file"""
    dm = DatasetManager()

    print("\n" + "=" * 60)
    print("SELECT DATASET CATEGORY")
    print("=" * 60)
    categories = dm.list_categories()
    for i, cat in enumerate(categories, 1):
        print(f"  {i}. {cat}")
    print("-" * 60)

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
    print(f"\n{'=' * 60}")
    print(f"AVAILABLE {category.upper()} DATASETS")
    print("=" * 60)
    for ds in datasets:
        print(f"  {ds['id']}. {ds['name']}  ({ds['size_mb']} MB)")
    print("-" * 60)

    while True:
        try:
            idx = int(input(f"Select dataset (1-{len(datasets)}): ").strip())
            if 1 <= idx <= len(datasets):
                return datasets[idx - 1]['path']
        except ValueError:
            pass
        print(f"  Enter a number between 1 and {len(datasets)}")

def select_model() -> str:
    """Select a TorchScript .pt model"""
    models_dir = Path(__file__).parent / 'models'
    models = sorted(models_dir.glob("*.pt")) if models_dir.exists() else []

    print(f"\n{'=' * 60}")
    print("MODEL SELECTION  (TorchScript .pt files only)")
    print("=" * 60)

    if models:
        for i, m in enumerate(models, 1):
            size_mb = m.stat().st_size / (1024 * 1024)
            print(f"  {i}. {m.name}  ({size_mb:.2f} MB)")
        print(f"  {len(models) + 1}. Enter custom path")
        print("-" * 60)

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
        print("  Note: Only TorchScript (.pt) models are supported.")
        print("  Convert .pth models using: torch.jit.trace() + torch.jit.save()")
        return input("  Enter full model path: ").strip()

def configure_backtest() -> BacktestConfig:
    """Configure backtesting parameters"""
    print(f"\n{'=' * 60}")
    print("BACKTESTING PARAMETERS")
    print("=" * 60)

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

def configure_statistical_backtest() -> dict:
    """Configure statistical backtesting parameters"""
    print(f"\n{'=' * 60}")
    print("STATISTICAL BACKTESTING PARAMETERS")
    print("=" * 60)

    print("\n  Confidence Levels: 95% and 99% (both evaluated)")
    confidence_levels = [0.95, 0.99]

    significance = float(
        input(f"  Significance level α for tests (default 0.05): ").strip() or "0.05")

    print("\n  VaR Estimation Method:")
    print("    1. Historical Simulation  (non-parametric, recommended)")
    print("    2. Parametric  (Normal distribution)")
    vm_choice = input("  Select (1 or 2, default 1): ").strip() or "1"
    var_method = 'parametric' if vm_choice == '2' else 'historical'

    print("\n  Rolling Window Size (100–250 days):")
    var_window = input("  Window size (default 250, 0=expanding): ").strip() or "250"
    var_window = int(var_window)
    if var_window == 0:
        var_window = None
    elif var_window < 100:
        print(f"  [WARNING] Window {var_window} is below recommended minimum (100). Using 100.")
        var_window = 100
    elif var_window > 250:
        print(f"  [WARNING] Window {var_window} exceeds recommended maximum (250). Using 250.")
        var_window = 250

    print(f"\n  Configuration:")
    print(f"    Confidence Levels : {', '.join(f'{cl*100:.0f}%' for cl in confidence_levels)}")
    print(f"    Significance (α)  : {significance}")
    print(f"    VaR Method        : {var_method}")
    print(f"    Rolling Window    : {var_window or 'Expanding'}")

    return {
        'confidence_levels': confidence_levels,
        'significance': significance,
        'var_method': var_method,
        'var_window': var_window,
    }

def main():
    """Execute backtesting workflow"""
    print_banner()

    dataset_path = select_dataset()

    model_path = select_model()
    if not Path(model_path).exists():
        print(f"\n  [ERROR] File not found: {model_path}")
        return

    config = configure_backtest()

    print(f"\n{'=' * 60}")
    print("DATA LOADING")
    print("=" * 60)
    dm = DatasetManager()
    stock_data = dm.load_dataset(dataset_path)

    tickers = list(stock_data.keys())
    print(f"\n  Portfolio stocks: {', '.join(tickers)} ({len(tickers)} total)")

    print(f"\n{'=' * 60}")
    print("MODEL LOADING  (TorchScript .pt — architecture embedded)")
    print("=" * 60)
    loader = UniversalModelLoader()
    model, model_info = loader.load(model_path)

    device = loader.device
    feature_columns = model_info['feature_columns']

    print(f"\n  Features used  : {feature_columns}")
    print(f"  Device         : {device}")

    predictor = PredictionController(model, device)
    mapper = ActionMapper(strategy=config.strategy,
                          threshold_pct=config.threshold_pct)

    print(f"\n{'=' * 60}")
    print("BACKTESTING SIMULATION")
    print("=" * 60)
    engine = BacktestingEngine(config, predictor, feature_columns, mapper)
    results = engine.run(stock_data)

    print(f"\n{'=' * 60}")
    print("PERFORMANCE METRICS")
    print("=" * 60)
    calc = MetricsCalculator(config.initial_capital)
    metrics = calc.calculate(results)

    print(f"\n{'=' * 60}")
    print("ROBUSTNESS VALIDATION")
    print("=" * 60)
    robustness_analyzer = RobustnessAnalyzer(
        n_folds=5, n_bootstrap=1000, n_simulations=1000, random_seed=42)
    robustness_results = robustness_analyzer.run(results, config.initial_capital)
    print_robustness_results(robustness_results)

    stat_config = configure_statistical_backtest()
    stat_backtester = StatisticalBacktester(
        confidence_levels=stat_config['confidence_levels'],
        significance=stat_config['significance'],
        var_method=stat_config['var_method'],
        var_window=stat_config['var_window'],
        initial_capital=config.initial_capital,
    )
    stat_results = stat_backtester.run(results)
    print_statistical_results(stat_results)

    results_dir = (Path(__file__).parent / 'results'
                   / f"backtest_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")

    print(f"\n{'=' * 60}")
    print("SAVING RESULTS")
    print("=" * 60)

    viz = BacktestVisualizer(str(results_dir))
    viz.generate_all(results)

    rob_viz = RobustnessVisualizer(str(results_dir))
    rob_viz.generate_all(robustness_results)

    stat_viz = StatisticalVisualizer(str(results_dir))
    stat_viz.generate_all(results, stat_results)

    rg = ReportGenerator(str(results_dir))
    rg.save(results, metrics, stat_results, robustness_results)

    print(f"\n{'=' * 60}")
    print("BACKTESTING COMPLETE!")
    print("=" * 60)
    print(f"\n  Results folder: {results_dir}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  [INTERRUPTED] Cancelled by user.")
    except Exception as e:
        print(f"\n\n  [ERROR] {e}")
        import traceback
        traceback.print_exc()

