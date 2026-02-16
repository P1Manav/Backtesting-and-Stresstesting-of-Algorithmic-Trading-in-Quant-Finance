import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from data_loader import DatasetManager
from model_loader import UniversalModelLoader
from model_interface import PredictionController, FeatureConverter, ActionMapper
from backtesting_engine import BacktestConfig, BacktestingEngine
from metrics import MetricsCalculator
from visualization import BacktestVisualizer, ReportGenerator

def print_banner():
    print("""
==================================================================

         UNIVERSAL STOCK PRICE BACKTESTING TOOL

       Loads TorchScript (.pt) models directly
       Architecture & weights embedded in the file
       Works with LSTM, GRU, RNN, BiLSTM, CNN …

==================================================================
    """)


def select_dataset() -> str:
    """Let user choose a dataset category and file."""
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
    """Let user pick a .pt model from the models/ folder or enter a path."""
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
    """Ask for backtesting parameters."""
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


def main():
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
    df = dm.load_dataset(dataset_path)

    print(f"\n{'=' * 60}")
    print("MODEL LOADING  (TorchScript .pt — architecture embedded)")
    print("=" * 60)
    loader = UniversalModelLoader()
    model, model_info = loader.load(model_path)

    device = loader.device
    feature_columns = model_info['feature_columns']

    print(f"\n  Features used  : {feature_columns}")
    print(f"  Device         : {device}")

    converter = FeatureConverter(
        feature_columns=feature_columns,
        sequence_length=config.sequence_length,
    )
    predictor = PredictionController(model, device)
    mapper = ActionMapper(strategy=config.strategy,
                          threshold_pct=config.threshold_pct)

    print(f"\n{'=' * 60}")
    print("BACKTESTING SIMULATION")
    print("=" * 60)
    engine = BacktestingEngine(config, predictor, converter, mapper)
    results = engine.run(df)

    print(f"\n{'=' * 60}")
    print("PERFORMANCE METRICS")
    print("=" * 60)
    calc = MetricsCalculator(config.initial_capital)
    metrics = calc.calculate(results)

    results_dir = (Path(__file__).parent / 'results'
                   / f"backtest_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")

    print(f"\n{'=' * 60}")
    print("SAVING RESULTS")
    print("=" * 60)

    viz = BacktestVisualizer(str(results_dir))
    viz.generate_all(results)

    rg = ReportGenerator(str(results_dir))
    rg.save(results, metrics)

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
