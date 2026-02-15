

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import modules
from data_loader import DatasetManager, DataValidator, KaggleFetcher
from model_loader import ModelLoader, ModelAnalyzer
from model_interface import PredictionController
from backtesting_engine import BacktestingEngine, BacktestingConfig
from metrics import MetricsCalculator, PerformanceMetrics
from visualization import BacktestVisualizer, ReportGenerator


def print_banner():
    """Print the application banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║              ALGORITHMIC TRADING BACKTESTING TOOL            ║
    ║                                                              ║
    ║         Historical Backtesting for ML/DL/RL Models           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def select_dataset_source() -> str:
    """
    Ask user to select dataset source.
    
    Returns:
        'local' or 'kaggle'
    """
    print("\n" + "="*60)
    print("STEP 1: SELECT DATA SOURCE")
    print("="*60)
    print("\nOptions:")
    print("  1. Use local dataset (from data_repository/backtesting/)")
    print("  2. Fetch dataset from Kaggle (requires API key)")
    print("  3. Generate sample dataset for testing")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return {'1': 'local', '2': 'kaggle', '3': 'sample'}[choice]
        print("Invalid choice. Please enter 1, 2, or 3.")


def get_local_dataset(dataset_manager: DatasetManager) -> Optional[str]:
    """
    Let user select a local dataset.
    
    Returns:
        Path to selected dataset or None
    """
    datasets = dataset_manager.list_available_datasets()
    
    if not datasets:
        print("\nNo datasets found in data_repository/backtesting/")
        print("Would you like to generate a sample dataset? (y/n): ", end="")
        if input().strip().lower() == 'y':
            return dataset_manager.create_sample_dataset()
        return None
    
    print("\nAvailable datasets:")
    for i, ds in enumerate(datasets, 1):
        print(f"  {i}. {ds['name']} ({ds['size_mb']:.2f} MB)")
    
    while True:
        try:
            choice = int(input(f"\nSelect dataset (1-{len(datasets)}): "))
            if 1 <= choice <= len(datasets):
                return datasets[choice - 1]['path']
        except ValueError:
            pass
        print("Invalid choice. Please enter a valid number.")


def get_kaggle_dataset(fetcher: KaggleFetcher) -> Optional[str]:
    """
    Let user fetch a dataset from Kaggle.
    
    Returns:
        Path to downloaded dataset or None
    """
    if not fetcher.is_available():
        print("\nKaggle API is not available.")
        print("Please install and configure it:")
        print("  pip install kaggle")
        print("  Configure ~/.kaggle/kaggle.json with your API credentials")
        return None
    
    print("\nPopular financial datasets:")
    popular = fetcher.list_popular_datasets()
    for i, ds in enumerate(popular, 1):
        print(f"  {i}. {ds['key']} ({ds['kaggle_path']})")
    print(f"  {len(popular)+1}. Enter custom Kaggle dataset path")
    
    while True:
        try:
            choice = int(input(f"\nSelect dataset (1-{len(popular)+1}): "))
            if 1 <= choice <= len(popular):
                folder = fetcher.download_popular_dataset(popular[choice-1]['key'])
                # Find first CSV file
                csv_files = list(Path(folder).glob('*.csv'))
                if csv_files:
                    return str(csv_files[0])
                print("No CSV files found in downloaded dataset.")
                return None
            elif choice == len(popular) + 1:
                path = input("Enter Kaggle dataset path (e.g., 'username/dataset'): ")
                folder = fetcher.download_dataset(path)
                csv_files = list(Path(folder).glob('*.csv'))
                if csv_files:
                    print(f"Found files: {[f.name for f in csv_files]}")
                    return str(csv_files[0])
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None


def get_model_path() -> str:
    """
    Ask user for the model file path.
    
    Returns:
        Path to the model file
    """
    print("\n" + "="*60)
    print("STEP 2: SELECT MODEL")
    print("="*60)
    print("\nEnter the path to your PyTorch model file (.pth)")
    print("Or press Enter to use a demo random model")
    
    while True:
        path = input("\nModel path: ").strip()
        
        if path == '':
            return 'demo'
        
        if os.path.exists(path):
            if path.endswith(('.pth', '.pt', '.pkl')):
                return path
            print("File must be a .pth, .pt, or .pkl file.")
        else:
            print(f"File not found: {path}")


def create_demo_model():
    """Create a simple demo model for testing."""
    import torch
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        """Simple LSTM model for demo purposes."""
        
        def __init__(self, input_size=5, hidden_size=32, output_size=3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            self.softmax = nn.Softmax(dim=-1)
        
        def forward(self, x):
            # x: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)
            # Take last output
            last_out = lstm_out[:, -1, :]
            out = self.fc(last_out)
            return self.softmax(out)
    
    model = SimpleModel()
    print("\nCreated demo LSTM model with:")
    print("  - Input: (batch, sequence, 5 features)")
    print("  - Output: 3 classes (Short, Flat, Long)")
    return model


def get_backtest_config() -> BacktestingConfig:
    """
    Get backtesting configuration from user.
    
    Returns:
        BacktestingConfig object
    """
    print("\n" + "="*60)
    print("STEP 3: CONFIGURE BACKTEST")
    print("="*60)
    
    print("\nUse default configuration? (y/n): ", end="")
    if input().strip().lower() == 'y':
        return BacktestingConfig()
    
    config = BacktestingConfig()
    
    try:
        val = input(f"Initial capital [{config.initial_capital}]: ").strip()
        if val:
            config.initial_capital = float(val)
        
        val = input(f"Commission rate % [{config.commission_rate*100}]: ").strip()
        if val:
            config.commission_rate = float(val) / 100
        
        val = input(f"Slippage % [{config.slippage*100}]: ").strip()
        if val:
            config.slippage = float(val) / 100
        
        val = input(f"Window size [{config.window_size}]: ").strip()
        if val:
            config.window_size = int(val)
        
        val = input(f"Allow short selling? (y/n) [{'y' if config.allow_short else 'n'}]: ").strip()
        if val:
            config.allow_short = val.lower() == 'y'
            
    except ValueError as e:
        print(f"Invalid input: {e}. Using defaults.")
    
    return config


def run_backtesting(data_path: str,
                    model,
                    config: BacktestingConfig,
                    results_path: Path) -> Dict[str, Any]:
    """
    Run the complete backtesting pipeline.
    
    Args:
        data_path: Path to dataset
        model: PyTorch model
        config: Backtesting configuration
        results_path: Path to save results
        
    Returns:
        Backtesting results dictionary
    """
    print("\n" + "="*60)
    print("RUNNING BACKTEST")
    print("="*60)
    
    # Load and validate data
    print("\n[1/5] Loading dataset...")
    dataset_manager = DatasetManager()
    data = dataset_manager.load_dataset(dataset_path=data_path)
    
    # Validate data
    validator = DataValidator()
    validator.validate_for_backtesting(data)
    
    info = dataset_manager.get_dataset_info(data)
    print(f"  Loaded {info['num_rows']} rows")
    if info['date_range']:
        print(f"  Date range: {info['date_range']['start']} to {info['date_range']['end']}")
    
    # Analyze model
    print("\n[2/5] Analyzing model...")
    analyzer = ModelAnalyzer(model)
    analysis = analyzer.analyze()
    print(f"  Model type: {analysis['model_type']}")
    print(f"  Output type: {analysis['output_type']}")
    print(f"  Parameters: {analysis['parameters']['total']:,}")
    
    # Initialize engine
    print("\n[3/5] Initializing backtesting engine...")
    engine = BacktestingEngine(config)
    
    model_config = {
        'window_size': config.window_size,
        'output_type': analysis['output_type']
    }
    
    engine.initialize(model, data, model_config)
    
    # Run backtest
    print("\n[4/5] Running backtest simulation...")
    results = engine.run(data)
    
    # Generate visualizations
    print("\n[5/5] Generating outputs...")
    
    # Create visualizer
    plots_path = results_path / "plots"
    visualizer = BacktestVisualizer(save_path=str(plots_path))
    visualizer.save_all_plots(results)
    
    # Generate report
    report_gen = ReportGenerator(str(results_path))
    report = report_gen.generate_full_report(results, save_files=True)
    
    return results


def main():
    """Main entry point for the backtesting tool."""
    print_banner()
    
    # Setup paths
    results_path = project_root / "results"
    results_path.mkdir(exist_ok=True)
    
    try:
        # Step 1: Select data source
        source = select_dataset_source()
        
        dataset_manager = DatasetManager()
        kaggle_fetcher = KaggleFetcher()
        
        if source == 'local':
            data_path = get_local_dataset(dataset_manager)
        elif source == 'kaggle':
            data_path = get_kaggle_dataset(kaggle_fetcher)
        else:  # sample
            data_path = dataset_manager.create_sample_dataset()
        
        if not data_path:
            print("\nNo dataset selected. Exiting.")
            return
        
        print(f"\nUsing dataset: {data_path}")
        
        # Step 2: Select model
        model_path = get_model_path()
        
        if model_path == 'demo':
            model = create_demo_model()
        else:
            loader = ModelLoader()
            model = loader.load_model(model_path)
        
        # Step 3: Configure backtest
        config = get_backtest_config()
        
        # Step 4: Run backtest
        results = run_backtesting(data_path, model, config, results_path)
        
        # Print final summary
        print("\n" + "="*60)
        print("BACKTEST COMPLETE")
        print("="*60)
        
        metrics = results.get('metrics', {})
        portfolio = results.get('portfolio', {})
        
        print(f"\n  Initial Capital:     ${portfolio.get('initial_capital', 0):>15,.2f}")
        print(f"  Final Value:         ${portfolio.get('final_value', 0):>15,.2f}")
        print(f"  Total Return:        {metrics.get('total_return_pct', 0):>15.2f}%")
        print(f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):>15.2f}")
        print(f"  Max Drawdown:        {metrics.get('max_drawdown_pct', 0):>15.2f}%")
        
        print(f"\n  Results saved to: {results_path}")
        print("\n" + "="*60)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
