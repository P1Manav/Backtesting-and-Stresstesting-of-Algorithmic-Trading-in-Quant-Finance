"""Universal backtesting tool for ML/DL/RL models trained on OHLCV data."""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from data_loader import DatasetManager
from data_loader.generic_preprocessor import GenericPreprocessor
from model_loader import ModelLoader
from model_loader.model_presets import PRESETS, get_preset, list_presets, create_custom_preset
from visualization import BacktestVisualizer
import matplotlib.pyplot as plt


def print_banner():
    banner = """
==================================================================

         UNIVERSAL STOCK PRICE BACKTESTING TOOL               

          Works with ANY model trained on OHLCV data          

==================================================================
    """
    print(banner)


def select_dataset_category():
    print("\n" + "=" * 70)
    print("SELECT DATASET CATEGORY")
    print("=" * 70)
    print("1. Backtesting Datasets (Normal market conditions)")
    print("2. Stress Testing Datasets (Crisis periods)")
    print("-" * 70)
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            return 'backtesting' if choice == '1' else 'stresstesting'
        print("Invalid choice. Enter 1 or 2.")


def list_and_select_dataset(category: str, data_root: Path):
    category_path = data_root / category
    
    datasets = []
    for idx, file in enumerate(sorted(category_path.glob("*.csv")), 1):
        size_mb = file.stat().st_size / (1024 * 1024)
        datasets.append({
            'id': idx,
            'name': file.name,
            'path': str(file),
            'size_mb': round(size_mb, 2)
        })
    
    print("\n" + "=" * 70)
    print(f"AVAILABLE {category.upper()} DATASETS")
    print("=" * 70)
    
    for ds in datasets:
        print(f"{ds['id']}. {ds['name']} ({ds['size_mb']} MB)")
    
    print("-" * 70)
    
    while True:
        try:
            choice = int(input(f"Select dataset (1-{len(datasets)}): ").strip())
            if 1 <= choice <= len(datasets):
                return datasets[choice - 1]['path']
            print(f"Enter a number between 1 and {len(datasets)}")
        except ValueError:
            print("Invalid input. Enter a number.")


def select_model_path():
    """Select model file."""
    models_dir = Path(__file__).parent / 'models'
    
    print("\n" + "=" * 70)
    print("MODEL SELECTION")
    print("=" * 70)
    
    models = list(models_dir.glob("*.pth")) if models_dir.exists() else []
    
    if models:
        print("\nAvailable models in models/ folder:")
        for idx, model in enumerate(models, 1):
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"{idx}. {model.name} ({size_mb:.2f} MB)")
        print(f"{len(models) + 1}. Enter custom path")
        print("-" * 70)
        
        while True:
            try:
                choice = int(input(f"Select model (1-{len(models) + 1}): ").strip())
                if 1 <= choice <= len(models):
                    return str(models[choice - 1])
                elif choice == len(models) + 1:
                    return input("Enter model path: ").strip()
                print(f"Enter number between 1 and {len(models) + 1}")
            except ValueError:
                print("Invalid input. Enter a number.")
    else:
        print("\nNo models found in models/ folder.")
        print("Please provide the full path to your .pth file.")
        return input("Model path: ").strip()


def select_model_config():
    print("\n" + "=" * 70)
    print("MODEL CONFIGURATION")
    print("=" * 70)
    print("\nHow was your model trained?")
    print("Select the preprocessing configuration that matches your training:")
    
    list_presets()
    print("\n8. Create custom configuration")
    print("-" * 70)
    
    while True:
        choice = input("Select configuration (1-8): ").strip()
        if choice in PRESETS:
            return get_preset(choice)
        elif choice == '8':
            return create_custom_preset()
        print("Invalid choice. Enter 1-8.")


def configure_backtesting():
    print("\n" + "=" * 70)
    print("BACKTESTING PARAMETERS")
    print("=" * 70)
    
    initial_capital = float(input("\nInitial capital (default: $100,000): ").strip() or "100000")
    commission_rate = float(input("Commission % per trade (default: 0.1): ").strip() or "0.1") / 100
    
    print("\nTrading Strategy:")
    print("1. Simple Threshold (Buy if predicted > current, Sell if predicted < current)")
    print("2. Percentage Threshold (Buy/Sell only if predicted change > threshold%)")
    strategy_choice = input("Select strategy (1 or 2, default: 1): ").strip() or "1"
    
    threshold = 0.0
    if strategy_choice == "2":
        threshold = float(input("Threshold % (default: 0.5): ").strip() or "0.5") / 100
    
    return {
        'initial_capital': initial_capital,
        'commission_rate': commission_rate,
        'strategy': 'simple' if strategy_choice == "1" else 'percentage',
        'threshold': threshold
    }


def load_and_preprocess_data(dataset_path: str, preprocessor: GenericPreprocessor):
    print("\n" + "=" * 70)
    print("DATA LOADING & PREPROCESSING")
    print("=" * 70)
    
    print(f"\nLoading: {Path(dataset_path).name}")
    
    try:
        df = pd.read_csv(dataset_path)
        
        # Normalize column names to handle case differences
        df.columns = df.columns.str.capitalize()
        
        # Find and parse date column
        date_cols = ['Date', 'Timestamp']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.sort_values(col).set_index(col)
                break
        
        # Filter to only stocks with sufficient data if multi-stock dataset
        if 'Name' in df.columns and len(df['Name'].unique()) > 1:
            print(f"  Multi-stock dataset detected: {len(df['Name'].unique())} stocks")
            # Use first stock with complete data
            stock_counts = df['Name'].value_counts()
            selected_stock = stock_counts.index[0]
            df = df[df['Name'] == selected_stock].copy()
            print(f"  Selected stock: {selected_stock} ({len(df)} rows)")
        
        print(f"[OK] Loaded {len(df)} rows")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        
        # Preprocess
        print("\n  Fitting preprocessor...")
        df_scaled = preprocessor.fit_transform(df)
        
        info = preprocessor.get_info()
        print(f"[OK] Preprocessing complete")
        print(f"  Features: {', '.join(info['feature_columns'])}")
        print(f"  Sequence length: {info['sequence_length']} days")
        print(f"  Scaling: {info['scaling_method']}")
        print(f"  Effective samples: {len(df) - info['sequence_length']}")
        
        return df
        
    except Exception as e:
        print(f"[ERROR] {e}")
        raise


def load_model(model_path: str, model_config: Optional[dict] = None):
    """Load ML/DL/RL model with auto-detection support."""
    from model_loader.universal_models import (
        UniversalModelLoader, auto_detect_architecture,
        get_default_config, list_available_architectures
    )
    import torch.nn as nn
    
    print("\n" + "=" * 70)
    print("MODEL LOADING")
    print("=" * 70)
    
    print(f"\nLoading: {Path(model_path).name}")
    
    # Try auto-detection if no config provided
    if model_config is None:
        detected_arch = auto_detect_architecture(model_path)
        if detected_arch:
            print(f"  Auto-detected architecture: {detected_arch}")
            
            print(f"\n  Enter input size (number of features):")
            print(f"  1 = Close only")
            print(f"  4 = OHLC")
            print(f"  5 = OHLCV")
            try:
                input_size = int(input("  > ").strip())
            except:
                input_size = 1
                print(f"  Using default: {input_size}")
            
            model_config = {
                'architecture': detected_arch,
                'input_size': input_size
            }
    
    try:
        loader = UniversalModelLoader()
        model = loader.load_model(model_path, model_config)
        
        print(f"[OK] Model loaded successfully")
        print(f"  Framework: {loader.framework}")
        print(f"  Type: {loader.model_type}")
        print(f"  Device: {loader.device}")
        
        if isinstance(model, nn.Module):
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, loader.device, loader
        
    except Exception as e:
        print(f"[ERROR] Loading model: {e}")
        print(f"\nAvailable PyTorch architectures: {list_available_architectures()}")
        print(f"\nTo load your model, provide model_config:")
        print(f"  Example: {{'architecture': 'lstm', 'input_size': 1, 'hidden_size': 64, 'num_layers': 2}}")
        raise


def run_backtesting(model, device, df_original, preprocessor, config, model_loader=None):
    """Run backtesting simulation.
        device: Device for PyTorch models
        df_original: Original dataframe
        preprocessor: Data preprocessor
        config: Backtesting configuration
        model_loader: UniversalModelLoader instance (optional, for universal prediction)
    """
    import torch.nn as nn
    
    print("\n" + "=" * 70)
    print("BACKTESTING SIMULATION")
    print("=" * 70)
    
    seq_len = preprocessor.sequence_length
    initial_capital = config['initial_capital']
    commission = config['commission_rate']
    
    # Determine model type
    is_pytorch = isinstance(model, nn.Module)
    
    # Initialize
    results = {
        'dates': [],
        'actual_prices': [],
        'predicted_prices': [],
        'positions': [],
        'portfolio_values': [],
        'cash': [],
        'holdings': [],
        'trades': []
    }
    
    cash = initial_capital
    holdings = 0
    position = 0
    
    print(f"\nInitial capital: ${initial_capital:,.2f}")
    print(f"Commission: {commission * 100}%")
    print(f"Strategy: {config['strategy']}")
    print(f"Model type: {'PyTorch' if is_pytorch else 'Sklearn/Other'}")
    print(f"\nStarting simulation from day {seq_len}...")
    
    # Run simulation
    if is_pytorch:
        with torch.no_grad():
            for i in range(seq_len, len(df_original)):
                current_date = df_original.index[i]
                current_price = df_original.iloc[i]['Close']
                
                # Get prediction window
                window = preprocessor.get_rolling_window(df_original, i, scaled=True)
                if window is None:
                    continue
                
                # Predict (PyTorch)
                X = torch.from_numpy(window).type(torch.float32).to(device)
                pred_scaled = model(X).cpu().numpy()
                predicted_price = preprocessor.inverse_transform(pred_scaled)[0, 0]
                
                # Trading logic
                trade_action = _execute_trade(
                    predicted_price, current_price, position, cash, holdings,
                    commission, config
                )
                
                if trade_action:
                    cash = trade_action['cash']
                    holdings = trade_action['holdings']
                    position = trade_action['position']
                    results['trades'].append(trade_action['message'])
                
                # Record results
                portfolio_value = cash + holdings * current_price
                results['dates'].append(current_date)
                results['actual_prices'].append(current_price)
                results['predicted_prices'].append(predicted_price)
                results['positions'].append(position)
                results['portfolio_values'].append(portfolio_value)
                results['cash'].append(cash)
                results['holdings'].append(holdings)
    
    else:  # Sklearn or other models
        for i in range(seq_len, len(df_original)):
            current_date = df_original.index[i]
            current_price = df_original.iloc[i]['Close']
            
            # Get prediction window
            window = preprocessor.get_rolling_window(df_original, i, scaled=True)
            if window is None:
                continue
            
            # Predict (Sklearn) - need to flatten for sklearn
            if model_loader:
                pred_scaled = model_loader.predict(window.reshape(1, -1))
            else:
                pred_scaled = model.predict(window.reshape(1, -1))
            
            predicted_price = preprocessor.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
            
            # Trading logic
            trade_action = _execute_trade(
                predicted_price, current_price, position, cash, holdings,
                commission, config
            )
            
            if trade_action:
                cash = trade_action['cash']
                holdings = trade_action['holdings']
                position = trade_action['position']
                results['trades'].append(trade_action['message'])
            
            # Record results
            portfolio_value = cash + holdings * current_price
            results['dates'].append(current_date)
            results['actual_prices'].append(current_price)
            results['predicted_prices'].append(predicted_price)
            results['positions'].append(position)
            results['portfolio_values'].append(portfolio_value)
            results['cash'].append(cash)
            results['holdings'].append(holdings)
    
    # Final portfolio value
    final_value = cash + holdings * results['actual_prices'][-1]
    
    print(f"\n{'=' * 70}")
    print(f"Simulation complete!")
    print(f"  Total days: {len(results['dates'])}")
    print(f"  Trades executed: {len(results['trades'])}")
    print(f"  Final portfolio value: ${final_value:,.2f}")
    print(f"  Return: {(final_value - initial_capital) / initial_capital * 100:.2f}%")
    
    return results


def _execute_trade(predicted_price, current_price, position, cash, holdings, commission, config):
    """Helper function to execute trades based on strategy."""
    
    if config['strategy'] == 'simple':
        if predicted_price > current_price and position == 0 and cash > current_price:
            # Buy
            shares = int(cash / (current_price * (1 + commission)))
            if shares > 0:
                cost = shares * current_price * (1 + commission)
                new_cash = cash - cost
                new_holdings = holdings + shares
                new_position = 1
                message = f"BUY {shares} @ ${current_price:.2f}"
                return {
                    'cash': new_cash,
                    'holdings': new_holdings,
                    'position': new_position,
                    'message': message
                }
        
        elif predicted_price < current_price and position == 1:
            # Sell
            revenue = holdings * current_price * (1 - commission)
            new_cash = cash + revenue
            new_holdings = 0
            new_position = 0
            message = f"SELL {holdings} @ ${current_price:.2f}"
            return {
                'cash': new_cash,
                'holdings': new_holdings,
                'position': new_position,
                'message': message
            }
    
    else:  # percentage strategy
        pred_return = (predicted_price - current_price) / current_price
        
        if pred_return > config['threshold'] and position == 0 and cash > current_price:
            shares = int(cash / (current_price * (1 + commission)))
            if shares > 0:
                cost = shares * current_price * (1 + commission)
                new_cash = cash - cost
                new_holdings = holdings + shares
                new_position = 1
                message = f"BUY {shares} @ ${current_price:.2f}"
                return {
                    'cash': new_cash,
                    'holdings': new_holdings,
                    'position': new_position,
                    'message': message
                }
        
        elif pred_return < -config['threshold'] and position == 1:
            revenue = holdings * current_price * (1 - commission)
            new_cash = cash + revenue
            new_holdings = 0
            new_position = 0
            message = f"SELL {holdings} @ ${current_price:.2f}"
            return {
                'cash': new_cash,
                'holdings': new_holdings,
                'position': new_position,
                'message': message
            }
    
    return None


def calculate_metrics(results, initial_capital):
    """Calculate performance metrics."""
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    
    pv = results['portfolio_values']
    returns = np.diff(pv) / pv[:-1]
    
    total_return = (pv[-1] - initial_capital) / initial_capital * 100
    n_years = len(pv) / 252
    ann_return = ((pv[-1] / initial_capital) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    cummax = np.maximum.accumulate(pv)
    drawdown = (pv - cummax) / cummax
    max_dd = np.min(drawdown) * 100
    
    metrics = {
        'Total Return (%)': total_return,
        'Annualized Return (%)': ann_return,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': max_dd,
        'Total Trades': len(results['trades']),
        'Final Portfolio ($)': pv[-1],
    }
    
    print()
    for k, v in metrics.items():
        if isinstance(v, float) and ('Return' in k or 'Drawdown' in k or 'Ratio' in k):
            print(f"  {k:.<50} {v:>12.2f}")
        elif isinstance(v, float):
            print(f"  {k:.<50} {v:>12,.2f}")
        else:
            print(f"  {k:.<50} {v:>12}")
    
    return metrics


def generate_visualizations(results, save_dir):
    """Generate plots."""
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    visualizer = BacktestVisualizer(save_path=str(save_dir))
    
    # 1. Equity curve
    print("\n  Creating equity curve...")
    fig = visualizer.plot_equity_curve(
        equity_curve=results['portfolio_values'],
        dates=results['dates'],
        title="Portfolio Equity Curve",
        save=True,
        filename="equity_curve.png"
    )
    plt.close(fig)
    
    # 2. Predictions vs Actual
    print("  Creating price prediction plot...")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(results['dates'], results['actual_prices'], label='Actual', color='blue', linewidth=1.5)
    ax.plot(results['dates'], results['predicted_prices'], label='Predicted', color='red', linewidth=1, alpha=0.7)
    ax.set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'predictions.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Drawdown
    print("  Creating drawdown plot...")
    pv = results['portfolio_values']
    cummax = np.maximum.accumulate(pv)
    dd = (pv - cummax) / cummax * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(results['dates'], dd, 0, color='red', alpha=0.3)
    ax.plot(results['dates'], dd, color='red', linewidth=1.5)
    ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'drawdown.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n[OK] Saved to: {save_dir}")


def save_results(results, metrics, save_dir):
    """Save results to CSV."""
    save_dir = Path(save_dir)
    
    df_results = pd.DataFrame({
        'Date': results['dates'],
        'Actual_Price': results['actual_prices'],
        'Predicted_Price': results['predicted_prices'],
        'Position': results['positions'],
        'Portfolio_Value': results['portfolio_values'],
        'Cash': results['cash'],
        'Holdings': results['holdings']
    })
    df_results.to_csv(save_dir / 'portfolio_history.csv', index=False)
    
    if results['trades']:
        pd.DataFrame(results['trades']).to_csv(save_dir / 'trades.csv', index=False)
    
    pd.DataFrame([metrics]).to_csv(save_dir / 'metrics.csv', index=False)
    
    print(f"  [OK] CSV files saved")


def main():
    """Main execution."""
    print_banner()
    
    # Step 1: Select dataset
    category = select_dataset_category()
    data_root = Path(__file__).parent / 'data_repository'
    dataset_path = list_and_select_dataset(category, data_root)
    
    # Step 2: Select model
    model_path = select_model_path()
    if not Path(model_path).exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        return
    
    # Step 3: Configure model preprocessing
    preset = select_model_config()
    preprocessor = GenericPreprocessor(
        sequence_length=preset.sequence_length,
        feature_columns=preset.feature_columns,
        scaling_method=preset.scaling_method,
        scaling_range=preset.scaling_range
    )
    
    # Step 4: Configure backtesting
    config = configure_backtesting()
    
    # Step 5: Load data
    df_original = load_and_preprocess_data(dataset_path, preprocessor)
    
    # Step 6: Load model (with optional config for notebook models)
    model_config = None  # Auto-detect or ask user
    # If you know your model architecture, specify it here:
    # model_config = {'architecture': 'lstm', 'input_size': 1, 'hidden_size': 64, 'num_layers': 2}
    model, device, model_loader = load_model(model_path, model_config)
    
    # Step 7: Run backtesting
    results = run_backtesting(model, device, df_original, preprocessor, config, model_loader)
    
    # Step 8: Calculate metrics
    metrics = calculate_metrics(results, config['initial_capital'])
    
    # Step 9: Visualize and save
    results_dir = Path(__file__).parent / 'results' / f"backtest_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    generate_visualizations(results, results_dir)
    save_results(results, metrics, results_dir)
    
    print("\n" + "=" * 70)
    print("BACKTESTING COMPLETE!")
    print("=" * 70)
    print(f"\nResults: {results_dir}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] User cancelled.")
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
