"""Load and manage stock market datasets"""
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from .data_validator import DataValidator

class DatasetManager:
    """Manage loading and selecting datasets"""

    def __init__(self, data_root: Optional[str] = None):
        """Initialize dataset manager"""
        if data_root is None:
            self.data_root = Path(__file__).parent.parent / "data_repository"
        else:
            self.data_root = Path(data_root)
        self.validator = DataValidator()

    def list_categories(self) -> List[str]:
        """List dataset categories"""
        categories = []
        for d in sorted(self.data_root.iterdir()):
            if d.is_dir():
                categories.append(d.name)
        return categories

    def list_datasets(self, category: str) -> List[Dict[str, Any]]:
        """List datasets in category"""
        category_path = self.data_root / category
        if not category_path.exists():
            return []

        datasets = []
        for idx, f in enumerate(sorted(category_path.glob("*.csv")), 1):
            size_mb = f.stat().st_size / (1024 * 1024)
            datasets.append({
                'id': idx,
                'name': f.name,
                'path': str(f),
                'size_mb': round(size_mb, 2)
            })
        return datasets

    def load_dataset(self, path: str) -> Dict[str, pd.DataFrame]:
        """Load dataset from CSV file"""
        print(f"\n  Loading: {Path(path).name}")
        df = pd.read_csv(path)

        df.columns = [c.strip().capitalize() for c in df.columns]

        date_col = None
        for candidate in ['Date', 'Timestamp', 'Datetime', 'Time']:
            if candidate in df.columns:
                date_col = candidate
                break

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df.sort_values(date_col).set_index(date_col)
        else:
            print("  [WARNING] No date column found, using row index")

        stock_col = None
        for candidate in ['Name', 'Ticker', 'Symbol', 'Stock', 'Instrument']:
            if candidate in df.columns and df[candidate].nunique() > 1:
                stock_col = candidate
                break

        MAX_DISPLAY = 30
        MAX_ALL_STOCKS = 50

        if stock_col is not None:
            unique_stocks = sorted(df[stock_col].unique())
            n = len(unique_stocks)
            print(f"\n  Multi-stock dataset detected ({n} stocks in '{stock_col}' column):")

            if n <= MAX_DISPLAY:
                for i, s in enumerate(unique_stocks, 1):
                    count = len(df[df[stock_col] == s])
                    print(f"    {i}. {s}  ({count} rows)")
            else:
                preview = unique_stocks[:10]
                tail = unique_stocks[-5:]
                for i, s in enumerate(preview, 1):
                    count = len(df[df[stock_col] == s])
                    print(f"    {i}. {s}  ({count} rows)")
                print(f"    ... ({n - 15} more stocks) ...")
                for i, s in enumerate(tail, n - 4):
                    count = len(df[df[stock_col] == s])
                    print(f"    {i}. {s}  ({count} rows)")

            print(f"\n  Portfolio options:")
            print(f"    A  — Select ALL {n} stocks for portfolio backtesting", end="")
            if n > MAX_ALL_STOCKS:
                print(f"  (⚠ {n} stocks — may be slow)")
            else:
                print()
            print(f"    S  — Select SPECIFIC stocks for portfolio backtesting")
            print(f"    T  — Select TOP N stocks by data availability")

            valid_choices = {'A', 'S', 'T'}

            while True:
                choice = input("  Enter choice (A/S/T): ").strip().upper()
                if choice in valid_choices:
                    break
                print(f"  Enter one of: A, S, T")

            if choice == 'A':
                selected = list(unique_stocks)
            elif choice == 'T':
                top_n = 0
                while top_n < 1:
                    try:
                        top_n = int(input(f"  How many top stocks? (2-{min(n, MAX_ALL_STOCKS)}): ").strip())
                        top_n = min(top_n, min(n, MAX_ALL_STOCKS))
                    except ValueError:
                        pass
                stock_counts = df[stock_col].value_counts()
                top_tickers = stock_counts.head(top_n).index.tolist()
                print(f"  Top {top_n} stocks by data availability:")
                for i, t in enumerate(top_tickers, 1):
                    print(f"    {i}. {t}  ({stock_counts[t]} rows)")
                selected = top_tickers
            else:
                if n > MAX_DISPLAY:
                    print(f"  Enter stock TICKER SYMBOLS separated by commas (e.g. AAPL,MSFT,GOOGL):")
                    while True:
                        raw = input("  > ").strip().upper()
                        picks = [s.strip() for s in raw.split(',') if s.strip()]
                        invalid = [s for s in picks if s not in unique_stocks]
                        if picks and not invalid:
                            selected = picks
                            break
                        if invalid:
                            print(f"  Invalid tickers: {invalid}. Try again.")
                        else:
                            print("  Enter at least one ticker.")
                else:
                    print(f"  Enter stock numbers separated by commas (e.g. 1,3,5):")
                    while True:
                        try:
                            indices = [int(x.strip()) for x in input("  > ").split(',')]
                            if all(1 <= idx <= n for idx in indices) and len(indices) > 0:
                                selected = [unique_stocks[idx - 1] for idx in indices]
                                break
                        except ValueError:
                            pass
                        print(f"  Enter valid numbers between 1 and {n}, separated by commas")

            print(f"\n  Loading {len(selected)} stocks ...")
            stock_dfs: Dict[str, pd.DataFrame] = {}
            for t in selected:
                sub = df[df[stock_col] == t].copy()
                sub = sub.drop(columns=[stock_col], errors='ignore')
                self.validator.validate(sub)
                stock_dfs[t] = sub
                print(f"  [OK] {t}: {len(sub)} rows  |  {sub.index[0]} -> {sub.index[-1]}")

            return stock_dfs

        self.validator.validate(df)
        print(f"  [OK] {len(df)} rows  |  {df.index[0]} -> {df.index[-1]}")
        return {'STOCK': df}

