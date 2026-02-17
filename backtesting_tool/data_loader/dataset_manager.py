import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from .data_validator import DataValidator


class DatasetManager:

    def __init__(self, data_root: Optional[str] = None):
        if data_root is None:
            self.data_root = Path(__file__).parent.parent / "data_repository"
        else:
            self.data_root = Path(data_root)
        self.validator = DataValidator()

    def list_categories(self) -> List[str]:
        """List available dataset categories."""
        categories = []
        for d in sorted(self.data_root.iterdir()):
            if d.is_dir():
                categories.append(d.name)
        return categories

    def list_datasets(self, category: str) -> List[Dict[str, Any]]:
        """List CSV datasets in a given category folder."""
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

    def load_dataset(self, path: str) -> pd.DataFrame:
        """Load a CSV dataset, standardize columns, parse dates, and validate."""
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

        # Detect the stock-identifier column (common names)
        stock_col = None
        for candidate in ['Name', 'Ticker', 'Symbol', 'Stock', 'Instrument']:
            if candidate in df.columns and df[candidate].nunique() > 1:
                stock_col = candidate
                break

        if stock_col is not None:
            unique_stocks = df[stock_col].unique()
            print(f"\n  Multi-stock dataset detected ({len(unique_stocks)} stocks in '{stock_col}' column):")
            for i, s in enumerate(unique_stocks, 1):
                count = len(df[df[stock_col] == s])
                print(f"    {i}. {s}  ({count} rows)")

            while True:
                try:
                    choice = int(input(f"  Select stock (1-{len(unique_stocks)}): ").strip())
                    if 1 <= choice <= len(unique_stocks):
                        break
                except ValueError:
                    pass
                print(f"  Enter a number between 1 and {len(unique_stocks)}")

            selected = unique_stocks[choice - 1]
            df = df[df[stock_col] == selected].copy()
            print(f"  Selected '{selected}' ({len(df)} rows)")

        self.validator.validate(df)

        print(f"  [OK] {len(df)} rows  |  {df.index[0]} -> {df.index[-1]}")
        return df
