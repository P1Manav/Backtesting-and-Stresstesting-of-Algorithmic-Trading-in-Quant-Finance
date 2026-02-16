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

        if 'Name' in df.columns and df['Name'].nunique() > 1:
            stock_counts = df['Name'].value_counts()
            selected = stock_counts.index[0]
            df = df[df['Name'] == selected].copy()
            print(f"  Multi-stock dataset: selected '{selected}' ({len(df)} rows)")

        self.validator.validate(df)

        print(f"  [OK] {len(df)} rows  |  {df.index[0]} -> {df.index[-1]}")
        return df
