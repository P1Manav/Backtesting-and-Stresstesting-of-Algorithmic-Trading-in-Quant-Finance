import os
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from .data_validator import DataValidator


class DatasetManager:
    
    REQUIRED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    def __init__(self, data_repository_path: Optional[str] = None):
        if data_repository_path is None:
            self.data_path = Path(__file__).parent.parent / "data_repository" / "backtesting"
        else:
            self.data_path = Path(data_repository_path)
        
        self.validator = DataValidator()
        self._datasets_cache: Dict[str, pd.DataFrame] = {}
    
    def list_available_datasets(self) -> List[Dict[str, Any]]:
        datasets = []
        
        if not self.data_path.exists():
            print(f"Warning: Data repository path does not exist: {self.data_path}")
            return datasets
        
        for file in self.data_path.glob("*.csv"):
            size_mb = file.stat().st_size / (1024 * 1024)
            datasets.append({
                'name': file.name,
                'path': str(file),
                'size_mb': round(size_mb, 2)
            })
        
        return datasets
    
    def load_dataset(self, 
                     dataset_name: Optional[str] = None,
                     dataset_path: Optional[str] = None,
                     parse_dates: bool = True,
                     fill_missing: bool = True) -> pd.DataFrame:
        if dataset_path is None and dataset_name is None:
            raise ValueError("Either dataset_name or dataset_path must be provided")
        
        if dataset_path is None:
            dataset_path = str(self.data_path / dataset_name)
        
        # Check cache
        cache_key = dataset_path
        if cache_key in self._datasets_cache:
            return self._datasets_cache[cache_key].copy()
        
        # Load the dataset
        df = self._load_csv(dataset_path, parse_dates)
        
        # Validate schema
        self.validator.validate_schema(df)
        
        # Normalize timestamps
        df = self._normalize_timestamps(df)
        
        # Handle missing values
        if fill_missing:
            df = self._handle_missing_values(df)
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Cache the dataset
        self._datasets_cache[cache_key] = df.copy()
        
        return df
    
    def _load_csv(self, path: str, parse_dates: bool = True) -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame.
        
        Args:
            path: Path to the CSV file
            parse_dates: Whether to parse date columns
            
        Returns:
            Raw DataFrame from CSV
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        
        # Try to detect date column
        df = pd.read_csv(path)
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        if parse_dates and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to expected format.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        # Common column name mappings
        column_mapping = {
            'date': 'Date',
            'datetime': 'Date',
            'timestamp': 'Date',
            'time': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'adj close': 'Close',
            'adj_close': 'Close',
            'adjusted_close': 'Close',
            'volume': 'Volume',
            'vol': 'Volume'
        }
        
        # Apply mapping (case-insensitive)
        new_columns = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in column_mapping:
                new_columns[col] = column_mapping[col_lower]
            else:
                new_columns[col] = col
        
        return df.rename(columns=new_columns)
    
    def _normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Date' not in df.columns:
            return df
        
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_missing = df.isnull().sum().sum()
        
        if initial_missing > 0:
            print(f"Found {initial_missing} missing values, applying forward/backward fill...")
        
        price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
        
        required_cols = [c for c in self.REQUIRED_COLUMNS if c in df.columns]
        df = df.dropna(subset=required_cols)
        
        return df
    
    def get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        info = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'date_range': None,
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.astype(str).to_dict()
        }
        
        if 'Date' in df.columns:
            info['date_range'] = {
                'start': str(df['Date'].min()),
                'end': str(df['Date'].max()),
                'days': (df['Date'].max() - df['Date'].min()).days
            }
        
        return info
    
    def create_sample_dataset(self, 
                               filename: str = "sample_ohlcv.csv",
                               num_days: int = 252) -> str:
        import numpy as np
        
        np.random.seed(42)
        
        dates = pd.date_range(start='2020-01-01', periods=num_days, freq='B')
        
        base_price = 100.0
        returns = np.random.normal(0.0005, 0.02, num_days)
        prices = base_price * np.cumprod(1 + returns)
        
        data = {
            'Date': dates,
            'Open': prices * (1 + np.random.uniform(-0.01, 0.01, num_days)),
            'High': prices * (1 + np.random.uniform(0, 0.02, num_days)),
            'Low': prices * (1 - np.random.uniform(0, 0.02, num_days)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, num_days)
        }
        
        df = pd.DataFrame(data)
        
        # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        
        # Save to data repository
        self.data_path.mkdir(parents=True, exist_ok=True)
        output_path = self.data_path / filename
        df.to_csv(output_path, index=False)
        
        print(f"Sample dataset created: {output_path}")
        return str(output_path)
    
    def clear_cache(self):
        self._datasets_cache.clear()
