"""
Kaggle Fetcher Module
Fetches datasets from Kaggle using the Kaggle API.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd


class KaggleFetcher:
    """
    Fetches and caches datasets from Kaggle.
    
    Requires Kaggle API credentials to be configured.
    """
    
    # Pre-defined popular financial datasets on Kaggle
    POPULAR_DATASETS = {
        'sp500': 'camnugent/sandp500',
        'crypto': 'sudalairajkumar/cryptocurrencypricehistory',
        'forex': 'brunotly/foreign-exchange-rates-per-dollar-20002019',
        'nasdaq': 'jacksoncrow/stock-market-dataset',
        'indian_stocks': 'rohanrao/nifty50-stock-market-data'
    }
    
    def __init__(self, cache_path: Optional[str] = None):
        """
        Initialize the Kaggle Fetcher.
        
        Args:
            cache_path: Path to cache downloaded datasets.
                        Defaults to ../data_repository/backtesting/
        """
        if cache_path is None:
            self.cache_path = Path(__file__).parent.parent / "data_repository" / "backtesting"
        else:
            self.cache_path = Path(cache_path)
        
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self._kaggle_available = self._check_kaggle_api()
    
    def _check_kaggle_api(self) -> bool:
        """Check if Kaggle API is available and configured."""
        try:
            result = subprocess.run(
                ['kaggle', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def is_available(self) -> bool:
        """Check if Kaggle fetching is available."""
        return self._kaggle_available
    
    def list_popular_datasets(self) -> List[Dict[str, str]]:
        """
        List pre-defined popular financial datasets.
        
        Returns:
            List of dataset information dictionaries
        """
        return [
            {'key': key, 'kaggle_path': path}
            for key, path in self.POPULAR_DATASETS.items()
        ]
    
    def search_datasets(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for datasets on Kaggle.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching datasets
        """
        if not self._kaggle_available:
            raise RuntimeError(
                "Kaggle API is not available. Please install and configure it: "
                "pip install kaggle && kaggle configure"
            )
        
        try:
            result = subprocess.run(
                ['kaggle', 'datasets', 'list', '-s', query, '--csv'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Kaggle search failed: {result.stderr}")
            
            # Parse CSV output
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return []
            
            headers = lines[0].split(',')
            datasets = []
            for line in lines[1:limit+1]:
                values = line.split(',')
                if len(values) >= len(headers):
                    datasets.append(dict(zip(headers, values)))
            
            return datasets
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Kaggle search timed out")
    
    def download_dataset(self, 
                          dataset_path: str,
                          unzip: bool = True,
                          force: bool = False) -> str:
        """
        Download a dataset from Kaggle.
        
        Args:
            dataset_path: Kaggle dataset path (e.g., 'username/dataset-name')
            unzip: Whether to unzip the downloaded files
            force: Force re-download even if cached
            
        Returns:
            Path to the downloaded dataset folder
        """
        if not self._kaggle_available:
            raise RuntimeError(
                "Kaggle API is not available. Please install and configure it."
            )
        
        # Create dataset-specific folder
        dataset_name = dataset_path.split('/')[-1]
        dataset_folder = self.cache_path / dataset_name
        
        # Check cache
        if dataset_folder.exists() and not force:
            csv_files = list(dataset_folder.glob('*.csv'))
            if csv_files:
                print(f"Using cached dataset: {dataset_folder}")
                return str(dataset_folder)
        
        # Download
        dataset_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading dataset: {dataset_path}")
        
        cmd = ['kaggle', 'datasets', 'download', '-d', dataset_path, '-p', str(dataset_folder)]
        if unzip:
            cmd.append('--unzip')
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for large datasets
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Download failed: {result.stderr}")
            
            print(f"Dataset downloaded to: {dataset_folder}")
            return str(dataset_folder)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Download timed out")
    
    def download_popular_dataset(self, key: str, force: bool = False) -> str:
        """
        Download a pre-defined popular dataset.
        
        Args:
            key: Key from POPULAR_DATASETS
            force: Force re-download
            
        Returns:
            Path to the downloaded dataset folder
        """
        if key not in self.POPULAR_DATASETS:
            raise ValueError(
                f"Unknown dataset key: {key}. Available: {list(self.POPULAR_DATASETS.keys())}"
            )
        
        return self.download_dataset(self.POPULAR_DATASETS[key], force=force)
    
    def convert_to_ohlcv(self, 
                          input_path: str,
                          output_name: Optional[str] = None,
                          date_col: str = 'Date',
                          open_col: str = 'Open',
                          high_col: str = 'High',
                          low_col: str = 'Low',
                          close_col: str = 'Close',
                          volume_col: str = 'Volume') -> str:
        """
        Convert a downloaded dataset to standardized OHLCV format.
        
        Args:
            input_path: Path to the input CSV file
            output_name: Name for the output file (optional)
            date_col: Name of the date column in input
            open_col: Name of the open price column
            high_col: Name of the high price column
            low_col: Name of the low price column
            close_col: Name of the close price column
            volume_col: Name of the volume column
            
        Returns:
            Path to the converted OHLCV file
        """
        # Load the input file
        df = pd.read_csv(input_path)
        
        # Create column mapping
        column_mapping = {
            date_col: 'Date',
            open_col: 'Open',
            high_col: 'High',
            low_col: 'Low',
            close_col: 'Close',
            volume_col: 'Volume'
        }
        
        # Select and rename columns
        available_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        df_ohlcv = df[list(available_columns.keys())].rename(columns=available_columns)
        
        # Parse dates
        if 'Date' in df_ohlcv.columns:
            df_ohlcv['Date'] = pd.to_datetime(df_ohlcv['Date'], infer_datetime_format=True)
            df_ohlcv = df_ohlcv.sort_values('Date')
        
        # Generate output path
        if output_name is None:
            input_name = Path(input_path).stem
            output_name = f"{input_name}_ohlcv.csv"
        
        output_path = self.cache_path / output_name
        df_ohlcv.to_csv(output_path, index=False)
        
        print(f"Converted to OHLCV format: {output_path}")
        return str(output_path)
    
    def get_cached_datasets(self) -> List[str]:
        """
        Get list of cached dataset folders.
        
        Returns:
            List of cached dataset folder paths
        """
        folders = []
        if self.cache_path.exists():
            for item in self.cache_path.iterdir():
                if item.is_dir():
                    folders.append(str(item))
        return folders
