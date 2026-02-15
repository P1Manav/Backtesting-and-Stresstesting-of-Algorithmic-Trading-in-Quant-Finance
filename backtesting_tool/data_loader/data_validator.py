"""
Data Validator Module
Validates dataset schema and data quality for backtesting.
"""

import pandas as pd
from typing import List, Optional, Dict, Any


class DataValidator:
    """
    Validates datasets for backtesting requirements.
    
    Validates:
    - Required columns presence
    - Data types
    - Value ranges
    - Data consistency
    """
    
    REQUIRED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    NUMERIC_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the Data Validator.
        
        Args:
            strict_mode: If True, raise exceptions on warnings
        """
        self.strict_mode = strict_mode
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame has the required schema.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        # Check required columns
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Required columns are: {self.REQUIRED_COLUMNS}"
            )
        
        # Check data types
        self._validate_data_types(df)
        
        # Check value ranges
        self._validate_value_ranges(df)
        
        # Check data consistency
        self._validate_consistency(df)
        
        # Report warnings
        if self.validation_warnings:
            for warning in self.validation_warnings:
                print(f"Warning: {warning}")
            if self.strict_mode:
                raise ValueError("Validation warnings found in strict mode")
        
        return True
    
    def _validate_data_types(self, df: pd.DataFrame):
        """Validate that columns have appropriate data types."""
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    self.validation_errors.append(
                        f"Column '{col}' should be numeric, got {df[col].dtype}"
                    )
        
        if self.validation_errors:
            raise ValueError(f"Data type errors: {self.validation_errors}")
    
    def _validate_value_ranges(self, df: pd.DataFrame):
        """Validate that values are within reasonable ranges."""
        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in df.columns:
                if (df[col] < 0).any():
                    self.validation_warnings.append(
                        f"Column '{col}' contains negative values"
                    )
        
        # Check for negative volume
        if 'Volume' in df.columns:
            if (df['Volume'] < 0).any():
                self.validation_warnings.append(
                    "Column 'Volume' contains negative values"
                )
    
    def _validate_consistency(self, df: pd.DataFrame):
        """Validate OHLC data consistency."""
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # High should be >= max(Open, Close)
            high_violations = (df['High'] < df[['Open', 'Close']].max(axis=1)).sum()
            if high_violations > 0:
                self.validation_warnings.append(
                    f"Found {high_violations} rows where High < max(Open, Close)"
                )
            
            # Low should be <= min(Open, Close)
            low_violations = (df['Low'] > df[['Open', 'Close']].min(axis=1)).sum()
            if low_violations > 0:
                self.validation_warnings.append(
                    f"Found {low_violations} rows where Low > min(Open, Close)"
                )
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing data quality metrics
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'duplicate_dates': 0,
            'price_consistency': {},
            'value_statistics': {}
        }
        
        # Missing values per column
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                report['missing_values'][col] = {
                    'count': int(missing),
                    'percentage': round(missing / len(df) * 100, 2)
                }
        
        # Duplicate dates
        if 'Date' in df.columns:
            report['duplicate_dates'] = int(df['Date'].duplicated().sum())
        
        # Price consistency checks
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            report['price_consistency'] = {
                'high_violations': int((df['High'] < df[['Open', 'Close']].max(axis=1)).sum()),
                'low_violations': int((df['Low'] > df[['Open', 'Close']].min(axis=1)).sum())
            }
        
        # Value statistics
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                report['value_statistics'][col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'zeros': int((df[col] == 0).sum()),
                    'negatives': int((df[col] < 0).sum())
                }
        
        return report
    
    def validate_for_backtesting(self, df: pd.DataFrame, min_rows: int = 30) -> bool:
        """
        Validate that dataset is suitable for backtesting.
        
        Args:
            df: Input DataFrame
            min_rows: Minimum number of rows required
            
        Returns:
            True if suitable for backtesting
        """
        # Basic schema validation
        self.validate_schema(df)
        
        # Check minimum rows
        if len(df) < min_rows:
            raise ValueError(
                f"Dataset has {len(df)} rows, minimum required is {min_rows}"
            )
        
        # Check for sufficient price variation
        if 'Close' in df.columns:
            price_std = df['Close'].std()
            if price_std == 0:
                raise ValueError("Close prices have no variation")
        
        # Check date ordering
        if 'Date' in df.columns:
            if not df['Date'].is_monotonic_increasing:
                print("Warning: Dates are not in ascending order, will be sorted")
        
        return True
