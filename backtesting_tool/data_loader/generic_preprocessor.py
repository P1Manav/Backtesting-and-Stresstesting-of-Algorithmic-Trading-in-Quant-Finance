"""Generic preprocessor for any model trained on OHLCV data."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch


class GenericPreprocessor:
    """Universal preprocessor for OHLCV data."""
    
    def __init__(self,
                 sequence_length: int = 60,
                 feature_columns: List[str] = None,
                 scaling_method: str = 'minmax',
                 scaling_range: Tuple[float, float] = (-1, 1),
                 target_column: str = 'Close'):
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns or ['Close']
        self.scaling_method = scaling_method
        self.scaling_range = scaling_range
        self.target_column = target_column
        
        self.scaler = None
        self.scaler_fitted = False
        
    def _initialize_scaler(self):
        if self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=self.scaling_range)
        elif self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = None
    
    def _normalize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        col_mapping = {col.lower(): col for col in df.columns}
        
        for expected_col in self.feature_columns + [self.target_column]:
            if expected_col not in df.columns and expected_col.lower() in col_mapping:
                actual_col = col_mapping[expected_col.lower()]
                df = df.rename(columns={actual_col: expected_col})
        
        return df
    
    def fit(self, data: pd.DataFrame) -> 'GenericPreprocessor':
        df = self._normalize_columns(data)
        
        missing = [col for col in self.feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")
        
        if self.scaling_method != 'none':
            self._initialize_scaler()
            values = df[self.feature_columns].values
            self.scaler.fit(values)
            self.scaler_fitted = True
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        if self.scaler_fitted:
            values = df[self.feature_columns].values
            df[self.feature_columns] = self.scaler.transform(values)
        
        return df
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)
    
    def inverse_transform(self, scaled_values: np.ndarray) -> np.ndarray:
        if not self.scaler_fitted or self.scaling_method == 'none':
            return scaled_values
        
        if scaled_values.ndim == 1:
            scaled_values = scaled_values.reshape(-1, 1)
        
        if len(self.feature_columns) == 1:
            return self.scaler.inverse_transform(scaled_values)
        
        if self.target_column in self.feature_columns:
            target_idx = self.feature_columns.index(self.target_column)
            n_features = len(self.feature_columns)
            
            dummy = np.zeros((scaled_values.shape[0], n_features))
            dummy[:, target_idx] = scaled_values.flatten()
            
            # Inverse transform
            inverse = self.scaler.inverse_transform(dummy)
            return inverse[:, target_idx].reshape(-1, 1)
        else:
            return scaled_values
    
    def create_sequences(self,
                        data: pd.DataFrame,
                        scaled: bool = False) -> Tuple[np.ndarray, np.ndarray, List]:
        """self._normalize_columns(data
        Create sequences for temporal models.
        
        Args:
            data: DataFrame with OHLCV data
            scaled: Whether to scale the data
            
        Returns:
            Tuple of (X, y, dates):
            - X: Input sequences (n_samples, sequence_length, n_features)
            - y: Target values (n_samples, 1)
            - dates: Corresponding dates
        """
        if scaled and not self.scaler_fitted:
            df = self.fit_transform(data)
        elif scaled:
            df = self.transform(data)
        else:
            df = data.copy()
        
        # Extract values
        feature_values = df[self.feature_columns].values
        target_values = df[self.target_column].values
        dates = df.index if isinstance(df.index, pd.DatetimeIndex) else df.get('Date', range(len(df)))
        
        X, y, date_indices = [], [], []
        
        # Create sequences
        for i in range(len(feature_values) - self.sequence_length):
            X.append(feature_values[i:i + self.sequence_length])
            y.append(target_values[i + self.sequence_length])
            date_indices.append(i + self.sequence_length)
        
        X = np.array(X)  # Shape: (n_samples, sequence_length, n_features)
        y = np.array(y).reshape(-1, 1)
        result_dates = [dates[i] if hasattr(dates, '__getitem__') else i for i in date_indices]
        
        return X, y, result_dates
    
    def get_rolling_window(self,
                          data: pd.DataFrame,
                          end_idx: int,
                          scaled: bool = True) -> Optional[np.ndarray]:
        if end_idx < self.sequence_length:
            return None
        
        window_data = data.iloc[end_idx - self.sequence_length:end_idx].copy()
        
        if scaled and not self.scaler_fitted:
            self.fit(data.iloc[:end_idx])
        
        if scaled:
            window_data = self.transform(window_data)
        else:
            window_data = self._normalize_columns(window_data)
        
        values = window_data[self.feature_columns].values
        window = values.reshape(1, self.sequence_length, len(self.feature_columns))
        
        return window
    
    def get_feature_count(self) -> int:
        return len(self.feature_columns)
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'sequence_length': self.sequence_length,
            'feature_columns': self.feature_columns,
            'n_features': len(self.feature_columns),
            'scaling_method': self.scaling_method,
            'scaling_range': self.scaling_range,
            'target_column': self.target_column,
            'fitted': self.scaler_fitted
        }
