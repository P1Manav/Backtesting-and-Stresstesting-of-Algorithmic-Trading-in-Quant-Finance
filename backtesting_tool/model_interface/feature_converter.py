"""
Feature Converter Module
Converts market data features to model input tensors.
"""

import torch
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union, Tuple
from enum import Enum


class FeatureType(Enum):
    """Types of features that can be extracted."""
    OHLCV = "ohlcv"
    RETURNS = "returns"
    NORMALIZED = "normalized"
    TECHNICAL = "technical"


class FeatureConverter:
    """
    Converts market features to model input tensors.
    
    Handles:
    - OHLCV data conversion
    - Feature normalization
    - Rolling window creation
    - Tensor formatting
    """
    
    def __init__(self, 
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the Feature Converter.
        
        Args:
            device: PyTorch device for tensors
            dtype: Data type for tensors
        """
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        
        # Normalization parameters
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._fitted = False
    
    def fit(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> 'FeatureConverter':
        """
        Fit normalization parameters from training data.
        
        Args:
            data: DataFrame to fit on
            columns: Columns to use for normalization
            
        Returns:
            Self for chaining
        """
        if columns is None:
            columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        cols_present = [c for c in columns if c in data.columns]
        values = data[cols_present].values
        
        self._mean = np.nanmean(values, axis=0)
        self._std = np.nanstd(values, axis=0)
        self._std[self._std == 0] = 1.0  # Prevent division by zero
        self._fitted = True
        
        return self
    
    def extract_features(self,
                         data: pd.DataFrame,
                         feature_type: FeatureType = FeatureType.OHLCV,
                         columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract features from market data.
        
        Args:
            data: DataFrame with market data
            feature_type: Type of features to extract
            columns: Specific columns to use
            
        Returns:
            NumPy array of features
        """
        if columns is None:
            columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        cols_present = [c for c in columns if c in data.columns]
        
        if feature_type == FeatureType.OHLCV:
            return self._extract_ohlcv(data, cols_present)
        elif feature_type == FeatureType.RETURNS:
            return self._extract_returns(data, cols_present)
        elif feature_type == FeatureType.NORMALIZED:
            return self._extract_normalized(data, cols_present)
        elif feature_type == FeatureType.TECHNICAL:
            return self._extract_technical(data)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def _extract_ohlcv(self, data: pd.DataFrame, columns: List[str]) -> np.ndarray:
        """Extract raw OHLCV features."""
        return data[columns].values.astype(np.float32)
    
    def _extract_returns(self, data: pd.DataFrame, columns: List[str]) -> np.ndarray:
        """Extract return-based features (percentage change)."""
        values = data[columns].values
        returns = np.diff(values, axis=0) / values[:-1]
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Add a zero row at the beginning to maintain shape
        returns = np.vstack([np.zeros((1, returns.shape[1])), returns])
        return returns.astype(np.float32)
    
    def _extract_normalized(self, data: pd.DataFrame, columns: List[str]) -> np.ndarray:
        """Extract normalized features using z-score."""
        values = data[columns].values
        
        if self._fitted:
            normalized = (values - self._mean) / self._std
        else:
            mean = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0)
            std[std == 0] = 1.0
            normalized = (values - mean) / std
        
        return np.nan_to_num(normalized, nan=0.0).astype(np.float32)
    
    def _extract_technical(self, data: pd.DataFrame) -> np.ndarray:
        """Extract technical indicator features."""
        features = []
        
        if 'Close' in data.columns:
            close = data['Close'].values
            
            # Simple Moving Averages
            sma_5 = self._rolling_mean(close, 5)
            sma_20 = self._rolling_mean(close, 20)
            
            # Price relative to SMAs
            features.append(close / sma_5 - 1)
            features.append(close / sma_20 - 1)
            
            # Volatility (rolling std)
            vol_20 = self._rolling_std(close, 20)
            features.append(vol_20 / close)
            
            # Returns
            returns = np.diff(close, prepend=close[0]) / close
            features.append(returns)
        
        if 'High' in data.columns and 'Low' in data.columns:
            # High-Low range
            hl_range = (data['High'].values - data['Low'].values) / data['Close'].values
            features.append(hl_range)
        
        if 'Volume' in data.columns:
            volume = data['Volume'].values
            vol_sma = self._rolling_mean(volume, 20)
            features.append(volume / vol_sma - 1)
        
        result = np.column_stack(features) if features else np.array([])
        return np.nan_to_num(result, nan=0.0).astype(np.float32)
    
    def _rolling_mean(self, arr: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling mean."""
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.mean(arr[i-window+1:i+1])
            elif i > 0:
                result[i] = np.mean(arr[:i+1])
        return result
    
    def _rolling_std(self, arr: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling standard deviation."""
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.std(arr[i-window+1:i+1])
            elif i > 0:
                result[i] = np.std(arr[:i+1])
        return result
    
    def create_rolling_windows(self,
                                features: np.ndarray,
                                window_size: int,
                                stride: int = 1) -> np.ndarray:
        """
        Create rolling windows from feature array.
        
        Args:
            features: Input features (timesteps, features)
            window_size: Size of each window
            stride: Step size between windows
            
        Returns:
            Array of shape (num_windows, window_size, features)
        """
        num_samples = features.shape[0]
        num_features = features.shape[1] if len(features.shape) > 1 else 1
        
        if num_samples < window_size:
            raise ValueError(
                f"Not enough samples ({num_samples}) for window size ({window_size})"
            )
        
        num_windows = (num_samples - window_size) // stride + 1
        
        windows = np.zeros((num_windows, window_size, num_features), dtype=np.float32)
        
        for i in range(num_windows):
            start = i * stride
            end = start + window_size
            if len(features.shape) == 1:
                windows[i, :, 0] = features[start:end]
            else:
                windows[i] = features[start:end]
        
        return windows
    
    def to_tensor(self, 
                  features: Union[np.ndarray, pd.DataFrame],
                  add_batch_dim: bool = True) -> torch.Tensor:
        """
        Convert features to PyTorch tensor.
        
        Args:
            features: Input features (numpy array or DataFrame)
            add_batch_dim: Whether to add batch dimension
            
        Returns:
            PyTorch tensor on the specified device
        """
        if isinstance(features, pd.DataFrame):
            features = features.values
        
        tensor = torch.tensor(features, dtype=self.dtype, device=self.device)
        
        if add_batch_dim and tensor.dim() < 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def prepare_batch(self,
                      data: pd.DataFrame,
                      window_size: int,
                      feature_type: FeatureType = FeatureType.NORMALIZED,
                      columns: Optional[List[str]] = None) -> torch.Tensor:
        """
        Prepare a batch of window samples for model input.
        
        Args:
            data: Market data DataFrame
            window_size: Size of rolling windows
            feature_type: Type of features to extract
            columns: Columns to use
            
        Returns:
            Tensor of shape (num_windows, window_size, num_features)
        """
        features = self.extract_features(data, feature_type, columns)
        windows = self.create_rolling_windows(features, window_size)
        return self.to_tensor(windows, add_batch_dim=False)
    
    def prepare_single(self,
                       data: pd.DataFrame,
                       feature_type: FeatureType = FeatureType.NORMALIZED,
                       columns: Optional[List[str]] = None) -> torch.Tensor:
        """
        Prepare a single sample for model input.
        
        Args:
            data: Market data DataFrame (single window)
            feature_type: Type of features to extract
            columns: Columns to use
            
        Returns:
            Tensor of shape (1, timesteps, num_features)
        """
        features = self.extract_features(data, feature_type, columns)
        return self.to_tensor(features, add_batch_dim=True)
