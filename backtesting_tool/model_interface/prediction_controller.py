"""
Prediction Controller Module
Main interface for model predictions and trading action generation.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List, Union

from .feature_converter import FeatureConverter, FeatureType
from .action_mapper import ActionMapper, TradingAction


class PredictionController:
    """
    Central controller for model predictions and trading actions.
    
    This interface is model-agnostic and handles:
    - Converting market features to model input
    - Running model inference
    - Converting model output to trading actions
    - Batch and rolling-window inference
    - Inference validation (NaN detection, shape checking)
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: Optional[torch.device] = None,
                 window_size: int = 30,
                 feature_columns: Optional[List[str]] = None,
                 feature_type: FeatureType = FeatureType.NORMALIZED,
                 output_type: str = 'auto'):
        """
        Initialize the Prediction Controller.
        
        Args:
            model: PyTorch model for predictions
            device: Device for inference
            window_size: Size of input window for the model
            feature_columns: Columns to use as features
            feature_type: Type of feature extraction
            output_type: Type of model output (auto-detected if 'auto')
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.window_size = window_size
        self.feature_columns = feature_columns or ['Open', 'High', 'Low', 'Close', 'Volume']
        self.feature_type = feature_type
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize converters
        self.feature_converter = FeatureConverter(device=self.device)
        self.action_mapper = ActionMapper(output_type=output_type)
        
        # Inference statistics
        self._inference_count = 0
        self._nan_count = 0
        self._shape_errors = 0
    
    def fit_normalizer(self, data: pd.DataFrame) -> 'PredictionController':
        """
        Fit the feature normalizer on training data.
        
        Args:
            data: DataFrame with training data
            
        Returns:
            Self for chaining
        """
        self.feature_converter.fit(data, self.feature_columns)
        return self
    
    def predict(self, 
                data: pd.DataFrame,
                validate: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions for a single window of data.
        
        Args:
            data: DataFrame with market data (should be window_size rows)
            validate: Whether to validate inputs/outputs
            
        Returns:
            Tuple of (actions, positions)
        """
        if validate:
            self._validate_input(data)
        
        # Extract features
        tensor = self.feature_converter.prepare_single(
            data, 
            self.feature_type, 
            self.feature_columns
        )
        
        # Run inference
        with torch.no_grad():
            output = self.model(tensor)
        
        # Handle tuple outputs (some models return hidden states)
        if isinstance(output, tuple):
            output = output[0]
        
        if validate:
            self._validate_output(output)
        
        # Map to trading actions
        actions, positions = self.action_mapper.map_output(output)
        
        self._inference_count += 1
        
        return actions, positions
    
    def predict_batch(self,
                      data: pd.DataFrame,
                      validate: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions for multiple windows (batch inference).
        
        Args:
            data: DataFrame with market data
            validate: Whether to validate inputs/outputs
            
        Returns:
            Tuple of (actions, positions) arrays
        """
        if validate:
            self._validate_input(data)
        
        # Prepare batch of windows
        tensor = self.feature_converter.prepare_batch(
            data,
            self.window_size,
            self.feature_type,
            self.feature_columns
        )
        
        # Run batch inference
        with torch.no_grad():
            output = self.model(tensor)
        
        if isinstance(output, tuple):
            output = output[0]
        
        if validate:
            self._validate_output(output)
        
        # Map to trading actions
        actions, positions = self.action_mapper.map_output(output)
        
        self._inference_count += len(actions)
        
        return actions, positions
    
    def predict_rolling(self,
                        data: pd.DataFrame,
                        stride: int = 1,
                        validate: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using a rolling window approach.
        
        Args:
            data: DataFrame with market data
            stride: Step size between windows
            validate: Whether to validate inputs/outputs
            
        Returns:
            Tuple of (actions, positions) for each window
        """
        if len(data) < self.window_size:
            raise ValueError(
                f"Data length ({len(data)}) must be >= window_size ({self.window_size})"
            )
        
        all_actions = []
        all_positions = []
        
        for i in range(0, len(data) - self.window_size + 1, stride):
            window_data = data.iloc[i:i + self.window_size]
            actions, positions = self.predict(window_data, validate=validate)
            all_actions.append(actions[-1])  # Take last prediction
            all_positions.append(positions[-1])
        
        return np.array(all_actions), np.array(all_positions)
    
    def predict_at_timestep(self,
                            data: pd.DataFrame,
                            timestep: int) -> Tuple[int, float]:
        """
        Make a prediction at a specific timestep using preceding data.
        
        Args:
            data: Full DataFrame with market data
            timestep: Index at which to make prediction
            
        Returns:
            Tuple of (action, position) for the timestep
        """
        if timestep < self.window_size - 1:
            # Not enough history, return flat position
            return TradingAction.FLAT.value, 0.0
        
        start_idx = timestep - self.window_size + 1
        window_data = data.iloc[start_idx:timestep + 1]
        
        actions, positions = self.predict(window_data, validate=False)
        
        return int(actions[-1]), float(positions[-1])
    
    def _validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        # Check for required columns
        missing_cols = [c for c in self.feature_columns if c not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        # Check for NaN values
        nan_count = data[self.feature_columns].isna().sum().sum()
        if nan_count > 0:
            self._nan_count += nan_count
            print(f"Warning: Input contains {nan_count} NaN values")
        
        # Check minimum length
        if len(data) < self.window_size:
            raise ValueError(
                f"Input length ({len(data)}) is less than window_size ({self.window_size})"
            )
    
    def _validate_output(self, output: torch.Tensor) -> None:
        """Validate model output."""
        # Check for NaN in output
        if torch.isnan(output).any():
            self._nan_count += 1
            raise ValueError("Model output contains NaN values")
        
        # Check for Inf in output
        if torch.isinf(output).any():
            raise ValueError("Model output contains infinite values")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return {
            'inference_count': self._inference_count,
            'nan_count': self._nan_count,
            'shape_errors': self._shape_errors,
            'window_size': self.window_size,
            'feature_columns': self.feature_columns,
            'feature_type': self.feature_type.value,
            'device': str(self.device)
        }
    
    def reset_statistics(self) -> None:
        """Reset inference statistics."""
        self._inference_count = 0
        self._nan_count = 0
        self._shape_errors = 0
    
    def set_window_size(self, window_size: int) -> None:
        """Update the window size."""
        self.window_size = window_size
    
    def set_feature_type(self, feature_type: FeatureType) -> None:
        """Update the feature extraction type."""
        self.feature_type = feature_type
    
    def set_output_type(self, output_type: str) -> None:
        """Update the action mapper output type."""
        self.action_mapper.output_type = output_type
    
    def get_action_name(self, action: int) -> str:
        """Get human-readable name for an action."""
        return self.action_mapper.get_action_name(action)


class ModelInterface:
    """
    High-level model interface wrapper for backtesting.
    
    Provides a simplified API for the backtesting engine.
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Model Interface.
        
        Args:
            model: PyTorch model
            config: Configuration dictionary with optional keys:
                - window_size: int
                - feature_columns: List[str]
                - feature_type: str ('normalized', 'returns', 'ohlcv')
                - output_type: str ('auto', 'signals', 'scores', etc.)
                - device: str
        """
        config = config or {}
        
        device = torch.device(config.get('device', 'cpu'))
        window_size = config.get('window_size', 30)
        feature_columns = config.get('feature_columns', 
                                      ['Open', 'High', 'Low', 'Close', 'Volume'])
        
        feature_type_str = config.get('feature_type', 'normalized')
        feature_type = FeatureType(feature_type_str) if isinstance(feature_type_str, str) else feature_type_str
        
        output_type = config.get('output_type', 'auto')
        
        self.controller = PredictionController(
            model=model,
            device=device,
            window_size=window_size,
            feature_columns=feature_columns,
            feature_type=feature_type,
            output_type=output_type
        )
        
        self._config = config
    
    def fit(self, training_data: pd.DataFrame) -> 'ModelInterface':
        """Fit normalizer on training data."""
        self.controller.fit_normalizer(training_data)
        return self
    
    def get_action(self, data: pd.DataFrame, timestep: int) -> Tuple[int, float]:
        """
        Get trading action at a specific timestep.
        
        Args:
            data: Market data DataFrame
            timestep: Current timestep index
            
        Returns:
            Tuple of (action, position_size)
        """
        return self.controller.predict_at_timestep(data, timestep)
    
    def get_all_actions(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get trading actions for all valid timesteps.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Tuple of (actions, positions) arrays
        """
        return self.controller.predict_rolling(data)
    
    @property
    def window_size(self) -> int:
        """Get the window size."""
        return self.controller.window_size
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return self.controller.get_statistics()
