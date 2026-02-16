import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler


class FeatureConverter:

    def __init__(self, feature_columns: List[str], sequence_length: int = 60,
                 scaling_range: Tuple[float, float] = (-1, 1)):
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.scaling_range = scaling_range

        self.scaler = MinMaxScaler(feature_range=scaling_range)
        self._fitted = False
        self.target_column = 'Close'

    def fit(self, df: pd.DataFrame) -> 'FeatureConverter':
        """Fit the scaler on the full dataset."""
        values = df[self.feature_columns].values
        self.scaler.fit(values)
        self._fitted = True
        return self

    def get_window(self, df: pd.DataFrame, end_idx: int) -> Optional[np.ndarray]:
        """Extract a scaled window of shape (1, sequence_length, n_features) ending at end_idx."""
        if end_idx < self.sequence_length:
            return None

        if not self._fitted:
            self.fit(df.iloc[:end_idx])

        window = df.iloc[end_idx - self.sequence_length:end_idx][self.feature_columns].values
        window_scaled = self.scaler.transform(window)
        return window_scaled.reshape(1, self.sequence_length, len(self.feature_columns))

    def inverse_transform_target(self, scaled_value: np.ndarray) -> float:
        """Inverse-transform a predicted value back to the original price scale."""
        if scaled_value.ndim == 0:
            scaled_value = scaled_value.reshape(1, 1)
        elif scaled_value.ndim == 1:
            scaled_value = scaled_value.reshape(-1, 1)

        n_features = len(self.feature_columns)

        if n_features == 1:
            return float(self.scaler.inverse_transform(scaled_value)[0, 0])

        target_idx = self.feature_columns.index(self.target_column)
        dummy = np.zeros((1, n_features))
        dummy[0, target_idx] = scaled_value.flatten()[0]
        inverse = self.scaler.inverse_transform(dummy)
        return float(inverse[0, target_idx])
