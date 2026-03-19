"""Stress scenario generation"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from abc import ABC, abstractmethod
class StressScenarioGenerator(ABC):
    """Base class for stress scenario generators."""
    def __init__(self, random_seed: int = 42):
        """Initialize scenario generator."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
    @abstractmethod
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
"""Generate stressed version of dataset. Args: data: Original OHLCV dataset with columns [Open, High, Low, Close, Volume] Returns: Stressed dataset with same structure as input"""
        pass
    def validate_ohlcv(self, data: pd.DataFrame) -> None:
        """Validate that data has required OHLCV columns."""
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(f"Missing OHLCV columns: {missing}")
    @staticmethod
    def _update_high_low(df: pd.DataFrame) -> pd.DataFrame:
"""Recalculate High and Low based on Open and Close. Ensures High >= max(Open, Close) and Low <= min(Open, Close)."""
        df_copy = df.copy()
        df_copy['High'] = df_copy[['Open', 'Close']].max(axis=1)
        df_copy['Low'] = df_copy[['Open', 'Close']].min(axis=1)
        return df_copy
class MarketCrashScenario(StressScenarioGenerator):
"""Generate market crash scenarios. Simulate sudden price drop with optional recovery."""
    def __init__(self, crash_percentage: float = 0.20, 
    """Initialize instance"""
                 shock_days: int = 5, 
                 recovery_days: int = 20,
                 volume_increase: float = 1.5,
                 random_seed: int = 42):
"""Args: crash_percentage: Fraction of price drop (e.g., 0.20 for 20%) shock_days: Number of days for crash to occur recovery_days: Days for complete recovery volume_increase: Multiplier for volume during crash random_seed: RNG seed"""
        super().__init__(random_seed)
        self.crash_percentage = crash_percentage
        self.shock_days = shock_days
        self.recovery_days = recovery_days
        self.volume_increase = volume_increase
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate market crash scenario."""
        self.validate_ohlcv(data)
        df = data.copy()
        n = len(df)
        crash_start = n // 3
        crash_end = min(crash_start + self.shock_days, n - self.recovery_days)
        recovery_end = min(crash_end + self.recovery_days, n)
        crash_factor = 1.0 - self.crash_percentage
        for i in range(crash_start, crash_end):
            progress = (i - crash_start) / (crash_end - crash_start)
            factor = 1.0 - self.crash_percentage * progress
            df.loc[df.index[i], 'Open'] *= factor
            df.loc[df.index[i], 'Close'] *= factor
        for i in range(crash_end, recovery_end):
            progress = (i - crash_end) / (recovery_end - crash_end)
            factor = crash_factor + (1.0 - crash_factor) * progress
            df.loc[df.index[i], 'Close'] = \
                df.loc[df.index[i], 'Close'] * factor / (crash_factor if i == crash_end else 1.0)
        df = self._update_high_low(df)
        for i in range(crash_start, crash_end):
            df.loc[df.index[i], 'Volume'] *= self.volume_increase
        return df
class VolatilityShockScenario(StressScenarioGenerator):
"""Generate volatility shock scenario. Amplify market returns volatility while maintaining price direction."""
    def __init__(self, volatility_factor: float = 2.0,
    """Initialize instance"""
                 shock_duration: int = 30,
                 normalization_window: int = 20,
                 random_seed: int = 42):
"""Args: volatility_factor: Multiplier for volatility (2.0 = 2x volatility) shock_duration: Number of days with elevated volatility normalization_window: Days to gradually revert to normal volatility random_seed: RNG seed"""
        super().__init__(random_seed)
        self.volatility_factor = volatility_factor
        self.shock_duration = shock_duration
        self.normalization_window = normalization_window
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility shock scenario."""
        self.validate_ohlcv(data)
        df = data.copy()
        n = len(df)
        returns = df['Close'].pct_change().fillna(0)
        shock_start = n // 3
        shock_end = min(shock_start + self.shock_duration, n - self.normalization_window)
        norm_end = min(shock_end + self.normalization_window, n)
        for i in range(shock_start, norm_end):
            if i < shock_end:
                factor = self.volatility_factor
            else:
                progress = (i - shock_end) / (norm_end - shock_end)
                factor = self.volatility_factor - (self.volatility_factor - 1.0) * progress
            amplified_return = returns.iloc[i] * factor
            prev_close = df['Close'].iloc[i - 1] if i > 0 else df['Close'].iloc[i]
            new_close = prev_close * (1 + amplified_return)
            df.loc[df.index[i], 'Close'] = new_close
            if i > 0:
                df.loc[df.index[i], 'Open'] = df['Close'].iloc[i - 1]
        df = self._update_high_low(df)
        return df
class RegimeShiftScenario(StressScenarioGenerator):
"""Generate regime shift scenario. Simulate change in market conditions (bearish, low volume, trend reversal)."""
    def __init__(self, regime_type: str = "bearish",
    """Initialize instance"""
                 duration: int = 30,
                 drift: float = -0.001,
                 volatility_factor: float = 1.5,
                 volume_factor: float = 1.0,
                 random_seed: int = 42):
"""Args: regime_type: "bearish", "low_volume", or "trend_reversal" duration: Number of days in regime drift: Daily drift to apply (negative for bearish) volatility_factor: Volatility multiplier volume_factor: Volume multiplier random_seed: RNG seed"""
        super().__init__(random_seed)
        self.regime_type = regime_type
        self.duration = duration
        self.drift = drift
        self.volatility_factor = volatility_factor
        self.volume_factor = volume_factor
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate regime shift scenario."""
        self.validate_ohlcv(data)
        df = data.copy()
        n = len(df)
        regime_start = n // 3
        regime_end = min(regime_start + self.duration, n)
        returns = df['Close'].pct_change().fillna(0)
        base_vol = returns.std()
        for i in range(regime_start, regime_end):
            current_price = df['Close'].iloc[i - 1] if i > 0 else df['Close'].iloc[i]
            drift_component = current_price * self.drift
            vol_shock = np.random.normal(0, base_vol * self.volatility_factor)
            new_close = current_price + drift_component + vol_shock
            new_close = max(new_close, current_price * 0.5)
            df.loc[df.index[i], 'Close'] = new_close
            if i > 0:
                df.loc[df.index[i], 'Open'] = df['Close'].iloc[i - 1]
            if self.regime_type == "low_volume":
                df.loc[df.index[i], 'Volume'] *= self.volume_factor
        df = self._update_high_low(df)
        return df
class SyntheticStressScenario(StressScenarioGenerator):
"""Generate synthetic stress scenarios using various methods."""
    def __init__(self, method: str = "gbm",
    """Initialize instance"""
                 num_paths: int = 100,
                 drift: float = -0.001,
                 volatility: float = 0.03,
                 shock_probability: float = 0.05,
                 shock_magnitude: float = 0.15,
                 random_seed: int = 42):
"""Args: method: "gbm" (Geometric Brownian Motion), "bootstrap", or "random_shock" num_paths: Number of simulated paths (for GBM) drift: Daily drift for GBM volatility: Volatility parameter for GBM shock_probability: Probability of shock per day shock_magnitude: Magnitude of shocks random_seed: RNG seed"""
        super().__init__(random_seed)
        self.method = method
        self.num_paths = num_paths
        self.drift = drift
        self.volatility = volatility
        self.shock_probability = shock_probability
        self.shock_magnitude = shock_magnitude
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic stress scenario."""
        self.validate_ohlcv(data)
        if self.method == "gbm":
            return self._generate_gbm(data)
        elif self.method == "bootstrap":
            return self._generate_bootstrap(data)
        elif self.method == "random_shock":
            return self._generate_random_shock(data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    def _generate_gbm(self, data: pd.DataFrame) -> pd.DataFrame:
        """Geometric Brownian Motion simulation."""
        df = data.copy()
        n = len(df)
        S0 = df['Close'].iloc[0]
        returns = df['Close'].pct_change().fillna(0)
        dt = 1.0 / 252
        prices = [S0]
        for _ in range(n - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            prices.append(
                prices[-1] * np.exp((self.drift - 0.5 * self.volatility**2) * dt 
                                   + self.volatility * dW)
            )
        price_array = np.array(prices)
        prices_scaled = price_array / np.mean(price_array[-20:]) * df['Close'].iloc[-1]
        df['Close'] = prices_scaled
        df['Open'] = prices_scaled
        df = self._update_high_low(df)
        return df
    def _generate_bootstrap(self, data: pd.DataFrame) -> pd.DataFrame:
        """Bootstrap resampling of returns."""
        df = data.copy()
        returns = df['Close'].pct_change().fillna(0).values
        bootstrap_returns = np.random.choice(returns, size=len(returns), replace=True)
        S0 = df['Close'].iloc[0]
        prices = [S0]
        for ret in bootstrap_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        df['Close'] = prices
        df['Open'] = prices
        df = self._update_high_low(df)
        return df
    def _generate_random_shock(self, data: pd.DataFrame) -> pd.DataFrame:
        """Random shock injection."""
        df = data.copy()
        prices = df['Close'].values.copy()
        for i in range(1, len(prices)):
            if np.random.random() < self.shock_probability:
                shock = np.random.choice([-1, 1]) * self.shock_magnitude
                prices[i] *= (1 + shock)
            else:
                normal_return = np.random.normal(self.drift, self.volatility)
                prices[i] *= (1 + normal_return)
        prices = np.abs(prices)
        df['Close'] = prices
        df['Open'] = prices
        df = self._update_high_low(df)
        return df

