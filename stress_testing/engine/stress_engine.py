"""Stress testing engine"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'backtesting_tool'))
from backtesting_engine import BacktestConfig, BacktestingEngine
from model_interface import PredictionController, ActionMapper
from metrics import MetricsCalculator
from stress_testing.scenarios import (MarketCrashScenario, VolatilityShockScenario,
                      RegimeShiftScenario, SyntheticStressScenario)
from stress_testing.utils.logger import setup_logger, log_section, log_subsection
@dataclass
class StressScenarioResult:
    """Container for scenario test results."""
    scenario_name: str
    scenario_type: str
    scenario_params: Dict[str, Any]
    baseline_metrics: Dict[str, float]
    stressed_metrics: Dict[str, float]
    performance_degradation: Dict[str, float]
    timestamp: str
class StressTestingEngine:
"""Orchestrate stress testing of trading models. Pipeline: 1. Load model 2. Load dataset 3. Run baseline backtest 4. Generate stressed scenarios 5. Run backtests on stressed data 6. Collect and analyze results 7. Generate reports"""
    def __init__(self, model_path: str, dataset_path: str, 
    """Initialize instance"""
                 predictor: PredictionController, action_mapper: ActionMapper,
                 config: BacktestConfig, feature_columns: List[str],
                 logger_name: str = "StressTestingEngine"):
"""Initialize stress testing engine. Args: model_path: Path to .pt model file dataset_path: Path to CSV dataset predictor: Prediction controller for model inference action_mapper: Mapper to convert predictions to actions config: Backtesting configuration feature_columns: Feature column names for model logger_name: Logger name"""
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.predictor = predictor
        self.action_mapper = action_mapper
        self.config = config
        self.feature_columns = feature_columns
        self.logger = setup_logger(logger_name)
        self.baseline_result = None
        self.baseline_metrics = None
        self.scenario_results: Dict[str, StressScenarioResult] = {}
    def load_dataset(self) -> Tuple[pd.DataFrame, str]:
"""Load and prepare dataset. Returns: Tuple of (DataFrame with ticker as index column, ticker symbol)"""
        log_section(self.logger, "LOADING DATASET")
        self.logger.info(f"Loading from: {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        ticker = Path(self.dataset_path).stem.upper()
        self.logger.info(f"Loaded {len(df)} trading days for {ticker}")
        self.logger.info(f"Columns: {list(df.columns)}")
        self.logger.info(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
        return df, ticker
    def run_baseline_backtest(self, data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
"""Run baseline backtest on original (unstressed) data. Args: data: Original OHLCV dataset ticker: Stock ticker symbol Returns: Baseline backtest results"""
        log_section(self.logger, "RUNNING BASELINE BACKTEST")
        stock_data = {ticker: data}
        engine = BacktestingEngine(
            config=self.config,
            predictor=self.predictor,
            feature_columns=self.feature_columns,
            action_mapper=self.action_mapper
        )
        self.logger.info(f"Running baseline backtest on {ticker}...")
        baseline_results = engine.run(stock_data)
        calculator = MetricsCalculator(self.config.initial_capital)
        baseline_metrics = calculator.calculate(baseline_results)
        self.baseline_result = baseline_results
        self.baseline_metrics = baseline_metrics.get('aggregate', {})
        log_subsection(self.logger, "Baseline Performance")
        self.logger.info(f"  Total Return: {self.baseline_metrics.get('Total Return (%)', 0):.2f}%")
        self.logger.info(f"  Sharpe Ratio: {self.baseline_metrics.get('Sharpe Ratio', 0):.4f}")
        self.logger.info(f"  Max Drawdown: {self.baseline_metrics.get('Max Drawdown (%)', 0):.2f}%")
        self.logger.info(f"  Volatility: {self.baseline_metrics.get('Annualized Volatility (%)', 0):.2f}%")
        return self.baseline_metrics
    def run_market_crash_scenarios(self, data: pd.DataFrame, ticker: str,
    """Execute main process"""
                                  crash_levels: List[float] = None) -> Dict[str, StressScenarioResult]:
        """Run market crash stress scenarios."""
        log_section(self.logger, "MARKET CRASH SCENARIOS")
        if crash_levels is None:
            crash_levels = [0.10, 0.20, 0.30, 0.50]
        results = {}
        for crash_pct in crash_levels:
            scenario_name = f"Market Crash ({crash_pct*100:.0f}%)"
            log_subsection(self.logger, scenario_name)
            generator = MarketCrashScenario(
                crash_percentage=crash_pct,
                shock_days=5,
                recovery_days=20
            )
            stressed_data = generator.generate(data)
            result = self._run_stress_backtest(
                stressed_data, ticker,
                scenario_name=scenario_name,
                scenario_type="market_crash",
                scenario_params={"crash_percentage": crash_pct}
            )
            results[scenario_name] = result
            self.scenario_results[scenario_name] = result
        return results
    def run_volatility_shock_scenarios(self, data: pd.DataFrame, ticker: str,
    """Execute main process"""
                                       vol_factors: List[float] = None) -> Dict[str, StressScenarioResult]:
        """Run volatility shock stress scenarios."""
        log_section(self.logger, "VOLATILITY SHOCK SCENARIOS")
        if vol_factors is None:
            vol_factors = [2.0, 3.0, 5.0]
        results = {}
        for vol_factor in vol_factors:
            scenario_name = f"Volatility Shock ({vol_factor:.1f}x)"
            log_subsection(self.logger, scenario_name)
            generator = VolatilityShockScenario(
                volatility_factor=vol_factor,
                shock_duration=30,
                normalization_window=20
            )
            stressed_data = generator.generate(data)
            result = self._run_stress_backtest(
                stressed_data, ticker,
                scenario_name=scenario_name,
                scenario_type="volatility_shock",
                scenario_params={"volatility_factor": vol_factor}
            )
            results[scenario_name] = result
            self.scenario_results[scenario_name] = result
        return results
    def run_regime_shift_scenarios(self, data: pd.DataFrame, ticker: str,
    """Execute main process"""
                                   regime_types: List[str] = None) -> Dict[str, StressScenarioResult]:
        """Run regime shift stress scenarios."""
        log_section(self.logger, "REGIME SHIFT SCENARIOS")
        if regime_types is None:
            regime_types = ["bearish", "low_volume", "trend_reversal"]
        regime_params = {
            "bearish": {"drift": -0.001, "volatility_factor": 1.5},
            "low_volume": {"drift": 0.0, "volatility_factor": 1.2, "volume_factor": 0.3},
            "trend_reversal": {"drift": -0.002, "volatility_factor": 1.3}
        }
        results = {}
        for regime_type in regime_types:
            scenario_name = f"Regime Shift ({regime_type})"
            log_subsection(self.logger, scenario_name)
            params = regime_params.get(regime_type, {})
            generator = RegimeShiftScenario(
                regime_type=regime_type,
                duration=30,
                **params
            )
            stressed_data = generator.generate(data)
            result = self._run_stress_backtest(
                stressed_data, ticker,
                scenario_name=scenario_name,
                scenario_type="regime_shift",
                scenario_params={"regime_type": regime_type, **params}
            )
            results[scenario_name] = result
            self.scenario_results[scenario_name] = result
        return results
    def run_synthetic_scenarios(self, data: pd.DataFrame, ticker: str,
    """Execute main process"""
                               methods: List[str] = None,
                               num_per_method: int = 3) -> Dict[str, StressScenarioResult]:
        """Run synthetic stress scenarios."""
        log_section(self.logger, "SYNTHETIC STRESS SCENARIOS")
        if methods is None:
            methods = ["gbm", "bootstrap", "random_shock"]
        results = {}
        for method in methods:
            for i in range(num_per_method):
                scenario_name = f"Synthetic {method.upper()} #{i+1}"
                log_subsection(self.logger, scenario_name)
                generator = SyntheticStressScenario(
                    method=method,
                    random_seed=42 + i
                )
                stressed_data = generator.generate(data)
                result = self._run_stress_backtest(
                    stressed_data, ticker,
                    scenario_name=scenario_name,
                    scenario_type="synthetic",
                    scenario_params={"method": method, "run": i+1}
                )
                results[scenario_name] = result
                self.scenario_results[scenario_name] = result
        return results
    def _run_stress_backtest(self, stressed_data: pd.DataFrame, ticker: str,
    """Execute main process"""
                            scenario_name: str, scenario_type: str,
                            scenario_params: Dict[str, Any]) -> StressScenarioResult:
"""Run backtest on stressed data and calculate degradation metrics. Args: stressed_data: Stressed OHLCV dataset ticker: Stock ticker scenario_name: Name of scenario scenario_type: Type of scenario scenario_params: Scenario configuration parameters Returns: StressScenarioResult with metrics and degradation analysis"""
        stock_data = {ticker: stressed_data}
        engine = BacktestingEngine(
            config=self.config,
            predictor=self.predictor,
            feature_columns=self.feature_columns,
            action_mapper=self.action_mapper
        )
        scenario_results = engine.run(stock_data)
        calculator = MetricsCalculator(self.config.initial_capital)
        all_scenario_metrics = calculator.calculate(scenario_results)
        scenario_metrics = all_scenario_metrics.get('aggregate', {})
        degradation = self._calculate_degradation(
            self.baseline_metrics, scenario_metrics
        )
        self.logger.info(f"  Return: {scenario_metrics.get('Total Return (%)', 0):.2f}% "
                        f"(d {degradation['total_return_drop']:.2f}%)")
        self.logger.info(f"  Sharpe: {scenario_metrics.get('Sharpe Ratio', 0):.4f} "
                        f"(d {degradation['sharpe_drop']:.4f})")
        self.logger.info(f"  Max DD: {scenario_metrics.get('Max Drawdown (%)', 0):.2f}% "
                        f"(d {degradation['max_drawdown_increase']:.2f}%)")
        return StressScenarioResult(
            scenario_name=scenario_name,
            scenario_type=scenario_type,
            scenario_params=scenario_params,
            baseline_metrics=self.baseline_metrics,
            stressed_metrics=scenario_metrics,
            performance_degradation=degradation,
            timestamp=datetime.now().isoformat()
        )
    @staticmethod
    def _calculate_degradation(baseline: Dict[str, float],
    """Perform calculations"""
                              stressed: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance degradation metrics."""
        return {
            'total_return_drop': baseline.get('Total Return (%)', 0) - stressed.get('Total Return (%)', 0),
            'sharpe_drop': baseline.get('Sharpe Ratio', 0) - stressed.get('Sharpe Ratio', 0),
            'sortino_drop': baseline.get('Sortino Ratio', 0) - stressed.get('Sortino Ratio', 0),
            'max_drawdown_increase': stressed.get('Max Drawdown (%)', 0) - baseline.get('Max Drawdown (%)', 0),
            'volatility_increase': stressed.get('Annualized Volatility (%)', 0) - baseline.get('Annualized Volatility (%)', 0),
            'calmar_drop': baseline.get('Calmar Ratio', 0) - stressed.get('Calmar Ratio', 0),
        }
    def get_scenario_results(self) -> Dict[str, StressScenarioResult]:
        """Get all scenario results."""
        return self.scenario_results
    def get_baseline_metrics(self) -> Dict[str, float]:
        """Get baseline metrics."""
        return self.baseline_metrics

