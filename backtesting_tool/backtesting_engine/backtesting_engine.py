import pandas as pd
from typing import Dict, List, Any

from .backtesting_config import BacktestConfig
from .portfolio import Portfolio
from .trade_executor import TradeExecutor
from model_interface.prediction_controller import PredictionController
from model_interface.feature_converter import FeatureConverter
from model_interface.action_mapper import ActionMapper


class BacktestingEngine:

    def __init__(self, config: BacktestConfig,
                 predictor: PredictionController,
                 feature_converter: FeatureConverter,
                 action_mapper: ActionMapper):
        self.config = config
        self.predictor = predictor
        self.converter = feature_converter
        self.mapper = action_mapper

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest over the full DataFrame and return results dict."""
        seq_len = self.config.sequence_length
        portfolio = Portfolio(self.config.initial_capital)
        executor = TradeExecutor(portfolio, self.config.commission_rate)

        self.converter.fit(df)

        results: Dict[str, List] = {
            'dates': [], 'actual_prices': [], 'predicted_prices': [],
            'positions': [], 'portfolio_values': [],
            'cash': [], 'holdings': [], 'trades': [],
        }

        print(f"\n  Initial capital : ${self.config.initial_capital:,.2f}")
        print(f"  Commission      : {self.config.commission_pct}%")
        print(f"  Strategy        : {self.config.strategy}")
        print(f"  Sequence length : {seq_len}")
        print(f"  Simulating from day {seq_len} to {len(df)} ...")

        for i in range(seq_len, len(df)):
            current_date = df.index[i]
            current_price = float(df.iloc[i]['Close'])

            window = self.converter.get_window(df, i)
            if window is None:
                continue

            pred_scaled = self.predictor.predict(window)
            predicted_price = self.converter.inverse_transform_target(pred_scaled)

            action = self.mapper.get_action(predicted_price, current_price,
                                            portfolio.position)

            executor.execute(action, current_price)

            results['dates'].append(current_date)
            results['actual_prices'].append(current_price)
            results['predicted_prices'].append(predicted_price)
            results['positions'].append(portfolio.position)
            results['portfolio_values'].append(portfolio.value(current_price))
            results['cash'].append(portfolio.cash)
            results['holdings'].append(portfolio.shares)

        results['trades'] = executor.trade_log

        final = portfolio.value(results['actual_prices'][-1])
        ret = (final - self.config.initial_capital) / self.config.initial_capital * 100
        print(f"\n  Simulation complete!")
        print(f"  Days simulated : {len(results['dates'])}")
        print(f"  Trades executed: {len(results['trades'])}")
        print(f"  Final value    : ${final:,.2f}  ({ret:+.2f}%)")

        return results
