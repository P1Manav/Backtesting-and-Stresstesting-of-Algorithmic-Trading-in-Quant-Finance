import pandas as pd
from typing import Dict, List, Any

from .backtesting_config import BacktestConfig
from .portfolio import Portfolio
from .trade_executor import TradeExecutor
from model_interface.prediction_controller import PredictionController
from model_interface.feature_converter import FeatureConverter
from model_interface.action_mapper import ActionMapper


class BacktestingEngine:
    """Runs a multi-stock portfolio backtest using a single predictive model."""

    def __init__(self, config: BacktestConfig,
                 predictor: PredictionController,
                 feature_columns: List[str],
                 action_mapper: ActionMapper):
        self.config = config
        self.predictor = predictor
        self.feature_columns = feature_columns
        self.mapper = action_mapper

    # ------------------------------------------------------------------
    def run(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run backtest over one or more stocks.

        Args:
            stock_data: ``{ticker: DataFrame}`` — each DataFrame has a
                        DatetimeIndex and at least an OHLCV set of columns.

        Returns:
            Results dict with aggregate portfolio values *and* per-stock
            price / prediction / position histories.
        """
        tickers = list(stock_data.keys())
        seq_len = self.config.sequence_length

        # --- Align all stocks to common trading dates ------------------
        # Start with the dates from the stock that has the most data,
        # then iteratively keep only stocks whose overlap is large enough.
        all_dates = {t: set(stock_data[t].index) for t in tickers}

        # Use the ticker with the most data as the reference
        ref_ticker = max(tickers, key=lambda t: len(all_dates[t]))
        common_dates = all_dates[ref_ticker]

        # Iteratively add stocks that share enough dates
        kept = [ref_ticker]
        dropped = []
        for t in tickers:
            if t == ref_ticker:
                continue
            overlap = common_dates & all_dates[t]
            if len(overlap) >= seq_len + 1:
                common_dates = overlap
                kept.append(t)
            else:
                dropped.append((t, len(overlap)))

        if dropped:
            print(f"\n  [WARNING] Dropped {len(dropped)} stock(s) with insufficient date overlap:")
            for t, cnt in dropped:
                print(f"    {t}: only {cnt} common dates (need {seq_len + 1})")

        tickers = kept
        n_stocks = len(tickers)

        if n_stocks == 0 or len(common_dates) < seq_len + 1:
            raise ValueError(
                f"No stocks have enough common trading dates "
                f"(need at least {seq_len + 1} for sequence_length={seq_len}). "
                f"Try selecting fewer stocks or reducing the sequence length."
            )

        common_dates = sorted(common_dates)
        common_idx = pd.DatetimeIndex(common_dates)

        print(f"\n  Stocks retained : {n_stocks} / {n_stocks + len(dropped)}")
        print(f"  Common dates    : {len(common_dates)} trading days")

        portfolio = Portfolio(self.config.initial_capital, tickers)
        executor = TradeExecutor(portfolio, self.config.commission_rate)

        aligned: Dict[str, pd.DataFrame] = {
            t: stock_data[t].loc[common_idx] for t in tickers
        }

        # --- One FeatureConverter (scaler) per stock -------------------
        converters: Dict[str, FeatureConverter] = {}
        for t in tickers:
            conv = FeatureConverter(self.feature_columns, seq_len)
            conv.fit(aligned[t])
            converters[t] = conv

        # --- Prepare results container ---------------------------------
        results: Dict[str, Any] = {
            'dates': [],
            'portfolio_values': [],
            'cash': [],
            'tickers': tickers,
            'trades': [],
            'per_stock': {
                t: {
                    'actual_prices': [], 'predicted_prices': [],
                    'positions': [], 'shares': [],
                }
                for t in tickers
            },
        }

        # --- Print simulation header -----------------------------------
        print(f"\n  Stocks          : {', '.join(tickers)} ({n_stocks} stock{'s' if n_stocks > 1 else ''})")
        print(f"  Initial capital : ${self.config.initial_capital:,.2f}")
        if n_stocks > 1:
            print(f"  Allocation      : Equal weight (${self.config.initial_capital / n_stocks:,.2f} per stock)")
        print(f"  Commission      : {self.config.commission_pct}%")
        print(f"  Strategy        : {self.config.strategy}")
        print(f"  Sequence length : {seq_len}")
        print(f"  Common dates    : {len(common_dates)} trading days")
        print(f"  Simulating from day {seq_len} to {len(common_dates)} ...")

        total_steps = len(common_dates) - seq_len
        report_interval = max(1, total_steps // 10)

        # --- Main simulation loop --------------------------------------
        for step, i in enumerate(range(seq_len, len(common_dates))):
            current_date = common_dates[i]
            current_prices: Dict[str, float] = {}
            actions: Dict[str, str] = {}
            predicted_prices: Dict[str, float] = {}

            # Phase 1 — Predict for every stock
            for t in tickers:
                df_t = aligned[t]
                current_prices[t] = float(df_t.iloc[i]['Close'])

                window = converters[t].get_window(df_t, i)
                if window is None:
                    actions[t] = 'HOLD'
                    predicted_prices[t] = current_prices[t]
                    continue

                pred_scaled = self.predictor.predict(window)
                predicted_prices[t] = converters[t].inverse_transform_target(pred_scaled)
                actions[t] = self.mapper.get_action(
                    predicted_prices[t], current_prices[t],
                    portfolio.positions[t])

            # Phase 2 — Execute SELLs first (free up cash)
            for t in tickers:
                if actions[t] == 'SELL':
                    executor.execute('SELL', t, current_prices[t], date=current_date)

            # Phase 3 — Execute BUYs with equal budget split
            buy_tickers = [t for t in tickers if actions[t] == 'BUY']
            if buy_tickers:
                budget_each = portfolio.cash / len(buy_tickers)
                for t in buy_tickers:
                    executor.execute('BUY', t, current_prices[t],
                                     date=current_date, budget=budget_each)

            # Record results
            port_val = portfolio.value(current_prices)
            results['dates'].append(current_date)
            results['portfolio_values'].append(port_val)
            results['cash'].append(portfolio.cash)

            for t in tickers:
                results['per_stock'][t]['actual_prices'].append(current_prices[t])
                results['per_stock'][t]['predicted_prices'].append(
                    predicted_prices.get(t, current_prices[t]))
                results['per_stock'][t]['positions'].append(portfolio.positions[t])
                results['per_stock'][t]['shares'].append(portfolio.shares[t])

            # Progress report
            if (step + 1) % report_interval == 0 or step == total_steps - 1:
                pct = (step + 1) / total_steps * 100
                print(f"    [{pct:5.1f}%]  Day {i}  |  Portfolio: ${port_val:,.2f}")

        # --- Final summary ---------------------------------------------
        results['trades'] = executor.trade_log

        final = portfolio.value(current_prices)
        ret = (final - self.config.initial_capital) / self.config.initial_capital * 100
        print(f"\n  Simulation complete!")
        print(f"  Days simulated : {len(results['dates'])}")
        print(f"  Trades executed: {len(results['trades'])}")
        print(f"  Final value    : ${final:,.2f}  ({ret:+.2f}%)")

        if n_stocks > 1:
            print(f"\n  Per-stock holdings at end:")
            for t in tickers:
                s = portfolio.shares[t]
                v = portfolio.stock_value(t, current_prices[t])
                print(f"    {t:10s}: {s:6d} shares  = ${v:>12,.2f}")
            print(f"    {'Cash':10s}: {'':6s}        = ${portfolio.cash:>12,.2f}")

        return results
