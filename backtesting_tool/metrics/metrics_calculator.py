from typing import Dict, Any
from .performance_metrics import PerformanceMetrics


class MetricsCalculator:

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital

    def calculate(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate and per-stock performance metrics."""
        tickers = results.get('tickers', ['STOCK'])
        n_stocks = len(tickers)
        trades = results.get('trades', [])

        # ---- Aggregate portfolio metrics --------------------------------
        pm = PerformanceMetrics(
            portfolio_values=results['portfolio_values'],
            initial_capital=self.initial_capital,
            num_trades=len(trades),
        )
        print("\n  === AGGREGATE PORTFOLIO METRICS ===")
        pm.display()

        all_metrics: Dict[str, Any] = {'aggregate': pm.as_dict()}

        # ---- Per-stock summary (multi-stock only) -----------------------
        if n_stocks > 1:
            print(f"\n  === PER-STOCK SUMMARY ===")
            for t in tickers:
                ps = results['per_stock'][t]
                prices = ps['actual_prices']
                stock_trades = [tr for tr in trades if tr.get('ticker') == t]
                buys  = sum(1 for tr in stock_trades if tr['action'] == 'BUY')
                sells = sum(1 for tr in stock_trades if tr['action'] == 'SELL')
                price_ret = (prices[-1] - prices[0]) / prices[0] * 100

                print(f"\n  {t}:")
                print(f"    Price : ${prices[0]:.2f} -> ${prices[-1]:.2f}  ({price_ret:+.2f}%)")
                print(f"    Trades: {buys} buys, {sells} sells  ({buys + sells} total)")
                print(f"    Final shares: {ps['shares'][-1]}")

                all_metrics[t] = {
                    'Start Price ($)': round(prices[0], 2),
                    'End Price ($)': round(prices[-1], 2),
                    'Price Return (%)': round(price_ret, 2),
                    'Buy Trades': buys,
                    'Sell Trades': sells,
                    'Total Trades': buys + sells,
                    'Final Shares': ps['shares'][-1],
                }

        return all_metrics
