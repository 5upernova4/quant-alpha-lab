"""
Backtester -- the simulation loop.

It has no market logic. It gets positions from a strategy, passes them to the
execution engine, passes the fills to the portfolio, and asks the performance
analyser for metrics. A new alpha needs no change to this file.

Timing (checked here)
-------------------------------------
    for each candle t:
        inputs_t   = signal rows up to and including t-1
        decision_t = strategy.generate_signal(inputs_t)
        fill        at open[t]

generate_signals() applies the t-1 cutoff: it only passes the decision frame,
which FeatureEngine built with a strict backward as-of join.
execute_signals() fills at the open only.

The loop above is written as array operations for speed. This is safe only
because the cutoff is built into the frame, not the loop index. To check this,
run_reference_loop() runs the slow per-candle loop and
assert_matches_reference() checks both give the same equity curve.
"""

import numpy as np
import pandas as pd

import config
from execution_engine import ExecutionEngine
from performance import PerformanceAnalyzer
from portfolio import Portfolio


class Backtester:
    """Runs one strategy over one decision frame under fixed execution rules."""

    def __init__(self, execution_engine=None, portfolio=None, performance=None):
        self.execution_engine = execution_engine or ExecutionEngine()
        self.portfolio = portfolio or Portfolio()
        self.performance = performance or PerformanceAnalyzer()
        self.signals = None
        self.executions = None
        self.results = {}

    # ------------------------------------------------------------------
    def run(self, data, strategy, market_data=None, label=None):
        """Run a historical simulation.

        `data` is the decision frame (signals only, already lagged).
        `market_data` is the price frame used for fills and accounting.
        """
        if market_data is None:
            raise ValueError(
                "Backtester.run needs market_data. Prices are never part of the "
                "decision frame, so they must be supplied separately."
            )
        self._assert_aligned(data, market_data)

        label = label or strategy.get_metadata().get("name", type(strategy).__name__)
        signals = self.generate_signals(data, strategy)
        executions = self.execute_signals(signals, market_data)
        self.update_portfolio(executions, market_data)
        self.results = self.analyze(self.portfolio, label=label)
        self.results["metadata"] = strategy.get_metadata()
        return self.results

    def generate_signals(self, data, strategy):
        """Generate strategy decisions under the t-1 information cutoff.

        The cutoff is already in `data`: row t holds signal values observed
        strictly before date t. We check it again here in case a frame was
        built by hand.
        """
        if "information_asof" in data.columns:
            stale = data["information_asof"] >= data["date"]
            if stale.any():
                raise AssertionError(
                    f"{int(stale.sum())} rows carry signal information dated on or after "
                    "their own candle. The t-1 cutoff has been violated."
                )
        inputs = data.drop(columns=["information_asof"], errors="ignore")
        self.signals = strategy.generate_signal(inputs)
        return self.signals

    def execute_signals(self, signals, data):
        """Simulate signal execution -- fills at candle t's open.

        The last candle has no next open, so a position on it cannot be held
        or exited in the sample. We set the last target flat before costs are
        charged, so the engine charges the exit cost on that candle (like
        closing the book on the last day). Setting it flat after costs would
        give a free exit.
        """
        targets = pd.Series(np.asarray(signals, dtype=float), index=data.index).copy()
        if len(targets):
            targets.iloc[-1] = 0.0
        self.executions = self.execution_engine.execute(targets, data)
        return self.executions

    def update_portfolio(self, executions, data):
        """Update portfolio state from the executions."""
        self.portfolio.update(executions, market_data=data)
        return self.portfolio

    def analyze(self, portfolio, label="Strategy"):
        """Analyze performance of the resulting return series."""
        returns = portfolio.get_returns()
        metrics = self.performance.compute_all(
            returns,
            positions=portfolio.get_positions(),
            costs=portfolio.history["cost"] if not portfolio.history.empty else None,
            label=label,
        )
        return {
            "label": label,
            "metrics": metrics,
            "returns": returns,
            "equity_curve": portfolio.get_equity_curve(),
            "positions": portfolio.get_positions(),
            "trade_log": self.execution_engine.get_trade_log(),
            "history": portfolio.history,
        }

    def get_results(self):
        return self.results

    # ------------------------------------------------------------------
    # correctness harness
    # ------------------------------------------------------------------
    def run_reference_loop(self, data, strategy, market_data):
        """The literal per-candle loop from the Technical Documentation.

        Slow but easy to check. Used to confirm the vectorised path matches.
        """
        inputs = data.drop(columns=["information_asof"], errors="ignore")
        positions = strategy.generate_signal(inputs).to_numpy()
        opens = market_data["open"].to_numpy()
        n = len(opens)
        c = self.execution_engine.cost_per_side

        equity, prev, curve = 1.0, 0.0, []
        for t in range(n):
            target = positions[t] if t < n - 1 else 0.0   # cannot hold the last candle
            cost = c * abs(target - prev)
            r = (opens[t + 1] / opens[t] - 1.0) if t < n - 1 else 0.0
            equity *= (1.0 + target * r - cost)
            curve.append(equity)
            prev = target
        return pd.Series(curve, index=pd.DatetimeIndex(market_data["date"]))

    def assert_matches_reference(self, data, strategy, market_data, tol=1e-12):
        """Confirm the vectorised engine and the reference loop agree exactly."""
        fast = self.run(data, strategy, market_data)["equity_curve"]
        slow = self.run_reference_loop(data, strategy, market_data)
        diff = float((fast - slow).abs().max())
        if diff > tol:
            raise AssertionError(
                f"Vectorised backtest differs from the reference loop by {diff:.3e}."
            )
        return diff

    @staticmethod
    def _assert_aligned(decision, market):
        if len(decision) != len(market):
            raise AssertionError(
                f"Decision frame ({len(decision)}) and market frame ({len(market)}) "
                "have different lengths."
            )
        if not (decision["date"].values == market["date"].values).all():
            raise AssertionError("Decision and market frames are not date-aligned.")
