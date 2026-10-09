"""
Portfolio -- position, cash and P&L accounting.

Deliberately thin, and deliberately the only place equity is computed. The
compounding convention is stated once, here, and every strategy and every
allocation method in the project inherits it, which is what makes the Task 3
comparison fair.

Convention
----------
Positions are expressed as a fraction of current equity, not a share count. For
a single instrument traded from a single account that is the natural
representation: a position of +1 means fully long, -0.5 means half short.

The period return for candle t is

    net_t = p_t * r_t - cost_t,      r_t = open[t+1] / open[t] - 1

and equity compounds as equity_t = equity_{t-1} * (1 + net_t).

`net_t` is earned over the interval [open_t, open_{t+1}], and the series is
indexed by t, the candle whose open opened the position. The final candle has no
open[t+1], so it cannot be held; the position is forced flat there and charged
its exit cost.
"""

import numpy as np
import pandas as pd

import config


class Portfolio:
    """Tracks equity, positions and P&L for one strategy or one allocation."""

    def __init__(self, initial_capital=1.0):
        self.initial_capital = float(initial_capital)
        self.reset()

    def reset(self):
        self.history = pd.DataFrame()
        self.equity_curve = pd.Series(dtype=float)
        self.returns = pd.Series(dtype=float)
        self.positions = pd.Series(dtype=float)
        self.pnl = pd.Series(dtype=float)

    # ------------------------------------------------------------------
    def update(self, executions, market_data=None):
        """Update portfolio state from an execution frame.

        `executions` comes from ExecutionEngine.execute; `market_data` supplies
        the open-to-open return that the held position earns.
        """
        if market_data is None:
            raise ValueError("Portfolio.update needs market_data to mark positions.")

        df = executions.copy().reset_index(drop=True)
        mkt = market_data.reset_index(drop=True)
        df["ret_oo"] = mkt["ret_oo"].values

        # The last candle has no forward open, so nothing can be earned on it.
        # The Backtester has already forced the target flat there so that the
        # exit cost was charged; all that is left is to neutralise the undefined
        # return. If a position is still open at this point something upstream
        # has skipped the terminal condition, so say so loudly.
        last = df.index[-1]
        if pd.isna(df.at[last, "ret_oo"]):
            if abs(float(df.at[last, "position"])) > 1e-12:
                raise AssertionError(
                    "Terminal candle still carries a position but has no forward open. "
                    "The exit was never costed."
                )
            df.at[last, "ret_oo"] = 0.0

        df["gross_return"] = df["position"] * df["ret_oo"]
        df["net_return"] = df["gross_return"] - df["cost"]
        df["equity"] = self.initial_capital * (1.0 + df["net_return"]).cumprod()
        df["pnl"] = df["equity"].diff().fillna(df["equity"] - self.initial_capital)

        self.history = df
        idx = pd.DatetimeIndex(df["date"])
        self.equity_curve = pd.Series(df["equity"].values, index=idx, name="equity")
        self.returns = pd.Series(df["net_return"].values, index=idx, name="net_return")
        self.positions = pd.Series(df["position"].values, index=idx, name="position")
        self.pnl = pd.Series(df["pnl"].values, index=idx, name="pnl")
        return self

    def mark_to_market(self, market_data):
        """Mark positions to market. Returns the equity series.

        Accounting here is already mark-to-market on every candle -- equity is
        recomputed from the realised open-to-open move each period rather than
        only when a trade closes -- so this returns the current curve.
        """
        if self.equity_curve.empty and market_data is not None:
            raise ValueError("Portfolio has not been updated with executions yet.")
        return self.equity_curve

    # ------------------------------------------------------------------
    def get_equity_curve(self):
        return self.equity_curve

    def get_positions(self):
        return self.positions

    def get_pnl(self):
        return self.pnl

    def get_returns(self):
        return self.returns

    def get_turnover(self):
        """Average traded notional per candle, as a fraction of equity."""
        if self.history.empty:
            return 0.0
        return float(self.history["traded_notional"].mean())

    def get_total_costs(self):
        return float(self.history["cost"].sum()) if not self.history.empty else 0.0
