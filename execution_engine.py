"""
ExecutionEngine -- turns target positions into fills, costs and a trade log.

Everything about how an order becomes money happens here and nowhere else. The
backtester decides *what* position it wants; this module decides what that costs
and at what price it happens. Keeping the two apart is what lets us re-run every
strategy under identical execution assumptions, which the problem statement
requires for the allocation comparison.

Cost model
----------
The mandated cost is 0.05% of trade notional per side. In return space, if the
position moves from p_{t-1} to p_t, the notional traded is |p_t - p_{t-1}|
multiplied by current equity, so the drag on that period's return is simply

    cost_t = c * |p_t - p_{t-1}|

A full round trip (0 -> 1 -> 0) therefore costs 0.10%, exactly as specified.

Slippage
--------
Modelled as an additional per-side cost on the same traded notional, defaulting
to zero. The reasoning is in the report: the problem statement fixes the cost
assumption and tells us not to substitute a different one, so the headline
numbers carry the mandated cost alone and slippage is applied only inside the
cost-sensitivity study, where it is varied rather than assumed.
"""

import numpy as np
import pandas as pd

import config


class ExecutionEngine:
    """Simulates fills at candle t's open and charges the mandated cost."""

    def __init__(self, cost_per_side=None, slippage_bps=None):
        self.cost_per_side = config.TRANSACTION_COST if cost_per_side is None else cost_per_side
        self.slippage_bps = config.SLIPPAGE_BPS if slippage_bps is None else slippage_bps
        self.trades = []

    # ------------------------------------------------------------------
    @property
    def total_cost_per_side(self):
        return self.cost_per_side + self.slippage_bps / 10_000.0

    def execute(self, signal, market_data):
        """Simulate execution of a target-position series.

        Returns a frame with, per date: the target position actually held, the
        traded amount, the cost charged, and the fill price.
        """
        positions = pd.Series(np.asarray(signal, dtype=float), index=market_data.index)
        prev = positions.shift(1).fillna(0.0)
        traded = (positions - prev).abs()

        fill_price = market_data[config.EXECUTION_PRICE_FIELD]

        out = pd.DataFrame(
            {
                "date": market_data["date"].values,
                "position": positions.values,
                "prev_position": prev.values,
                "traded_notional": traded.values,
                "fill_price": fill_price.values,
            }
        )
        # Mandated cost and slippage are charged separately so both can be seen,
        # then summed. Keeping them apart in the output is what lets the
        # cost-sensitivity study vary one without disturbing the other.
        out["mandated_cost"] = self.apply_transaction_cost(out["traded_notional"])
        out["slippage_cost"] = self.apply_slippage(out["traded_notional"])
        out["cost"] = out["mandated_cost"] + out["slippage_cost"]
        self._log(out)
        return out

    def apply_transaction_cost(self, traded_notional):
        """Apply trading costs to a series of traded notionals (fractions of equity)."""
        return np.asarray(traded_notional, dtype=float) * self.cost_per_side

    def apply_slippage(self, traded_notional):
        """Apply execution slippage on the same traded notional.

        Kept separate from the mandated cost so that the two can always be
        reported and stressed independently.
        """
        return np.asarray(traded_notional, dtype=float) * (self.slippage_bps / 10_000.0)

    def record_trade(self, trade):
        self.trades.append(dict(trade))

    def get_trade_log(self):
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame(
            columns=["date", "side", "from_position", "to_position", "size", "price", "cost"]
        )

    # ------------------------------------------------------------------
    def _log(self, executions):
        """Record only the rows where the position actually changed."""
        self.trades = []
        changed = executions[executions["traded_notional"] > 1e-12]
        for row in changed.itertuples(index=False):
            self.record_trade(
                {
                    "date": row.date,
                    "side": "BUY" if row.position > row.prev_position else "SELL",
                    "from_position": row.prev_position,
                    "to_position": row.position,
                    "size": row.traded_notional,
                    "price": row.fill_price,
                    "cost": row.cost,
                }
            )
