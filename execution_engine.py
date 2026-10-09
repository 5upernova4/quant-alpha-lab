"""
ExecutionEngine -- turns target positions into fills, costs and a trade log.

The backtester decides what position it wants; this module decides the fill
price and the cost. Keeping these apart means every strategy runs under the
same execution rules, which the allocation comparison needs.

Cost model
----------
The mandated cost is 0.05% of trade notional per side. If the position moves
from p_{t-1} to p_t, the notional traded is |p_t - p_{t-1}| times current
equity, so the cost in return terms is

    cost_t = c * |p_t - p_{t-1}|

A full round trip (0 -> 1 -> 0) costs 0.10%, as specified.

Slippage
--------
An extra per-side cost on the same traded notional, zero by default. The
problem statement fixes the cost and says not to replace it, so the headline
numbers use only the mandated cost. Slippage is used only in the
cost-sensitivity study, where it is varied. See the report.
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

        Returns a frame with, per date: the position held, the traded amount,
        the cost charged and the fill price.
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
        # Mandated cost and slippage are kept in separate columns, then summed,
        # so the cost-sensitivity study can vary one without the other.
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

        Kept separate from the mandated cost so each can be reported and
        stressed on its own.
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
        """Record only the rows where the position changed."""
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
