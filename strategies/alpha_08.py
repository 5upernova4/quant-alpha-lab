"""
Alpha 08 - Next-Day Continuation.

Added for v2, after the competition. See reports/v2_improvements.md.

Where the idea came from
------------------------
The obvious way to use the mean-reversion finding is to fade yesterday's
move. I tested that on the development window first and it lost money in
every variant, in every year (2018, 2019 and 2020 separately). The sign was
the other way round: at a one-day horizon, yesterday's direction tends to
carry into the next open-to-open day. The reversion this instrument is known
for shows up over 5-10 days, and comes from the trend-state flags, not from
one day's move.

So this strategy trades the one-day continuation, and the rest of the book
keeps trading the slower reversion.

Signals used
------------
PB07 - distance of price from a medium-term reference. Its one-day change is
       almost the same as the previous session's return (correlation 0.97 on
       dev), because the reference moves slowly. So the change in PB07 gives
       us yesterday's direction without touching price.
BB07 - recent volatility level. Used only as an optional on/off gate.

Rule
----
position = sign_from_fit * sign(change in PB07), optionally only on days when
BB07 is above its own trailing median. Whether to use the gate is chosen in
fit() by net Sharpe on the dev window, the same way alpha_02 picks its hold.

Where it fails
--------------
It trades a lot (about 125x a year on dev), so costs take over a third of
its gross return, and it breaks even at about 12.5 bps per side, 2.5x the
mandated cost. On its own it is weak: Newey-West p = 0.16 on dev, and only
an 18% chance of beating the best of the ~18 variants I tried by luck. It
is in the book as a small sleeve because it is nearly uncorrelated with
the reversion strategies, not because it is strong.
"""

import numpy as np
import pandas as pd

import config
from strategy import BaseStrategy
from strategies._fitting import direction_and_strength


class Alpha08NextDayContinuation(BaseStrategy):
    name = "Alpha08_NextDayContinuation"
    hypothesis = (
        "The previous session's direction carries into the next open-to-open "
        "day, while the multi-day trend state still mean-reverts."
    )
    signals_used = ["PB07", "BB07"]
    fit_horizon = 1

    GATE_GRID = (False, True)
    PARAM_GRID = {"vol_gate": [False, True], "gate_window": [63, 126, 252]}

    def __init__(self, vol_gate=False, gate_window=126, **params):
        super().__init__(vol_gate=vol_gate, gate_window=gate_window, **params)
        self.vol_gate = vol_gate
        self.gate_window = gate_window
        self.sign_ = 1.0

    # ------------------------------------------------------------------
    @staticmethod
    def _move(data):
        # one-day change in PB07 ~ yesterday's return; NaN on the first row
        return pd.Series(data["PB07"], dtype=float).diff()

    def _gate(self, data, use_gate):
        if not use_gate:
            return pd.Series(1.0, index=data.index)
        vol = pd.Series(data["BB07"], dtype=float)
        median = vol.rolling(self.gate_window, min_periods=self.gate_window // 3).median()
        # NaN (warm-up) compares as False, so the gate starts closed
        return (vol > median).astype(float)

    def _compose(self, data, use_gate):
        direction = np.sign(self._move(data)).fillna(0.0)
        return (self.sign_ * direction * self._gate(data, use_gate)).clip(-1.0, 1.0)

    # ------------------------------------------------------------------
    def fit(self, data, target=None, daily_target=None):
        """Estimate the sign on the 1-day return, then choose the gate."""
        if target is None:
            self.sign_ = 1.0
            self.is_fitted = True
            return self

        sign, t, rho = direction_and_strength(self._move(data), target, horizon=1)
        self.sign_ = sign
        self.fitted_["PB07_change"] = {"sign": sign, "t_stat": round(t, 3),
                                       "spearman": round(rho, 4)}

        y = daily_target if daily_target is not None else target
        y = pd.Series(np.asarray(y, dtype=float), index=data.index)
        best, best_score = self.vol_gate, -np.inf
        scores = {}
        for g in self.GATE_GRID:
            pos = self._compose(data, g)
            traded = pos.diff().abs().fillna(pos.abs())
            pnl = (pos * y - config.TRANSACTION_COST * traded).dropna()
            if len(pnl) < 60 or pnl.std(ddof=1) == 0:
                continue
            score = float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(config.TRADING_DAYS_PER_YEAR))
            scores[str(g)] = round(score, 3)
            if score > best_score:
                best, best_score = g, score
        self.vol_gate = best
        self.params["vol_gate"] = best
        self.fitted_["vol_gate"] = {"chosen": best, "dev_sharpe_net_of_cost": scores}
        self.is_fitted = True
        return self

    def _signal(self, features):
        return self._compose(features, self.vol_gate)
