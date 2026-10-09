"""
Alpha 02 -- Oscillator Reversal.

Hypothesis
----------
A bounded momentum oscillator at an extreme tells us about positioning, not
value. When the overbought flag fires, most buyers have already bought; when
the oversold flag fires, forced sellers are mostly done. Either way the next
few days tend to move back the other way.

This is a different hypothesis from Alpha 01, though both are contrarian.
Alpha 01 is continuous and always on, always leaning against the current
stretch. Alpha 02 is an event strategy: flat most of the time, trading only
when a flag fires. The information comes on different days and fades at a
different speed, so it is tested separately.

Signals used
------------
BB03 -- overbought condition from a bounded oscillator.
BB04 -- oversold condition from a bounded oscillator.

Trading rule
------------
When a flag fires, take a position in the direction the development window
found profitable and hold it for `hold_days`, shrinking linearly. Overlapping
triggers add up to the position bound. The holding period is picked on the
development window from a small grid set in advance.

Where it should fail
--------------------
An oscillator extreme that marks the start of a repricing, not the end.
These cluster around regime changes, so losses should come in bursts, not
evenly. The robustness tests check this.
"""

import numpy as np
import pandas as pd

import config
from strategy import BaseStrategy
from strategies._fitting import event_response


class Alpha02OscillatorReversal(BaseStrategy):
    name = "Alpha02_OscillatorReversal"
    hypothesis = (
        "Bounded-oscillator extremes mark exhaustion of a positioning flow; "
        "fade the extreme for a few days."
    )
    signals_used = ["BB03", "BB04"]
    fit_horizon = 5    # Oscillator exhaustion unwinds over roughly a week.

    # The grid used by the robustness sweep. Set on the class so the sweep
    # tests settings the hypothesis allows, not a range picked at report time.
    PARAM_GRID = {'hold_days': [2, 3, 5, 8, 10, 15]}

    HOLD_GRID = (3, 5, 10)      # set in advance, not widened after looking

    def __init__(self, hold_days=5, **params):
        super().__init__(hold_days=hold_days, **params)
        self.hold_days = hold_days
        self.signs_ = {}

    # ------------------------------------------------------------------
    def _decay_kernel(self, flag, hold):
        """Spread a one-day trigger over `hold` days with a linear decay."""
        f = pd.Series(flag, dtype=float).fillna(0.0)
        weights = np.array([(hold - k) / hold for k in range(hold)])
        acc = pd.Series(0.0, index=f.index)
        for k, w in enumerate(weights):
            acc = acc + w * f.shift(k).fillna(0.0)
        return acc / weights.sum()

    def fit(self, data, target=None, daily_target=None):
        """Estimate each flag's direction, then pick the holding period.

        The direction comes from the horizon label. The holding period is
        chosen on the one-day return net of the mandated cost, so it picks the
        hold that makes the most money after paying for its turnover, not the
        one that best matches a gross five-day return.
        """
        self.signs_ = {}
        if target is None:
            self.signs_ = {"BB03": -1.0, "BB04": 1.0}
            self.is_fitted = True
            return self

        for c in self.signals_used:
            sign, t, spread = event_response(data[c], target, horizon=self.fit_horizon)
            self.signs_[c] = sign
            self.fitted_[c] = {"sign": sign, "t_stat": round(t, 3), "spread_bps": round(spread * 1e4, 2)}

        best, best_score = self.hold_days, -np.inf
        scoring_label = daily_target if daily_target is not None else target
        y = pd.Series(np.asarray(scoring_label, dtype=float), index=data.index)
        for h in self.HOLD_GRID:
            pos = self._compose(data, h)
            traded = pos.diff().abs().fillna(pos.abs())
            pnl = (pos * y - config.TRANSACTION_COST * traded).dropna()
            if len(pnl) < 60 or pnl.std(ddof=1) == 0:
                continue
            score = float(pnl.mean() / pnl.std(ddof=1)
                          * np.sqrt(config.TRADING_DAYS_PER_YEAR))
            if score > best_score:
                best, best_score = h, score
        self.hold_days = best
        self.params["hold_days"] = best
        self.fitted_["hold_days"] = {"chosen": best,
                                     "dev_sharpe_net_of_cost": round(best_score, 3),
                                     "grid": list(self.HOLD_GRID),
                                     "scored_on": "1-day net" if daily_target is not None
                                     else "horizon label (fallback)"}
        self.is_fitted = True
        return self

    def _compose(self, data, hold):
        if not self.signs_:
            self.signs_ = {"BB03": -1.0, "BB04": 1.0}
        score = pd.Series(0.0, index=data.index)
        for c in self.signals_used:
            s = self.signs_.get(c, 0.0)
            if s != 0.0:
                score = score + s * self._decay_kernel(data[c], hold)
        return score.clip(-1.0, 1.0)

    def _signal(self, features):
        return self._compose(features, self.hold_days)
