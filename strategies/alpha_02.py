"""
Alpha 02 -- Oscillator Reversal.

Hypothesis
----------
A bounded momentum oscillator reaching an extreme is a statement about
positioning, not about value. When the overbought flag fires, the buyers who
were going to buy have already bought; when the oversold flag fires, the forced
sellers are largely done. Either way the next few days lean back the other way.

This is a *different* hypothesis from Alpha 01 even though both are contrarian.
Alpha 01 is continuous and always on -- it is permanently leaning against
whatever stretch exists. Alpha 02 is an event strategy: it is flat most of the
time and only takes a position when a discrete condition fires. The information
arrives on different days and decays on a different schedule, which is what
makes it worth testing separately.

Signals used
------------
BB03 -- overbought condition from a bounded oscillator.
BB04 -- oversold condition from a bounded oscillator.

Trading rule
------------
When a flag fires, take a position in the direction the development window says
is profitable and hold it for `hold_days`, decaying linearly. Overlapping
triggers accumulate up to the position bound. The holding period is chosen on
the development window from a small, pre-declared grid -- not swept until
something looked good.

Where it should fail
--------------------
An oscillator extreme that marks the *start* of a repricing rather than the end
of one. Those cluster in regime breaks, so failures should arrive in bursts
rather than evenly. That clustering is tested rather than assumed.
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

    # The grid the robustness sweep explores. Declared on the class so the
    # sweep tests settings the hypothesis actually permits, rather than an
    # arbitrary range invented at report time.
    PARAM_GRID = {'hold_days': [2, 3, 5, 8, 10, 15]}

    HOLD_GRID = (3, 5, 10)      # declared in advance, not expanded after looking

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

        The direction comes from the horizon label; the holding period is chosen
        on the one-day realised return net of the mandated cost, so the choice
        answers the question a trader would ask -- which hold makes the most
        money after paying for the turnover it creates -- rather than which hold
        correlates best with a gross five-day number.
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
