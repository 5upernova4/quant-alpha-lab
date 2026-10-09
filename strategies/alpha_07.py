"""
Alpha 07 - Drift-Anchored Dual Reversion.

Added at Task 3, after the Task 2 holdout had been opened (see
strategies/__init__.py). Its parameters are still fitted on the development
window only, but the idea came from looking at the Task 2 results, so its
holdout numbers are not a clean out-of-sample test.

Hypothesis
----------
Alphas 01-06 all trade reversion around zero. Since most of the reversion
signals are "trend is up, so fade it", they end up net short more often than
long: their average net position on the development window was -0.06, in an
instrument that went up 9.7% a year over the same window. The drift was simply
left on the table.

So this strategy separates two decisions:
  1. what to hold when there is no view  -> a long anchor, size fitted on dev
  2. how far to lean away from it         -> contrarian overlays

The overlays:
  * BB03/BB04 oscillator extremes, faded with a linear decay. On dev only the
    BB03 (overbought) side cleared the screen; BB04 got a zero sign.
  * PB01 + PB03 agreeing, faded with its own decay.

BB03 fires on about 14% of days, while the trend consensus is non-zero on about
65% of them, so the two overlays mostly act on different days. On dev: anchor
alone (i.e. buy and hold) Sharpe 0.62, anchor + oscillator 0.97, anchor + trend
0.82, all three 1.07.

With the fitted settings (anchor 1.0, gain 1.25) the overlays are big enough to
flip the position, not just trim it. On dev it was fully long on 41% of days
and short on 34% (mostly when both trend flags had been on for a while), with
an average net position of +0.35. So a good part of its return is market
exposure, and the Task 3 factor model separates that from the overlays' alpha.

Signals used: BB03, BB04, PB01, PB03.

Where it fails
--------------
Down or sideways years. The anchor loses and the overlays only soften it. In
2019 it lost 5.5% against 5.8% for buy and hold, so the overlays barely helped.
"""

import numpy as np
import pandas as pd

import config
from strategy import BaseStrategy
from strategies._fitting import direction_and_strength, event_response


class Alpha07DriftAnchoredDualReversion(BaseStrategy):
    name = "Alpha07_DriftAnchoredDualReversion"
    hypothesis = (
        "Hold the instrument's drift as the default position and lean against "
        "oscillator extremes and trend-flag crowding, instead of trading "
        "reversion from flat."
    )
    signals_used = ["BB03", "BB04", "PB01", "PB03"]
    fit_horizon = 5

    # Grids fixed before fitting. The anchor grid includes 0.0, so if the dev
    # window had shown no drift the fit could switch the anchor off and this
    # would become a plain market-neutral overlay.
    ANCHOR_GRID = (0.0, 0.3, 0.5, 0.7, 1.0)
    OSC_HOLD_GRID = (3, 5, 8, 10)
    TREND_HOLD_GRID = (3, 5, 8)
    GAIN_GRID = (0.75, 1.0, 1.25)

    PARAM_GRID = {
        "anchor": list(ANCHOR_GRID),
        "osc_hold": list(OSC_HOLD_GRID),
        "trend_hold": list(TREND_HOLD_GRID),
        "gain": list(GAIN_GRID),
    }

    OSC_SIGNALS = ["BB03", "BB04"]

    # Signs used only if generate_signal is called before fit(). They are the
    # Task 2 findings (overbought -> down, oversold -> up, trend consensus is
    # contrarian); fit() re-estimates all three on the dev window.
    PRIOR_SIGNS = {"BB03": -1.0, "BB04": 1.0, "trend": -1.0}

    def __init__(self, anchor=1.0, osc_hold=5, trend_hold=3, gain=1.0, **params):
        super().__init__(anchor=anchor, osc_hold=osc_hold,
                         trend_hold=trend_hold, gain=gain, **params)
        self.anchor = anchor
        self.osc_hold = osc_hold
        self.trend_hold = trend_hold
        self.gain = gain
        self.signs_ = dict(self.PRIOR_SIGNS)

    # ------------------------------------------------------------------
    @staticmethod
    def _decay_kernel(flag, hold):
        """Spread a trigger over `hold` days, weight falling linearly to zero.

        A step function would treat day 5 as being as fresh as day 1, and the
        event-response profile on dev doesn't look like that.
        """
        f = pd.Series(flag, dtype=float).fillna(0.0)
        weights = np.array([(hold - k) / hold for k in range(hold)])
        acc = pd.Series(0.0, index=f.index)
        for k, w in enumerate(weights):
            acc = acc + w * f.shift(k).fillna(0.0)
        return acc / weights.sum()

    @staticmethod
    def _consensus(data):
        # PB01 + PB03 - 1 is +1 when both trend flags are on, -1 when both are
        # off, and 0 when they disagree (no crowd to lean against)
        return (pd.Series(data["PB01"], dtype=float).fillna(0.0)
                + pd.Series(data["PB03"], dtype=float).fillna(0.0) - 1.0)

    def _osc_overlay(self, data, hold):
        out = pd.Series(0.0, index=data.index)
        for c in self.OSC_SIGNALS:
            s = self.signs_.get(c, 0.0)
            if s != 0.0:
                out = out + s * self._decay_kernel(data[c], hold)
        return out

    def _trend_overlay(self, data, hold):
        s = self.signs_.get("trend", 0.0)
        if s == 0.0:
            return pd.Series(0.0, index=data.index)
        return s * self._decay_kernel(self._consensus(data), hold)

    def _compose(self, data, anchor, osc_hold, trend_hold, gain):
        pos = (anchor
               + gain * self._osc_overlay(data, osc_hold)
               + gain * self._trend_overlay(data, trend_hold))
        return pos.clip(-1.0, 1.0)

    # ------------------------------------------------------------------
    def fit(self, data, target=None, daily_target=None):
        """Estimate the three overlay signs, then pick the four settings.

        Signs come from the 5-day forward return, same screen as the other
        alphas. The settings are then scored on the 1-day return net of the
        mandated cost, because the anchor changes turnover as well as return and
        a gross multi-day label would favour whichever setting trades most.

        5 x 4 x 3 x 3 = 180 combinations on ~770 days. That is a real
        multiple-testing exposure. The Task 3 run reports a deflated Sharpe for
        this grid (using the Sharpe spread across it) as well as the plateau:
        how many of the 180 settings are profitable at all.
        """
        if target is None:
            self.signs_ = dict(self.PRIOR_SIGNS)
            self.is_fitted = True
            return self

        self.signs_ = {}
        for c in self.OSC_SIGNALS:
            sign, t, spread = event_response(data[c], target, horizon=self.fit_horizon)
            self.signs_[c] = sign
            self.fitted_[c] = {"sign": sign, "t_stat": round(t, 3),
                               "spread_bps": round(spread * 1e4, 2)}

        sign, t, rho = direction_and_strength(self._consensus(data), target,
                                              horizon=self.fit_horizon)
        self.signs_["trend"] = sign
        self.fitted_["trend_consensus"] = {"sign": sign, "t_stat": round(t, 3),
                                           "spearman": round(rho, 4)}

        y = daily_target if daily_target is not None else target
        y = pd.Series(np.asarray(y, dtype=float), index=data.index)

        best = (self.anchor, self.osc_hold, self.trend_hold, self.gain)
        best_score = -np.inf
        for a in self.ANCHOR_GRID:
            for ho in self.OSC_HOLD_GRID:
                for ht in self.TREND_HOLD_GRID:
                    for g in self.GAIN_GRID:
                        pos = self._compose(data, a, ho, ht, g)
                        traded = pos.diff().abs().fillna(pos.abs())
                        pnl = (pos * y - config.TRANSACTION_COST * traded).dropna()
                        if len(pnl) < 60 or pnl.std(ddof=1) == 0:
                            continue
                        score = float(pnl.mean() / pnl.std(ddof=1)
                                      * np.sqrt(config.TRADING_DAYS_PER_YEAR))
                        if score > best_score:
                            best, best_score = (a, ho, ht, g), score

        self.anchor, self.osc_hold, self.trend_hold, self.gain = best
        self.params.update({"anchor": self.anchor, "osc_hold": self.osc_hold,
                            "trend_hold": self.trend_hold, "gain": self.gain})
        self.fitted_["selection"] = {
            "anchor": self.anchor,
            "osc_hold": self.osc_hold,
            "trend_hold": self.trend_hold,
            "gain": self.gain,
            "dev_sharpe_net_of_cost": round(best_score, 3),
            "combinations_searched": (len(self.ANCHOR_GRID) * len(self.OSC_HOLD_GRID)
                                      * len(self.TREND_HOLD_GRID) * len(self.GAIN_GRID)),
        }
        self.is_fitted = True
        return self

    def _signal(self, features):
        return self._compose(features, self.anchor, self.osc_hold,
                             self.trend_hold, self.gain)
