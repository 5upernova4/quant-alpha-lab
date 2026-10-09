"""
Alpha 03 -- Trend-State Fade.

Hypothesis
----------
The price-based state flags show where the trend has been, not where it is
going. When short-term returns mean-revert, many trend flags reading "up"
describe a move that is already priced in. The hypothesis is that agreement
among the flags is a contrarian signal: the more of them point the same way,
the more crowded the trade and the weaker the forward return.

This differs from Alpha 01 in what it reads. Alpha 01 measures how far price
has moved from a reference (continuous distance). Alpha 03 counts how many
separate trend definitions agree (discrete state). A market can be very
stretched with one flag on, or barely stretched with every flag on. These are
different signals, and they change at very different speeds.

Signals used
------------
PB01, PB02 -- short and long-term trend-direction flags.
PB03, PB04 -- price above short and long-term average reference levels.
PB05 -- sign of recent short-horizon momentum.

Trading rule
------------
Build a consensus score as the signed average of the flags, each with the
sign the development window supports, and hold a position in proportion to it.
State flags change slowly, so positions change slowly and turnover is low.
That is the practical reason to hold this alpha next to the faster ones.

Where it should fail
--------------------
A long, steady trend where every flag stays on for months. The strategy is
then short the whole way up. This is like Alpha 01's failure mode, and the
orthogonality analysis checks whether the two fail together.
"""

import numpy as np
import pandas as pd

from strategy import BaseStrategy
from strategies._fitting import MIN_ABS_T, event_response


class Alpha03TrendStateFade(BaseStrategy):
    name = "Alpha03_TrendStateFade"
    hypothesis = (
        "Agreement among trend-state flags marks a crowded, already-paid-for move; "
        "position against the consensus."
    )
    signals_used = ["PB01", "PB02", "PB03", "PB04", "PB05"]
    fit_horizon = 10    # Trend-state flags are persistent; the crowding they describe unwinds slowly.

    # The grid used by the robustness sweep. Set on the class so the sweep
    # tests settings the hypothesis allows, not a range picked at report time.
    PARAM_GRID = {'min_abs_t': [0.5, 0.75, 1.0, 1.5, 2.0]}

    def __init__(self, min_abs_t=None, **params):
        # Defaults to the shared screening floor, so all six strategies are
        # screened the same way. It is still a parameter because the
        # robustness sweep varies it.
        self.min_abs_t = MIN_ABS_T if min_abs_t is None else min_abs_t
        super().__init__(min_abs_t=self.min_abs_t, **params)
        self.signs_ = {}

    def fit(self, data, target=None, daily_target=None):
        self.signs_ = {}
        if target is None:
            self.signs_ = {c: -1.0 for c in self.signals_used}
            self.is_fitted = True
            return self
        for c in self.signals_used:
            sign, t, spread = event_response(data[c], target, min_abs_t=self.min_abs_t, horizon=self.fit_horizon)
            self.signs_[c] = sign
            self.fitted_[c] = {"sign": sign, "t_stat": round(t, 3), "spread_bps": round(spread * 1e4, 2)}
        kept = [c for c, s in self.signs_.items() if s != 0.0]
        self.fitted_["flags_kept"] = kept
        self.is_fitted = True
        return self

    def _signal(self, features):
        if not self.signs_:
            self.signs_ = {c: -1.0 for c in self.signals_used}
        parts = []
        for c in self.signals_used:
            s = self.signs_.get(c, 0.0)
            if s != 0.0:
                # centre each flag so "off" is -1 and "on" is +1; otherwise a flag
                # that is on 80% of the time gives a constant long bias.
                centred = 2.0 * features[c].fillna(0.5) - 1.0
                parts.append(s * centred)
        if not parts:
            return pd.Series(0.0, index=features.index)
        return pd.concat(parts, axis=1).mean(axis=1).clip(-1.0, 1.0)
