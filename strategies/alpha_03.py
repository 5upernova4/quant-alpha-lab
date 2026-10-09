"""
Alpha 03 -- Trend-State Fade.

Hypothesis
----------
The price-based state flags describe where the trend *has been*, not where it is
going. In an instrument whose short-horizon returns mean-revert, a cluster of
trend flags reading "up" is a description of a move that has already been paid
for. The hypothesis is that the *agreement* of those flags is itself a
contrarian variable: the more of them that point the same way, the more crowded
the position and the weaker the forward return.

This differs from Alpha 01 in what it reads. Alpha 01 measures *how far* price
has travelled from a reference, using continuous distance. Alpha 03 measures
*how many independent trend definitions currently agree*, using discrete state.
A market can be strongly stretched with only one flag on, or barely stretched
with every flag on -- the two are not the same statement, and they turn over at
very different speeds.

Signals used
------------
PB01, PB02 -- short and long-term trend-direction flags.
PB03, PB04 -- price above short and long-term average reference levels.
PB05 -- sign of recent short-horizon momentum.

Trading rule
------------
Build a consensus score as the signed average of the flags, each flipped to the
direction the development window supports, then hold a position proportional to
the consensus. Because state flags are persistent, positions are persistent and
turnover is low -- which is the practical reason to carry this alpha alongside
the faster ones.

Where it should fail
--------------------
A long, orderly trend where every flag stays on for months. The strategy is then
short the whole way up. This is the mirror image of Alpha 01's failure mode, and
whether the two fail *together* is exactly what the orthogonality analysis has
to answer.
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

    # The grid the robustness sweep explores. Declared on the class so the
    # sweep tests settings the hypothesis actually permits, rather than an
    # arbitrary range invented at report time.
    PARAM_GRID = {'min_abs_t': [0.5, 0.75, 1.0, 1.5, 2.0]}

    def __init__(self, min_abs_t=None, **params):
        # Defaults to the shared screening floor rather than carrying its own,
        # so that all six strategies are screened on identical terms. The
        # parameter stays exposed because the robustness sweep varies it.
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
                # centre each flag so that "off" is -1 and "on" is +1: a flag that
                # is on 80% of the time otherwise builds a permanent long bias.
                centred = 2.0 * features[c].fillna(0.5) - 1.0
                parts.append(s * centred)
        if not parts:
            return pd.Series(0.0, index=features.index)
        return pd.concat(parts, axis=1).mean(axis=1).clip(-1.0, 1.0)
