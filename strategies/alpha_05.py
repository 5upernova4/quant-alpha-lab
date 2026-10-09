"""
Alpha 05 -- Volume-Flow Continuation.

Hypothesis
----------
This is the only strategy in the set that is not contrarian, by design.

Volume carries information that price alone does not. A move on heavy, rising
volume is more likely an institution filling an order over several days than a
one-off gap in liquidity; a move on thin volume is more likely noise that gets
reversed. The hypothesis is that volume-backed flow continues over a very short
horizon, because large orders take days to fill.

This is the other side of the trade from Alphas 01-04. Those earn a premium
for providing liquidity to over-extended moves. This one earns by stepping
aside from, or joining, moves with real size behind them. If the hypothesis
holds, it should make money when the reversion strategies struggle. That is
why it is in the set, and the orthogonality analysis tests this.

Signals used
------------
VB01 -- above-average trading participation.
VB02 -- rising trend in cumulative volume flow.
VB03 -- a price move unusually well supported by volume.
VB04 -- a secondary participation flag.
VB05 -- normalised measure of how unusual current participation is.

Trading rule
------------
Weight each volume signal by the direction and strength found on the
development window, and hold the weighted average. Weak drivers get zero
weight, not a small one.

Where it should fail
--------------------
Volume spikes that mark capitulation, not accumulation: the end of a move,
not the middle. These cluster at turning points, so losses should be few,
large and bunched together.
"""

import numpy as np
import pandas as pd

from strategy import BaseStrategy
from strategies._fitting import direction_and_strength, event_response, squash, trailing_z


class Alpha05VolumeFlowContinuation(BaseStrategy):
    name = "Alpha05_VolumeFlowContinuation"
    hypothesis = (
        "Moves backed by unusual participation reflect orders still being worked "
        "and continue at short horizon."
    )
    signals_used = ["VB01", "VB02", "VB03", "VB04", "VB05"]
    fit_horizon = 1    # Order-working is a next-day effect; if it needs a fortnight it is not flow.

    # The grid used by the robustness sweep. Set on the class so the sweep
    # tests settings the hypothesis allows, not a range picked at report time.
    PARAM_GRID = {'z_window': [30, 60, 90], 'scale': [0.7, 1.0, 1.5]}
    BOOLEAN_SIGNALS = ["VB01", "VB02", "VB03", "VB04"]
    CONTINUOUS_SIGNALS = ["VB05"]

    def __init__(self, z_window=60, scale=1.0, **params):
        super().__init__(z_window=z_window, scale=scale, **params)
        self.z_window = z_window
        self.scale = scale
        self.signs_ = {}
        self.weights_ = {}

    def generate_features(self, data):
        out = data[self.signals_used].copy()
        for c in self.CONTINUOUS_SIGNALS:
            out[c + "_z"] = trailing_z(data[c], self.z_window)
        return out

    def fit(self, data, target=None, daily_target=None):
        feats = self.generate_features(data)
        self.signs_, self.weights_ = {}, {}
        if target is None:
            self.signs_ = {c: 1.0 for c in self.signals_used}
            self.weights_ = {c: 1.0 for c in self.signals_used}
            self.is_fitted = True
            return self

        for c in self.BOOLEAN_SIGNALS:
            s, t, spread = event_response(feats[c], target, horizon=self.fit_horizon)
            self.signs_[c] = s
            self.weights_[c] = min(abs(t), 3.0)
            self.fitted_[c] = {"sign": s, "t_stat": round(t, 3), "spread_bps": round(spread * 1e4, 2)}
        for c in self.CONTINUOUS_SIGNALS:
            s, t, rho = direction_and_strength(feats[c + "_z"], target, horizon=self.fit_horizon)
            self.signs_[c] = s
            self.weights_[c] = min(abs(t), 3.0)
            self.fitted_[c] = {"sign": s, "t_stat": round(t, 3), "spearman": round(rho, 4)}
        self.is_fitted = True
        return self

    def _signal(self, features):
        if not self.signs_:
            self.signs_ = {c: 1.0 for c in self.signals_used}
            self.weights_ = {c: 1.0 for c in self.signals_used}

        num = pd.Series(0.0, index=features.index)
        den = 0.0
        for c in self.BOOLEAN_SIGNALS:
            s, w = self.signs_.get(c, 0.0), self.weights_.get(c, 0.0)
            if s != 0.0 and w > 0:
                num = num + s * w * (2.0 * features[c].fillna(0.5) - 1.0)
                den += w
        for c in self.CONTINUOUS_SIGNALS:
            s, w = self.signs_.get(c, 0.0), self.weights_.get(c, 0.0)
            if s != 0.0 and w > 0:
                num = num + s * w * pd.Series(
                    squash(features[c + "_z"].fillna(0.0), 1.5), index=features.index
                )
                den += w
        if den == 0:
            return pd.Series(0.0, index=features.index)
        return (num / den).clip(-1.0, 1.0)
