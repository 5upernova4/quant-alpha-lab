"""
Alpha 05 -- Volume-Flow Continuation.

Hypothesis
----------
This is the one strategy in the set that is not contrarian, and it is here on
purpose.

Participation carries information that price alone does not. A move on heavy,
rising volume is more likely to be an institution working an order over several
days than a one-off liquidity air pocket; a move on thin volume is more likely
to be noise that gets retraced. The hypothesis is that volume-confirmed flow
*continues* at a very short horizon, because large orders take days to complete.

Economically this is the opposite side of the trade from Alphas 01-04. Those
earn a premium for providing liquidity to over-extended moves. This one earns by
stepping aside from, or joining, the moves that have real size behind them. If
the hypothesis holds, the strategy should make money in exactly the periods
where the reversion book struggles -- which is the entire reason for carrying
it, and the claim the orthogonality analysis has to adjudicate.

Signals used
------------
VB01 -- above-average trading participation.
VB02 -- rising trend in cumulative volume flow.
VB03 -- a price move unusually well supported by volume.
VB04 -- a secondary participation flag.
VB05 -- normalised measure of how unusual current participation is.

Trading rule
------------
Weight each volume signal by the direction and confidence the development window
supports, and hold the weighted consensus. Weak drivers get a zero weight rather
than a small one.

Where it should fail
--------------------
Volume spikes that mark capitulation rather than accumulation -- the end of a
move rather than the middle of one. Those cluster at turning points, so the
failures should be few, large and concentrated.
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

    # The grid the robustness sweep explores. Declared on the class so the
    # sweep tests settings the hypothesis actually permits, rather than an
    # arbitrary range invented at report time.
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
