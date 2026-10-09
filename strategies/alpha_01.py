"""
Alpha 01 -- Trend-Stretch Reversion.

Hypothesis
----------
Price does not move away from its own trend reference indefinitely. When the
normalised gap between price and a medium-term reference becomes unusually wide,
the marginal buyer at that level is paying a premium for a move that has already
happened, and the gap closes. The trade is to lean against the stretch: short
when price is far above its reference, long when it is far below.

This is short-horizon overreaction, and it is the most widely documented effect
in liquid single-instrument daily data -- liquidity provision earns a premium
precisely because someone has to take the other side of a crowded move.

Signals used
------------
PB07 -- normalised distance of price from a medium-term trend reference.
PB08 -- normalised separation between two trend references (trend strength).

Both are continuous and bounded, which makes them suitable as a *sizing*
variable rather than a binary switch: a large stretch should produce a large
position, a small one a small position.

Trading rule
------------
Standardise each driver against its own trailing window, flip it to the sign the
development window says is profitable, average the two, and squash into a
position. Horizon is a few days -- the position decays naturally as the stretch
closes, with no explicit exit rule.

Where it should fail
--------------------
In a genuine sustained trend, "stretched" stays stretched and the strategy
fights it the whole way. That is the systematic failure mode and it is examined
directly in the robustness section.
"""

import numpy as np
import pandas as pd

from strategy import BaseStrategy
from strategies._fitting import direction_and_strength, squash, trailing_z


class Alpha01TrendStretchReversion(BaseStrategy):
    name = "Alpha01_TrendStretchReversion"
    hypothesis = (
        "Price stretched far from its medium-term trend reference reverts; "
        "fade the size of the stretch."
    )
    signals_used = ["PB07", "PB08"]
    fit_horizon = 10    # Stretch closes over days, not overnight; PB07/PB08 are medium-term references.

    # The grid the robustness sweep explores. Declared on the class so the
    # sweep tests settings the hypothesis actually permits, rather than an
    # arbitrary range invented at report time.
    PARAM_GRID = {'z_window': [30, 45, 60, 90, 120], 'scale': [0.8, 1.0, 1.2, 1.6]}

    def __init__(self, z_window=60, scale=1.2, **params):
        super().__init__(z_window=z_window, scale=scale, **params)
        self.z_window = z_window
        self.scale = scale
        self.signs_ = {}

    def generate_features(self, data):
        out = pd.DataFrame(index=data.index)
        for c in self.signals_used:
            out[c + "_z"] = trailing_z(data[c], self.z_window)
        return out

    def fit(self, data, target=None, daily_target=None):
        """Read the direction of each driver off the development window."""
        feats = self.generate_features(data)
        self.signs_ = {}
        if target is None:
            self.signs_ = {c: -1.0 for c in self.signals_used}   # prior: reversion
        else:
            for c in self.signals_used:
                sign, t, rho = direction_and_strength(feats[c + "_z"], target, horizon=self.fit_horizon)
                self.signs_[c] = sign
                self.fitted_[c] = {"sign": sign, "t_stat": round(t, 3), "spearman": round(rho, 4)}
        self.is_fitted = True
        return self

    def _signal(self, features):
        if not self.signs_:
            self.signs_ = {c: -1.0 for c in self.signals_used}
        parts = []
        for c in self.signals_used:
            s = self.signs_.get(c, 0.0)
            if s != 0.0:
                parts.append(s * features[c + "_z"])
        if not parts:
            return pd.Series(0.0, index=features.index)
        score = pd.concat(parts, axis=1).mean(axis=1)
        return pd.Series(squash(score.fillna(0.0), self.scale), index=features.index)
