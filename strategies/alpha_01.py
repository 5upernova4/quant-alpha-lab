"""
Alpha 01 -- Trend-Stretch Reversion.

Hypothesis
----------
Price does not stay far from its trend reference forever. When the normalised
gap between price and a medium-term reference gets unusually wide, buyers at
that level are paying for a move that already happened, and the gap closes.
The trade goes against the stretch: short when price is far above its
reference, long when far below.

This is short-term overreaction, a well-documented effect in liquid daily
data. Providing liquidity earns a premium because someone has to take the
other side of a crowded move.

Signals used
------------
PB07 -- normalised distance of price from a medium-term trend reference.
PB08 -- normalised separation between two trend references (trend strength).

Both are continuous and bounded, so they can size the position instead of
acting as an on/off switch: a large stretch gives a large position, a small
one a small position.

Trading rule
------------
Z-score each driver over its own trailing window, set its sign to the one
the development window found profitable, average the two, and squash into a
position. Horizon is a few days. The position shrinks as the stretch closes,
with no separate exit rule.

Where it should fail
--------------------
In a real sustained trend, the stretch stays wide and the strategy fights it
the whole way. This is the main failure mode and is checked in the robustness
section.
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

    # The grid used by the robustness sweep. Set on the class so the sweep
    # tests settings the hypothesis allows, not a range picked at report time.
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
