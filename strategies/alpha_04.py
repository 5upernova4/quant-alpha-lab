"""
Alpha 04 -- Band Position Reversion.

Hypothesis
----------
A volatility band shows what a normal move is under current conditions. So
where price sits in the band is a volatility-adjusted measure of extension.
Unlike a raw distance, it tightens in calm markets and widens in volatile
ones. The hypothesis is that position in the band reverts, and that touches of
the band edges are more often exhaustion than continuation.

Why this is not the same as Alpha 01: Alpha 01 measures stretch against a
trend reference, scaled by price only. Alpha 04 measures it against a
volatility band, so the same 1% move gives a large reading in a quiet market
and a small one in a volatile market. The two disagree most when volatility is
changing.

Signals used
------------
BB06 -- normalised position of price within its recent volatility band.
BB01 -- breakout above the upper band.
BB02 -- breakout below the lower band.

Trading rule
------------
Core position from band location, with the sign from the development window.
Band breakouts add an on/off overlay in the direction the development window
supports, so the strategy does not assume breakouts fail.

Where it should fail
--------------------
A real volatility expansion, where price rides the upper band for weeks. The
band widens but position in the band stays high, and the strategy is short the
whole time.
"""

import numpy as np
import pandas as pd

from strategy import BaseStrategy
from strategies._fitting import direction_and_strength, event_response, squash


class Alpha04BandPositionReversion(BaseStrategy):
    name = "Alpha04_BandPositionReversion"
    hypothesis = (
        "Volatility-adjusted position within the band reverts; band-edge "
        "breakouts are more often exhaustion than continuation."
    )
    signals_used = ["BB06", "BB01", "BB02"]
    fit_horizon = 10    # Band position normalises over a similar span to the band window itself.

    # The grid used by the robustness sweep. Set on the class so the sweep
    # tests settings the hypothesis allows, not a range picked at report time.
    PARAM_GRID = {'scale': [0.5, 0.8, 1.2], 'breakout_weight': [0.0, 0.2, 0.4, 0.6]}

    def __init__(self, scale=0.8, breakout_weight=0.4, **params):
        super().__init__(scale=scale, breakout_weight=breakout_weight, **params)
        self.scale = scale
        self.breakout_weight = breakout_weight
        self.signs_ = {}

    def fit(self, data, target=None, daily_target=None):
        self.signs_ = {}
        if target is None:
            self.signs_ = {"BB06": -1.0, "BB01": -1.0, "BB02": 1.0}
            self.is_fitted = True
            return self

        sign, t, rho = direction_and_strength(data["BB06"], target, horizon=self.fit_horizon)
        self.signs_["BB06"] = sign
        self.fitted_["BB06"] = {"sign": sign, "t_stat": round(t, 3), "spearman": round(rho, 4)}

        for c in ("BB01", "BB02"):
            s, t, spread = event_response(data[c], target, horizon=self.fit_horizon)
            self.signs_[c] = s
            self.fitted_[c] = {"sign": s, "t_stat": round(t, 3), "spread_bps": round(spread * 1e4, 2)}
        self.is_fitted = True
        return self

    def _signal(self, features):
        if not self.signs_:
            self.signs_ = {"BB06": -1.0, "BB01": -1.0, "BB02": 1.0}

        core = self.signs_.get("BB06", 0.0) * features["BB06"].fillna(0.0)
        score = pd.Series(squash(core, self.scale), index=features.index)

        overlay = pd.Series(0.0, index=features.index)
        for c in ("BB01", "BB02"):
            s = self.signs_.get(c, 0.0)
            if s != 0.0:
                overlay = overlay + s * features[c].fillna(0.0)
        return (score + self.breakout_weight * overlay).clip(-1.0, 1.0)
