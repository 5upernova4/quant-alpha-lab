"""
Alpha 06 -- Volatility-Regime Conditioned Reversion.

Hypothesis
----------
Mean reversion is not constant. It is payment for providing liquidity, and
liquidity is scarcest (so the payment is largest) when volatility is high and
when a quiet period has just ended. This hypothesis is not about direction; it
says the volatility state tells you how much of the reversion trade to do.

So this strategy is different from the rest. Alphas 01, 03 and 04 answer
"which way?". This one uses a simple reversion core and puts all its signal
into "how much?". That is a separate question, and it can help a portfolio
even when the direction call is shared.

Signals used
------------
BB07 -- normalised recent volatility level (the conditioning variable).
BB05 -- a low-volatility squeeze (its release is the second conditioner).
BB06 -- position within the band, used only as the plain directional core.

Trading rule
------------
Direction comes from the band-position core. Size comes from a multiplier
based on the volatility state: high volatility and a just-released squeeze
both scale the position up; calm conditions scale it down. The multiplier is
capped so leverage stays limited.

Where it should fail
--------------------
A regime where high volatility comes with a strong trend, not choppy moves:
a crash. The multiplier then sizes the strategy up just when the core is most
wrong. This is the most dangerous failure mode in the set and is checked
against the March 2020 window.
"""

import numpy as np
import pandas as pd

from strategy import BaseStrategy
from strategies._fitting import direction_and_strength, squash, trailing_z


class Alpha06VolatilityRegimeReversion(BaseStrategy):
    name = "Alpha06_VolatilityRegimeReversion"
    hypothesis = (
        "Reversion pays most when volatility is elevated or a squeeze has just "
        "released; use the volatility state to size a plain reversion core."
    )
    signals_used = ["BB07", "BB05", "BB06"]
    fit_horizon = 10    # The band-position core reverts over about a fortnight; the
                        # volatility conditioning is measured on the same window.

    # The grid used by the robustness sweep. Set on the class so the sweep
    # tests settings the hypothesis allows, not a range picked at report time.
    PARAM_GRID = {'z_window': [30, 60, 90], 'max_mult': [1.2, 1.6, 2.0]}

    def __init__(self, z_window=60, min_mult=0.3, max_mult=1.6, **params):
        super().__init__(z_window=z_window, min_mult=min_mult, max_mult=max_mult, **params)
        self.z_window = z_window
        self.min_mult = min_mult
        self.max_mult = max_mult
        self.core_sign_ = 0.0
        self.vol_sign_ = 0.0

    def generate_features(self, data):
        out = pd.DataFrame(index=data.index)
        out["core"] = data["BB06"]
        out["vol_z"] = trailing_z(data["BB07"], self.z_window)
        # The signal is a squeeze that has just ended, not the squeeze itself:
        # BB05 on yesterday and off today is a release.
        sq = data["BB05"].fillna(0.0)
        out["squeeze_release"] = ((sq.shift(1) > 0.5) & (sq <= 0.5)).astype(float)
        return out

    def fit(self, data, target=None, daily_target=None):
        feats = self.generate_features(data)
        if target is None:
            self.is_fitted = True
            return self

        s, t, rho = direction_and_strength(feats["core"], target, horizon=self.fit_horizon)
        # If the fit cannot find a direction, hold no position. Do not fall
        # back on the sign we expected; that would let the prior override the
        # data.
        self.core_sign_ = s
        self.fitted_["BB06_core"] = {"sign": self.core_sign_, "t_stat": round(t, 3),
                                     "spearman": round(rho, 4)}

        # Does high volatility make the core more profitable? Test the
        # interaction instead of assuming the sign.
        core = self.core_sign_ * feats["core"]
        y = pd.Series(np.asarray(target, dtype=float), index=data.index)
        base = (core * y)
        interaction = (core * feats["vol_z"].fillna(0.0) * y)
        ok = base.notna() & interaction.notna()
        if ok.sum() > 60 and interaction[ok].std(ddof=1) > 0:
            t_int = float(interaction[ok].mean() / (interaction[ok].std(ddof=1) / np.sqrt(ok.sum())))
            self.vol_sign_ = float(np.sign(t_int)) if abs(t_int) >= 1.0 else 0.0
            self.fitted_["vol_interaction"] = {"sign": self.vol_sign_, "t_stat": round(t_int, 3)}
        self.is_fitted = True
        return self

    def _signal(self, features):
        if self.core_sign_ == 0.0:
            return pd.Series(0.0, index=features.index)
        core = pd.Series(
            squash(self.core_sign_ * features["core"].fillna(0.0), 0.8), index=features.index
        )
        vz = features["vol_z"].fillna(0.0)
        mult = 1.0 + self.vol_sign_ * 0.35 * vz + 0.25 * features["squeeze_release"]
        mult = mult.clip(self.min_mult, self.max_mult)
        return (core * mult).clip(-1.0, 1.0)
