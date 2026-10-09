"""
BaselineStrategy -- the obvious rule, included so there is something to beat.

Hypothesis: the short-term trend-direction flag (PB01) means what it says, so
being long while it is on and flat while it is off should capture the trend.

This is the naive reading of a trend flag and it is deliberately not tuned. It
exists to complete the Task 1 pipeline end-to-end and to act as the reference
point for Task 2: if the researched alphas cannot beat a one-line rule after
costs, they are not worth the complexity.
"""

import numpy as np
import pandas as pd

from strategy import BaseStrategy


class BaselineStrategy(BaseStrategy):
    name = "Baseline_PB01_Long_Only"
    hypothesis = "PB01 flags an up-trend; hold the asset while it is on."
    signals_used = ["PB01"]

    def _signal(self, features):
        if "PB01" not in features.columns:
            return pd.Series(0.0, index=features.index)
        return features["PB01"].fillna(0.0).clip(0.0, 1.0)
