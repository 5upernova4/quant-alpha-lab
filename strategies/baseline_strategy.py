"""
BaselineStrategy -- a simple rule to compare the alphas against.

Hypothesis: the short-term trend-direction flag (PB01) means what it says, so
being long while it is on and flat while it is off should capture the trend.

This is the simple reading of a trend flag and it is not tuned. It completes
the Task 1 pipeline and is the reference for Task 2: if the alphas cannot beat
a one-line rule after costs, they are not worth the extra complexity.
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
