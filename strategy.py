"""
BaseStrategy -- the contract every alpha implements.

The backtester, execution engine and portfolio do not know about any specific
strategy. They only know that a strategy turns a decision frame into a position
series in [-1, +1]. So a new alpha can be added without changing engine code.

Two rules are checked here:

1. generate_signal only gets the decision frame, which has no price or volume.
   A strategy cannot look at price.
2. The output is clipped to the position bound and NaN becomes flat, so a
   broken signal gives "no position" instead of bad values in the portfolio.
"""

import numpy as np
import pandas as pd

import config


class BaseStrategy:
    """Abstract base for every alpha in strategies/."""

    name = "BaseStrategy"
    hypothesis = "Not stated."
    signals_used = []

    # The horizon of the hypothesis, in trading days. It is set on the class as
    # part of the idea, not tuned after seeing results. The forward-return label
    # used in fit() is built at this horizon. A signal about the next two weeks
    # should not be judged on tomorrow's return.
    fit_horizon = 1

    def __init__(self, **params):
        self.params = params
        self.fitted_ = {}
        self.is_fitted = False

    # ------------------------------------------------------------------
    # required interface
    # ------------------------------------------------------------------
    def generate_features(self, data):
        """Define strategy inputs.

        Default: return the decision frame as is. Strategies that need derived
        columns (rolling on-rates, composite scores) override this and build
        them from the signal columns only.
        """
        return data

    def generate_signal(self, data):
        """Generate trading decisions: a target position per row, in [-1, 1].

        Subclasses implement `_signal`. This wrapper runs the checks so each
        alpha does not have to.
        """
        self._assert_no_price_columns(data)
        features = self.generate_features(data)
        raw = self._signal(features)
        return self._sanitise(raw, index=data.index)

    def fit(self, data, target=None, daily_target=None):
        """Fit parameters/models where applicable.

        Called with development-window data only. Strategies with nothing to
        fit still set the flag, so the pipeline can check fit() ran before any
        holdout run.

        Two labels are passed in:

        * `target` is the forward return at the strategy's horizon, used to
          estimate which way each driver points.
        * `daily_target` is the one-day open-to-open return, which is what the
          strategy earns each candle. Any choice of holding period must be
          scored on this, net of cost, because hold length changes turnover. A
          choice made on a gross multi-day label would favour whichever setting
          trades most.
        """
        self.is_fitted = True
        return self

    def predict(self, data):
        """Produce strategy outputs -- here, the target position series."""
        return self.generate_signal(data)

    def get_metadata(self):
        return {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "signals_used": list(self.signals_used),
            "params": dict(self.params),
            "fitted": dict(self.fitted_),
        }

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _signal(self, features):
        raise NotImplementedError(f"{type(self).__name__} must implement _signal().")

    @staticmethod
    def _assert_no_price_columns(data):
        leaked = [c for c in config.FORBIDDEN_STRATEGY_INPUTS if c in data.columns]
        if leaked:
            raise ValueError(
                f"Strategy input frame contains forbidden columns {leaked}. "
                "Price and volume may not drive a trading decision."
            )

    @staticmethod
    def _sanitise(raw, index):
        s = pd.Series(np.asarray(raw, dtype=float), index=index)
        s = s.fillna(0.0).clip(-config.MAX_GROSS_POSITION, config.MAX_GROSS_POSITION)
        return s

    # ------------------------------------------------------------------
    # small helpers shared by the alphas
    # ------------------------------------------------------------------
    @staticmethod
    def _zscore(series, window, min_periods=None):
        """Trailing z-score. Backward-looking window only."""
        mp = min_periods or window
        mu = series.rolling(window, min_periods=mp).mean()
        sd = series.rolling(window, min_periods=mp).std()
        return (series - mu) / sd.replace(0.0, np.nan)

    @staticmethod
    def _rank_pct(series, window, min_periods=None):
        """Trailing percentile rank of the latest value within its own window."""
        mp = min_periods or window
        return series.rolling(window, min_periods=mp).apply(
            lambda a: (a[-1] > a[:-1]).mean() if len(a) > 1 else 0.5, raw=True
        )
