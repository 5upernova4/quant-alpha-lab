"""
BaseStrategy -- the contract every alpha implements.

The point of this class is replaceability. The backtester, execution engine and
portfolio know nothing about any particular strategy; they know that a strategy
turns a decision frame into a position series in [-1, +1]. That is the whole
interface, and it is why adding alpha_07 later would not require touching a
single line of engine code.

Two rules are enforced here rather than left to good intentions:

1. generate_signal only ever receives the decision frame, which by construction
   contains no price or volume. A strategy physically cannot look at price.
2. Whatever a strategy returns is clipped to the configured position bound and
   NaN-filled to flat, so a half-finished idea degrades to "no position" instead
   of quietly injecting garbage into the portfolio.
"""

import numpy as np
import pandas as pd

import config


class BaseStrategy:
    """Abstract base for every alpha in strategies/."""

    name = "BaseStrategy"
    hypothesis = "Not stated."
    signals_used = []

    # The horizon the hypothesis is about, in trading days. It is declared on the
    # class -- part of the idea, not a knob turned after seeing results -- and it
    # is the horizon the forward-return label is built at when the strategy is
    # fitted. A reversion signal that is about the next fortnight cannot be
    # calibrated against tomorrow's return and then be said to have failed.
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

        Default: pass the decision frame through untouched. Strategies that need
        derived columns (rolling on-rates, composite scores) override this and
        build them from the signal columns only.
        """
        return data

    def generate_signal(self, data):
        """Generate trading decisions: a target position per row, in [-1, 1].

        Subclasses implement `_signal`. This wrapper does the guarding so that
        no individual alpha can forget it.
        """
        self._assert_no_price_columns(data)
        features = self.generate_features(data)
        raw = self._signal(features)
        return self._sanitise(raw, index=data.index)

    def fit(self, data, target=None, daily_target=None):
        """Fit parameters/models where applicable.

        Called with development-window data only. Strategies with no fitted
        component leave this as a no-op but still flip the flag, so the pipeline
        can assert that fit() was called before any holdout run.

        Two labels are supplied, and they do different jobs:

        * `target` is the forward return at the strategy's declared horizon, used
          to estimate which way each driver points.
        * `daily_target` is the one-day open-to-open return -- what the strategy
          actually earns each candle. Any choice about *how long to hold* has to
          be scored against this one and net of trading cost, because a longer
          hold changes turnover, and a selection made on a gross multi-day label
          would systematically prefer whichever setting trades most.
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
