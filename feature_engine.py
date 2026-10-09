"""
FeatureEngine -- builds the decision frame and owns the information cutoff.

This is the single most important module for research integrity. Everything a
strategy is allowed to see passes through here, and it passes through exactly
one function, so there is one place to audit rather than six.

The contract it enforces:

    the decision for candle t may use signal-library values observed
    up to and including candle t-1's close, and nothing else.

Two separate things are handled, and it is worth keeping them apart:

1. Calendar alignment. Signals exist on some dates that have no price row. For
   each tradeable date we take the most recent signal observation strictly
   before it, which is what a desk would actually have on screen at the open.
2. The forward-return target. Price is used here -- and only here -- to build
   the supervised-learning label and to drive execution. It never becomes a
   feature. The problem statement is explicit about that split, so the module
   enforces it rather than trusting us to remember.
"""

import numpy as np
import pandas as pd

import config


class LookAheadError(AssertionError):
    """Raised when a frame handed to a strategy could contain future information."""


class FeatureEngine:
    """Constructs the lagged signal panel, derived features and return targets."""

    def __init__(self):
        self.registered = {}      # name -> the callable, so transform() can re-apply it
        self.descriptions = {}    # name -> human-readable description for the report
        self.audit = {}

    # ------------------------------------------------------------------
    # required interface
    # ------------------------------------------------------------------
    def add_feature(self, data, name, function):
        """Register a derived feature and attach it to `data`.

        The function is stored so that get_report / the research notebook can
        state exactly how each engineered column was produced.
        """
        out = data.copy()
        out[name] = function(out)
        self.registered[name] = function
        self.descriptions[name] = getattr(function, "__doc__", None) or repr(function)
        return out

    def transform(self, data):
        """Apply the configured transformations.

        Only two derived families are used across the strategy set, and both are
        causal by construction: rolling means of a signal (its recent 'on' rate)
        and rolling z-scores of a continuous signal. Both are computed with
        trailing windows on an already-lagged panel, so they cannot reach
        forward.
        """
        out = data.copy()
        for name, fn in self.registered.items():
            if name not in out.columns:
                out[name] = fn(out)
        return out

    def compute_returns(self, prices):
        """Construct the return series used for accounting and as the ML label.

        Open-to-open is the correct convention here and the choice matters. The
        mandated rule is that a decision made after candle t-1 executes at
        candle t's open. A position entered at open[t] and held one candle is
        exited at open[t+1], so open-to-open is the return that is actually
        capturable. Using close-to-close instead would credit the strategy with
        the overnight gap between close[t] and open[t+1], which it never had the
        chance to trade.
        """
        out = prices.copy()
        out["ret_oo"] = out["open"].shift(-1) / out["open"] - 1.0   # realised over [open_t, open_t+1]
        out["ret_cc"] = out["close"].pct_change()                    # diagnostic only
        out["ret_log"] = np.log(out["close"]).diff()                 # diagnostic only
        return out

    def rolling_feature(self, data, window, function, column="close"):
        """Construct a trailing rolling feature. Window is backward-looking."""
        return data[column].rolling(window, min_periods=window).apply(function, raw=True)

    def validate_no_lookahead(self, decision_frame, raw_signals, sample=250):
        """Check information availability -- empirically, not by assertion alone.

        The test: pick rows at random, and for each one confirm that the value
        sitting in the decision frame for date t equals the raw signal value on
        some date strictly earlier than t. If any row matched date t itself, the
        lag has been lost somewhere and the whole backtest is invalid.
        """
        rng = np.random.default_rng(config.RANDOM_SEED)
        raw = raw_signals.set_index("date")
        dates = decision_frame["date"].to_numpy()
        idx = rng.choice(len(dates), size=min(sample, len(dates)), replace=False)

        checked = 0
        for i in sorted(idx):
            t = decision_frame["date"].iloc[i]
            src = decision_frame["information_asof"].iloc[i]
            if pd.isna(src):
                continue
            if src >= t:
                raise LookAheadError(
                    f"Decision row for {t.date()} is sourced from signal date "
                    f"{pd.Timestamp(src).date()}, which is not strictly earlier."
                )
            for col in config.ALL_SIGNALS:
                a = decision_frame[col].iloc[i]
                b = raw.at[src, col] if src in raw.index else np.nan
                if pd.isna(a) and pd.isna(b):
                    continue
                if not np.isclose(float(a), float(b), equal_nan=True):
                    raise LookAheadError(
                        f"{col} on {t.date()} does not match its claimed source row "
                        f"{pd.Timestamp(src).date()} ({a} vs {b})."
                    )
            checked += 1

        leaked = [c for c in config.FORBIDDEN_STRATEGY_INPUTS if c in decision_frame.columns]
        if leaked:
            raise LookAheadError(
                f"Decision frame exposes price/volume columns {leaked}; the problem "
                "statement forbids them as strategy inputs."
            )

        self.audit["lookahead_rows_checked"] = checked
        self.audit["lookahead_status"] = "PASS"
        return True

    # ------------------------------------------------------------------
    # the main construction
    # ------------------------------------------------------------------
    def build_decision_frame(self, prices, signals):
        """Return the frame a strategy is allowed to see, plus the market frame.

        Returns (decision_frame, market_frame):

        * decision_frame -- one row per tradeable date. Contains only the 20
          signal columns, the date, and `information_asof` (the signal date the
          row was actually sourced from, kept for auditing). No prices.
        * market_frame -- date, OHLCV and the return series. Used by the
          execution engine, the portfolio and the research label. Never given
          to a strategy.
        """
        prices = prices.sort_values("date").reset_index(drop=True)
        signals = signals.sort_values("date").reset_index(drop=True)

        sig = signals.copy()
        sig["information_asof"] = sig["date"]

        # merge_asof with allow_exact_matches=False is the whole ballgame: for
        # tradeable date t it takes the latest signal row with date STRICTLY
        # less than t. That is the t-1 cutoff, expressed once.
        decision = pd.merge_asof(
            prices[["date"]],
            sig,
            on="date",
            direction="backward",
            allow_exact_matches=False,
        )

        keep = ["date", "information_asof"] + config.ALL_SIGNALS
        decision = decision[keep]

        market = self.compute_returns(prices)

        self.audit["tradeable_dates"] = len(decision)
        self.audit["rows_without_prior_signal"] = int(decision["information_asof"].isna().sum())
        self.audit["mean_information_age_days"] = float(
            (decision["date"] - decision["information_asof"]).dt.days.mean()
        )
        return decision, market

    def get_report(self):
        lines = ["FEATURE / INFORMATION AUDIT", "=" * 62]
        for k, v in self.audit.items():
            lines.append(f"  {k:32s}: {v}")
        if self.registered:
            lines.append("  registered derived features: " + ", ".join(self.registered))
        return "\n".join(lines)
