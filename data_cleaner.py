"""
DataCleaner -- turns the raw files into a research-ready panel.

The supplied price file is not clean, and the defects are not accidents: the
rows are out of chronological order, a handful of dates are written in a
different format from the rest, some rows appear twice, and some trading dates
present in the signal file have no price row at all. Each of those breaks a
backtest in a different way, so each is handled explicitly and counted in the
data-quality report rather than being silently absorbed by a generic dropna().
"""

import numpy as np
import pandas as pd

import config


class DataCleaner:
    """Validates, repairs and audits the raw inputs."""

    # The supplied file mixes ISO dates with day-first dates. Parsing with a
    # single format silently produces NaT; parsing with dayfirst inference
    # silently mis-reads 2018-01-02 as 2 January vs 1 February depending on the
    # row. We therefore try the formats we have actually observed, in order.
    DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"]

    def __init__(self):
        self.report_lines = []
        self.diagnostics = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def validate(self, data, kind="price"):
        """Return a dict of data-quality findings. Never mutates `data`."""
        findings = {
            "rows": len(data),
            "duplicate_full_rows": int(data.duplicated().sum()),
            "duplicate_dates": int(data["date"].duplicated().sum()),
            "null_cells": int(data.isnull().sum().sum()),
            "null_by_column": {
                k: int(v) for k, v in data.isnull().sum().items() if v > 0
            },
        }
        if kind == "price":
            px = data[config.OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")
            findings["non_positive_prices"] = int(
                (px[["open", "high", "low", "close"]] <= 0).sum().sum()
            )
            findings["non_positive_volume"] = int((px["volume"] <= 0).sum())
            findings["high_below_low"] = int((px["high"] < px["low"]).sum())
            findings["high_below_open_or_close"] = int(
                (px["high"] < px[["open", "close"]].max(axis=1)).sum()
            )
            findings["low_above_open_or_close"] = int(
                (px["low"] > px[["open", "close"]].min(axis=1)).sum()
            )
        return findings

    def clean(self, prices, signals):
        """Return (clean_prices, clean_signals), both chronological and unique.

        Order matters. Dates are normalised first, because everything else --
        sorting, duplicate detection, alignment -- is meaningless while the same
        calendar day can be spelled two different ways.
        """
        self.report_lines = []
        prices = self._parse_dates(prices, "price")
        signals = self._parse_dates(signals, "signal")

        # Profiled AFTER the dates are normalised. Counting duplicates on the raw
        # strings undercounts them, because the same calendar day written two
        # different ways does not compare equal -- and the report would then
        # disagree with the repair line immediately below it.
        self.diagnostics = {
            "price_raw": self.validate(prices, "price"),
            "signal_raw": self.validate(signals, "signal"),
        }

        prices = self._coerce_numeric(prices, config.OHLCV_COLUMNS)
        signals = self._coerce_numeric(signals, config.ALL_SIGNALS)

        prices = self.remove_duplicates(prices, "price")
        signals = self.remove_duplicates(signals, "signal")

        prices = self.sort_chronologically(prices)
        signals = self.sort_chronologically(signals)

        prices = self.handle_missing(prices, "price")
        signals = self.handle_missing(signals, "signal")

        self._audit_calendars(prices, signals)
        self.diagnostics["price_clean"] = self.validate(prices, "price")
        self.diagnostics["signal_clean"] = self.validate(signals, "signal")
        return prices, signals

    # ------------------------------------------------------------------
    # individual repairs -- each one is also a required interface method
    # ------------------------------------------------------------------
    def sort_chronologically(self, data):
        """Enforce time order.

        The raw price file arrives shuffled. Every downstream calculation --
        returns, rolling windows, the t-1 information cutoff -- assumes the row
        order *is* the time order, so this has to happen before anything else
        touches the frame.
        """
        before = bool(data["date"].is_monotonic_increasing)
        out = data.sort_values("date", kind="mergesort").reset_index(drop=True)
        if not before:
            self.report_lines.append(
                "Re-sorted rows into chronological order (raw file was shuffled)."
            )
        return out

    def handle_missing(self, data, kind="price"):
        """Handle missing observations.

        Two different situations, handled two different ways:

        * Leading NaNs in the continuous signals (BB06, BB07, VB05) are warm-up
          periods -- the indicator simply does not exist yet. Filling them
          backwards would invent information that did not exist on the day, so
          they are left as NaN and every strategy treats NaN as "no opinion".
        * Interior NaNs, if any appear, are forward-filled only. Forward fill
          uses the last value that was genuinely known at the time; a backward
          fill would copy tomorrow's value into today and is look-ahead.
        """
        cols = config.OHLCV_COLUMNS if kind == "price" else config.ALL_SIGNALS
        cols = [c for c in cols if c in data.columns]
        out = data.copy()

        leading = {}
        interior_filled = 0
        for c in cols:
            isna = out[c].isna()
            if not isna.any():
                continue
            first_valid = out[c].first_valid_index()
            if first_valid is None:
                self.report_lines.append(f"Column {c} is entirely missing; left as NaN.")
                continue
            n_lead = int(isna.loc[: first_valid - 1].sum()) if first_valid > 0 else 0
            if n_lead:
                leading[c] = n_lead
            tail = out[c].loc[first_valid:]
            n_int = int(tail.isna().sum())
            if n_int:
                out.loc[first_valid:, c] = tail.ffill()
                interior_filled += n_int

        if leading:
            self.report_lines.append(
                "Left warm-up NaNs in place (no backward fill) for: "
                + ", ".join(f"{k} x{v}" for k, v in leading.items())
            )
        if interior_filled:
            self.report_lines.append(
                f"Forward-filled {interior_filled} interior missing values (no backward fill)."
            )
        if kind == "price":
            # A price row we cannot trade on is worse than a missing row: it
            # would produce a fabricated fill. Drop it and say so.
            bad = out[config.OHLCV_COLUMNS[:4]].isna().any(axis=1)
            if bad.any():
                self.report_lines.append(f"Dropped {int(bad.sum())} price rows with unusable OHLC.")
                out = out.loc[~bad].reset_index(drop=True)
        return out

    def remove_duplicates(self, data, kind="price"):
        """Handle duplicate observations.

        The price file contains exact repeat rows for a number of dates. Because
        they are byte-identical we can drop them without choosing a winner; if
        they had disagreed we would have had to state a rule, so we check that
        first and shout if the assumption ever stops holding.
        """
        out = data.copy()
        dup_dates = out["date"].duplicated(keep=False)
        n_conflicting = 0
        if dup_dates.any():
            value_cols = [c for c in out.columns if c != "date"]
            grouped = out.loc[dup_dates].groupby("date")[value_cols].nunique()
            n_conflicting = int((grouped > 1).any(axis=1).sum())

        n_before = len(out)
        out = out.drop_duplicates(subset="date", keep="first").reset_index(drop=True)
        removed = n_before - len(out)
        if removed:
            note = f"Removed {removed} duplicate {kind} rows (one per repeated date)."
            if n_conflicting:
                note += (
                    f" WARNING: {n_conflicting} of those dates had conflicting values; "
                    "kept the first occurrence."
                )
            self.report_lines.append(note)
        return out

    def get_report(self):
        """Return data-quality diagnostics as printable text."""
        lines = ["DATA QUALITY REPORT", "=" * 62]
        raw_p = self.diagnostics.get("price_raw", {})
        raw_s = self.diagnostics.get("signal_raw", {})
        lines.append(
            f"Raw price file  : {raw_p.get('rows', 0)} rows, "
            f"{raw_p.get('duplicate_dates', 0)} duplicate dates, "
            f"{raw_p.get('null_cells', 0)} null cells"
        )
        lines.append(
            f"Raw signal file : {raw_s.get('rows', 0)} rows, "
            f"{raw_s.get('duplicate_dates', 0)} duplicate dates, "
            f"{raw_s.get('null_cells', 0)} null cells"
        )
        lines.append("-" * 62)
        for line in self.report_lines:
            lines.append("  - " + line)
        lines.append("-" * 62)
        cp = self.diagnostics.get("price_clean", {})
        lines.append(
            f"Clean price panel: {cp.get('rows', 0)} tradeable dates, "
            f"OHLC consistency violations: {cp.get('high_below_low', 0)}"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _parse_dates(self, data, kind):
        """Normalise a mixed-format date column without guessing."""
        out = data.copy()
        raw = out["date"].astype(str).str.strip()
        parsed = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
        used = {}
        for fmt in self.DATE_FORMATS:
            todo = parsed.isna()
            if not todo.any():
                break
            attempt = pd.to_datetime(raw[todo], format=fmt, errors="coerce")
            n = int(attempt.notna().sum())
            if n:
                used[fmt] = n
                parsed.loc[todo] = attempt

        if parsed.isna().any():
            bad = raw[parsed.isna()].unique()[:5]
            raise ValueError(
                f"Unparseable dates in the {kind} file, e.g. {list(bad)}. "
                f"Add the format to DataCleaner.DATE_FORMATS."
            )
        if len(used) > 1:
            detail = ", ".join(f"{k} x{v}" for k, v in used.items())
            self.report_lines.append(
                f"Normalised mixed date formats in the {kind} file ({detail})."
            )
        out["date"] = parsed
        return out

    @staticmethod
    def _coerce_numeric(data, columns):
        out = data.copy()
        for c in columns:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    def _audit_calendars(self, prices, signals):
        """Compare the two calendars and record the mismatch.

        Some dates carry a signal row but no price row. Those days are not
        tradeable -- there is no open to fill against -- so the tradeable
        calendar is the price calendar. The signals on those days are not
        thrown away: they stay in the information set and are picked up by the
        as-of alignment in FeatureEngine as the last observation before the
        next tradeable day.
        """
        p_dates, s_dates = set(prices["date"]), set(signals["date"])
        only_signal = sorted(s_dates - p_dates)
        only_price = sorted(p_dates - s_dates)
        self.diagnostics["dates_signal_without_price"] = len(only_signal)
        self.diagnostics["dates_price_without_signal"] = len(only_price)
        if only_signal:
            self.report_lines.append(
                f"{len(only_signal)} dates carry signals but no price row; they are not "
                "tradeable, so they are excluded from the trading calendar but retained "
                "in the information set."
            )
        if only_price:
            self.report_lines.append(
                f"{len(only_price)} dates carry a price but no signal row; the most recent "
                "prior signal observation is carried into those decisions."
            )
        gaps = prices["date"].diff().dt.days.dropna()
        self.diagnostics["max_calendar_gap_days"] = int(gaps.max()) if len(gaps) else 0
