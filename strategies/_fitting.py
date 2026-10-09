"""
Shared fitting helpers for the alpha strategies.

Every alpha needs the same step: given a driver built from the signal library
and a forward-return label, find which way the relationship goes and how
strong it is. Writing this once means the six strategies differ in their
hypothesis, not in their statistics code, which is what Task 2 asks for.

Two rules:

* The direction is estimated, never hard-coded. Nothing here says "PB01 is
  contrarian"; the fit reads it from the development window.
* A weak estimate gets zero weight, not a small one. A t-statistic below the
  floor means we cannot tell the relationship from noise, so we do not trade
  it at all.
"""

import numpy as np
import pandas as pd

# This is a screening floor, not a significance threshold, because the
# effective sample is small. A 10-day forward return measured daily over a
# 770-day window has about 77 independent observations, and the t-stats below
# are already divided by sqrt(horizon) for this. Requiring |t| >= 2 on 77
# points would drop drivers whose sign is stable in every sub-period and makes
# economic sense, just because the window is short. That is worse than letting
# a weak one through, since the strategy-level gates catch weak ones anyway.
#
# The real significance test comes later: the block permutation test and the
# Newey-West t on the backtest's 770 daily returns, in
# AlphaResearch.SELECTION_RULE. Nothing is selected on this screen alone.
MIN_ABS_T = 0.5     # overlap-corrected; a screen, not a significance claim
MIN_OBS = 60        # below this there is not enough development history to fit


def overlap_correction(horizon):
    """Shrink a t-statistic computed on overlapping forward windows.

    A 10-day forward return measured every day shares nine days with the
    next row, so rows are far from independent. Treating them as independent
    inflates the t-statistic by about sqrt(h). Dividing by sqrt(h) is the
    standard first-order correction.
    """
    return np.sqrt(max(1.0, float(horizon)))


def direction_and_strength(driver, target, min_abs_t=MIN_ABS_T, horizon=1):
    """Estimate the sign of driver -> forward-return, plus how sure we are.

    Returns (sign, t_stat, correlation). `sign` is +1, -1 or 0, where 0 means
    "not distinguishable from noise on this window".

    Uses Spearman rank correlation instead of Pearson so a few large return
    days cannot set the sign alone. The t-statistic is already corrected for
    the overlap from `horizon`.
    """
    d = pd.Series(driver, dtype=float)
    y = pd.Series(np.asarray(target, dtype=float), index=d.index)
    ok = d.notna() & y.notna()
    d, y = d[ok], y[ok]

    if len(d) < MIN_OBS or d.nunique() < 2:
        return 0.0, 0.0, 0.0

    rho = float(d.rank().corr(y.rank()))
    if not np.isfinite(rho) or abs(rho) >= 1.0:
        return 0.0, 0.0, 0.0

    t = rho * np.sqrt(len(d) - 2) / np.sqrt(max(1e-12, 1.0 - rho ** 2))
    t = t / overlap_correction(horizon)
    sign = float(np.sign(rho)) if abs(t) >= min_abs_t else 0.0
    return sign, float(t), rho


def event_response(flag, target, min_abs_t=MIN_ABS_T, horizon=1):
    """Same idea for a boolean flag: mean forward return when on minus when off.

    The t-statistic carries the same overlapping-window correction.
    """
    f = pd.Series(flag, dtype=float)
    y = pd.Series(np.asarray(target, dtype=float), index=f.index)
    ok = f.notna() & y.notna()
    f, y = f[ok], y[ok]

    on, off = y[f > 0.5], y[f <= 0.5]
    if len(on) < 20 or len(off) < 20:
        return 0.0, 0.0, 0.0

    spread = float(on.mean() - off.mean())
    se = np.sqrt(on.var(ddof=1) / len(on) + off.var(ddof=1) / len(off))
    t = (spread / se if se > 0 else 0.0) / overlap_correction(horizon)
    sign = float(np.sign(spread)) if abs(t) >= min_abs_t else 0.0
    return sign, float(t), spread


def squash(score, scale=1.0):
    """Map an unbounded score into (-1, 1) without a hard clip.

    tanh instead of a clip, so the position changes smoothly with signal
    strength and an extreme reading gives a bigger trade than a strong one.
    """
    return np.tanh(np.asarray(score, dtype=float) / max(scale, 1e-9))


def trailing_z(series, window, min_periods=None):
    """Trailing z-score. Uses past values only."""
    s = pd.Series(series, dtype=float)
    mp = min_periods or max(20, window // 2)
    mu = s.rolling(window, min_periods=mp).mean()
    sd = s.rolling(window, min_periods=mp).std(ddof=1)
    return ((s - mu) / sd.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
