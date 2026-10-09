"""
Make a fake price_train.csv and signals_train.csv with the same columns as
the competition files.

The real data comes from the Inter-IIT prepathon and is not in this repo.
This lets the tests and CI run the whole pipeline without it. The numbers
mean nothing; only the shape matches.

It also plants the same four kinds of defect the real files had, so the
cleaner has something to fix:
  - price rows shuffled
  - some dates written DD-MM-YYYY instead of YYYY-MM-DD
  - some price rows duplicated
  - some dates in the signal file with no price row

    python tools/make_synthetic_data.py              # writes to data/
    python tools/make_synthetic_data.py --out /tmp/x
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START = "2018-01-02"
N_DATES = 1000          # rows in the signal file, same as the real one
N_MISSING_PRICE = 14    # signal dates with no price row
N_DUPLICATES = 14       # repeated price rows
N_DDMMYYYY = 27         # price dates written the other way round


def _rsi(close, n=14):
    diff = close.diff()
    up = diff.clip(lower=0).rolling(n).mean()
    down = (-diff.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def make_frames(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(START, periods=N_DATES)

    # Daily returns: small drift, ~1% vol, slight negative autocorrelation so
    # the reversion strategies have something weak to find.
    eps = rng.normal(0.0004, 0.010, N_DATES)
    ret = np.empty(N_DATES)
    ret[0] = eps[0]
    for i in range(1, N_DATES):
        ret[i] = eps[i] - 0.05 * ret[i - 1]
    close = pd.Series(250 * np.cumprod(1 + ret), index=dates)

    gap = rng.normal(0, 0.002, N_DATES)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + gap)
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, N_DATES)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, N_DATES)))
    volume = pd.Series(rng.lognormal(9.5, 0.4, N_DATES).round(), index=dates)

    prices = pd.DataFrame({
        "date": dates, "open": open_.round(2).values, "high": hi.round(2).values,
        "low": lo.round(2).values, "close": close.round(2).values,
        "volume": volume.astype(int).values,
    })

    # Signals built from the fake prices, roughly in the spirit of the real
    # library (trend flags, bands, oscillator, volume flags).
    ma5, ma10 = close.rolling(5).mean(), close.rolling(10).mean()
    ma20, ma50 = close.rolling(20).mean(), close.rolling(50).mean()
    sd20 = close.rolling(20).std()
    r = close.pct_change()
    rsi = _rsi(close)
    width = (4 * sd20 / ma20)
    obv = (np.sign(r).fillna(0) * volume).cumsum()
    vma20, vma50 = volume.rolling(20).mean(), volume.rolling(50).mean()

    s = pd.DataFrame(index=dates)
    s["PB01"] = (ma5.diff() > 0)
    s["PB02"] = (ma50.diff() > 0)
    s["PB03"] = (close > ma20)
    s["PB04"] = (close > ma50)
    s["PB05"] = (close.pct_change(5) > 0)
    s["PB06"] = (close >= close.rolling(20).max())
    s["PB07"] = (close / ma20 - 1).round(5)
    s["PB08"] = (ma10 / ma50 - 1).round(5)
    s["BB01"] = (close > ma20 + 2 * sd20)
    s["BB02"] = (close < ma20 - 2 * sd20)
    s["BB03"] = (rsi > 70)
    s["BB04"] = (rsi < 30)
    s["BB05"] = (width < width.rolling(120, min_periods=20).median())
    s["BB06"] = ((close - ma20) / (2 * sd20)).round(4)
    s["BB07"] = r.rolling(14).std().round(5)
    s["VB01"] = (volume > vma20)
    s["VB02"] = (obv.rolling(10).mean().diff() > 0)
    s["VB03"] = ((r.abs() > 2 * r.rolling(20).std()) & (volume > 1.5 * vma20))
    s["VB04"] = (volume > vma50)
    s["VB05"] = (volume / vma20 - 1).round(4)

    flags = [c for c in s.columns if c not in ("PB07", "PB08", "BB06", "BB07", "VB05")]
    s[flags] = s[flags].astype(int)
    for c in ("PB07", "PB08"):
        s[c] = s[c].fillna(0.0)
    # BB06, BB07 and VB05 keep their warm-up NaNs, like the real file
    signals = s.reset_index().rename(columns={"index": "date"})
    return prices, signals


def plant_defects(prices, signals, seed=0):
    rng = np.random.default_rng(seed + 1)

    # some signal dates have no price row
    drop = rng.choice(np.arange(60, len(prices) - 1), N_MISSING_PRICE, replace=False)
    prices = prices.drop(index=prices.index[drop]).reset_index(drop=True)

    # exact duplicate price rows, so the price file ends up with N_DATES rows
    dup = rng.choice(len(prices), N_DUPLICATES, replace=False)
    prices = pd.concat([prices, prices.iloc[dup]], ignore_index=True)

    # dates as strings, a few in DD-MM-YYYY
    prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")
    signals["date"] = signals["date"].dt.strftime("%Y-%m-%d")
    odd = rng.choice(len(prices), N_DDMMYYYY, replace=False)
    prices.loc[odd, "date"] = pd.to_datetime(prices.loc[odd, "date"]).dt.strftime("%d-%m-%Y")

    # shuffled rows
    prices = prices.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return prices, signals


def write(out_dir, seed=0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prices, signals = plant_defects(*make_frames(seed), seed=seed)
    prices.to_csv(out / "price_train.csv", index=False)
    signals.to_csv(out / "signals_train.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    target = Path(args.out)
    if (target / "price_train.csv").exists():
        raise SystemExit(f"{target}/price_train.csv already exists; refusing to overwrite it.")
    print(f"wrote synthetic data to {write(target, args.seed)}")
