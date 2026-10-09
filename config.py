"""
Central configuration for the Multi-Alpha Research Lab.

Paths, the mandated cost and the research/holdout split all live here. No
other module hard-codes a path or a cost.

Paths are relative to this file, so the project runs from any working
directory (a submission requirement).
"""

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Both can be set from the environment. The tests use this to run on
# synthetic data without touching data/ or the committed results/.
DATA_DIR = Path(os.environ.get("QUANT_DATA_DIR", PROJECT_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("QUANT_RESULTS_DIR", PROJECT_ROOT / "results"))
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

PRICE_DATA_PATH = DATA_DIR / "price_train.csv"
SIGNAL_DATA_PATH = DATA_DIR / "signals_train.csv"

for _d in (RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Execution assumptions, fixed by the Technical Documentation.
# Do not change or override these.
# ----------------------------------------------------------------------------
TRANSACTION_COST = 0.0005          # 0.05% of notional per side; round trip = 0.10%
EXECUTION_PRICE_FIELD = "open"     # trades fill at candle t's open
SIGNAL_LAG = 1                     # decision for candle t sees signals through t-1

# Slippage must be modelled, but the headline numbers use only the mandated
# cost so they compare with other teams. Slippage is used only in the
# cost-sensitivity study.
SLIPPAGE_BPS = 0.0                 # headline: none beyond the mandated cost
SLIPPAGE_STRESS_BPS = [0.0, 2.0, 5.0, 10.0]   # per side, basis points

# ----------------------------------------------------------------------------
# Research protocol
# ----------------------------------------------------------------------------
# All parameter choices, sign estimates and model fits use only the
# development window. The holdout is used once, at the end.
DEV_END_DATE = "2020-12-31"        # development window: start .. this date
HOLDOUT_START_DATE = "2021-01-01"  # untouched during research

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0               # excess-return convention; stated in the report

# Position bounds. Shorting is allowed: the problem statement does not forbid
# it and the main effect in this instrument works both ways.
MAX_GROSS_POSITION = 1.0

# Reproducibility
RANDOM_SEED = 20260918

# Column groups in the supplied signal library
PRICE_BASED_SIGNALS = ["PB01", "PB02", "PB03", "PB04", "PB05", "PB06", "PB07", "PB08"]
BAND_BASED_SIGNALS = ["BB01", "BB02", "BB03", "BB04", "BB05", "BB06", "BB07"]
VOLUME_BASED_SIGNALS = ["VB01", "VB02", "VB03", "VB04", "VB05"]
ALL_SIGNALS = PRICE_BASED_SIGNALS + BAND_BASED_SIGNALS + VOLUME_BASED_SIGNALS

# Columns a strategy may never see. This is checked in code.
FORBIDDEN_STRATEGY_INPUTS = ["open", "high", "low", "close", "volume"]

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
