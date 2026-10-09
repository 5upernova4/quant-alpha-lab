"""
DataLoader -- reads the raw files and checks their shape.

It only loads and checks; it does not fix anything. All fixes happen in
DataCleaner, so one module owns the data-quality report and we can always say
what was wrong with the raw file.
"""

import pandas as pd

import config


class DataLoader:
    """Loads the price series and the anonymised signal library."""

    REQUIRED_PRICE_COLUMNS = ["date"] + config.OHLCV_COLUMNS
    REQUIRED_SIGNAL_COLUMNS = ["date"] + config.ALL_SIGNALS

    def __init__(self):
        self.load_report = {}

    def load(self, source=None, signal_source=None):
        """Load both files and return them untouched, as (prices, signals).

        `source` and `signal_source` default to the paths in config, so main.py
        does not need to know where the data lives.
        """
        price_path = source or config.PRICE_DATA_PATH
        signal_path = signal_source or config.SIGNAL_DATA_PATH

        for path, label in ((price_path, "price"), (signal_path, "signal")):
            if not str(path) or not pd.io.common.file_exists(str(path)):
                raise FileNotFoundError(
                    f"Could not find the {label} file at {path}. "
                    f"Expected it under {config.DATA_DIR}."
                )

        prices = pd.read_csv(price_path)
        signals = pd.read_csv(signal_path)

        self.load_report = {
            "price_path": str(price_path),
            "signal_path": str(signal_path),
            "price_rows_raw": len(prices),
            "signal_rows_raw": len(signals),
            "price_columns": list(prices.columns),
            "signal_columns": list(signals.columns),
        }
        return prices, signals

    def validate_schema(self, data, kind="price"):
        """Check that the expected columns are present.

        Returns (is_valid, missing_columns) so the caller can see what is
        missing, not just that something is.
        """
        required = (
            self.REQUIRED_PRICE_COLUMNS if kind == "price" else self.REQUIRED_SIGNAL_COLUMNS
        )
        missing = [c for c in required if c not in data.columns]
        return (len(missing) == 0), missing

    def get_report(self):
        return dict(self.load_report)
