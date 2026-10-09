"""
DataLoader -- gets raw files off disk and asserts they are shaped as expected.

Deliberately dumb. It loads and it checks; it does not repair anything. All
repair work belongs to DataCleaner, so that the data-quality report has a single
owner and we can always answer "what exactly was wrong with the raw file?".
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

        `source` and `signal_source` default to the configured paths so that
        main.py never has to know where the data lives.
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

        Returns (is_valid, missing_columns) rather than a bare bool, because a
        silent False tells the operator nothing about what to fix.
        """
        required = (
            self.REQUIRED_PRICE_COLUMNS if kind == "price" else self.REQUIRED_SIGNAL_COLUMNS
        )
        missing = [c for c in required if c not in data.columns]
        return (len(missing) == 0), missing

    def get_report(self):
        return dict(self.load_report)
