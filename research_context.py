"""
ResearchContext -- builds the data once and hands out consistent slices.

Not a required module. Without it, each script would rebuild the panel in
its own way and tables in the report could disagree. Task 2 research, Task 3
allocation and the final evaluation all get their data from here.

It also owns the development/holdout split. Every sign estimate, parameter
choice and model fit uses the development slice, and the holdout is scored
once at the end. Keeping the split in one shared object means every script
uses the same dates.
"""

import numpy as np
import pandas as pd

import config
from backtester import Backtester
from data_cleaner import DataCleaner
from data_loader import DataLoader
from execution_engine import ExecutionEngine
from feature_engine import FeatureEngine
from performance import PerformanceAnalyzer
from portfolio import Portfolio


class ResearchContext:
    """Loads, cleans, aligns and splits the data; runs strategies on a slice."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.features = FeatureEngine()
        self.perf = PerformanceAnalyzer()

        self.prices = None
        self.signals_clean = None
        self.decision = None
        self.market = None
        self.build()

    # ------------------------------------------------------------------
    def build(self):
        raw_prices, raw_signals = self.loader.load()

        ok_p, missing_p = self.loader.validate_schema(raw_prices, "price")
        ok_s, missing_s = self.loader.validate_schema(raw_signals, "signal")
        if not (ok_p and ok_s):
            raise ValueError(f"Schema validation failed. Missing: {missing_p + missing_s}")

        self.prices, self.signals_clean = self.cleaner.clean(raw_prices, raw_signals)
        self.decision, self.market = self.features.build_decision_frame(
            self.prices, self.signals_clean
        )
        self.features.validate_no_lookahead(self.decision, self.signals_clean)
        return self

    # ------------------------------------------------------------------
    # slicing
    # ------------------------------------------------------------------
    def _mask(self, split):
        d = self.decision["date"]
        if split == "dev":
            return d <= pd.Timestamp(config.DEV_END_DATE)
        if split == "holdout":
            return d >= pd.Timestamp(config.HOLDOUT_START_DATE)
        if split == "full":
            return pd.Series(True, index=d.index)
        raise ValueError(f"Unknown split {split!r}; use 'dev', 'holdout' or 'full'.")

    def slice(self, split="full"):
        """Return (decision_frame, market_frame) for a split, index reset."""
        m = self._mask(split)
        return (
            self.decision.loc[m].reset_index(drop=True),
            self.market.loc[m].reset_index(drop=True),
        )

    def target(self, split="dev", horizon=1):
        """The forward-return label used to fit strategies.

        Price is used here only as a label, never as a feature. The label is
        the open-to-open return the strategy would earn, so strategies are
        fitted and scored on the same quantity.
        """
        _, market = self.slice(split)
        if horizon == 1:
            y = market["ret_oo"].copy()
        else:
            y = market["open"].shift(-horizon) / market["open"] - 1.0

        # The last `horizon` rows of a window would use prices from the next
        # window. On the dev split, that means holdout opens leak into the
        # label. Setting them to NaN loses a few rows and removes the leak.
        y = y.copy()
        y.iloc[-horizon:] = np.nan
        return y.reset_index(drop=True)

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------
    def positions_on_full_history(self, strategy):
        """Generate the position path once, over the whole history.

        Several strategies use trailing windows (z-scores, decay kernels). If
        the data is sliced before positions are made, those windows restart at
        the slice start. The holdout would then start with empty windows on
        1 January, and the same date could get two different positions
        depending on the slice.

        Making positions on the full history and slicing after fixes this, and
        matches live trading, where a strategy keeps its warm-up. There is no
        look-ahead: each position still uses only signals from before its
        candle.
        """
        inputs = self.decision.drop(columns=["information_asof"], errors="ignore")
        return strategy.generate_signal(inputs)

    def run_strategy(self, strategy, split="full", cost=None, slippage_bps=None, label=None):
        """Backtest one strategy on one split under the standard assumptions.

        Positions come from the full history (see positions_on_full_history),
        then the window is cut. The first candle is entered from flat, so the
        entry cost is charged.
        """
        mask = self._mask(split).to_numpy()
        positions = self.positions_on_full_history(strategy)
        market = self.market.loc[mask].reset_index(drop=True)
        pos = pd.Series(positions.to_numpy()[mask], index=market.index)

        engine = ExecutionEngine(cost_per_side=cost, slippage_bps=slippage_bps)
        bt = Backtester(execution_engine=engine, portfolio=Portfolio(),
                        performance=PerformanceAnalyzer())
        label = label or strategy.get_metadata()["name"]

        executions = bt.execute_signals(pos, market)
        bt.update_portfolio(executions, market)
        results = bt.analyze(bt.portfolio, label=label)
        results["metadata"] = strategy.get_metadata()
        return results

    def fit_strategies(self, strategies, split="dev", horizon=None):
        """Fit every strategy on the development window. Never on the holdout.

        Each strategy is fitted at its own forward-return horizon, since the
        horizon is part of the hypothesis. Pass `horizon` to override it for a
        test.

        Caveat (also in the report): horizons over one day give overlapping
        windows, so the t-stats at the fitting stage overstate significance and
        are only for screening. Significance is tested later on the daily
        backtest returns, with autocorrelation-robust standard errors.
        """
        if split != "dev":
            raise ValueError(
                f"Refusing to fit on split {split!r}. Fitting is permitted on the "
                "development window only -- 'full' includes the holdout, and the "
                "holdout is scored once, at the end."
            )
        decision, _ = self.slice(split)
        inputs = decision.drop(columns=["information_asof"], errors="ignore")
        items = strategies.values() if isinstance(strategies, dict) else strategies
        daily = self.target(split, 1)
        for s in items:
            h = horizon if horizon is not None else getattr(s, "fit_horizon", 1)
            s.fit(inputs, self.target(split, h), daily_target=daily)
            s.fitted_["fit_horizon_days"] = h
        return strategies

    def returns_frame(self, strategies, split="full"):
        """Net return series for a set of strategies, aligned on one index."""
        out = {}
        for key, strat in (strategies.items() if isinstance(strategies, dict)
                           else [(s.get_metadata()["name"], s) for s in strategies]):
            out[key] = self.run_strategy(strat, split=split)["returns"]
        return pd.DataFrame(out).dropna(how="any")

    def positions_frame(self, strategies, split="full"):
        out = {}
        for key, strat in (strategies.items() if isinstance(strategies, dict)
                           else [(s.get_metadata()["name"], s) for s in strategies]):
            out[key] = self.run_strategy(strat, split=split)["positions"]
        return pd.DataFrame(out).dropna(how="any")

    # ------------------------------------------------------------------
    def benchmark_returns(self, split="full"):
        """Buy-and-hold on the same execution convention, for reference."""
        _, market = self.slice(split)
        r = market["ret_oo"].fillna(0.0).copy()
        r.iloc[-1] = 0.0
        cost = pd.Series(0.0, index=r.index)
        cost.iloc[0] = config.TRANSACTION_COST      # buy once
        cost.iloc[-1] = config.TRANSACTION_COST     # sell once
        return pd.Series((r - cost).values, index=pd.DatetimeIndex(market["date"]),
                         name="buy_and_hold")

    def summary(self):
        lines = [
            self.cleaner.get_report(),
            "",
            self.features.get_report(),
            "",
            "SPLITS",
            "=" * 62,
        ]
        for split in ("dev", "holdout", "full"):
            d, _ = self.slice(split)
            lines.append(
                f"  {split:<8}: {len(d):4d} candles   "
                f"{d['date'].min().date()} -> {d['date'].max().date()}"
            )
        return "\n".join(lines)
