"""
ResearchContext -- builds the data once and hands out consistent slices.

Not a mandated module, but the alternative is every script rebuilding the panel
with its own slightly different assumptions, which is how two tables in the same
report end up disagreeing. Everything downstream -- Task 2 research, Task 3
allocation, the final evaluation -- pulls its data from here.

It also owns the development/holdout boundary. That boundary is the single most
important discipline in the project: every sign estimate, parameter choice and
model fit happens on the development slice, and the holdout is scored exactly
once at the end. Making the split a property of a shared object rather than a
convention in each script means it cannot quietly drift.
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

        Price is legitimate here and only here -- as a supervised-learning label,
        never as a feature. The label is the open-to-open return the strategy
        would actually capture, so what a strategy is fitted to and what it is
        scored on are the same quantity.
        """
        _, market = self.slice(split)
        if horizon == 1:
            y = market["ret_oo"].copy()
        else:
            y = market["open"].shift(-horizon) / market["open"] - 1.0

        # The last `horizon` rows of a window would be labelled with prices from
        # the window that follows it. On the development split that means the
        # label for the final candles is computed from holdout opens, which is a
        # small but genuine leak across the boundary the whole protocol rests on.
        # Blanking them costs a handful of observations and removes the leak.
        y = y.copy()
        y.iloc[-horizon:] = np.nan
        return y.reset_index(drop=True)

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------
    def positions_on_full_history(self, strategy):
        """Generate the position path once, over the whole history.

        This exists because of a subtle and expensive mistake: several strategies
        use trailing windows (z-scores, decay kernels), and if the data is sliced
        *before* the positions are generated, every one of those windows restarts
        at the slice boundary. The holdout would then be scored on a strategy
        that had amnesia on 1 January, and the same date would get two different
        positions depending on which window it was evaluated in.

        Generating on the full history and slicing afterwards fixes that, and it
        is also what actually happens in production -- a live strategy does not
        forget its warm-up because the evaluation period changed. No look-ahead is
        introduced: every position still depends only on signals strictly older
        than its own candle.
        """
        inputs = self.decision.drop(columns=["information_asof"], errors="ignore")
        return strategy.generate_signal(inputs)

    def run_strategy(self, strategy, split="full", cost=None, slippage_bps=None, label=None):
        """Backtest one strategy on one split under the standard assumptions.

        Positions come from the full history (see positions_on_full_history), then
        the window is cut. The first candle of the window is entered from flat, so
        the entry cost is charged rather than inherited.
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

        Each strategy is fitted against the forward-return horizon it declares,
        because the horizon is part of the hypothesis. Pass `horizon` to override
        that for a controlled experiment.

        One caveat carried into the report: horizons longer than a day produce
        overlapping windows, so the t-statistics reported at the fitting stage
        are screening statistics and overstate significance. Real significance is
        established later on the non-overlapping daily return series of the
        backtest, with autocorrelation-robust standard errors.
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
