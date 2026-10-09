"""
RobustnessTester -- does the edge hold under changed conditions?

Significance tests ask whether an effect differs from noise on this sample.
Robustness asks whether we would still find it if the sample, costs or
parameters were a bit different. A strategy that only works at one cost level,
in one year, at one parameter setting is likely a coincidence.

Five checks, each for a specific way a backtest can mislead:

1. Sub-period stability -- splits results by calendar year and by regime.
   Catches: an edge that is really one good quarter.
2. Cost and slippage sensitivity -- re-runs the backtest at higher costs and
   reports the break-even level. Catches: a real signal that costs too much to
   trade.
3. Parameter sensitivity -- moves each parameter across a grid and reports the
   shape of the results, not just the best point. Catches: a result that only
   works at one exact setting (a curve fit).
4. Signal-noise robustness -- randomly flips or jitters some signal inputs.
   Catches: depending on a few exact readings.
5. Failure analysis -- looks at the worst drawdowns and checks whether the
   losses share a cause. Catches: a systematic exposure mistaken for bad luck.
"""

import numpy as np
import pandas as pd

import config
from backtester import Backtester
from execution_engine import ExecutionEngine
from performance import PerformanceAnalyzer
from portfolio import Portfolio


class RobustnessTester:
    """Re-runs a strategy under varied assumptions and reports the spread."""

    def __init__(self, decision_frame, market_frame, seed=None):
        self.decision = decision_frame.reset_index(drop=True)
        self.market = market_frame.reset_index(drop=True)
        self.seed = config.RANDOM_SEED if seed is None else seed
        self.perf = PerformanceAnalyzer()
        self.results_ = {}

    # ------------------------------------------------------------------
    def _run(self, strategy, decision=None, market=None, cost=None, slippage_bps=None):
        d = self.decision if decision is None else decision
        m = self.market if market is None else market
        engine = ExecutionEngine(cost_per_side=cost, slippage_bps=slippage_bps)
        bt = Backtester(execution_engine=engine, portfolio=Portfolio(),
                        performance=PerformanceAnalyzer())
        return bt.run(d, strategy, market_data=m)

    # ------------------------------------------------------------------
    def decay_analysis(self, strategy, n_chunks=6):
        """Is the edge fading *within* the development window?

        Splits the window into equal chunks, measures Sharpe in each, and fits
        a straight line. A strongly negative slope means the effect was already
        fading during development, which is an early warning that it may fail
        on the holdout. A strategy can pass every "is it positive?" test and
        still fail this one.
        """
        res = self._run(strategy)
        r = res["returns"]
        if len(r) < n_chunks * 20:
            return {}
        chunks = np.array_split(np.arange(len(r)), n_chunks)
        sharpes = []
        for idx in chunks:
            sub = r.iloc[idx]
            sd = sub.std(ddof=1)
            sharpes.append(float(sub.mean() / sd * np.sqrt(config.TRADING_DAYS_PER_YEAR))
                           if sd > 0 else 0.0)
        x = np.arange(n_chunks, dtype=float)
        y = np.array(sharpes)
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        se = (np.sqrt((resid ** 2).sum() / max(1, n_chunks - 2))
              / np.sqrt(((x - x.mean()) ** 2).sum()))
        out = {
            "n_chunks": n_chunks,
            "chunk_sharpes": sharpes,
            "first_half_sharpe": float(np.mean(sharpes[: n_chunks // 2])),
            "second_half_sharpe": float(np.mean(sharpes[n_chunks // 2:])),
            "decay_slope": float(slope),
            "decay_t": float(slope / se) if se > 0 else 0.0,
        }
        self.results_["decay"] = pd.DataFrame([out])
        return out

    def subperiod_analysis(self, strategy, by="year"):
        """Performance broken out by calendar year (or an explicit split)."""
        res = self._run(strategy)
        r = res["returns"]
        pos = res["positions"]
        groups = r.index.year if by == "year" else by

        rows = []
        for key, sub in r.groupby(groups):
            if len(sub) < 20:
                continue
            m = self.perf.compute_all(sub, positions=pos.loc[sub.index], label=str(key))
            rows.append({
                "period": str(key), "n": m["n_periods"],
                "ann_return": m["annualized_return"], "volatility": m["volatility"],
                "sharpe": m["sharpe"], "sortino": m["sortino"],
                "max_drawdown": m["max_drawdown"], "hit_rate": m.get("hit_rate", np.nan),
            })
        out = pd.DataFrame(rows)
        if not out.empty:
            out["positive"] = out["sharpe"] > 0
        self.results_["subperiod"] = out
        return out

    def regime_analysis(self, strategy, vol_window=21):
        """Split by realised-volatility tercile and by market direction.

        Realised volatility uses price. That is fine here: this is analysis
        after the fact of where a strategy worked, not an input to any trade.
        """
        res = self._run(strategy)
        r, pos = res["returns"], res["positions"]
        mkt = self.market.set_index(pd.DatetimeIndex(self.market["date"]))
        rv = mkt["ret_oo"].rolling(vol_window, min_periods=vol_window).std()
        trend = mkt["ret_oo"].rolling(63, min_periods=63).mean()

        common = r.index.intersection(rv.dropna().index)
        rv, trend, r2 = rv.loc[common], trend.loc[common], r.loc[common]
        rows = []

        terciles = pd.qcut(rv, 3, labels=["low vol", "mid vol", "high vol"])
        for key, sub in r2.groupby(terciles, observed=True):
            if len(sub) < 20:
                continue
            m = self.perf.compute_all(sub, positions=pos.loc[sub.index], label=str(key))
            rows.append({"regime": str(key), "n": m["n_periods"], "ann_return": m["annualized_return"],
                         "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"]})

        direction = pd.Series(
            np.where(trend.isna(), "undefined",
                     np.where(trend > 0, "rising market", "falling market")),
            index=common,
        )
        direction = direction[direction != "undefined"]
        for key, sub in r2.reindex(direction.index).groupby(direction, observed=True):
            if len(sub) < 20:
                continue
            m = self.perf.compute_all(sub, positions=pos.loc[sub.index], label=str(key))
            rows.append({"regime": str(key), "n": m["n_periods"], "ann_return": m["annualized_return"],
                         "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"]})

        out = pd.DataFrame(rows)
        self.results_["regime"] = out
        return out

    def cost_sensitivity(self, strategy, cost_multipliers=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0),
                         slippage_grid=None):
        """Re-run at escalating costs and find the break-even cost level."""
        base = config.TRANSACTION_COST
        rows = []
        for k in cost_multipliers:
            res = self._run(strategy, cost=base * k)
            m = res["metrics"]
            rows.append({"cost_multiple": k, "cost_per_side_bps": base * k * 1e4,
                         "ann_return": m["annualized_return"], "sharpe": m["sharpe"],
                         "turnover": m.get("annual_turnover", np.nan)})
        for bps in (slippage_grid or config.SLIPPAGE_STRESS_BPS):
            res = self._run(strategy, cost=base, slippage_bps=bps)
            m = res["metrics"]
            rows.append({"cost_multiple": np.nan, "slippage_bps": bps,
                         "cost_per_side_bps": base * 1e4 + bps,
                         "ann_return": m["annualized_return"], "sharpe": m["sharpe"],
                         "turnover": m.get("annual_turnover", np.nan)})

        out = pd.DataFrame(rows)
        priced = out.dropna(subset=["cost_multiple"]).sort_values("cost_per_side_bps")
        breakeven = np.nan
        pos_rows = priced[priced["ann_return"] > 0]
        neg_rows = priced[priced["ann_return"] <= 0]
        if len(pos_rows) and len(neg_rows):
            lo = pos_rows["cost_per_side_bps"].max()
            hi = neg_rows[neg_rows["cost_per_side_bps"] > lo]["cost_per_side_bps"]
            breakeven = float((lo + hi.min()) / 2) if len(hi) else float(lo)
        elif len(pos_rows) and not len(neg_rows):
            # Still profitable at the highest cost tested, so this number is
            # only a lower bound on the break-even cost. Flag it as such.
            breakeven = float(priced["cost_per_side_bps"].max())
            out.attrs["breakeven_is_lower_bound"] = True
        out.attrs["breakeven_cost_bps"] = breakeven
        out.attrs.setdefault("breakeven_is_lower_bound", False)
        self.results_["cost"] = out
        return out

    def parameter_sensitivity(self, strategy_class, param_grid, fit_target=None,
                              fit_data=None, refit=True):
        """Sweep parameters and report the full surface, peak and dispersion.

        The spread matters more than the peak. A stable strategy sits on a
        broad plateau where nearby settings all work; an overfit one sits on a
        spike. Reporting only the spike hides a curve fit.

        Two details:

        * The strategy is refitted at each grid point, so a parameter that
          changes how features are built also changes the fitted directions.
        * A parameter that the strategy's own fit() re-chooses (for example
          Alpha 02's holding period) is pinned for the sweep. Otherwise every
          grid point ends at the same value and the surface looks flat.
        """
        from itertools import product

        fit_data = self.decision if fit_data is None else fit_data
        fit_data = fit_data.drop(columns=["information_asof"], errors="ignore")

        keys = list(param_grid)
        rows = []
        for combo in product(*(param_grid[k] for k in keys)):
            params = dict(zip(keys, combo))
            strat = strategy_class(**params)
            if refit and fit_target is not None:
                strat.fit(fit_data, fit_target)
                # restore the swept values that fit() may have re-chosen
                for k, v in params.items():
                    if hasattr(strat, k):
                        setattr(strat, k, v)
            m = self._run(strat)["metrics"]
            rows.append({**params, "sharpe": m["sharpe"],
                         "ann_return": m["annualized_return"],
                         "max_drawdown": m["max_drawdown"],
                         "annual_turnover": m.get("annual_turnover", np.nan)})

        out = pd.DataFrame(rows)
        if not out.empty:
            out.attrs["sharpe_mean"] = float(out["sharpe"].mean())
            out.attrs["sharpe_std"] = float(out["sharpe"].std(ddof=1)) if len(out) > 1 else 0.0
            out.attrs["sharpe_min"] = float(out["sharpe"].min())
            out.attrs["sharpe_max"] = float(out["sharpe"].max())
            out.attrs["pct_positive"] = float((out["sharpe"] > 0).mean())
            # A plateau scores near 1; a spike scores near 0.
            out.attrs["plateau_score"] = (
                float(out["sharpe"].mean() / out["sharpe"].max())
                if out["sharpe"].max() > 1e-9 else 0.0
            )
        self.results_["parameters"] = out
        return out

    def signal_noise_robustness(self, strategy, flip_rates=(0.01, 0.05, 0.10), n_draws=20):
        """Corrupt a fraction of the signal inputs and re-measure.

        Boolean flags are flipped; continuous signals get Gaussian noise scaled
        to their own standard deviation. A strategy that depends on a few exact
        readings gets much worse; a robust one gets worse slowly.
        """
        rng = np.random.default_rng(self.seed)
        baseline = self._run(strategy)["metrics"]["sharpe"]
        rows = []
        for rate in flip_rates:
            draws = []
            for _ in range(n_draws):
                d = self.decision.copy()
                for c in config.ALL_SIGNALS:
                    col = d[c]
                    mask = rng.random(len(col)) < rate
                    if col.dropna().isin([0, 1]).all():
                        d.loc[mask, c] = 1.0 - col[mask]
                    else:
                        sd = col.std(ddof=1)
                        d.loc[mask, c] = col[mask] + rng.normal(0, sd if sd > 0 else 1.0, mask.sum())
                draws.append(self._run(strategy, decision=d)["metrics"]["sharpe"])
            # Retention only makes sense when the clean strategy made money.
            # With a negative baseline, "got worse" would show as a number
            # above 1 and look like robustness.
            retention = (float(np.mean(draws) / baseline) if baseline > 1e-9 else np.nan)
            rows.append({"flip_rate": rate, "sharpe_mean": float(np.mean(draws)),
                         "sharpe_std": float(np.std(draws, ddof=1)),
                         "sharpe_p05": float(np.quantile(draws, 0.05)),
                         "retention": retention,
                         "sharpe_drop": float(baseline - np.mean(draws))})
        out = pd.DataFrame(rows)
        out.attrs["baseline_sharpe"] = baseline
        self.results_["noise"] = out
        return out

    def failure_analysis(self, strategy, n_worst=5, window=21):
        """Isolate the worst stretches and describe what they had in common."""
        res = self._run(strategy)
        r, pos = res["returns"], res["positions"]
        rolling = r.rolling(window).sum()
        worst_idx = rolling.nsmallest(n_worst * window).index
        # de-cluster: keep the worst window, then skip forward
        chosen, blocked = [], set()
        for ts in rolling.sort_values().index:
            if ts in blocked:
                continue
            chosen.append(ts)
            loc = r.index.get_loc(ts)
            blocked.update(r.index[max(0, loc - window): loc + window])
            if len(chosen) >= n_worst:
                break

        mkt = self.market.set_index(pd.DatetimeIndex(self.market["date"]))
        rows = []
        for ts in sorted(chosen):
            loc = r.index.get_loc(ts)
            sl = slice(max(0, loc - window + 1), loc + 1)
            seg_r, seg_pos = r.iloc[sl], pos.iloc[sl]
            seg_mkt = mkt["ret_oo"].reindex(seg_r.index)
            rows.append({
                "window_end": ts.date(),
                "strategy_return": float(seg_r.sum()),
                "market_return": float(seg_mkt.sum()),
                "avg_position": float(seg_pos.mean()),
                "market_vol_ann": float(seg_mkt.std(ddof=1) * np.sqrt(config.TRADING_DAYS_PER_YEAR)),
                "fought_the_market": bool(np.sign(seg_pos.mean()) != np.sign(seg_mkt.sum())),
            })
        out = pd.DataFrame(rows)
        if not out.empty:
            out.attrs["pct_fighting_trend"] = float(out["fought_the_market"].mean())
        self.results_["failure"] = out
        return out

    # ------------------------------------------------------------------
    def report(self):
        lines = ["ROBUSTNESS", "=" * 76]
        for key in ("subperiod", "regime", "cost", "parameters", "noise", "failure"):
            df = self.results_.get(key)
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            lines.append(f"\n[{key}]")
            lines.append(df.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
            for k, v in df.attrs.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)
