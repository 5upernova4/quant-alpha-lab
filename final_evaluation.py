"""
FinalEvaluator - the head-to-head between allocation methods.

The PS requires five methods on the same data, execution rules, cost and
evaluation period:

    1. best individual strategy
    2. equal weight
    3. risk-based
    4. optimisation-based
    5. the proposed dynamic allocation

Every method's return series is produced upstream by portfolio_backtest, which
nets the strategy positions into one book and runs it through the normal
ExecutionEngine. So costs (including the cost of changing weights) are already
inside the returns handed to this class, charged the same way for everyone.
This class only scores and compares.

"Best individual" is picked on the development window and carried into the
holdout unchanged. Picking it on the holdout would be choosing the winner after
the race.
"""

import numpy as np
import pandas as pd

from performance import PerformanceAnalyzer
from statistics import StatisticalTester


class FinalEvaluator:
    """Scores and compares every allocation method on identical assumptions."""

    def __init__(self, ppy=None):
        self.perf = PerformanceAnalyzer(periods_per_year=ppy)
        self.stats = StatisticalTester()
        self.methods_ = {}
        self.comparison_ = None
        self.robustness_ = None

    # ------------------------------------------------------------------
    def add_method(self, name, returns, weights=None, note="", metrics=None):
        """Register one method's net daily returns.

        `metrics` is the backtest's own metrics dict, if there is one; it is
        where turnover and cost drag come from, since those need positions.
        """
        r = pd.Series(returns, dtype=float).dropna()
        self.methods_[name] = {"returns": r, "weights": weights, "note": note,
                               "metrics": metrics or {}}
        return self

    def evaluate(self, strategy_set=None, name=None):
        """Metrics for one registered method, or for a return series passed in."""
        if name is not None:
            return self.perf.compute_all(self.methods_[name]["returns"], label=name)
        return self.perf.compute_all(pd.Series(strategy_set, dtype=float).dropna(),
                                     label="strategy_set")

    def compare(self, methods=None, benchmark=None):
        """One table, all methods, same metrics."""
        keys = list(methods or self.methods_)
        rows = []
        for k in keys:
            info = self.methods_[k]
            m = self.perf.compute_all(info["returns"], label=k)
            bt = info.get("metrics", {})
            row = {
                "method": k,
                "n": m["n_periods"],
                "total_return": m["total_return"],
                "ann_return": m["annualized_return"],
                "volatility": m["volatility"],
                "sharpe": m["sharpe"],
                "sortino": m["sortino"],
                "calmar": m["calmar"],
                "max_drawdown": m["max_drawdown"],
                "max_dd_days": m["max_dd_duration_days"],
                "annual_turnover": bt.get("annual_turnover", np.nan),
                "cost_drag": bt.get("annualized_cost_drag", np.nan),
                "note": info.get("note", ""),
            }
            w = info.get("weights")
            if w is not None:
                wa = pd.DataFrame(w) if not isinstance(w, pd.Series) else w.to_frame().T
                arr = wa.to_numpy(dtype=float)
                row["avg_herfindahl"] = float(np.mean((arr ** 2).sum(axis=1)))
                row["effective_n"] = (1.0 / row["avg_herfindahl"]
                                      if row["avg_herfindahl"] > 0 else np.nan)
            rows.append(row)

        out = pd.DataFrame(rows)
        if benchmark and benchmark in self.methods_:
            b = self.methods_[benchmark]["returns"]
            deltas = []
            for k in keys:
                r = self.methods_[k]["returns"]
                common = r.index.intersection(b.index)
                d = r.loc[common] - b.loc[common]
                sd = d.std(ddof=1)
                deltas.append({
                    "method": k,
                    "excess_ann_return": float(d.mean() * self.perf.ppy),
                    "information_ratio": float(d.mean() / sd * np.sqrt(self.perf.ppy))
                    if sd > 0 else 0.0,
                })
            out = out.merge(pd.DataFrame(deltas), on="method", how="left")
        self.comparison_ = out
        return out

    def robustness_test(self, results=None, n_splits=3):
        """Sharpe in each third of the evaluation window.

        The organisers say they may re-score on sub-periods, so we do it first.
        A method that only wins in one third hasn't really won.
        """
        rows = []
        for k, info in self.methods_.items():
            r = info["returns"]
            if len(r) < n_splits * 20:
                continue
            sharpes = []
            for i, idx in enumerate(np.array_split(np.arange(len(r)), n_splits)):
                sub = r.iloc[idx]
                m = self.perf.compute_all(sub, label=f"{k}_p{i+1}")
                sharpes.append(m["sharpe"])
                rows.append({"method": k, "subperiod": i + 1,
                             "start": sub.index[0].date(), "end": sub.index[-1].date(),
                             "n": len(sub), "sharpe": m["sharpe"],
                             "ann_return": m["annualized_return"],
                             "max_drawdown": m["max_drawdown"]})
            rows.append({"method": k, "subperiod": "worst", "start": r.index[0].date(),
                         "end": r.index[-1].date(), "n": len(r),
                         "sharpe": float(np.min(sharpes)),
                         "ann_return": np.nan, "max_drawdown": np.nan})
        out = pd.DataFrame(rows)
        self.robustness_ = out
        return out

    def significance_vs(self, method, benchmark):
        """Newey-West t and a bootstrap interval on the daily return difference."""
        a = self.methods_[method]["returns"]
        b = self.methods_[benchmark]["returns"]
        common = a.index.intersection(b.index)
        d = a.loc[common] - b.loc[common]
        res = self.stats.newey_west_t(d)
        boot = self.stats.stationary_bootstrap(d)
        return {
            "method": method, "benchmark": benchmark, "n": len(d),
            "mean_excess_ann": float(d.mean() * self.perf.ppy),
            "nw_t_stat": res["t_stat"], "nw_p_value": res["p_value"],
            "boot_ci_low": boot["ci_low"], "boot_ci_high": boot["ci_high"],
        }

    # ------------------------------------------------------------------
    def report(self, title="FINAL EVALUATION"):
        lines = [title, "=" * 128]
        if self.comparison_ is not None:
            cols = [c for c in ["method", "n", "total_return", "ann_return", "volatility",
                                "sharpe", "sortino", "calmar", "max_drawdown", "max_dd_days",
                                "annual_turnover", "cost_drag", "effective_n",
                                "excess_ann_return", "information_ratio"]
                    if c in self.comparison_.columns]
            lines.append(self.comparison_[cols].to_string(
                index=False, float_format=lambda v: f"{v:9.4f}"))
            notes = self.comparison_[["method", "note"]]
            notes = notes[notes["note"].astype(bool)]
            if len(notes):
                lines.append("\n[Notes]")
                for _, row in notes.iterrows():
                    lines.append(f"    {row['method']:<28} {row['note']}")
        if self.robustness_ is not None and not self.robustness_.empty:
            lines.append("\n[Sharpe by third of the evaluation window]")
            lines.append(self.robustness_.to_string(
                index=False, float_format=lambda v: f"{v:9.4f}"))
        return "\n".join(lines)
