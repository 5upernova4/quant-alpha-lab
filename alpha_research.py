"""
AlphaResearch -- the Task 2 pipeline, start to finish.

Runs every candidate strategy through the same tests and ends with a
selection. The steps:

    1. Describe.    What does each strategy do, how often, at what cost?
    2. Test.        Is the performance different from noise?
    3. Stress.      Does it hold across sub-periods, costs, parameters, noise?
    4. Decompose.   Are these different alphas, or the same alpha six times?
    5. Select.      Which pass all four, taken together?

Selection uses a fixed written rule on the development window, not a judgement
made after seeing the holdout. The holdout is scored once, after the set is
fixed. If the selection does badly, the report says so; the set is not changed.
"""

import numpy as np
import pandas as pd

import config
from orthogonality import OrthogonalityAnalyzer
from performance import PerformanceAnalyzer
from robustness import RobustnessTester
from statistics import StatisticalTester


class AlphaResearch:
    """Runs the full Task 2 research programme over a set of candidate alphas."""

    # The selection rule, written down before the holdout is opened.
    #
    # Gates first, then a rank, instead of one threshold per metric, because
    # the problem statement asks for the five points to be weighed together. A
    # strategy with a modest Sharpe but new information is worth more to a
    # portfolio than a slightly better one that copies something already held.
    # A single threshold per metric cannot capture that.
    #
    # The three gates are yes/no questions:
    #   G1  does the performance come from timing, or just from holding exposure?
    #   G2  does it survive the cost it has to pay?
    #   G3  does it add anything the rest of the book does not already have?
    # Failing any gate excludes the strategy, whatever its return.
    #
    # The rest are matters of degree, so they are scored and ranked.
    SELECTION_RULE = {
        # gates
        "max_permutation_p": 0.30,
        "min_breakeven_cost_bps": 5.0,        # must survive the mandated 0.05% per side
        "min_residual_sharpe": 0.0,           # must earn after the others are hedged out
        # ranking
        "score_components": ["dev_sharpe", "subperiods_positive", "decay_slope",
                             "noise_retention", "turnover_efficiency"],
        "min_composite_score": 0.0,           # at or above the candidate-set average
        "min_strategies": 3,                  # a portfolio needs something to diversify
    }

    def __init__(self, context, strategies, benchmark_label="buy_and_hold"):
        self.ctx = context
        self.strategies = strategies
        self.perf = PerformanceAnalyzer()
        self.stats = StatisticalTester()
        self.benchmark_label = benchmark_label
        self.tables = {}

    # ------------------------------------------------------------------
    # 1. describe
    # ------------------------------------------------------------------
    def performance_table(self, split="dev"):
        rows = []
        for key, strat in self.strategies.items():
            res = self.ctx.run_strategy(strat, split=split)
            m = res["metrics"]
            rows.append({
                "strategy": key, "name": strat.get_metadata()["name"],
                "ann_return": m["annualized_return"], "volatility": m["volatility"],
                "sharpe": m["sharpe"], "sortino": m["sortino"], "calmar": m["calmar"],
                "max_drawdown": m["max_drawdown"], "max_dd_days": m["max_dd_duration_days"],
                "hit_rate_active": m.get("hit_rate_active", np.nan),
                "pct_days_active": m.get("pct_days_active", np.nan),
                "annual_turnover": m.get("annual_turnover", np.nan),
                "cost_drag": m.get("annualized_cost_drag", np.nan),
            })
        bh = self.ctx.benchmark_returns(split)
        bm = self.perf.compute_all(bh, label=self.benchmark_label)
        rows.append({
            "strategy": self.benchmark_label, "name": "Buy and hold",
            "ann_return": bm["annualized_return"], "volatility": bm["volatility"],
            "sharpe": bm["sharpe"], "sortino": bm["sortino"], "calmar": bm["calmar"],
            "max_drawdown": bm["max_drawdown"], "max_dd_days": bm["max_dd_duration_days"],
            "hit_rate_active": np.nan, "pct_days_active": 1.0,
            "annual_turnover": 0.0, "cost_drag": np.nan,
        })
        out = pd.DataFrame(rows)
        self.tables[f"performance_{split}"] = out
        return out

    # ------------------------------------------------------------------
    # 2. test
    # ------------------------------------------------------------------
    def significance_table(self, split="dev", n_trials=None):
        _, market = self.ctx.slice(split)
        asset_r = market["ret_oo"].fillna(0.0)
        # Task 3 checks alpha_07 on its own, but it was still the 7th idea
        # tried, so the deflated Sharpe must count it
        n_trials = n_trials or len(self.strategies)
        rows = []
        for key, strat in self.strategies.items():
            res = self.ctx.run_strategy(strat, split=split)
            row = self.stats.full_battery(
                res["returns"], positions=res["positions"],
                asset_returns=asset_r, label=key, n_trials=n_trials,
            )
            rows.append(row)
        out = pd.DataFrame(rows)
        # multiple-testing correction across the candidates we tested
        adj, reject = self.stats.benjamini_hochberg(out["nw_p_value"].fillna(1.0))
        out["nw_p_fdr"] = adj
        out["survives_fdr_5pct"] = reject
        self.tables[f"significance_{split}"] = out
        return out

    # ------------------------------------------------------------------
    # 3. stress
    # ------------------------------------------------------------------
    def robustness_suite(self, split="dev", noise_draws=12):
        decision, market = self.ctx.slice(split)
        summary, detail = [], {}
        for key, strat in self.strategies.items():
            rt = RobustnessTester(decision, market)
            sub = rt.subperiod_analysis(strat)
            reg = rt.regime_analysis(strat)
            cost = rt.cost_sensitivity(strat)
            noise = rt.signal_noise_robustness(strat, n_draws=noise_draws)
            fail = rt.failure_analysis(strat)
            decay = rt.decay_analysis(strat)
            grid = getattr(type(strat), "PARAM_GRID", None)
            params = (rt.parameter_sensitivity(type(strat), grid,
                                               fit_target=self.ctx.target(split,
                                                                          strat.fit_horizon),
                                               fit_data=decision)
                      if grid else pd.DataFrame())
            perf_m = self.ctx.run_strategy(strat, split=split)["metrics"]
            detail[key] = {"subperiod": sub, "regime": reg, "cost": cost,
                           "noise": noise, "failure": fail, "parameters": params,
                           "decay": pd.DataFrame([decay]) if decay else pd.DataFrame()}
            summary.append({
                "strategy": key,
                "n_subperiods": len(sub),
                "pct_subperiods_positive": float(sub["positive"].mean()) if len(sub) else np.nan,
                "worst_subperiod_sharpe": float(sub["sharpe"].min()) if len(sub) else np.nan,
                "breakeven_cost_bps": cost.attrs.get("breakeven_cost_bps", np.nan),
                "sharpe_at_2x_cost": float(
                    cost.loc[cost["cost_multiple"] == 2.0, "sharpe"].iloc[0]
                ) if (cost["cost_multiple"] == 2.0).any() else np.nan,
                "noise_retention_5pct": float(
                    noise.loc[noise["flip_rate"] == 0.05, "retention"].iloc[0]
                ) if (noise["flip_rate"] == 0.05).any() else np.nan,
                "pct_failures_fighting_trend": fail.attrs.get("pct_fighting_trend", np.nan),
                "param_plateau_score": params.attrs.get("plateau_score", np.nan)
                if len(params) else np.nan,
                "param_pct_positive": params.attrs.get("pct_positive", np.nan)
                if len(params) else np.nan,
                "param_sharpe_spread": params.attrs.get("sharpe_std", np.nan)
                if len(params) else np.nan,
                "decay_slope": decay.get("decay_slope", np.nan),
                "decay_t": decay.get("decay_t", np.nan),
                "first_half_sharpe": decay.get("first_half_sharpe", np.nan),
                "second_half_sharpe": decay.get("second_half_sharpe", np.nan),
                "annual_turnover": perf_m.get("annual_turnover", np.nan),
                "cost_to_gross": self._cost_to_gross(perf_m),
            })
        out = pd.DataFrame(summary)
        self.tables[f"robustness_{split}"] = out
        self.tables[f"robustness_detail_{split}"] = detail
        return out, detail

    @staticmethod
    def _cost_to_gross(metrics):
        """What fraction of gross return the costs ate. NaN for a flat strategy."""
        net = metrics.get("annualized_return")
        drag = metrics.get("annualized_cost_drag")
        if net is None or drag is None:
            return np.nan
        gross = abs(net + drag)
        return float(drag / gross) if gross > 1e-9 else np.nan

    # ------------------------------------------------------------------
    # 4. decompose
    # ------------------------------------------------------------------
    def orthogonality_suite(self, split="dev"):
        R = self.ctx.returns_frame(self.strategies, split=split)
        oa = OrthogonalityAnalyzer(R)
        oa.correlation_matrix()
        dim = oa.effective_dimensionality()
        qr_tab = oa.pivoted_qr()
        inc = oa.incremental_contribution()
        gs, resid = oa.gram_schmidt_residuals()
        stab = oa.stability_through_time(window=min(252, max(60, len(R) // 3)),
                                         step=max(5, len(R) // 30))
        self.tables[f"orthogonality_{split}"] = {
            "analyzer": oa, "returns": R, "dimensionality": dim,
            "pivoted_qr": qr_tab, "incremental": inc, "gram_schmidt": gs,
            "residuals": resid, "stability": stab,
            "correlation": oa.results_["correlation"],
        }
        return self.tables[f"orthogonality_{split}"]

    # ------------------------------------------------------------------
    # 5. select
    # ------------------------------------------------------------------
    def select(self, split="dev"):
        """Apply the pre-declared rule and return (selected_keys, scorecard).

        Everything uses the development window. The scorecard records each
        gate result and each score component, so the report can explain why a
        strategy was excluded.
        """
        perf = self.tables.get(f"performance_{split}")
        if perf is None:
            perf = self.performance_table(split)
        sig = self.tables.get(f"significance_{split}")
        if sig is None:
            sig = self.significance_table(split)
        rob = self.tables.get(f"robustness_{split}")
        if rob is None:
            rob, _ = self.robustness_suite(split)
        orth = self.tables.get(f"orthogonality_{split}") or self.orthogonality_suite(split)

        perf_i = perf.set_index("strategy")
        sig_i = sig.set_index("label")
        rob_i = rob.set_index("strategy")
        inc_i = orth["incremental"].set_index("strategy")
        rule = self.SELECTION_RULE

        rows = []
        for key in self.strategies:
            turnover = float(rob_i.at[key, "annual_turnover"])
            row = {
                "strategy": key,
                "dev_sharpe": float(perf_i.at[key, "sharpe"]),
                "permutation_p": float(sig_i.at[key, "perm_p_value"])
                if "perm_p_value" in sig_i.columns else np.nan,
                "breakeven_cost_bps": float(rob_i.at[key, "breakeven_cost_bps"]),
                "residual_sharpe": float(inc_i.at[key, "residual_sharpe"])
                if key in inc_i.index else np.nan,
                "subperiods_positive": float(rob_i.at[key, "pct_subperiods_positive"]),
                "decay_slope": float(rob_i.at[key, "decay_slope"]),
                "noise_retention": float(rob_i.at[key, "noise_retention_5pct"]),
                "annual_turnover": turnover,
                # cheaper strategies score higher; log because 10x vs 20x
                # matters more than 80x vs 90x
                "turnover_efficiency": float(-np.log(max(turnover, 1.0))),
            }
            row["pass_G1_timing"] = bool(row["permutation_p"] <= rule["max_permutation_p"])
            row["pass_G2_cost"] = bool(row["breakeven_cost_bps"] >= rule["min_breakeven_cost_bps"])
            row["pass_G3_incremental"] = bool(row["residual_sharpe"] > rule["min_residual_sharpe"])
            row["passes_gates"] = all(row[f"pass_{g}"] for g in
                                      ("G1_timing", "G2_cost", "G3_incremental"))
            rows.append(row)

        sc = pd.DataFrame(rows)

        # Composite score: each component is standardised across the
        # candidates, so the score means "better than the others on this
        # measure" and does not depend on each metric's units.
        for c in rule["score_components"]:
            v = sc[c].astype(float)
            sd = v.std(ddof=1)
            sc[f"z_{c}"] = (v - v.mean()) / sd if sd > 1e-12 else 0.0
        sc["composite_score"] = sc[[f"z_{c}" for c in rule["score_components"]]].mean(axis=1)

        eligible = sc[sc["passes_gates"]].sort_values("composite_score", ascending=False)
        chosen = eligible[eligible["composite_score"] >= rule["min_composite_score"]]["strategy"].tolist()
        if len(chosen) < rule["min_strategies"]:
            chosen = eligible["strategy"].head(rule["min_strategies"]).tolist()

        sc["selected"] = sc["strategy"].isin(chosen)
        sc["excluded_because"] = [
            "" if r.selected else
            ", ".join(
                [g for g, ok in (("fails timing test", r.pass_G1_timing),
                                 ("not cost-viable", r.pass_G2_cost),
                                 ("adds nothing incremental", r.pass_G3_incremental))
                 if not ok]
            ) or "ranked below the composite cut"
            for r in sc.itertuples()
        ]
        self.tables["selection"] = sc
        self.tables["selected_keys"] = chosen
        return chosen, sc

    # ------------------------------------------------------------------
    def report(self, split="dev"):
        lines = [f"TASK 2 -- ALPHA RESEARCH ({split} window)", "=" * 118]
        perf = self.tables.get(f"performance_{split}")
        if perf is not None:
            lines.append("\n[Performance]")
            lines.append(perf.drop(columns=["name"]).to_string(
                index=False, float_format=lambda v: f"{v:8.3f}"))
        sig = self.tables.get(f"significance_{split}")
        if sig is not None:
            lines.append("\n[Significance]")
            cols = ["label", "t_t_stat", "nw_t_stat", "nw_p_value", "nw_p_fdr",
                    "boot_point", "boot_ci_low", "boot_ci_high", "psr_psr",
                    "dsr_dsr", "perm_p_value"]
            lines.append(sig[[c for c in cols if c in sig.columns]].to_string(
                index=False, float_format=lambda v: f"{v:8.3f}"))
        rob = self.tables.get(f"robustness_{split}")
        if rob is not None:
            lines.append("\n[Robustness]")
            lines.append(rob.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
        orth = self.tables.get(f"orthogonality_{split}")
        if orth is not None:
            lines.append("\n" + orth["analyzer"].report())
        sel = self.tables.get("selection")
        if sel is not None:
            lines.append("\n[Selection -- rule declared before the holdout was opened]")
            lines.append("    GATES (all must pass)")
            lines.append(f"      G1 timing      : permutation p <= {self.SELECTION_RULE['max_permutation_p']}")
            lines.append(f"      G2 cost        : break-even cost >= {self.SELECTION_RULE['min_breakeven_cost_bps']} bps per side")
            lines.append(f"      G3 incremental : residual Sharpe > {self.SELECTION_RULE['min_residual_sharpe']} after hedging out the others")
            lines.append(f"    RANK on the mean z-score of {', '.join(self.SELECTION_RULE['score_components'])}")
            lines.append(f"      keep composite >= {self.SELECTION_RULE['min_composite_score']}, "
                         f"minimum {self.SELECTION_RULE['min_strategies']} strategies")
            show = ["strategy", "dev_sharpe", "permutation_p", "breakeven_cost_bps",
                    "residual_sharpe", "decay_slope", "annual_turnover",
                    "passes_gates", "composite_score", "selected", "excluded_because"]
            lines.append(sel[show].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
            lines.append(f"\n  SELECTED SET: {', '.join(self.tables.get('selected_keys', []))}")
        return "\n".join(lines)
