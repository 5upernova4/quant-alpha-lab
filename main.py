"""
main.py - runs the whole thing end to end.

    python main.py                # Tasks 1, 2 and 3
    python main.py --task 1       # data and backtesting infrastructure
    python main.py --task 2       # alpha discovery and research
    python main.py --task 3       # portfolio construction and allocation (runs Task 2 first)
    python main.py --quick        # fewer bootstrap / permutation draws, for a fast check

Every table printed is also saved to results/ as a CSV, and the console output
goes to results/run_log.txt, so any number in the reports can be traced to a file.
"""

import argparse
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import config
from alpha_research import AlphaResearch
from dynamic_allocator import DynamicAllocator, tilt_null
from factor_model import FactorModel
from final_evaluation import FinalEvaluator
from meta_model import AlphaMetaModel
from orthogonality import OrthogonalityAnalyzer
import plots
import portfolio_backtest as pbt
from performance import PerformanceAnalyzer
from portfolio_optimizer import PortfolioOptimizer
from research_context import ResearchContext
from statistics import StatisticalTester
from strategies import BaselineStrategy, build_all, build_task3_additions

# show each pandas deprecation once instead of hiding it
warnings.simplefilter("once", FutureWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)


class Tee:
    """Echo everything printed to both the console and the run log."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def banner(text):
    print("\n" + "#" * 100)
    print(f"# {text}")
    print("#" * 100)


def save(df, name):
    """Write a table to results/ and return it unchanged."""
    path = config.RESULTS_DIR / f"{name}.csv"
    # keep the index when it carries labels (strategy names, dates), drop it
    # when it's just row numbers
    keep = (isinstance(df, pd.Series) or df.index.name is not None
            or not pd.api.types.is_integer_dtype(df.index))
    pd.DataFrame(df).to_csv(path, index=keep)
    return df


# ===========================================================================
# TASK 1
# ===========================================================================
def run_task1(ctx):
    banner("TASK 1 -- DATA & BACKTESTING INFRASTRUCTURE")
    print(ctx.summary())

    print("\n" + "-" * 100)
    print("RESEARCH-INTEGRITY CHECKS")
    print("-" * 100)

    checks = []

    # 1. the t-1 cutoff, verified row by row against the raw signal file
    ctx.features.validate_no_lookahead(ctx.decision, ctx.signals_clean, sample=400)
    checks.append(("Signal information is strictly older than the candle it trades",
                   f"PASS ({ctx.features.audit['lookahead_rows_checked']} rows re-derived "
                   f"from the raw file)"))

    # 2. no price or volume can reach a strategy
    leaked = [c for c in config.FORBIDDEN_STRATEGY_INPUTS if c in ctx.decision.columns]
    checks.append(("Decision frame exposes no price or volume column",
                   "PASS" if not leaked else f"FAIL {leaked}"))

    # 3. the vectorised engine equals the literal per-candle loop
    from backtester import Backtester
    diff = Backtester().assert_matches_reference(ctx.decision, BaselineStrategy(), ctx.market)
    checks.append(("Vectorised engine reproduces the per-candle reference loop",
                   f"PASS (max equity difference {diff:.2e})"))

    # 4. the mandated cost is exactly what a round trip costs
    from execution_engine import ExecutionEngine
    eng = ExecutionEngine()
    round_trip = eng.apply_transaction_cost(np.array([1.0])) + eng.apply_transaction_cost(np.array([1.0]))
    checks.append(("A full round trip costs 0.10% of notional",
                   f"PASS ({float(round_trip[0]) * 100:.3f}%)"))

    # 5. fills happen at the open
    checks.append(("Trades fill at candle t's open",
                   f"PASS (execution price field = '{config.EXECUTION_PRICE_FIELD}')"))

    # 6. development and holdout do not overlap
    dev, _ = ctx.slice("dev")
    hold, _ = ctx.slice("holdout")
    overlap = set(dev["date"]) & set(hold["date"])
    checks.append(("Development and holdout windows are disjoint",
                   "PASS" if not overlap else f"FAIL ({len(overlap)} shared dates)"))

    for name, status in checks:
        print(f"  [{status.split()[0]:<4}] {name:<62} {status}")

    print("\n" + "-" * 100)
    print("BASELINE STRATEGY")
    print("-" * 100)
    baseline = BaselineStrategy()
    res = ctx.run_strategy(baseline, split="full")
    print(f"  hypothesis: {baseline.hypothesis}")
    print()
    print(ctx.perf.report(res["returns"], positions=res["positions"],
                          costs=res["history"]["cost"], label="Baseline (PB01 long-only)"))
    print()
    print(ctx.perf.report(ctx.benchmark_returns("full"), label="Buy and hold (reference)"))

    save(res["history"], "task1_baseline_history")
    save(res["trade_log"], "task1_baseline_trades")
    with open(config.RESULTS_DIR / "task1_data_quality_report.txt", "w") as f:
        f.write(ctx.cleaner.get_report() + "\n\n" + ctx.features.get_report())

    print("\n  The baseline loses money after costs while the asset itself rose. That is the")
    print("  motivating result for Task 2: the obvious reading of a trend flag is wrong here.")
    return res


# ===========================================================================
# TASK 2
# ===========================================================================
def run_task2(ctx, quick=False):
    banner("TASK 2 -- ALPHA DISCOVERY & RESEARCH")

    alphas = build_all()
    ctx.fit_strategies(alphas, split="dev")

    print("CANDIDATE STRATEGIES (directions estimated on the development window only)")
    print("-" * 100)
    for key, s in alphas.items():
        md = s.get_metadata()
        kept = [f"{k}{v['sign']:+.0f}" for k, v in md["fitted"].items()
                if isinstance(v, dict) and v.get("sign", 0) != 0]
        print(f"  {key}  {md['name']}")
        print(f"      {md['hypothesis']}")
        print(f"      signals {', '.join(md['signals_used'])} | horizon "
              f"{md['fitted'].get('fit_horizon_days')}d | retained {', '.join(kept) or 'none'}")

    research = AlphaResearch(ctx, alphas)
    if quick:
        research.stats = StatisticalTester(n_boot=300)

    print("\n" + "-" * 100)
    print("DEVELOPMENT-WINDOW RESULTS")
    print("-" * 100)
    save(research.performance_table("dev"), "task2_performance_dev")
    save(research.significance_table("dev"), "task2_significance_dev")
    rob, rob_detail = research.robustness_suite("dev", noise_draws=5 if quick else 12)
    save(rob, "task2_robustness_dev")
    orth = research.orthogonality_suite("dev")
    save(orth["correlation"], "task2_correlation_dev")
    save(orth["pivoted_qr"], "task2_pivoted_qr_dev")
    save(orth["incremental"], "task2_incremental_dev")
    if not orth["stability"].empty:
        save(orth["stability"], "task2_qr_stability_dev")

    selected, scorecard = research.select("dev")
    save(scorecard, "task2_selection_scorecard")
    print(research.report("dev"))

    for key, detail in rob_detail.items():
        save(detail["subperiod"], f"task2_subperiod_{key}")
        save(detail["cost"], f"task2_cost_sensitivity_{key}")
        save(detail["regime"], f"task2_regime_{key}")
        save(detail["noise"], f"task2_noise_{key}")
        save(detail["failure"], f"task2_failure_{key}")
        if len(detail["parameters"]):
            save(detail["parameters"], f"task2_parameter_sweep_{key}")

    print("\n" + "-" * 100)
    print("HOLDOUT -- opened once, after the set above was frozen")
    print("-" * 100)
    hold = save(research.performance_table("holdout"), "task2_performance_holdout")
    print(hold.drop(columns=["name"]).to_string(index=False,
                                                float_format=lambda v: f"{v:8.3f}"))

    # ---- figures ----
    full_returns = {k: ctx.run_strategy(s, split="full")["returns"] for k, s in alphas.items()}
    plots.equity_curves(full_returns, "Candidate strategies -- net of the mandated cost",
                        "fig_task2_equity_curves",
                        benchmark=ctx.benchmark_returns("full"),
                        split_date=config.HOLDOUT_START_DATE)
    plots.drawdowns(full_returns, "Candidate strategies -- drawdowns", "fig_task2_drawdowns")
    plots.correlation_heatmap(orth["correlation"],
                              "Pairwise correlation of strategy returns (development)",
                              "fig_task2_correlation")
    plots.qr_contribution(orth["pivoted_qr"],
                          "Pivoted QR -- what each strategy adds that the earlier picks did not",
                          "fig_task2_qr_contribution")
    plots.variance_spectrum(orth["dimensionality"],
                            "Effective dimensionality of the strategy-return space",
                            "fig_task2_variance_spectrum")
    plots.cost_sensitivity({k: d["cost"] for k, d in rob_detail.items()},
                           "Cost sensitivity -- Sharpe against transaction cost",
                           "fig_task2_cost_sensitivity")

    dev_tab = research.tables["performance_dev"].set_index("strategy")
    rows = []
    for key in alphas:
        rows.append({
            "strategy": key,
            "dev_sharpe": float(dev_tab.at[key, "sharpe"]),
            "holdout_sharpe": float(hold.set_index("strategy").at[key, "sharpe"]),
            "selected": key in selected,
        })
    decay = pd.DataFrame(rows)
    decay["sharpe_decay"] = decay["dev_sharpe"] - decay["holdout_sharpe"]
    save(decay, "task2_dev_vs_holdout")
    print("\n[Development vs holdout -- the overfitting check]")
    print(decay.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    plots.dev_vs_holdout(decay, "Development vs holdout Sharpe", "fig_task2_dev_vs_holdout")

    return alphas, research, selected


# ===========================================================================
# TASK 3
# ===========================================================================
def vet_addition(ctx, alphas, universe, key, strat, quick=False):
    """Put a late addition through the same dev-window checks as the Task 2
    candidates. Nothing here decides anything; it's so its evidence can be
    read next to theirs."""
    print("\n" + "-" * 100)
    print(f"LATE ADDITION: {key} (written after the Task 2 holdout was opened)")
    print("-" * 100)
    md = strat.get_metadata()
    print(f"  {md['hypothesis']}")
    print(f"  settings chosen on dev : {md['fitted'].get('selection')}")
    signs = {k: v["sign"] for k, v in md["fitted"].items()
             if isinstance(v, dict) and "sign" in v}
    print(f"  overlay signs on dev   : {signs}")

    research = AlphaResearch(ctx, {key: strat})
    if quick:
        research.stats = StatisticalTester(n_boot=300)
    perf = research.performance_table("dev")
    sig = research.significance_table("dev", n_trials=len(alphas) + 1)
    rob, detail = research.robustness_suite("dev", noise_draws=5 if quick else 12)
    inc = OrthogonalityAnalyzer(ctx.returns_frame(universe, split="dev")).incremental_contribution()
    hold = research.performance_table("holdout")

    p, s, r = (perf.set_index("strategy").loc[key], sig.set_index("label").loc[key],
               rob.set_index("strategy").loc[key])

    # The deflated Sharpe above counts 7 hypotheses with a default spread. The
    # parameter grid is a second, bigger search, so deflate for that too using
    # the Sharpe spread actually observed across the grid.
    grid = detail[key]["parameters"]
    dev_r = ctx.run_strategy(strat, split="dev")["returns"]
    dsr_grid = research.stats.deflated_sharpe_ratio(
        dev_r, n_trials=len(grid), variance_of_trial_sharpes=float(grid["sharpe"].var(ddof=1)))
    card = {
        "strategy": key,
        "dev_sharpe": p["sharpe"], "dev_ann_return": p["ann_return"],
        "dev_max_drawdown": p["max_drawdown"],
        "nw_t": s["nw_t_stat"], "nw_p": s["nw_p_value"],
        "boot_ci_low": s["boot_ci_low"], "boot_ci_high": s["boot_ci_high"],
        "deflated_sharpe_prob": s["dsr_dsr"],
        "deflated_sharpe_prob_grid": dsr_grid["dsr"],
        "grid_expected_max_sharpe": dsr_grid["expected_max_sharpe_under_null"],
        "permutation_p": s["perm_p_value"],
        "breakeven_cost_bps": r["breakeven_cost_bps"], "sharpe_at_2x_cost": r["sharpe_at_2x_cost"],
        "subperiods_positive": r["pct_subperiods_positive"],
        "param_pct_positive": r["param_pct_positive"], "param_plateau": r["param_plateau_score"],
        "noise_retention_5pct": r["noise_retention_5pct"], "annual_turnover": r["annual_turnover"],
        "residual_sharpe_vs_book": float(inc.set_index("strategy").at[key, "residual_sharpe"]),
        "holdout_sharpe": float(hold.set_index("strategy").at[key, "sharpe"]),
        "holdout_ann_return": float(hold.set_index("strategy").at[key, "ann_return"]),
    }
    card = pd.DataFrame([card])
    print(card.T.to_string(header=False, float_format=lambda v: f"{v:10.3f}"))
    print("\n  Same gates as Task 2 would have applied: permutation p <= 0.30, break-even")
    print("  >= 5 bps per side, residual Sharpe > 0 after the rest of the book.")
    save(card, f"task3_addition_{key}_scorecard")
    for name, tab in detail[key].items():
        if len(tab):
            save(tab, f"task3_addition_{key}_{name}")
    save(inc, "task3_incremental_dev")
    return card


def run_task3(ctx, alphas, selected, quick=False):
    banner("TASK 3 -- ALPHA PORTFOLIO CONSTRUCTION & DYNAMIC ALLOCATION")

    task2_book = {k: alphas[k] for k in selected} if selected else dict(alphas)
    additions = build_task3_additions()
    ctx.fit_strategies(additions, split="dev")
    chosen = {**task2_book, **additions}
    print(f"  Task 2 set (frozen before the holdout) : {', '.join(task2_book)}")
    print(f"  Added at Task 3                        : {', '.join(additions)}")
    print(f"  Allocating across                      : {', '.join(chosen)}")

    for key, strat in additions.items():
        vet_addition(ctx, alphas, chosen, key, strat, quick=quick)

    R_dev = ctx.returns_frame(chosen, split="dev")
    R_hold = ctx.returns_frame(chosen, split="holdout")
    R_full = ctx.returns_frame(chosen, split="full")
    _, market_full = ctx.slice("full")

    # ---------------- factor model ----------------
    # Fitted on dev for the allocation work. The holdout fit is a diagnostic:
    # it shows whether holdout returns were alpha or just market exposure.
    for split, R in (("dev", R_dev), ("holdout", R_hold)):
        print("\n" + "-" * 100)
        print(f"[{split} window]")
        _, mkt = ctx.slice(split)
        fm = FactorModel().fit(R, FactorModel.build_factors(mkt))
        print(fm.report())
        save(fm.diagnostics_, f"task3_factor_model_{split}")
        if split == "dev":
            save(fm.risk_decomposition(), "task3_risk_decomposition_dev")

    # ---------------- static allocations ----------------
    print("\n" + "-" * 100)
    opt = PortfolioOptimizer(shrinkage=0.5, max_weight=0.60)
    opt.fit(R_dev)
    weights = opt.all_methods()
    weights.index.name = "strategy"
    save(weights, "task3_static_weights")
    risk_share = pd.DataFrame({m: opt.risk_contributions(weights[m]) for m in weights.columns})
    risk_share.index.name = "strategy"
    save(risk_share, "task3_risk_contributions_dev")
    print("STATIC ALLOCATIONS (fitted on the development window)")
    print(weights.to_string(float_format=lambda v: f"{v:7.3f}"))
    print("\n[Share of portfolio variance from each strategy -- concentration in risk terms]")
    print(risk_share.to_string(float_format=lambda v: f"{v:7.3f}"))
    print()
    print(opt.report())

    # ---------------- learned component ----------------
    print("\n" + "-" * 100)
    decision_full, _ = ctx.slice("full")
    state = AlphaMetaModel.build_market_state(decision_full)

    mm = AlphaMetaModel(block=21, min_train_blocks=6)
    X_dev, y_dev, meta_dev = mm.build_dataset(R_dev, state)
    mm.fit(None, None, X=X_dev, y=y_dev, blocks=meta_dev["block"].to_numpy())
    folds, _ = mm.walk_forward(X_dev, y_dev, meta_dev)
    null = mm.null_baseline(X_dev, y_dev, meta_dev, n_perm=50 if quick else 200)
    print(mm.report())
    save(folds, "task3_meta_walkforward")
    save(pd.DataFrame([{**null, **{k: v for k, v in folds.attrs.items()}}]),
         "task3_meta_null_baseline")
    save(mm.get_feature_importance(), "task3_meta_feature_importance")

    # ---------------- dynamic allocation ----------------
    print("\n" + "-" * 100)
    base_w = weights["risk_parity"]
    alloc_kwargs = dict(tilt_strength=0.5, blend_with_base=0.5,
                        no_trade_band=0.05, max_weight=0.50)

    # Run once, continuously, over the whole history. Restarting at the holdout
    # boundary would throw away everything the model had learned, which a live
    # allocator wouldn't do. Inside the holdout it is trained on dev plus the
    # holdout blocks that have already closed, all strictly in the past.
    X_full, y_full, meta_full = mm.build_dataset(R_full, state)
    allocator = DynamicAllocator(base_weights=base_w, **alloc_kwargs)
    dyn_full = allocator.run(R_full, AlphaMetaModel(block=21, min_train_blocks=6),
                             X_full, y_full, meta_full, base_weights=base_w,
                             min_train_blocks=6)
    print("DYNAMIC ALLOCATION (gated, run continuously across the whole history)")
    print(allocator.report())
    if dyn_full["first_live_date"] is not None:
        print(f"\n  first model-driven allocation: {dyn_full['first_live_date'].date()} "
              f"(earlier blocks hold the base weights while history builds up)")
    save(allocator.get_allocation_history(), "task3_allocation_history")

    # ---------------- allocation-level null ----------------
    # The gate keeps the dynamic method on its base weights, so the comparison
    # table can't tell us whether the model is any good. Here the gate is off
    # and the model's tilt is compared with random tilts of the same size.
    P_full = ctx.positions_frame(chosen, split="full")

    def evaluate(w_path):
        w = w_path.reindex(pd.DatetimeIndex(market_full["date"])).ffill()
        return pbt.run_allocation(P_full, w, market_full)["returns"]

    live = dyn_full["first_live_date"]
    windows = {"dev (after warm-up)": (live, pd.Timestamp(config.DEV_END_DATE)),
               "holdout": (pd.Timestamp(config.DEV_END_DATE), R_full.index[-1])}
    tn = tilt_null(R_full, X_full, y_full, meta_full, base_w, evaluate,
                   model_factory=lambda: AlphaMetaModel(block=21, min_train_blocks=6),
                   allocator_kwargs=alloc_kwargs, windows=windows,
                   n_draws=50 if quick else 200, min_train_blocks=6)
    print("\nALLOCATION-LEVEL NULL: ungated model tilt vs random tilts through the same rules")
    print(tn.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    save(tn, "task3_tilt_null")

    # ---------------- cost netting ----------------
    nb = pbt.netting_benefit(P_full, weights["equal_weight"], market_full)
    print("\nCOST NETTING (one account instead of N separate ones, equal weight, full history)")
    for k, v in nb.items():
        print(f"    {k:<34} {v:10.4f}")
    save(pd.DataFrame([nb]), "task3_netting_benefit")

    # ---------------- the required comparison ----------------
    dev_sharpes = {k: PerformanceAnalyzer().sharpe_ratio(R_dev[k]) for k in R_full.columns}
    best_key = max(dev_sharpes, key=dev_sharpes.get)
    task2_keys = list(task2_book)
    static = {
        "2_equal_weight": ("equal_weight", ""),
        "3_risk_based": ("risk_parity", "risk parity (equal risk contribution)"),
        "4_optimisation_based": ("max_sharpe", "shrunk mean-variance, max Sharpe"),
    }

    results, stress_rows, sig_rows = {}, [], []
    for split, R in (("dev", R_dev), ("holdout", R_hold), ("full", R_full)):
        _, market = ctx.slice(split)
        P = ctx.positions_frame(chosen, split=split)
        ev = FinalEvaluator()

        best = ctx.run_strategy(chosen[best_key], split=split)
        ev.add_method("1_best_individual", best["returns"], metrics=best["metrics"],
                      note=f"'{best_key}', picked on dev Sharpe")
        for key, (wcol, note) in static.items():
            res = pbt.run_allocation(P, weights[wcol], market, label=key)
            ev.add_method(key, res["returns"], weights=weights[wcol], note=note,
                          metrics=res["metrics"])

        dyn_w = dyn_full["weights"].reindex(pd.DatetimeIndex(market["date"])).ffill()
        dyn_res = pbt.run_allocation(P, dyn_w, market, label="5_dynamic")
        n_reb = int(sum(1 for h in allocator.get_allocation_history().to_dict("records")
                        if h["traded"] and h["date"] is not None
                        and market["date"].iloc[0] <= pd.Timestamp(h["date"]) <= market["date"].iloc[-1]))
        ev.add_method("5_dynamic", dyn_res["returns"], weights=dyn_w, metrics=dyn_res["metrics"],
                      note=f"rebalanced {n_reb}x in this window; skill-gated tilt on risk parity")

        ew2 = pd.Series(1.0 / len(task2_keys), index=task2_keys)
        ref = pbt.run_allocation(P[task2_keys], ew2, market, label="task2_book")
        ev.add_method("R_task2_book_ew", ref["returns"], metrics=ref["metrics"],
                      note="reference: equal weight over the frozen Task 2 set, no alpha_07")
        ev.add_method("R_buy_and_hold", ctx.benchmark_returns(split),
                      note="reference, not a required baseline")

        comp = ev.compare(benchmark="2_equal_weight")
        ev.robustness_test()
        results[split] = {"evaluator": ev, "comparison": comp}
        save(comp, f"task3_comparison_{split}")
        save(ev.robustness_, f"task3_subperiods_{split}")

        print(f"\nALLOCATION COMPARISON -- {split.upper()}")
        if split == "dev":
            print("  (static weights were fitted on this window, so these are in-sample)")
        print(ev.report(title=f"FINAL EVALUATION -- {split}"))

        if split in ("holdout", "full"):
            pairs = ([("5_dynamic", b) for b in ("1_best_individual", "2_equal_weight",
                                                 "3_risk_based", "4_optimisation_based")]
                     + [(m, "1_best_individual") for m in ("2_equal_weight", "3_risk_based",
                                                           "4_optimisation_based")]
                     + [("2_equal_weight", "3_risk_based"),
                        ("2_equal_weight", "4_optimisation_based"),
                        ("2_equal_weight", "R_task2_book_ew")])
            print(f"\n[{split}: are the differences distinguishable from noise?]")
            for a, b in pairs:
                st = ev.significance_vs(a, b)
                sig_rows.append({"split": split, **st})
                print(f"    {a:<22} vs {b:<22} excess {st['mean_excess_ann']:+.2%}/yr  "
                      f"NW t {st['nw_t_stat']:+.2f}  p {st['nw_p_value']:.3f}")

            # the organisers may re-run with harsher execution assumptions
            for mult in (1, 2, 4):
                c = config.TRANSACTION_COST * mult
                row = {"split": split, "cost_multiple": mult, "cost_bps_per_side": c * 1e4,
                       "1_best_individual": ctx.run_strategy(chosen[best_key], split=split,
                                                             cost=c)["metrics"]["sharpe"]}
                for key, (wcol, _) in static.items():
                    row[key] = pbt.run_allocation(P, weights[wcol], market, cost=c)["metrics"]["sharpe"]
                row["5_dynamic"] = pbt.run_allocation(P, dyn_w, market, cost=c)["metrics"]["sharpe"]
                stress_rows.append(row)

    stress = pd.DataFrame(stress_rows)
    print("\n[Sharpe under higher transaction costs]")
    print(stress.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    save(stress, "task3_cost_stress")
    save(pd.DataFrame(sig_rows), "task3_significance")
    results["full"]["dynamic"] = dyn_full

    # ---- figures ----
    plots.walk_forward_folds(folds, "Meta-model walk-forward folds -- train vs out-of-sample IC",
                             "fig_task3_walkforward")
    plots.null_baseline(null, "Learned component against its permuted-label null",
                        "fig_task3_null_baseline", null_draws=mm.null_draws_)
    plots.tilt_null(tn, "Ungated meta-model tilt vs random tilts (same rules)",
                    "fig_task3_tilt_null")
    plots.weight_path(dyn_full["weights"], "Dynamic allocation -- weight path",
                      "fig_task3_weight_path")
    plots.allocation_comparison(results["holdout"]["comparison"],
                                "Allocation methods on the holdout -- Sharpe",
                                "fig_task3_allocation_holdout")
    hold_methods = {k: v["returns"] for k, v in results["holdout"]["evaluator"].methods_.items()}
    plots.equity_curves(hold_methods, "Allocation methods -- holdout", "fig_task3_equity_holdout")
    full_methods = {k: v["returns"] for k, v in results["full"]["evaluator"].methods_.items()}
    plots.equity_curves(full_methods, "Allocation methods -- full history",
                        "fig_task3_equity_full", split_date=config.HOLDOUT_START_DATE)
    plots.drawdowns(full_methods, "Allocation methods -- drawdowns", "fig_task3_drawdowns")
    save(dyn_full["weights"], "task3_dynamic_weight_path")
    return results


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Multi-Alpha Research Lab")
    ap.add_argument("--task", type=int, choices=[1, 2, 3], default=None,
                    help="run a single task instead of all three")
    ap.add_argument("--quick", action="store_true",
                    help="fewer bootstrap/permutation draws for a fast check")
    args = ap.parse_args()

    log = Tee(config.RESULTS_DIR / "run_log.txt")
    sys.stdout = log
    np.random.seed(config.RANDOM_SEED)

    try:
        print(f"Multi-Alpha Research Lab -- run started {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"Transaction cost {config.TRANSACTION_COST:.4%} per side | "
              f"fills at candle t's {config.EXECUTION_PRICE_FIELD} | "
              f"signals lagged {config.SIGNAL_LAG} candle")

        ctx = ResearchContext()
        alphas = research = selected = None

        if args.task in (None, 1):
            run_task1(ctx)
        if args.task in (None, 2, 3):
            alphas, research, selected = run_task2(ctx, quick=args.quick)
        if args.task in (None, 3):
            run_task3(ctx, alphas, selected, quick=args.quick)

        banner("RUN COMPLETE")
        print(f"  Tables written to {config.RESULTS_DIR}")
        print(f"  Console log      {config.RESULTS_DIR / 'run_log.txt'}")
    finally:
        sys.stdout = log.stdout
        log.close()


if __name__ == "__main__":
    main()
