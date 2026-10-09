"""
run_v2.py - the post-competition v2 research, in two separate stages.

    python run_v2.py --stage dev       # walk-forward on dev, pick and freeze v2,
                                       # run the bias checks. Never reads 2021.
    python run_v2.py --stage holdout   # score the frozen v2 on 2021, once

main.py is untouched by this and still reproduces the submitted results.
Outputs go to results/v2/ and results/figures/fig_v2_*.png.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

import config
import plots
import v2_portfolio as v2
from main import Tee
from research_context import ResearchContext

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
fmt = lambda v: f"{v:8.3f}"   # noqa: E731

# Rough count of alpha_08 variants looked at on dev before settling on the
# final rule (reversal and continuation versions, thresholds, gates). It goes
# into the deflated Sharpe so the winner is judged as the best of that many.
ALPHA08_VARIANTS_TRIED = 18

# Same idea for the book: the 30 configurations in the walk-forward grid,
# plus about 30 more overlay/weighting variants looked at on dev while
# deciding what to put in the grid.
BOOK_VARIANTS_TRIED = 60


def save(df, name):
    v2.V2_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(v2.V2_DIR / f"{name}.csv", index=False)
    return df


def stage_dev(ctx):
    print("=" * 90)
    print("v2 DEV STAGE: walk-forward inside 2018-2020. The holdout is not read.")
    print("=" * 90)

    folds = v2.walk_forward(ctx)
    save(folds, "walk_forward_folds")
    cfg, wide = v2.select(folds)
    save(wide.reset_index(), "walk_forward_selection")
    print("\nValidation Sharpe by configuration (fold = validation year):")
    print(wide.to_string(float_format=fmt))
    ref = folds[folds["config"] == "v1 as submitted (unscaled)"].set_index("fold")["sharpe"]
    print(f"\nv1 as submitted, unscaled: {', '.join(f'{y}: {s:.3f}' for y, s in ref.items())}")
    print(f"\nSelected: {v2.config_name(cfg)}")

    frozen = v2.freeze(ctx, cfg, wide)
    print("Frozen to results/v2/frozen_config.json")
    print(json.dumps({k: frozen[k] for k in ("config", "weights", "scale", "dev_sharpe")},
                     indent=2, default=str))

    print("\n" + "-" * 90)
    print("BIAS CHECKS (dev window)")
    print("-" * 90)
    strategies, P, pos, info = v2.fit_frozen(ctx, cfg)
    from statistics import StatisticalTester
    dev_r = v2.run_position(ctx, pos, v2.date_mask(ctx, end=config.DEV_END_DATE))["returns"]
    dsr = StatisticalTester().deflated_sharpe_ratio(
        dev_r, n_trials=BOOK_VARIANTS_TRIED,
        variance_of_trial_sharpes=float(wide["mean_val_sharpe"].var(ddof=1)))
    print(f"  dev Sharpe of the frozen book {dsr['observed_sharpe']:.2f} is in-sample. Best of "
          f"{BOOK_VARIANTS_TRIED} by luck would be ~{dsr['expected_max_sharpe_under_null']:.2f}; "
          f"probability it beats that: {dsr['dsr']:.2f}")
    save(pd.DataFrame([dsr]), "deflated_sharpe_dev")
    cause = v2.causality_check(ctx, cfg, strategies)
    print(f"  truncation test: {cause['status']} ({cause['cuts_checked']} cut dates, "
          f"max change {cause['max_position_change']:.1e})")
    shifted = v2.shifted_signal_check(ctx, cfg, strategies, info)
    print(shifted.to_string(index=False, float_format=fmt))
    save(pd.DataFrame([cause]), "bias_truncation")
    save(shifted, "bias_shifted_signals")

    print("\n" + "-" * 90)
    print("alpha_08 on its own (same dev battery as the Task 2 strategies)")
    print("-" * 90)
    card, detail = v2.alpha08_scorecard(ctx, ALPHA08_VARIANTS_TRIED)
    fitted = card.pop("fitted")
    print(pd.Series(card).to_string(float_format=fmt))
    print(f"  fitted: {fitted}")
    save(pd.DataFrame([card]), "alpha_08_scorecard")
    for name, tab in detail.items():
        if len(tab):
            save(tab.reset_index(drop=True) if name != "subperiod" else tab.reset_index(),
                 f"alpha_08_{name}")

    plots_walk_forward(wide)
    if cause["status"] != "PASS":
        raise SystemExit("Truncation test failed; not continuing.")


def stage_holdout(ctx):
    print("=" * 90)
    print("v2 HOLDOUT STAGE: the frozen configuration, scored on 2021")
    print("=" * 90)
    frozen = v2.load_frozen()
    cfg, strategies, P, pos = v2.rebuild_frozen(ctx, frozen)
    print(f"  frozen config: {v2.config_name(cfg)}  (weights and scale re-derived and matched)")

    dev_mask = v2.date_mask(ctx, end=config.DEV_END_DATE)
    v1_scaled, _ = v2.build_book(ctx, v2.DEFAULT, P, dev_mask)
    books = {
        "v1 as submitted": v2.unscaled_v1(P),
        "v1 scaled to 10% vol": v1_scaled,
        "v2": pos,
    }
    splits = {
        "dev": dev_mask,
        "holdout": v2.date_mask(ctx, start=config.HOLDOUT_START_DATE),
        "full": v2.date_mask(ctx),
    }

    table = save(v2.book_table(ctx, books, splits), "v1_vs_v2")
    print("\n" + table.to_string(index=False, float_format=fmt))

    sig = pd.concat([v2.significance(ctx, books, splits[s]).assign(split=s)
                     for s in ("dev", "holdout")], ignore_index=True)
    save(sig, "significance")
    print("\nSignificance (Newey-West t, stationary-bootstrap Sharpe CI, block permutation):")
    print(sig.to_string(index=False, float_format=fmt))

    stress = save(v2.cost_stress(ctx, books, {k: splits[k] for k in ("dev", "holdout")}),
                  "cost_stress")
    print("\nSharpe at 1x / 2x / 4x the mandated cost:")
    print(stress.to_string(index=False, float_format=fmt))

    # figures
    returns = {k: v2.run_position(ctx, p, splits["full"])["returns"] for k, p in books.items()}
    plots.equity_curves(returns, "v1 vs v2, net of costs (dev 2018-2020, holdout 2021)",
                        "fig_v2_equity_full", benchmark=ctx.benchmark_returns("full"),
                        split_date=config.HOLDOUT_START_DATE)
    plots.drawdowns(returns, "v1 vs v2, drawdowns", "fig_v2_drawdowns")
    hold = {k: v2.run_position(ctx, p, splits["holdout"])["returns"] for k, p in books.items()}
    plots.equity_curves(hold, "v1 vs v2 on the 2021 holdout", "fig_v2_equity_holdout",
                        benchmark=ctx.benchmark_returns("holdout"))


def plots_walk_forward(wide):
    import matplotlib.pyplot as plt
    years = [y for _, y in v2.FOLDS]
    top = wide.copy()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(top))))
    y = np.arange(len(top))
    for i, yr in enumerate(years):
        ax.barh(y + (i - 0.5) * 0.4, top[yr], height=0.4, label=f"validate {yr}")
    default = v2.config_name(v2.DEFAULT)
    labels = [f"{c}  *" if c == default else c for c in top.index]
    ax.set_yticks(y, labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("validation Sharpe (net of costs)")
    ax.set_title("v2 walk-forward inside dev (* = v1 weighting, scaled)")
    ax.legend(loc="lower right")
    plots._save(fig, "fig_v2_walkforward")


def main():
    ap = argparse.ArgumentParser(description="v2 research, dev and holdout stages")
    ap.add_argument("--stage", choices=["dev", "holdout"], required=True)
    args = ap.parse_args()
    np.random.seed(config.RANDOM_SEED)
    v2.V2_DIR.mkdir(parents=True, exist_ok=True)
    log = Tee(v2.V2_DIR / f"{args.stage}_stage_log.txt")
    sys.stdout = log
    try:
        ctx = ResearchContext(verbose=False)
        if args.stage == "dev":
            stage_dev(ctx)
        else:
            stage_holdout(ctx)
    finally:
        sys.stdout = log.stdout
        log.close()


if __name__ == "__main__":
    sys.exit(main())
