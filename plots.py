"""
Figure generation for the research reports.

Kept out of the analysis modules on purpose: nothing here computes a number that
appears in a table. Every figure is drawn from a series that has already been
saved to results/, so a chart and the table beside it can never disagree.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
})

PALETTE = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#b83280", "#4a5568", "#d69e2e"]


def _save(fig, name):
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def equity_curves(returns_dict, title, name, benchmark=None, split_date=None):
    """Cumulative growth of one unit, log scale."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (label, r) in enumerate(returns_dict.items()):
        eq = (1.0 + pd.Series(r).fillna(0.0)).cumprod()
        ax.plot(eq.index, eq.values, label=label, color=PALETTE[i % len(PALETTE)], lw=1.4)
    if benchmark is not None:
        eq = (1.0 + pd.Series(benchmark).fillna(0.0)).cumprod()
        ax.plot(eq.index, eq.values, label="buy and hold", color="#a0aec0",
                lw=1.2, ls="--")
    if split_date is not None:
        ax.axvline(pd.Timestamp(split_date), color="#e53e3e", lw=1.0, ls=":")
        ax.text(pd.Timestamp(split_date), ax.get_ylim()[1], "  holdout begins",
                color="#e53e3e", va="top", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("growth of 1 unit (log scale)")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    return _save(fig, name)


def drawdowns(returns_dict, title, name):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for i, (label, r) in enumerate(returns_dict.items()):
        eq = (1.0 + pd.Series(r).fillna(0.0)).cumprod()
        dd = eq / eq.cummax() - 1.0
        ax.plot(dd.index, dd.values * 100, label=label, color=PALETTE[i % len(PALETTE)], lw=1.2)
    ax.set_ylabel("drawdown (%)")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    return _save(fig, name)


def correlation_heatmap(corr, title, name):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.55 else "black")
    ax.grid(False)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, name)


def qr_contribution(qr_table, title, name):
    """How much each strategy adds that the ones before it did not have."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    d = qr_table.sort_values("selection_order")
    x = np.arange(len(d))
    ax.bar(x, d["independent_fraction"] * 100, color=PALETTE[0], label="new information")
    ax.bar(x, d["explained_by_earlier"] * 100, bottom=d["independent_fraction"] * 100,
           color="#cbd5e0", label="already spanned by earlier picks")
    ax.set_xticks(x)
    ax.set_xticklabels(d["strategy"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% of the strategy's own variation")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, name)


def variance_spectrum(dimensionality, title, name):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    share = np.array(dimensionality["variance_share"]) * 100
    cum = np.array(dimensionality["cumulative_variance"]) * 100
    x = np.arange(1, len(share) + 1)
    ax.bar(x, share, color=PALETTE[0], label="variance explained")
    ax.plot(x, cum, color=PALETTE[1], marker="o", ms=4, lw=1.4, label="cumulative")
    ax.axhline(95, color="#718096", ls=":", lw=1)
    ax.set_xlabel("direction in the strategy-return space")
    ax.set_ylabel("% of variance")
    ax.set_title(title + f"  (participation ratio {dimensionality['participation_ratio']:.2f})")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, name)


def cost_sensitivity(cost_tables, title, name):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (label, d) in enumerate(cost_tables.items()):
        d = d.dropna(subset=["cost_multiple"]).sort_values("cost_per_side_bps")
        ax.plot(d["cost_per_side_bps"], d["sharpe"], marker="o", ms=3.5,
                color=PALETTE[i % len(PALETTE)], label=label, lw=1.3)
    ax.axvline(config.TRANSACTION_COST * 1e4, color="#e53e3e", ls="--", lw=1.1)
    ax.text(config.TRANSACTION_COST * 1e4, ax.get_ylim()[1], " mandated 5 bps",
            color="#e53e3e", va="top", fontsize=8)
    ax.axhline(0, color="#718096", lw=0.8)
    ax.set_xlabel("transaction cost per side (bps)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    return _save(fig, name)


def dev_vs_holdout(decay_table, title, name):
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for _, row in decay_table.iterrows():
        colour = PALETTE[2] if row["selected"] else "#a0aec0"
        ax.plot([0, 1], [row["dev_sharpe"], row["holdout_sharpe"]],
                marker="o", ms=5, color=colour, lw=1.5)
        ax.text(1.02, row["holdout_sharpe"], row["strategy"], fontsize=8,
                va="center", color=colour)
    ax.axhline(0, color="#718096", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["development", "holdout"])
    ax.set_xlim(-0.15, 1.45)
    ax.set_ylabel("Sharpe ratio")
    ax.set_title(title + "\n(green = selected; a line sloping down is decay)")
    return _save(fig, name)


def allocation_comparison(comparison, title, name, metric="sharpe"):
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    d = comparison.sort_values("method")
    colours = [PALETTE[0] if not m.startswith("5") else PALETTE[2] for m in d["method"]]
    ax.bar(d["method"], d[metric], color=colours)
    ax.axhline(0, color="#718096", lw=0.8)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    return _save(fig, name)


def weight_path(weights, title, name):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    w = pd.DataFrame(weights)
    ax.stackplot(w.index, *[w[c].values for c in w.columns], labels=list(w.columns),
                 colors=PALETTE[: len(w.columns)], alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_ylabel("portfolio weight")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower left")
    return _save(fig, name)


def walk_forward_folds(folds, title, name):
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.bar(folds["fold_block"] - 0.2, folds["train_ic"], width=0.4,
           color=PALETTE[0], label="train IC")
    ax.bar(folds["fold_block"] + 0.2, folds["test_ic"], width=0.4,
           color=PALETTE[1], label="test IC (out of sample)")
    ax.axhline(0, color="#718096", lw=0.8)
    ax.set_xlabel("walk-forward fold (rebalance block)")
    ax.set_ylabel("information coefficient")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, name)


def null_baseline(null_result, title, name, null_draws=None):
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    if null_draws is not None and len(null_draws):
        ax.hist(null_draws, bins=25, color="#cbd5e0", edgecolor="white",
                label="permuted-label null")
    ax.axvline(null_result["real_mean_test_ic"], color=PALETTE[1], lw=2,
               label=f"actual model ({null_result['real_mean_test_ic']:+.3f})")
    ax.axvline(null_result["null_p95_test_ic"], color=PALETTE[4], lw=1.4, ls="--",
               label=f"null 95th pct ({null_result['null_p95_test_ic']:+.3f})")
    ax.set_xlabel("mean out-of-sample information coefficient")
    ax.set_ylabel("permutations")
    ax.set_title(title + f"   p = {null_result['p_value']:.3f}")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, name)


def tilt_null(null_table, title, name):
    """Ungated model tilt vs random tilts, one panel per window."""
    draws = null_table.attrs.get("null_draws", {})
    n = len(null_table)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.3), squeeze=False)
    for ax, (_, row) in zip(axes[0], null_table.iterrows()):
        d = draws.get(row["window"], [])
        if len(d):
            ax.hist(d, bins=25, color="#cbd5e0", edgecolor="white", label="random scores")
        ax.axvline(row["model_sharpe_minus_base"], color=PALETTE[1], lw=2,
                   label=f"meta-model ({row['model_sharpe_minus_base']:+.2f})")
        ax.axvline(row["null_p95"], color=PALETTE[4], lw=1.4, ls="--",
                   label=f"null 95th pct ({row['null_p95']:+.2f})")
        ax.set_title(f"{row['window']}   p = {row['p_value']:.2f}", fontsize=9)
        ax.set_xlabel("Sharpe minus base allocation's Sharpe")
        ax.legend(frameon=False, fontsize=7)
    axes[0][0].set_ylabel("draws")
    fig.suptitle(title, fontsize=10)
    return _save(fig, name)
