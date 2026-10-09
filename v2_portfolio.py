"""
v2 portfolio - research done after the competition, on the dev window only.

v1 (submitted) is the equal-weight book of alpha_02, alpha_03, alpha_05 and
alpha_07. v2 asks whether a few simple changes improve it:

    which strategies   v1 set, or v1 set + alpha_08 (next-day continuation)
    how to weight      equal, inverse volatility, or Sharpe-weighted shrunk
                       halfway back to equal
    an overlay         none, a volatility-regime tilt, volatility targeting,
                       or a no-trade band (two widths) to cut turnover

Every combination is also scaled to a fixed 10% annual volatility target.
The target is a risk choice, not a fitted number: v1 ran at about 7% vol on
dev and only used a third of the position limit, so scaling is the plain
way to use more of it. Sharpe does not change with scaling, so it is the
number to judge skill by. Return does change, so v1 scaled the same way is
reported next to v2 as a control.

How a configuration is chosen (decided before running anything):
  1. Walk forward inside dev. Fold 1 fits on 2018 and scores 2019. Fold 2
     fits on 2018-2019 and scores 2020. Strategies are refitted in each fold.
  2. Score = average validation Sharpe across the two folds.
  3. A configuration only replaces "v1 set, equal weight, no overlay" if it
     beats it in both folds. Otherwise v2 is just v1 with the scaling.
  4. The winner is refitted on the whole dev window and frozen to
     results/v2/frozen_config.json. Only then is the holdout scored.

Holdout (2021) numbers are never used to choose anything here.
"""

import json
from itertools import product

import numpy as np
import pandas as pd

import config
import portfolio_backtest as pbt
from strategies import ALPHA_REGISTRY, TASK3_ADDITIONS, V2_ADDITIONS

V1_KEYS = ["alpha_02", "alpha_03", "alpha_05", "alpha_07"]
STRATEGY_SETS = {
    "v1_set": V1_KEYS,
    "v1_set+alpha_08": V1_KEYS + ["alpha_08"],
}
WEIGHTINGS = ("equal", "inverse_vol", "sharpe_shrunk")
OVERLAYS = ("none", "regime_tilt", "vol_target", "band_05", "band_10")

TARGET_VOL = 0.10           # annualised, for every configuration
MAX_SCALE = 4.0             # never lever the raw book more than this
REGIME_WINDOW = 126         # trailing window for the BB07 median
VOL_WINDOW = 63             # trailing window for realised book vol

# (last fit date, validation year)
FOLDS = (("2018-12-31", 2019), ("2019-12-31", 2020))

DEFAULT = ("v1_set", "equal", "none")
V2_DIR = config.RESULTS_DIR / "v2"


def all_configs():
    return list(product(STRATEGY_SETS, WEIGHTINGS, OVERLAYS))


def config_name(cfg):
    return " / ".join(cfg)


# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------
def new_strategies(keys):
    registry = {**ALPHA_REGISTRY, **TASK3_ADDITIONS, **V2_ADDITIONS}
    return {k: registry[k]() for k in keys}


def fit_until(ctx, strategies, last_fit_date):
    """Fit strategies on data up to `last_fit_date`, which must be inside dev.

    Same as ResearchContext.fit_strategies, but for any cut-off inside dev, so
    the walk-forward can fit on 2018 only, or on 2018-2019. The forward-return
    label is built inside the window, so the last few rows have no label
    rather than one that peeks past the cut-off.
    """
    end = pd.Timestamp(last_fit_date)
    if end > pd.Timestamp(config.DEV_END_DATE):
        raise ValueError(f"Refusing to fit past the dev window ({end.date()}).")
    mask = (ctx.decision["date"] <= end).to_numpy()
    inputs = ctx.decision.loc[mask].drop(columns=["information_asof"]).reset_index(drop=True)
    opens = ctx.market.loc[mask, "open"].reset_index(drop=True)
    daily = opens.shift(-1) / opens - 1.0
    for s in strategies.values():
        h = s.fit_horizon
        s.fit(inputs, opens.shift(-h) / opens - 1.0, daily_target=daily)
    return strategies


def position_frame(ctx, strategies, decision=None):
    """Each strategy's position on every date (generated on the full history,
    like the rest of the project, so trailing windows don't restart)."""
    decision = ctx.decision if decision is None else decision
    inputs = decision.drop(columns=["information_asof"], errors="ignore")
    out = {k: s.generate_signal(inputs).to_numpy() for k, s in strategies.items()}
    return pd.DataFrame(out, index=pd.DatetimeIndex(decision["date"]))


# ---------------------------------------------------------------------------
# running a position series through the project's own engine
# ---------------------------------------------------------------------------
def run_position(ctx, position, mask, cost=None):
    """Backtest one position series on the dates in `mask`.

    Goes through portfolio_backtest.run_allocation, so it pays the mandated
    cost, fills at the open and starts the window flat, like every other
    number in the project.
    """
    mask = np.asarray(mask, dtype=bool)
    market = ctx.market.loc[mask].reset_index(drop=True)
    P = pd.DataFrame({"book": np.asarray(position, dtype=float)[mask]})
    return pbt.run_allocation(P, pd.Series({"book": 1.0}), market, cost=cost)


def date_mask(ctx, start=None, end=None):
    d = ctx.market["date"]
    m = pd.Series(True, index=d.index)
    if start is not None:
        m &= d >= pd.Timestamp(start)
    if end is not None:
        m &= d <= pd.Timestamp(end)
    return m.to_numpy()


# ---------------------------------------------------------------------------
# building a book
# ---------------------------------------------------------------------------
def weights_for(scheme, fit_returns):
    """Strategy weights from returns on the fit window only."""
    keys = list(fit_returns.columns)
    equal = pd.Series(1.0 / len(keys), index=keys)
    if scheme == "equal":
        return equal
    if scheme == "inverse_vol":
        inv = 1.0 / fit_returns.std(ddof=1).replace(0.0, np.nan)
        return (inv / inv.sum()).fillna(0.0)
    if scheme == "sharpe_shrunk":
        sr = (fit_returns.mean() / fit_returns.std(ddof=1)).clip(lower=0.0).fillna(0.0)
        if sr.sum() <= 0:
            return equal
        return 0.5 * equal + 0.5 * sr / sr.sum()
    raise ValueError(scheme)


def regime_multiplier(decision):
    """1.5 when BB07 (recent volatility) is above its trailing median, else 0.5.

    Task 2 found reversion pays more when volatility is high (alpha_06's
    idea), so this sizes the whole book up in those periods. BB07 is a
    signal, already lagged a day, so this uses no price.
    """
    vol = pd.Series(decision["BB07"].to_numpy(), dtype=float)
    med = vol.rolling(REGIME_WINDOW, min_periods=REGIME_WINDOW // 3).median()
    high = (vol > med).astype(float)
    return 0.5 + high.to_numpy()


def trailing_book_vol(base_position, market):
    """Recent realised vol of the book's own P&L, as known at each decision.

    Shifted two days: the return from open t-2 to open t-1 is the latest one
    fully known when the decision for open t is made. Using the book's own
    past P&L to size it is the same reading of the rules as the Task 3
    allocator (a strategy's own P&L can size it; price can't drive a signal).
    """
    pnl = pd.Series(np.asarray(base_position) * market["ret_oo"].fillna(0.0).to_numpy())
    return pnl.shift(2).rolling(VOL_WINDOW, min_periods=20).std().to_numpy()


def vol_target_multiplier(base_position, market, fit_mask):
    """Scale down when the book's recent vol is high, up when it is low (at
    most 2x). The reference level is the median over the fit window."""
    vol = trailing_book_vol(base_position, market)
    ref = float(np.nanmedian(vol[fit_mask]))
    return np.nan_to_num(np.minimum(ref / vol, 2.0), nan=1.0)


def no_trade_band(position, width):
    """Only trade when the target moves more than `width` away from what we
    hold, and then only to the edge of the band. Cuts small, frequent trades."""
    target = np.asarray(position, dtype=float)
    out = np.zeros_like(target)
    held = 0.0
    for i, x in enumerate(target):
        if abs(x - held) > width:
            held = x - np.sign(x - held) * width
        out[i] = held
    return out


def build_book(ctx, cfg, P, fit_mask, decision=None, market=None):
    """Turn a configuration into one position series.

    Weights, the vol-target reference and the scale factor all come from the
    fit window. Returns (position, details).
    """
    decision = ctx.decision if decision is None else decision
    market = ctx.market if market is None else market
    set_name, scheme, overlay = cfg
    keys = STRATEGY_SETS[set_name]

    fit_returns = pd.DataFrame({k: run_position(ctx, P[k].to_numpy(), fit_mask)["returns"].to_numpy()
                                for k in keys})
    w = weights_for(scheme, fit_returns)
    base = (P[keys] * w).sum(axis=1).to_numpy()

    if overlay == "regime_tilt":
        base = base * regime_multiplier(decision)
    elif overlay == "vol_target":
        base = base * vol_target_multiplier(base, market, fit_mask[:len(base)])

    fit_vol = run_position(ctx, base, fit_mask)["metrics"]["volatility"]
    scale = float(np.clip(TARGET_VOL / fit_vol, 0.0, MAX_SCALE)) if fit_vol > 0 else 1.0
    pos = np.clip(scale * base, -config.MAX_GROSS_POSITION, config.MAX_GROSS_POSITION)

    if overlay.startswith("band_"):
        pos = no_trade_band(pos, int(overlay.split("_")[1]) / 100.0)

    return pos, {"weights": w.to_dict(), "scale": scale}


def unscaled_v1(P):
    """The submitted book: equal weight over the four v1 strategies."""
    return P[V1_KEYS].mean(axis=1).to_numpy()


# ---------------------------------------------------------------------------
# 1. walk-forward inside dev
# ---------------------------------------------------------------------------
def summarise(res):
    m = res["metrics"]
    return {
        "ann_return": m["annualized_return"],
        "volatility": m["volatility"],
        "sharpe": m["sharpe"],
        "max_drawdown": m["max_drawdown"],
        "annual_turnover": m.get("annual_turnover", np.nan),
        "hit_rate_active": m.get("hit_rate_active", np.nan),
    }


def walk_forward(ctx):
    """Score every configuration on the two dev validation years."""
    rows = []
    keys = sorted(set(sum(STRATEGY_SETS.values(), [])))
    for last_fit, val_year in FOLDS:
        strategies = fit_until(ctx, new_strategies(keys), last_fit)
        P = position_frame(ctx, strategies)
        fit_mask = date_mask(ctx, end=last_fit)
        val_mask = date_mask(ctx, f"{val_year}-01-01", f"{val_year}-12-31")

        ref = run_position(ctx, unscaled_v1(P), val_mask)
        rows.append({"fold": val_year, "config": "v1 as submitted (unscaled)",
                     "set": "v1_set", "weighting": "equal", "overlay": "none",
                     "scale": 1.0, **summarise(ref)})
        for cfg in all_configs():
            pos, info = build_book(ctx, cfg, P, fit_mask)
            res = run_position(ctx, pos, val_mask)
            rows.append({"fold": val_year, "config": config_name(cfg), "set": cfg[0],
                         "weighting": cfg[1], "overlay": cfg[2], "scale": info["scale"],
                         **summarise(res)})
    return pd.DataFrame(rows)


def select(folds):
    """Apply the selection rule written at the top of this file."""
    grid = folds[folds["config"] != "v1 as submitted (unscaled)"]
    wide = grid.pivot_table(index="config", columns="fold", values="sharpe")
    wide["mean_val_sharpe"] = wide.mean(axis=1)
    default = wide.loc[config_name(DEFAULT)]
    years = [y for _, y in FOLDS]
    wide["beats_default_both_folds"] = (wide[years] > default[years]).all(axis=1)
    wide = wide.sort_values("mean_val_sharpe", ascending=False)

    eligible = wide[wide["beats_default_both_folds"]]
    chosen = eligible.index[0] if len(eligible) else config_name(DEFAULT)
    return tuple(chosen.split(" / ")), wide


# ---------------------------------------------------------------------------
# 2. freeze on the full dev window
# ---------------------------------------------------------------------------
def fit_frozen(ctx, cfg):
    """Refit the chosen configuration's strategies on the whole dev window."""
    keys = sorted(set(STRATEGY_SETS[cfg[0]]) | set(V1_KEYS))
    strategies = new_strategies(keys)
    ctx.fit_strategies(strategies, split="dev")
    P = position_frame(ctx, strategies)
    dev_mask = date_mask(ctx, end=config.DEV_END_DATE)
    pos, info = build_book(ctx, cfg, P, dev_mask)
    return strategies, P, pos, info


def freeze(ctx, cfg, wide):
    strategies, P, pos, info = fit_frozen(ctx, cfg)
    dev = run_position(ctx, pos, date_mask(ctx, end=config.DEV_END_DATE))
    frozen = {
        "config": {"set": cfg[0], "weighting": cfg[1], "overlay": cfg[2]},
        "strategies": STRATEGY_SETS[cfg[0]],
        "weights": info["weights"],
        "scale": info["scale"],
        "target_vol": TARGET_VOL,
        "dev_sharpe": dev["metrics"]["sharpe"],
        "fitted": {k: s.get_metadata()["fitted"] for k, s in strategies.items()
                   if k in STRATEGY_SETS[cfg[0]]},
        "walk_forward_mean_val_sharpe": float(wide.loc[config_name(cfg), "mean_val_sharpe"]),
        "configs_compared": len(wide),
    }
    V2_DIR.mkdir(parents=True, exist_ok=True)
    with open(V2_DIR / "frozen_config.json", "w") as f:
        json.dump(frozen, f, indent=2, default=_jsonable)
    return frozen


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    return str(x)


def load_frozen():
    path = V2_DIR / "frozen_config.json"
    if not path.exists():
        raise FileNotFoundError("No frozen v2 config. Run `python run_v2.py --stage dev` first.")
    with open(path) as f:
        return json.load(f)


def rebuild_frozen(ctx, frozen):
    """Rebuild the frozen book and check nothing moved since it was frozen."""
    cfg = (frozen["config"]["set"], frozen["config"]["weighting"], frozen["config"]["overlay"])
    strategies, P, pos, info = fit_frozen(ctx, cfg)
    if abs(info["scale"] - frozen["scale"]) > 1e-9:
        raise AssertionError("Scale differs from the frozen config.")
    for k, w in frozen["weights"].items():
        if abs(info["weights"][k] - w) > 1e-9:
            raise AssertionError(f"Weight for {k} differs from the frozen config.")
    return cfg, strategies, P, pos


# ---------------------------------------------------------------------------
# the reversal idea that didn't work
# ---------------------------------------------------------------------------
def reversal_check(ctx):
    """Fade vs follow yesterday's move, each dev year on its own.

    The change in PB07 stands in for yesterday's return (signals only). No
    fitting: the sign is fixed, so this is a straight look at the data.
    """
    move = np.sign(ctx.decision["PB07"].astype(float).diff()).fillna(0.0).to_numpy()
    trend_on = ctx.decision["PB01"].fillna(0.0).to_numpy() > 0.5
    rules = {
        "fade yesterday (reversal)": -move,
        "fade up-days only while PB01 is on": -((move > 0) & trend_on).astype(float),
        "fade down-days only while PB01 is off": ((move < 0) & ~trend_on).astype(float),
        "follow yesterday (continuation)": move,
    }
    rows = []
    for name, pos in rules.items():
        for year in (2018, 2019, 2020):
            mask = date_mask(ctx, f"{year}-01-01", f"{year}-12-31")
            net = run_position(ctx, pos, mask)["metrics"]
            gross = run_position(ctx, pos, mask, cost=0.0)["metrics"]
            rows.append({"rule": name, "year": year, "sharpe_net": net["sharpe"],
                         "sharpe_before_costs": gross["sharpe"],
                         "ann_return_net": net["annualized_return"],
                         "annual_turnover": net["annual_turnover"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. bias checks (dev window only)
# ---------------------------------------------------------------------------
def causality_check(ctx, cfg, strategies, n_cuts=5):
    """Cut the data at several dates and rebuild the book. Positions up to the
    cut must not change. If they did, the book would be using later data."""
    dev_end = int(date_mask(ctx, end=config.DEV_END_DATE).sum())
    cuts = np.linspace(300, dev_end - 1, n_cuts).astype(int)
    P_full = position_frame(ctx, strategies)
    worst = 0.0
    for cut in cuts:
        dec = ctx.decision.iloc[:cut].reset_index(drop=True)
        P_cut = position_frame(ctx, strategies, decision=dec)
        # strategies only see signals, so their positions must match exactly
        worst = max(worst, float(np.abs(P_cut.to_numpy() - P_full.iloc[:cut].to_numpy()).max()))
        # the book-level overlays add their own trailing windows; check those
        # on the truncated frames too
        keys = STRATEGY_SETS[cfg[0]]
        base_cut = P_cut[keys].mean(axis=1).to_numpy()
        # rebuild returns from the truncated opens, so the last row really
        # doesn't know the next open
        mk = ctx.market.iloc[:cut].reset_index(drop=True)
        mk["ret_oo"] = mk["open"].shift(-1) / mk["open"] - 1.0
        worst = max(worst, float(np.nanmax(np.abs(
            regime_multiplier(dec) - regime_multiplier(ctx.decision)[:cut]))))
        worst = max(worst, float(np.nanmax(np.abs(
            trailing_book_vol(base_cut, mk) - trailing_book_vol(P_full[keys].mean(axis=1), ctx.market)[:cut]))))
    return {"cuts_checked": len(cuts), "max_position_change": worst,
            "status": "PASS" if worst < 1e-12 else "FAIL"}


def shifted_signal_check(ctx, cfg, strategies, info):
    """Re-run dev with the signals moved one day later, and one day earlier.

    Later (an extra day of delay) should hurt: the edge decays with age.
    Earlier means using signals from the day being traded, which is
    look-ahead. If that doesn't blow the Sharpe up, the harness couldn't
    catch a leak. Weights and scale stay frozen; strategies are not refitted.
    """
    dev_mask = date_mask(ctx, end=config.DEV_END_DATE)
    keys = STRATEGY_SETS[cfg[0]]
    w = pd.Series(info["weights"])
    rows = []
    for label, shift in (("as built", 0), ("one extra day of delay", 1),
                         ("one day of look-ahead (should be caught)", -1)):
        dec = ctx.decision.copy()
        dec[config.ALL_SIGNALS] = dec[config.ALL_SIGNALS].shift(shift)
        P = position_frame(ctx, strategies, decision=dec)
        base = (P[keys] * w).sum(axis=1).to_numpy()
        if cfg[2] == "regime_tilt":
            base = base * regime_multiplier(dec)
        elif cfg[2] == "vol_target":
            base = base * vol_target_multiplier(base, ctx.market, dev_mask)
        pos = np.clip(info["scale"] * base, -1, 1)
        if cfg[2].startswith("band_"):
            pos = no_trade_band(pos, int(cfg[2].split("_")[1]) / 100.0)
        m = run_position(ctx, pos, dev_mask)["metrics"]
        rows.append({"test": label, "signal_shift_days": shift,
                     "dev_sharpe": m["sharpe"], "dev_ann_return": m["annualized_return"]})
    return pd.DataFrame(rows)


def alpha08_scorecard(ctx, n_variants_tried):
    """alpha_08 through the same dev battery as the Task 2 strategies."""
    from alpha_research import AlphaResearch
    s = new_strategies(["alpha_08"])
    ctx.fit_strategies(s, split="dev")
    research = AlphaResearch(ctx, s)
    perf = research.performance_table("dev").set_index("strategy").loc["alpha_08"]
    sig = research.significance_table("dev", n_trials=n_variants_tried).set_index("label").loc["alpha_08"]
    rob, detail = research.robustness_suite("dev", noise_draws=12)
    rob = rob.set_index("strategy").loc["alpha_08"]
    card = {
        "dev_sharpe": perf["sharpe"], "dev_ann_return": perf["ann_return"],
        "dev_max_drawdown": perf["max_drawdown"], "annual_turnover": perf["annual_turnover"],
        "cost_drag": perf["cost_drag"],
        "nw_t": sig["nw_t_stat"], "nw_p": sig["nw_p_value"],
        "boot_sharpe_ci_low": sig["boot_ci_low"], "boot_sharpe_ci_high": sig["boot_ci_high"],
        "permutation_p": sig["perm_p_value"],
        "deflated_sharpe_prob": sig["dsr_dsr"], "variants_tried": n_variants_tried,
        "breakeven_cost_bps": rob["breakeven_cost_bps"], "sharpe_at_2x_cost": rob["sharpe_at_2x_cost"],
        "subperiods_positive": rob["pct_subperiods_positive"],
        "noise_retention_5pct": rob["noise_retention_5pct"],
        "param_pct_positive": rob["param_pct_positive"],
        "fitted": s["alpha_08"].get_metadata()["fitted"],
    }
    return card, detail["alpha_08"]


# ---------------------------------------------------------------------------
# 4. the one holdout evaluation
# ---------------------------------------------------------------------------
def book_table(ctx, books, splits):
    rows = []
    for split, mask in splits.items():
        for name, pos in books.items():
            res = run_position(ctx, pos, mask)
            rows.append({"book": name, "split": split, **summarise(res)})
        bh = ctx.benchmark_returns(split)
        from performance import PerformanceAnalyzer
        m = PerformanceAnalyzer().compute_all(bh)
        rows.append({"book": "buy and hold", "split": split,
                     "ann_return": m["annualized_return"], "volatility": m["volatility"],
                     "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                     "annual_turnover": 0.0, "hit_rate_active": m.get("hit_rate", np.nan)})
    return pd.DataFrame(rows)


def significance(ctx, books, mask, control="v1 scaled to 10% vol"):
    """Is v2 different from zero, and from v1 at the same risk?"""
    from statistics import StatisticalTester
    st = StatisticalTester()
    asset = ctx.market.loc[mask, "ret_oo"].fillna(0.0).reset_index(drop=True)
    out = []
    returns = {k: run_position(ctx, p, mask)["returns"].reset_index(drop=True)
               for k, p in books.items()}
    for name, pos in books.items():
        r = returns[name]
        nw = st.newey_west_t(r)
        boot = st.stationary_bootstrap(r)
        perm = st.permutation_test(pd.Series(np.asarray(pos)[mask]), asset)
        row = {"book": name, "nw_t": nw["t_stat"], "nw_p": nw["p_value"],
               "boot_sharpe": boot["point"], "boot_ci_low": boot["ci_low"],
               "boot_ci_high": boot["ci_high"], "permutation_p": perm["p_value"]}
        if name != control and control in returns:
            diff = r - returns[control]
            d = st.newey_west_t(diff)
            row.update({"excess_vs_control_ann": float(diff.mean() * config.TRADING_DAYS_PER_YEAR),
                        "excess_nw_t": d["t_stat"], "excess_nw_p": d["p_value"]})
        out.append(row)
    return pd.DataFrame(out)


def cost_stress(ctx, books, splits, multiples=(1, 2, 4)):
    rows = []
    for split, mask in splits.items():
        for mult in multiples:
            row = {"split": split, "cost_multiple": mult}
            for name, pos in books.items():
                c = config.TRANSACTION_COST * mult
                row[name] = run_position(ctx, pos, mask, cost=c)["metrics"]["sharpe"]
            rows.append(row)
    return pd.DataFrame(rows)
