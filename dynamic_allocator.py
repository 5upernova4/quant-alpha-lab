"""
DynamicAllocator - turns the meta-model's scores into weights.

A prediction isn't an allocation. Three choices sit in between and they matter
more than the model does:

1. How far the scores may move the weights. We tilt a fixed base allocation
   (risk parity) instead of rebuilding from scratch every month.
2. How often to trade. A no-trade band keeps the current weights unless the
   target has moved by more than 5 points in some strategy.
3. What to refuse. Weights are capped at 50%.

On top of that the tilt is scaled by how much skill the model has actually shown
on blocks that have already closed (the "skill gate", see _confidence). If it
hasn't shown any, the allocator just holds the base weights.

tilt_null() at the bottom is the allocation-level null baseline: the ungated
model against random scores pushed through the same tilt / band / cap rules.
"""

import numpy as np
import pandas as pd

import config


class DynamicAllocator:
    """Converts strategy scores into target weights, with turnover control."""

    def __init__(self, base_weights=None, tilt_strength=0.5, max_weight=0.50,
                 min_weight=0.0, no_trade_band=0.05, blend_with_base=0.5,
                 confidence_scaling=True, target_ic=0.10, min_skill_folds=6,
                 min_skill_t=1.0):
        self.base_weights = base_weights
        self.tilt_strength = tilt_strength
        # With confidence_scaling on, the tilt is multiplied by a number in
        # [0, 1] that depends on the model's realised out-of-sample IC so far.
        # Turning it off gives the "ungated" allocator used in tilt_null().
        self.confidence_scaling = confidence_scaling
        self.target_ic = target_ic
        self.min_skill_folds = min_skill_folds
        self.min_skill_t = min_skill_t
        self.confidence_ = 1.0
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.no_trade_band = no_trade_band
        self.blend_with_base = blend_with_base
        self.history_ = []

    # ------------------------------------------------------------------
    def allocate(self, strategy_scores, constraints=None):
        """Scores -> target weights.

        Scores are z-scored across strategies first. If the model thinks next
        month is bad for everything, that says something about the market, not
        about which strategy to prefer, so only the relative part is used.
        """
        s = pd.Series(strategy_scores, dtype=float).dropna()
        if s.empty:
            return pd.Series(dtype=float)

        c = constraints or {}
        max_w = c.get("max_weight", self.max_weight)
        min_w = c.get("min_weight", self.min_weight)

        base = (pd.Series(self.base_weights).reindex(s.index)
                if self.base_weights is not None else pd.Series(1.0 / len(s), index=s.index))
        base = base.fillna(0.0)
        if base.sum() > 0:
            base = base / base.sum()

        sd = s.std(ddof=1)
        z = (s - s.mean()) / sd if sd > 1e-12 else pd.Series(0.0, index=s.index)
        z = z.clip(-2.0, 2.0)

        strength = self.tilt_strength * (self.confidence_ if self.confidence_scaling else 1.0)

        # multiplicative tilt, so weights stay positive without another clip
        tilted = base * np.exp(strength * z)
        if tilted.sum() > 0:
            tilted = tilted / tilted.sum()

        w = self.blend_with_base * base + (1.0 - self.blend_with_base) * tilted
        w = w.clip(min_w, max_w)
        return w / w.sum() if w.sum() > 0 else base

    def rebalance(self, current_weights, target_weights, date=None):
        """Apply the no-trade band. Returns (applied_weights, turnover)."""
        cur = pd.Series(current_weights, dtype=float)
        tgt = pd.Series(target_weights, dtype=float).reindex(cur.index).fillna(0.0)
        cur = cur.reindex(tgt.index).fillna(0.0)

        drift = (tgt - cur).abs()
        if drift.max() < self.no_trade_band:
            applied, turnover = cur, 0.0
        else:
            applied = tgt.copy()
            turnover = float((applied - cur).abs().sum())

        self.history_.append({
            "date": date,
            "turnover": turnover,
            "traded": turnover > 0,
            "confidence": float(self.confidence_),
            **{f"w_{k}": float(v) for k, v in applied.items()},
        })
        return applied, turnover

    def _confidence(self, realised_ic, min_folds=6, min_t=1.0):
        """Skill gate: map realised out-of-sample IC to a tilt multiplier in [0, 1].

        Both conditions must hold before the model moves any capital:
          1. at least `min_folds` closed blocks with a positive mean IC
          2. that mean has a t-stat of at least `min_t` across those blocks
        Otherwise the multiplier is 0 and the base weights are held.

        This exists because the model fails its permuted-label null. If a
        component can't show skill it shouldn't move money, and that should be
        decided by a rule written in advance, not by us after seeing the result.
        Only blocks that have already closed are used, so nothing looks ahead.
        """
        if not self.confidence_scaling:
            return 1.0
        ic = np.asarray(realised_ic, dtype=float)
        ic = ic[np.isfinite(ic)]
        if len(ic) < min_folds:
            return 0.0
        mean_ic = float(ic.mean())
        sd = float(ic.std(ddof=1))
        t = mean_ic / (sd / np.sqrt(len(ic))) if sd > 0 else 0.0
        if mean_ic <= 0 or t < min_t:
            return 0.0
        return float(np.clip(mean_ic / self.target_ic, 0.0, 1.0))

    def get_allocation_history(self):
        return pd.DataFrame(self.history_)

    # ------------------------------------------------------------------
    def run(self, strategy_returns, meta_model, X, y, meta, base_weights=None,
            min_train_blocks=8):
        """Walk the allocator forward through time.

        At each block: refit the model on every earlier block, score the
        strategies, tilt the base weights, hold them through the next block.
        Until there is enough history to fit anything, hold the base weights.
        """
        R = pd.DataFrame(strategy_returns)
        names = list(R.columns)
        base = (pd.Series(base_weights).reindex(names).fillna(0.0)
                if base_weights is not None else pd.Series(1.0 / len(names), index=names))
        base = base / base.sum() if base.sum() > 0 else pd.Series(1.0 / len(names), index=names)
        self.base_weights = base

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        block_ids = meta["block"].to_numpy()

        current = base.copy()
        weight_path = pd.DataFrame(np.nan, index=R.index, columns=names)
        rebalance_cost = pd.Series(0.0, index=R.index)
        self.history_ = []
        first_live = None
        realised_ic = []          # cross-sectional IC of blocks that have closed
        self.confidence_ = 0.0 if self.confidence_scaling else 1.0

        for k in sorted(np.unique(block_ids)):
            te = block_ids == k
            asof = meta.loc[te, "asof"].iloc[0]          # last date we may use
            end = meta.loc[te, "label_end"].iloc[0]      # weights held until here
            window = R.index[(R.index > asof) & (R.index <= end)]
            if len(window) == 0:
                continue

            tr = block_ids < k
            if k < min_train_blocks or tr.sum() < 10:
                target = base                            # warm-up
            else:
                meta_model.fit(None, None, X=X_arr[tr], y=y_arr[tr], blocks=block_ids[tr])
                self.confidence_ = self._confidence(
                    realised_ic, min_folds=self.min_skill_folds, min_t=self.min_skill_t
                )
                pred_now = np.asarray(meta_model.predict(X=X_arr[te]), dtype=float)
                scores = pd.Series(pred_now, index=meta.loc[te, "strategy"].to_numpy()
                                   ).reindex(names)
                target = self.allocate(scores)
                if first_live is None:
                    first_live = asof

                # y for this block is the outcome over the window we are about
                # to hold. It gets appended now but is only read at the next
                # block, by which time that window has closed.
                a, p = pd.Series(y_arr[te]), pd.Series(pred_now)
                if a.nunique() > 1 and p.nunique() > 1:
                    realised_ic.append(float(p.rank().corr(a.rank())))

            applied, turnover = self.rebalance(current, target, date=asof)
            weight_path.loc[window, :] = applied.reindex(names).fillna(0.0).to_numpy()
            rebalance_cost.loc[window[0]] = turnover * config.TRANSACTION_COST
            current = applied

        # A decision at the end of block k uses R[asof], and R[asof] is the return
        # from open[asof] to open[asof+1], so it is only known at that next open.
        # Trading at that same open would use a price we fill at. So every
        # decision is pushed back one day and trades at the open after.
        weight_path = weight_path.shift(1)
        rebalance_cost = rebalance_cost.shift(1).fillna(0.0)

        # before the first block: base weights; after the last: last decision
        weight_path = weight_path.ffill()
        weight_path = weight_path.fillna(
            pd.DataFrame([base.to_numpy()] * len(R), index=R.index, columns=names)
        )

        gross = (R * weight_path).sum(axis=1)
        return {
            "returns": gross - rebalance_cost,
            "gross_returns": gross,
            "weights": weight_path,
            "rebalance_cost": rebalance_cost,
            "total_rebalance_cost": float(rebalance_cost.sum()),
            "n_rebalances": int(sum(h["traded"] for h in self.history_)),
            "first_live_date": first_live,
            "realised_ic": realised_ic,
        }

    # ------------------------------------------------------------------
    def report(self):
        h = self.get_allocation_history()
        lines = ["DYNAMIC ALLOCATION", "=" * 72]
        lines.append(f"  tilt strength   : {self.tilt_strength} "
                     f"(scaled by demonstrated skill: {'on' if self.confidence_scaling else 'off'})")
        lines.append(f"  final confidence: {self.confidence_:.3f} "
                     f"(0 = hold base weights, 1 = full tilt)")
        lines.append(f"  skill gate      : >= {self.min_skill_folds} closed folds "
                     f"and IC t-stat >= {self.min_skill_t}")
        if not h.empty and "confidence" in h.columns:
            lines.append(f"  gate open at    : {(h['confidence'] > 0).mean():.1%} of rebalances")
        lines.append(f"  blend with base : {self.blend_with_base}")
        lines.append(f"  no-trade band   : {self.no_trade_band}")
        lines.append(f"  weight bounds   : [{self.min_weight}, {self.max_weight}]")
        if not h.empty:
            lines.append(f"  rebalance points: {len(h)}  (traded at {int(h['traded'].sum())})")
            lines.append(f"  total turnover  : {h['turnover'].sum():.2f}x notional")
            wcols = [c for c in h.columns if c.startswith("w_")]
            if wcols:
                lines.append("\n[Average weights]")
                for c in wcols:
                    lines.append(f"    {c[2:]:<28} {h[c].mean():7.2%}  "
                                 f"(min {h[c].min():.2%}, max {h[c].max():.2%})")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# allocation-level null baseline
# ----------------------------------------------------------------------
class RandomScores:
    """Drop-in for the meta-model that returns noise. Used only for the null."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def fit(self, *args, **kwargs):
        return self

    def predict(self, X=None, **kwargs):
        return self.rng.standard_normal(len(X))


def tilt_null(strategy_returns, X, y, meta, base_weights, evaluate, model_factory,
              allocator_kwargs=None, windows=None, n_draws=200, seed=None,
              min_train_blocks=6):
    """Does the model's tilt beat a random tilt of the same size?

    The skill gate keeps the real dynamic method at its base weights almost all
    the time, so comparing it with the static methods says nothing about the
    model. Here the gate is switched off, the model is allowed to tilt freely,
    and the result is compared against the same allocator fed random scores.
    Random scores go through the same z-score, tilt, blend, cap and no-trade
    band, so their turnover is comparable by construction.

    `evaluate(weight_path)` must return a daily net return series (the netted
    portfolio, costed like everything else). `windows` maps a name to a
    (start, end) date pair; the Sharpe difference vs the base allocation is
    computed on each.
    """
    kw = dict(allocator_kwargs or {})
    kw["confidence_scaling"] = False
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)

    def run_once(model):
        alloc = DynamicAllocator(base_weights=base_weights, **kw)
        res = alloc.run(strategy_returns, model, X, y, meta,
                        base_weights=base_weights, min_train_blocks=min_train_blocks)
        h = alloc.get_allocation_history()
        return evaluate(res["weights"]), float(h["turnover"].sum()), res["first_live_date"]

    base_path = pd.DataFrame(
        [pd.Series(base_weights).reindex(strategy_returns.columns).to_numpy()]
        * len(strategy_returns), index=strategy_returns.index,
        columns=strategy_returns.columns)
    base_r = evaluate(base_path)

    model_r, model_turn, first_live = run_once(model_factory())
    if windows is None:
        windows = {"live": (first_live, strategy_returns.index[-1])}

    def sharpe(r):
        r = r.dropna()
        sd = r.std(ddof=1)
        return float(r.mean() / sd * np.sqrt(config.TRADING_DAYS_PER_YEAR)) if sd > 0 else 0.0

    def excess(r, lo, hi):
        m = (r.index > pd.Timestamp(lo)) & (r.index <= pd.Timestamp(hi))
        return sharpe(r[m]) - sharpe(base_r[m])

    null = {w: [] for w in windows}
    null_turn = []
    for _ in range(n_draws):
        r, turn, _ = run_once(RandomScores(int(rng.integers(1 << 31))))
        null_turn.append(turn)
        for w, (lo, hi) in windows.items():
            null[w].append(excess(r, lo, hi))

    rows = []
    for w, (lo, hi) in windows.items():
        d = np.array(null[w])
        real = excess(model_r, lo, hi)
        rows.append({
            "window": w,
            "start": pd.Timestamp(lo).date(),
            "end": pd.Timestamp(hi).date(),
            "model_sharpe_minus_base": real,
            "null_mean": float(d.mean()),
            "null_p95": float(np.quantile(d, 0.95)),
            "p_value": float(((d >= real).sum() + 1) / (len(d) + 1)),
            "model_turnover": model_turn,
            "null_mean_turnover": float(np.mean(null_turn)),
        })
    out = pd.DataFrame(rows)
    out.attrs["null_draws"] = null
    return out
