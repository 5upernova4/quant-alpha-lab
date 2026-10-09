"""
AlphaMetaModel - the learned part of the allocator.

What it predicts
----------------
Not the price. The Technical Documentation rules out a plain next-price model,
and an allocator doesn't need one anyway: it needs to know which *strategy* is
about to do well. So the label, for strategy i at rebalance block k, is

    y_i(k) = mean net return of strategy i over block k+1 / its trailing vol

which is roughly an information ratio for the next month and is comparable
across strategies.

Why the model is this small
---------------------------
~770 development days at a 21-day rebalance is about 36 decisions. With four
strategies that is ~140 rows, but they are not 140 independent points: all four
strategies share the same ~36 market blocks. With that little data anything
flexible will fit noise, so:

* ridge regression (linear, one penalty, no interactions)
* 10 slow features, fixed up front
* the ridge penalty is picked inside each training fold by time-ordered CV, so
  the amount of regularisation is never chosen with future data (the first few
  folds have fewer than 8 blocks, too few to split, and use a fixed penalty of 10)
* features are standardised with training-fold statistics only

Features
--------
From each strategy's own P&L record: trailing mean return over 1 and 3 blocks,
trailing vol, recent/longer vol ratio, hit rate, current drawdown.
From the signal library (averaged over the last block): mean of the PB, BB and
VB flags, and the level of BB07.

Price and volume never go in. Using a strategy's own past P&L to size it is an
allocation input, not a trading signal; the report states this assumption.

Which IC we report
------------------
The allocator z-scores the predictions across strategies inside each block
before tilting, so only the *ranking within a block* reaches the weights. The
statistic that matters is therefore the cross-sectional IC: rank correlation
between predicted and actual effectiveness across the strategies in one block,
averaged over blocks. That is what the null baseline is tested on. A pooled IC
over all rows is also printed, but it mixes in "is next month good for every
strategy" timing that the allocator throws away.
"""

import numpy as np
import pandas as pd

import config


class AlphaMetaModel:
    """Ridge model of next-block strategy effectiveness, plus the checks on it."""

    ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)     # ridge penalties tried in-fold

    def __init__(self, block=21, lookback_blocks=3, min_train_blocks=8, seed=None):
        self.block = block
        self.lookback_blocks = lookback_blocks
        self.min_train_blocks = min_train_blocks
        self.seed = config.RANDOM_SEED if seed is None else seed

        self.coef_ = None
        self.feature_names_ = None
        self.scaler_ = None
        self.chosen_alpha_ = None
        self.walk_forward_ = None
        self.null_ = None
        self.null_draws_ = None
        self.capacity_ = None
        self.n_strategies_ = None

    # ------------------------------------------------------------------
    # dataset
    # ------------------------------------------------------------------
    def build_dataset(self, strategy_returns, market_state):
        """One row per strategy per rebalance block.

        Row (i, k) only uses data up to the end of block k and is labelled with
        strategy i's effectiveness over block k+1. The final block has no label
        yet, so it is dropped.
        """
        R = pd.DataFrame(strategy_returns).dropna(how="any")
        S = pd.DataFrame(market_state).reindex(R.index).ffill()
        n = len(R)
        b, L = self.block, self.lookback_blocks

        edges = list(range(0, n - b + 1, b))
        rows, labels, keys = [], [], []

        for k, start in enumerate(edges):
            end = start + b                       # block k is [start, end)
            if end + b > n:
                break                             # nothing to label it with
            hist = R.iloc[:end]
            if len(hist) < L * b:
                continue

            state = S.iloc[start:end]
            market_feats = {
                "state_pb": float(state.get("pb_mean", pd.Series(dtype=float)).mean()),
                "state_bb": float(state.get("bb_mean", pd.Series(dtype=float)).mean()),
                "state_vb": float(state.get("vb_mean", pd.Series(dtype=float)).mean()),
                "state_vol": float(state.get("vol_level", pd.Series(dtype=float)).mean()),
            }

            nxt = R.iloc[end:end + b]
            for name in R.columns:
                h = hist[name]
                recent = h.iloc[-b:]
                longer = h.iloc[-L * b:]
                sd_recent = float(recent.std(ddof=1))
                sd_long = float(longer.std(ddof=1))
                eq = (1.0 + h).cumprod()
                feats = {
                    "ret_1b": float(recent.mean()),
                    "ret_3b": float(longer.mean()),
                    "vol_1b": sd_recent,
                    "vol_ratio": float(sd_recent / sd_long) if sd_long > 0 else 1.0,
                    "hit_1b": float((recent > 0).mean()),
                    "drawdown": float(eq.iloc[-1] / eq.cummax().iloc[-1] - 1.0),
                    **market_feats,
                }
                fwd = nxt[name]
                y = float(fwd.mean() / sd_long) if sd_long > 0 else 0.0
                rows.append(feats)
                labels.append(y)
                keys.append((k, name, R.index[end - 1], R.index[min(end + b, n) - 1]))

        X = pd.DataFrame(rows)
        y = pd.Series(labels, name="effectiveness")
        meta = pd.DataFrame(keys, columns=["block", "strategy", "asof", "label_end"])
        self.feature_names_ = list(X.columns)
        self.n_strategies_ = R.shape[1]
        return X, y, meta

    @staticmethod
    def build_market_state(decision_frame):
        """Squash the signal library into four market-state numbers per day."""
        d = decision_frame.set_index(pd.DatetimeIndex(decision_frame["date"]))
        pb = [c for c in config.PRICE_BASED_SIGNALS if c in d.columns]
        bb = [c for c in config.BAND_BASED_SIGNALS if c in d.columns]
        vb = [c for c in config.VOLUME_BASED_SIGNALS if c in d.columns]
        out = pd.DataFrame(index=d.index)
        out["pb_mean"] = d[pb].mean(axis=1)
        out["bb_mean"] = d[bb].mean(axis=1)
        out["vb_mean"] = d[vb].mean(axis=1)
        out["vol_level"] = d["BB07"] if "BB07" in d.columns else 0.0
        return out.ffill().fillna(0.0)

    # ------------------------------------------------------------------
    # ridge, written out so the penalty is visible
    # ------------------------------------------------------------------
    @staticmethod
    def _fit_ridge(X, y, alpha):
        Xc = np.column_stack([np.ones(len(X)), X])
        P = np.eye(Xc.shape[1]) * alpha
        P[0, 0] = 0.0                              # don't shrink the intercept
        return np.linalg.solve(Xc.T @ Xc + P, Xc.T @ y)

    @staticmethod
    def _apply(coef, X):
        return np.column_stack([np.ones(len(X)), X]) @ coef

    def _blocks_for(self, n_rows, blocks=None):
        if blocks is not None:
            return np.asarray(blocks)
        # rows come out of build_dataset grouped by block, N strategies at a time
        if self.n_strategies_:
            return np.arange(n_rows) // self.n_strategies_
        return np.arange(n_rows)

    def _select_alpha(self, X, y, blocks=None, n_folds=4):
        """Pick the ridge penalty by time-ordered CV inside the training data.

        Folds are cut on block boundaries so a block is never split between
        fitting and validation. Returns the chosen penalty together with the
        validation R2 and cross-sectional IC it achieved - that is the
        "validation" number in the train / validation / test comparison.
        """
        blocks = self._blocks_for(len(X), blocks)
        ub = np.unique(blocks)
        fallback = {"alpha": self.ALPHAS[len(self.ALPHAS) // 2],
                    "val_r2": np.nan, "val_ic": np.nan}
        if len(ub) < n_folds * 2:
            return fallback

        cuts = np.array_split(ub, n_folds)
        best = None
        for a in self.ALPHAS:
            preds, actual, vblocks = [], [], []
            for f in range(1, n_folds):
                tr = np.isin(blocks, np.concatenate(cuts[:f]))
                va = np.isin(blocks, cuts[f])
                if tr.sum() < 10 or va.sum() < 3:
                    continue
                mu, sd = X[tr].mean(0), X[tr].std(0)
                sd = np.where(sd > 0, sd, 1.0)
                coef = self._fit_ridge((X[tr] - mu) / sd, y[tr], a)
                preds.append(self._apply(coef, (X[va] - mu) / sd))
                actual.append(y[va])
                vblocks.append(blocks[va])
            if not preds:
                continue
            p, yv, bv = np.concatenate(preds), np.concatenate(actual), np.concatenate(vblocks)
            mse = float(np.mean((p - yv) ** 2))
            if best is None or mse < best["mse"]:
                best = {"alpha": a, "mse": mse, "val_r2": self._r2(yv, p),
                        "val_ic": self._xs_ic(yv, p, bv)}
        return best or fallback

    # ------------------------------------------------------------------
    # required interface
    # ------------------------------------------------------------------
    def fit(self, strategy_history, market_state=None, X=None, y=None, blocks=None):
        """Fit on everything available up to now."""
        if X is None or y is None:
            X, y, meta = self.build_dataset(strategy_history, market_state)
            blocks = meta["block"].to_numpy()
        Xv = np.asarray(X, dtype=float)
        yv = np.asarray(y, dtype=float)
        if len(Xv) < 10:
            self.coef_ = np.zeros(Xv.shape[1] + 1)
            self.chosen_alpha_ = None
            return self

        blocks = self._blocks_for(len(Xv), blocks)
        self.chosen_alpha_ = self._select_alpha(Xv, yv, blocks)["alpha"]
        mu, sd = Xv.mean(0), Xv.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        self.scaler_ = (mu, sd)
        self.coef_ = self._fit_ridge((Xv - mu) / sd, yv, self.chosen_alpha_)
        n_blocks = len(np.unique(blocks))
        self.capacity_ = {
            "model": "ridge regression (linear, L2-regularised)",
            "n_features": Xv.shape[1],
            "n_observations": len(Xv),
            "observations_per_feature": len(Xv) / max(1, Xv.shape[1]),
            "independent_market_blocks": n_blocks,
            "blocks_per_feature": n_blocks / max(1, Xv.shape[1]),
            "ridge_alpha": self.chosen_alpha_,
        }
        return self

    def predict(self, strategy_history=None, market_state=None, X=None):
        """Predicted effectiveness for each row."""
        if self.coef_ is None:
            raise RuntimeError("AlphaMetaModel.fit must be called first.")
        if X is None:
            X, _, _ = self.build_dataset(strategy_history, market_state)
        Xv = np.asarray(X, dtype=float)
        mu, sd = self.scaler_ if self.scaler_ else (0.0, 1.0)
        return self._apply(self.coef_, (Xv - mu) / sd)

    def score(self, strategies=None, X=None, meta=None):
        """Strategy-level assessments as a table."""
        pred = self.predict(X=X)
        out = meta.copy() if meta is not None else pd.DataFrame()
        out["predicted_effectiveness"] = pred
        return out

    def get_feature_importance(self):
        """Standardised coefficients (inputs are scaled, so they compare directly)."""
        if self.coef_ is None or self.feature_names_ is None:
            return pd.DataFrame()
        return (
            pd.DataFrame({"feature": self.feature_names_, "coefficient": self.coef_[1:]})
            .assign(abs_coefficient=lambda d: d["coefficient"].abs())
            .sort_values("abs_coefficient", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # the model-discipline evidence
    # ------------------------------------------------------------------
    def walk_forward(self, X, y, meta, min_train=None):
        """Expanding-window refit: train on blocks < k, predict block k, repeat.

        Each fold records three numbers per metric: in-sample (train), inner-CV
        (validation, used only to pick the penalty) and the block it had not seen
        (test).
        """
        min_train = min_train or self.min_train_blocks
        Xa, ya = np.asarray(X, float), np.asarray(y, float)
        blocks_all = meta["block"].to_numpy()
        rows, preds = [], []

        for k in sorted(np.unique(blocks_all)):
            if k < min_train:
                continue
            tr = blocks_all < k
            te = blocks_all == k
            if tr.sum() < 10 or te.sum() == 0:
                continue

            Xtr, ytr, btr = Xa[tr], ya[tr], blocks_all[tr]
            Xte, yte = Xa[te], ya[te]

            sel = self._select_alpha(Xtr, ytr, btr)
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd = np.where(sd > 0, sd, 1.0)
            coef = self._fit_ridge((Xtr - mu) / sd, ytr, sel["alpha"])
            p_tr = self._apply(coef, (Xtr - mu) / sd)
            p_te = self._apply(coef, (Xte - mu) / sd)

            rows.append({
                "fold_block": int(k),
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "ridge_alpha": sel["alpha"],
                "train_r2": self._r2(ytr, p_tr),
                "val_r2": sel["val_r2"],
                "test_r2": self._r2(yte, p_te),
                "train_ic": self._xs_ic(ytr, p_tr, btr),
                "val_ic": sel["val_ic"],
                "test_ic": self._ic(yte, p_te),
            })
            for j, i in enumerate(np.where(te)[0]):
                preds.append({
                    "block": int(k),
                    "strategy": meta["strategy"].iloc[i],
                    "asof": meta["asof"].iloc[i],
                    "predicted": float(p_te[j]),
                    "actual": float(yte[j]),
                })

        folds = pd.DataFrame(rows)
        pred_df = pd.DataFrame(preds)
        if not folds.empty:
            # A test fold is one block, i.e. one point per strategy. Its IC is a
            # proper cross-sectional IC (coarse, but that's what the allocator
            # uses). Its R2 on 3-4 points is close to meaningless, so R2 is
            # reported pooled over every out-of-sample prediction instead.
            a = folds.attrs
            a["mean_test_ic"] = float(folds["test_ic"].mean())      # the null's statistic
            a["pct_folds_positive_ic"] = float((folds["test_ic"] > 0).mean())
            a["n_folds"] = int(len(folds))
            if not pred_df.empty:
                a["pooled_test_ic"] = self._ic(pred_df["actual"], pred_df["predicted"])
                a["n_pooled_predictions"] = int(len(pred_df))

            # The train / validation / test comparison has to use one set of
            # folds. The earliest folds have too few blocks for inner CV (they
            # fall back to a fixed penalty and have no validation score), so the
            # comparison is made on the folds that do have one.
            cv = folds[folds["val_r2"].notna()]
            cv_pred = pred_df[pred_df["block"].isin(cv["fold_block"])] if not pred_df.empty else pred_df
            a["n_cv_folds"] = int(len(cv))
            if len(cv) and not cv_pred.empty:
                a["mean_train_r2"] = float(cv["train_r2"].mean())
                a["mean_val_r2"] = float(cv["val_r2"].mean())
                a["pooled_test_r2"] = self._r2(cv_pred["actual"], cv_pred["predicted"])
                a["mean_train_ic"] = float(cv["train_ic"].mean())
                a["mean_val_ic"] = float(cv["val_ic"].mean())
                a["cv_folds_test_ic"] = float(cv["test_ic"].mean())
                a["train_val_r2_gap"] = a["mean_train_r2"] - a["mean_val_r2"]
                a["val_test_r2_gap"] = a["mean_val_r2"] - a["pooled_test_r2"]
                a["train_test_r2_gap"] = a["mean_train_r2"] - a["pooled_test_r2"]
                a["train_test_ic_gap"] = a["mean_train_ic"] - a["cv_folds_test_ic"]
        self.walk_forward_ = (folds, pred_df)
        return folds, pred_df

    def null_baseline(self, X, y, meta, n_perm=200):
        """Re-run the whole walk-forward on labels shuffled inside each block.

        Shuffling within a block keeps each month's set of outcomes and destroys
        only the link between a strategy's features and its own outcome, which
        is exactly the thing the allocator would be relying on. The statistic is
        the mean cross-sectional test IC, same as the real model's headline.
        """
        real_folds, _ = self.walk_forward(X, y, meta)
        if real_folds.empty:
            return {}
        real_ic = float(real_folds.attrs["mean_test_ic"])

        rng = np.random.default_rng(self.seed)
        y_arr = np.asarray(y, dtype=float)
        blocks = meta["block"].to_numpy()
        draws = []
        for _ in range(n_perm):
            y_perm = y_arr.copy()
            for b in np.unique(blocks):
                m = blocks == b
                y_perm[m] = rng.permutation(y_perm[m])
            folds, _ = self.walk_forward(X, pd.Series(y_perm), meta)
            if not folds.empty:
                draws.append(float(folds.attrs["mean_test_ic"]))

        draws = np.array(draws)
        out = {
            "statistic": "mean cross-sectional out-of-sample IC",
            "real_mean_test_ic": real_ic,
            "null_mean_test_ic": float(draws.mean()) if len(draws) else np.nan,
            "null_std_test_ic": float(draws.std(ddof=1)) if len(draws) > 1 else np.nan,
            "null_p95_test_ic": float(np.quantile(draws, 0.95)) if len(draws) else np.nan,
            # (count + 1) / (n + 1) so a p-value can never be exactly zero
            "p_value": float(((draws >= real_ic).sum() + 1) / (len(draws) + 1))
            if len(draws) else np.nan,
            "n_permutations": int(len(draws)),
        }
        out["clears_null"] = bool(out["p_value"] < 0.05) if len(draws) else False
        self.null_ = out
        self.null_draws_ = draws
        self.walk_forward(X, y, meta)    # the loop above overwrote the real result
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _r2(y, p):
        y, p = np.asarray(y, float), np.asarray(p, float)
        ss = float(((y - y.mean()) ** 2).sum())
        return float(1.0 - ((y - p) ** 2).sum() / ss) if ss > 0 else 0.0

    @staticmethod
    def _ic(y, p):
        y, p = pd.Series(np.asarray(y, float)), pd.Series(np.asarray(p, float))
        if y.nunique() < 2 or p.nunique() < 2:
            return 0.0
        c = y.rank().corr(p.rank())
        return float(c) if np.isfinite(c) else 0.0

    @classmethod
    def _xs_ic(cls, y, p, blocks):
        """Rank IC across strategies within each block, averaged over blocks."""
        y, p, blocks = np.asarray(y, float), np.asarray(p, float), np.asarray(blocks)
        ics = [cls._ic(y[blocks == b], p[blocks == b]) for b in np.unique(blocks)
               if (blocks == b).sum() > 1]
        return float(np.mean(ics)) if ics else 0.0

    # ------------------------------------------------------------------
    def report(self):
        """Capacity, walk-forward and null baseline, in that order."""
        lines = ["ALPHA META-MODEL -- learned effectiveness component", "=" * 84]

        lines.append("\n[1. Capacity]")
        cap = self.capacity_ or {}
        lines.append(f"  model                      : {cap.get('model', 'ridge regression')}")
        lines.append(f"  features                   : {cap.get('n_features', 0)}")
        lines.append(f"  rows (strategy x block)    : {cap.get('n_observations', 0)}")
        lines.append(f"  independent market blocks  : {cap.get('independent_market_blocks', 0)}")
        lines.append(f"  blocks per feature         : {cap.get('blocks_per_feature', 0):.1f}")
        lines.append(f"  ridge penalty (in-fold CV) : {cap.get('ridge_alpha')}")
        lines.append("  About three independent blocks per feature. A linear, penalised model")
        lines.append("  on a handful of slow features is as much capacity as that supports.")

        lines.append("\n[2. Walk-forward validation]")
        if self.walk_forward_ is not None:
            folds, _ = self.walk_forward_
            if not folds.empty:
                a = folds.attrs
                lines.append(folds.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
                lines.append("")
                lines.append(f"  on the {a.get('n_cv_folds', 0)} folds with an inner-CV penalty "
                             f"(the first {a.get('n_folds', 0) - a.get('n_cv_folds', 0)} are too "
                             "short to cross-validate and use a fixed penalty):")
                lines.append("                     train      validation   test")
                lines.append(f"  R2              {a.get('mean_train_r2', 0):+9.4f}   "
                             f"{a.get('mean_val_r2', 0):+9.4f}   {a.get('pooled_test_r2', 0):+9.4f}"
                             "   (test pooled over those folds' predictions)")
                lines.append(f"  x-sectional IC  {a.get('mean_train_ic', 0):+9.4f}   "
                             f"{a.get('mean_val_ic', 0):+9.4f}   {a.get('cv_folds_test_ic', 0):+9.4f}"
                             "   (mean over blocks)")
                lines.append(f"\n  train -> test R2 gap    : {a.get('train_test_r2_gap', 0):+.4f}")
                lines.append(f"  train -> test IC gap    : {a.get('train_test_ic_gap', 0):+.4f}")
                lines.append(f"  all {a.get('n_folds', 0)} folds: mean test IC "
                             f"{a.get('mean_test_ic', 0):+.4f} (the null baseline's statistic)")
                lines.append(f"  pooled test IC          : {a.get('pooled_test_ic', 0):+.4f} over "
                             f"{a.get('n_pooled_predictions', 0)} predictions "
                             "(includes timing the allocator discards)")
                lines.append(f"  folds with test IC > 0  : {a.get('pct_folds_positive_ic', 0):.1%}")

        lines.append("\n[3. Null baseline: same walk-forward on within-block shuffled labels]")
        if self.null_:
            n = self.null_
            lines.append(f"  real mean test IC       : {n['real_mean_test_ic']:+.4f}")
            lines.append(f"  null mean test IC       : {n['null_mean_test_ic']:+.4f} "
                         f"(sd {n['null_std_test_ic']:.4f})")
            lines.append(f"  null 95th percentile IC : {n['null_p95_test_ic']:+.4f}")
            lines.append(f"  permutation p-value     : {n['p_value']:.4f} "
                         f"over {n['n_permutations']} permutations")
            verdict = ("clears the null baseline" if n["clears_null"]
                       else "DOES NOT clear the null baseline -> report as "
                            "indistinguishable from noise")
            lines.append(f"  verdict                 : {verdict}")

        imp = self.get_feature_importance()
        if not imp.empty:
            lines.append("\n[Standardised coefficients, full dev fit]")
            lines.append(imp.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        return "\n".join(lines)
