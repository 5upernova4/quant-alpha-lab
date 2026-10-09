"""
OrthogonalityAnalyzer -- are these actually different sources of alpha?

The problem statement is precise about what it wants here, and it is not a
correlation matrix. Two strategies can be only mildly correlated pairwise and
still add nothing to each other once a third is in the book; conversely a pair
can be highly correlated and still carry a useful residual. The question is about
the *span* of the return vectors, not about pairs.

The geometry, plainly
---------------------
Stack the strategy return streams as columns of a T x N matrix R. Each column is
a vector in T-dimensional space. Two strategies that always move together point
in the same direction, so one of them adds no new direction to the space the
others already cover. The number of genuinely different directions in R is its
effective rank, and that is the honest count of how many distinct alphas we have,
regardless of how many files are in strategies/.

Method
------
QR decomposition with column pivoting. At each step it picks the remaining column
with the largest component orthogonal to everything already selected -- that is,
the one carrying the most information the book does not yet have -- and records
how much of that column was genuinely new. The pivot order is therefore a
ranking by incremental contribution, computed jointly rather than pairwise, and
the diagonal of R gives the size of each new direction directly.

Three cautions are built into the reporting:

* A direction that is mathematically new can still be economically the same
  trade. Linear independence is necessary evidence, not sufficient, and the
  report says so.
* Rank is computed against a tolerance, and near-degenerate directions are
  flagged rather than silently counted.
* Stability through time is checked by re-running the decomposition on rolling
  windows. A basis that reshuffles every quarter is not a basis.
"""

import numpy as np
import pandas as pd
from scipy.linalg import qr

import config


class OrthogonalityAnalyzer:
    """Studies the linear structure of the strategy-return space."""

    def __init__(self, returns_frame, tol=None):
        """`returns_frame`: DataFrame of strategy net returns, one column each."""
        self.R = returns_frame.dropna(how="any").copy()
        self.names = list(self.R.columns)
        self.tol = tol
        self.results_ = {}

    # ------------------------------------------------------------------
    @property
    def matrix(self):
        """Mean-centred return matrix.

        Centring matters: without it the first direction found is dominated by
        whichever strategies simply have the largest average return, which is a
        statement about profitability rather than about shared behaviour.
        """
        M = self.R.to_numpy(dtype=float)
        return M - M.mean(axis=0, keepdims=True)

    def correlation_matrix(self):
        """Pairwise correlation. Reported as context, never as the conclusion."""
        c = self.R.corr()
        self.results_["correlation"] = c
        return c

    def effective_dimensionality(self):
        """How many distinct directions does the return space actually have?

        Three complementary readings, because any single one can mislead:
        * numerical rank at a tolerance;
        * the number of principal components needed for 95% of the variance;
        * the participation ratio, a smooth 'effective number of independent
          streams' that does not depend on an arbitrary cutoff.
        """
        M = self.matrix
        if M.shape[0] < 2 or M.shape[1] < 1:
            return {}
        s = np.linalg.svd(M, compute_uv=False)
        tol = self.tol if self.tol is not None else max(M.shape) * np.finfo(float).eps * s[0]
        rank = int((s > tol).sum())

        var = s ** 2
        share = var / var.sum()
        cum = np.cumsum(share)
        n95 = int(np.searchsorted(cum, 0.95) + 1)
        participation = float((var.sum() ** 2) / (var ** 2).sum())

        out = {
            "n_strategies": len(self.names),
            "numerical_rank": rank,
            "components_for_95pct_variance": n95,
            "participation_ratio": participation,
            "condition_number": float(s[0] / s[-1]) if s[-1] > 0 else np.inf,
            "singular_values": s.tolist(),
            "variance_share": share.tolist(),
            "cumulative_variance": cum.tolist(),
        }
        self.results_["dimensionality"] = out
        return out

    def pivoted_qr(self):
        """QR with column pivoting -- the ranking by incremental contribution.

        Returns a frame ordered by selection: the first row is the strategy that
        spans the most on its own, each subsequent row is the strategy adding the
        most that the ones above it do not already contain.
        """
        M = self.matrix
        Q, Rm, piv = qr(M, mode="economic", pivoting=True)
        diag = np.abs(np.diag(Rm))
        total = np.linalg.norm(M, axis=0)

        rows = []
        for order, (j, d) in enumerate(zip(piv, diag)):
            name = self.names[j]
            norm = total[j]
            rows.append({
                "selection_order": order + 1,
                "strategy": name,
                "incremental_norm": float(d),
                "own_norm": float(norm),
                # Share of this strategy's own VARIANCE that was not already
                # available from the strategies picked before it. Norms are
                # squared first so this is on the same footing as the R-squared
                # reported by the Gram-Schmidt table -- comparing a norm ratio
                # with a variance ratio in the same report would make two
                # correct numbers look like a contradiction.
                "independent_fraction": float((d / norm) ** 2) if norm > 0 else 0.0,
                "explained_by_earlier": float(1.0 - (d / norm) ** 2) if norm > 0 else 1.0,
                "relative_to_first": float(d / diag[0]) if diag[0] > 0 else 0.0,
            })
        out = pd.DataFrame(rows)
        self.results_["pivoted_qr"] = out
        return out

    def gram_schmidt_residuals(self, order=None):
        """Sequentially orthogonalise and report each residual stream.

        Complements the pivoted QR: here *we* choose the order (for example, by
        conviction or by standalone Sharpe) and read off what is left of each
        strategy once the ones before it have been removed. The residual series
        themselves are returned, so their Sharpe can be measured -- a strategy
        whose residual still earns is genuinely additive.
        """
        names = order or self.names
        M = self.R[names].to_numpy(dtype=float)
        M = M - M.mean(axis=0, keepdims=True)

        basis, rows, residuals = [], [], {}
        for k, name in enumerate(names):
            v = M[:, k].copy()
            own = np.linalg.norm(v)
            for b in basis:
                v = v - (v @ b) * b
            res_norm = np.linalg.norm(v)
            rows.append({
                "order": k + 1, "strategy": name,
                "residual_fraction": float(res_norm / own) if own > 0 else 0.0,
                "r_squared_vs_earlier": float(1.0 - (res_norm / own) ** 2) if own > 0 else 1.0,
            })
            residuals[name] = pd.Series(v, index=self.R.index)
            if res_norm > 1e-12:
                basis.append(v / res_norm)

        out = pd.DataFrame(rows)
        self.results_["gram_schmidt"] = out
        self.results_["residual_series"] = pd.DataFrame(residuals)
        return out, pd.DataFrame(residuals)

    def incremental_contribution(self, ppy=None):
        """What each strategy still earns after projecting out the others.

        For each strategy, regress its returns on *all* the others and keep the
        residual. If the residual still has a positive mean, the strategy is
        carrying return the rest of the book cannot replicate -- which is a
        stronger claim than low correlation.
        """
        ppy = ppy or config.TRADING_DAYS_PER_YEAR
        rows = []
        for name in self.names:
            y = self.R[name].to_numpy(dtype=float)
            others = [c for c in self.names if c != name]
            if not others:
                continue
            X = self.R[others].to_numpy(dtype=float)
            X = np.column_stack([np.ones(len(X)), X])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            eps = y - X @ beta

            # The tradeable residual stream is the intercept plus the noise, not
            # the noise alone. An OLS residual with a fitted intercept has mean
            # exactly zero by construction, so scoring `eps` would report a
            # residual Sharpe of zero for every strategy no matter how additive
            # it is. What a desk would actually hold is this strategy hedged with
            # the others -- and that position earns the intercept.
            hedged = beta[0] + eps
            sd = hedged.std(ddof=1)
            ss_tot = float(((y - y.mean()) ** 2).sum())
            rows.append({
                "strategy": name,
                "alpha_ann": float(beta[0] * ppy),
                "residual_sharpe": float(hedged.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0,
                "r_squared_vs_others": float(1.0 - (eps ** 2).sum() / ss_tot) if ss_tot > 0 else 0.0,
                "residual_share_of_vol": float(sd / y.std(ddof=1)) if y.std(ddof=1) > 0 else 0.0,
            })
        out = pd.DataFrame(rows).sort_values("residual_sharpe", ascending=False)
        self.results_["incremental"] = out
        return out

    def stability_through_time(self, window=252, step=21):
        """Does the basis hold still, or does it reshuffle every quarter?

        Re-runs the pivoted QR on rolling windows and records which strategy is
        picked first and how the pivot order moves. A set of directions that is
        genuinely structural keeps roughly the same ordering.
        """
        rows = []
        n = len(self.R)
        if n < window + step:
            return pd.DataFrame()
        for start in range(0, n - window + 1, step):
            sub = self.R.iloc[start:start + window]
            M = sub.to_numpy(dtype=float)
            M = M - M.mean(axis=0, keepdims=True)
            _, Rm, piv = qr(M, mode="economic", pivoting=True)
            s = np.linalg.svd(M, compute_uv=False)
            var = s ** 2
            rows.append({
                "window_start": sub.index[0].date(),
                "window_end": sub.index[-1].date(),
                "first_pick": self.names[piv[0]],
                "pivot_order": "|".join(self.names[j] for j in piv),
                "participation_ratio": float((var.sum() ** 2) / (var ** 2).sum()),
                "components_95pct": int(np.searchsorted(np.cumsum(var / var.sum()), 0.95) + 1),
            })
        out = pd.DataFrame(rows)
        if not out.empty:
            modal = out["pivot_order"].mode()
            out.attrs["modal_pivot_order"] = modal.iloc[0] if len(modal) else ""
            out.attrs["pivot_order_stability"] = float(
                (out["pivot_order"] == out.attrs["modal_pivot_order"]).mean()
            )
            out.attrs["first_pick_stability"] = float(
                out["first_pick"].value_counts(normalize=True).iloc[0]
            )
        self.results_["stability"] = out
        return out

    # ------------------------------------------------------------------
    def report(self):
        lines = ["INDEPENDENT-ALPHA / RETURN-SPACE ANALYSIS", "=" * 82]
        dim = self.results_.get("dimensionality", {})
        if dim:
            lines.append(
                f"  {dim['n_strategies']} strategies span {dim['numerical_rank']} numerically "
                f"independent directions."
            )
            lines.append(
                f"  {dim['components_for_95pct_variance']} components explain 95% of the "
                f"variance; participation ratio {dim['participation_ratio']:.2f} "
                f"(an effective count of independent streams)."
            )
            lines.append(f"  Condition number {dim['condition_number']:.1f}.")
        for key, title in (("pivoted_qr", "Pivoted QR -- incremental contribution"),
                           ("incremental", "Residual after projecting out all others"),
                           ("gram_schmidt", "Sequential Gram-Schmidt residuals")):
            df = self.results_.get(key)
            if df is not None and not df.empty:
                lines.append(f"\n[{title}]")
                lines.append(df.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        st = self.results_.get("stability")
        if st is not None and not st.empty:
            lines.append("\n[Stability through time]")
            lines.append(f"  modal pivot order : {st.attrs.get('modal_pivot_order','')}")
            lines.append(f"  order held in     : {st.attrs.get('pivot_order_stability',0):.1%} of windows")
            lines.append(f"  first pick held in: {st.attrs.get('first_pick_stability',0):.1%} of windows")
        lines.append(
            "\n  Caveat carried into the selection: a mathematically new direction is not "
            "proof of\n  economic independence. These results are read together with the "
            "hypothesis and the\n  robustness evidence, never on their own."
        )
        return "\n".join(lines)
