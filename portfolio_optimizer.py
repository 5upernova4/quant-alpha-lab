"""
PortfolioOptimizer - how much of each strategy to hold.

All the static methods live here so they get the same returns, the same risk
model and the same constraints. Otherwise the Task 3 comparison means nothing.

equal_weight        1/N. Hard to beat, and it estimates nothing.
inverse_volatility  weight by 1/sigma. Ignores correlation.
risk_parity         weights where every strategy contributes the same share of
                    *portfolio* risk, correlations included. This is our
                    "risk-based" baseline.
max_sharpe          maximise shrunk expected return / risk. Our
                    "optimisation-based" baseline.
mean_variance       mu'w - (lambda/2) w'Sw, with an optional turnover penalty.

Assumptions
-----------
* Sample means over ~770 days have a standard error about as big as the means
  themselves. Every optimising method shrinks them towards the cross-sectional
  average (James-Stein style; main.py uses shrinkage=0.5, i.e. halfway),
  otherwise the optimiser just piles into whichever strategy got lucky.
* The covariance is shrunk towards a constant-correlation target by the same
  amount (Ledoit-Wolf style), for the same reason.
* Weights are long-only, sum to 1 and are capped at 60%. Shorting a whole
  strategy is a strong claim on this little data.
* risk_contributions() reports concentration in risk terms, because weight-based
  concentration is misleading when the strategies' vols differ by 3x.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config


class PortfolioOptimizer:
    """Allocates capital across strategies under an explicit objective."""

    METHODS = ("equal_weight", "inverse_volatility", "risk_parity",
               "max_sharpe", "mean_variance")

    def __init__(self, ppy=None, shrinkage=0.5, allow_short=False,
                 max_weight=0.60, risk_aversion=5.0, turnover_penalty=0.0):
        self.ppy = ppy or config.TRADING_DAYS_PER_YEAR
        self.shrinkage = shrinkage
        self.allow_short = allow_short
        self.max_weight = max_weight
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty

        self.returns_ = None
        self.mu_ = None
        self.cov_ = None
        self.weights_ = None
        self.method_ = None
        self.diagnostics_ = {}

    # ------------------------------------------------------------------
    def fit(self, strategy_returns, risk_model=None, constraints=None):
        """Configure the allocation problem: estimate mu and Sigma, set bounds."""
        R = pd.DataFrame(strategy_returns).dropna(how="any")
        self.returns_ = R
        self.names_ = list(R.columns)
        n = len(self.names_)

        raw_mu = R.mean().to_numpy(dtype=float)
        # with this little history, differences between strategy means are
        # mostly noise, so pull them towards their average
        grand = raw_mu.mean()
        self.mu_ = (1 - self.shrinkage) * raw_mu + self.shrinkage * grand

        sample_cov = R.cov().to_numpy(dtype=float) if risk_model is None else np.asarray(risk_model)
        self.cov_ = self._shrink_covariance(sample_cov)

        if constraints:
            self.max_weight = constraints.get("max_weight", self.max_weight)
            self.allow_short = constraints.get("allow_short", self.allow_short)
            self.risk_aversion = constraints.get("risk_aversion", self.risk_aversion)
            self.turnover_penalty = constraints.get("turnover_penalty", self.turnover_penalty)

        self.diagnostics_ = {
            "n_strategies": n, "n_observations": len(R),
            "observations_per_parameter": len(R) / max(1, n * (n + 3) / 2),
            "mean_shrinkage": self.shrinkage,
            "raw_mu_annual": dict(zip(self.names_, raw_mu * self.ppy)),
            "shrunk_mu_annual": dict(zip(self.names_, self.mu_ * self.ppy)),
            "condition_number": float(np.linalg.cond(self.cov_)),
        }
        return self

    def _shrink_covariance(self, S):
        """Shrink towards a constant-correlation matrix with the same vols, by `shrinkage`."""
        n = S.shape[0]
        var = np.diag(S).copy()
        sd = np.sqrt(np.maximum(var, 1e-18))
        corr = S / np.outer(sd, sd)
        off = corr[~np.eye(n, dtype=bool)]
        avg_corr = float(off.mean()) if off.size else 0.0
        target_corr = np.full((n, n), avg_corr)
        np.fill_diagonal(target_corr, 1.0)
        target = target_corr * np.outer(sd, sd)

        lam = self.shrinkage
        shrunk = (1 - lam) * S + lam * target
        # keep it positive definite
        w, V = np.linalg.eigh(shrunk)
        w = np.maximum(w, 1e-12)
        return V @ np.diag(w) @ V.T

    # ------------------------------------------------------------------
    def _bounds(self, n):
        lo = -self.max_weight if self.allow_short else 0.0
        return [(lo, self.max_weight)] * n

    def optimize(self, method="max_sharpe", current_weights=None):
        """Determine portfolio weights under the chosen method."""
        if self.returns_ is None:
            raise RuntimeError("PortfolioOptimizer.fit must be called first.")
        n = len(self.names_)
        self.method_ = method

        if method == "equal_weight":
            w = np.full(n, 1.0 / n)
        elif method == "inverse_volatility":
            sd = np.sqrt(np.diag(self.cov_))
            inv = 1.0 / np.where(sd > 0, sd, np.inf)
            w = inv / inv.sum()
        elif method == "risk_parity":
            w = self._solve_risk_parity()
        elif method in ("max_sharpe", "mean_variance"):
            w = self._solve(method, current_weights)
        else:
            raise ValueError(f"Unknown method {method!r}. Use one of {self.METHODS}.")

        w = np.clip(w, -self.max_weight if self.allow_short else 0.0, self.max_weight)
        total = w.sum()
        w = w / total if abs(total) > 1e-12 else np.full(n, 1.0 / n)
        self.weights_ = pd.Series(w, index=self.names_, name=method)
        return self.weights_

    def _solve_risk_parity(self):
        """Equalise each strategy's contribution to total portfolio risk."""
        n = len(self.names_)
        cov = self.cov_

        def objective(w):
            port_var = float(w @ cov @ w)
            if port_var <= 0:
                return 1e9
            contrib = w * (cov @ w) / np.sqrt(port_var)
            return float(((contrib - contrib.mean()) ** 2).sum())

        w0 = np.full(n, 1.0 / n)
        res = minimize(objective, w0, method="SLSQP", bounds=[(1e-6, 1.0)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                       options={"maxiter": 500, "ftol": 1e-12})
        return res.x if res.success else w0

    def _solve(self, method, current_weights=None):
        n = len(self.names_)
        mu, cov = self.mu_, self.cov_
        prev = np.zeros(n) if current_weights is None else np.asarray(current_weights, dtype=float)

        def neg_sharpe(w):
            vol = np.sqrt(max(1e-18, float(w @ cov @ w)))
            turn = self.turnover_penalty * np.abs(w - prev).sum()
            return -((float(mu @ w) - turn) / vol)

        def neg_utility(w):
            turn = self.turnover_penalty * np.abs(w - prev).sum()
            return -(float(mu @ w) - turn - 0.5 * self.risk_aversion * float(w @ cov @ w))

        objective = neg_sharpe if method == "max_sharpe" else neg_utility
        w0 = np.full(n, 1.0 / n)
        res = minimize(objective, w0, method="SLSQP", bounds=self._bounds(n),
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                       options={"maxiter": 800, "ftol": 1e-12})
        return res.x if res.success else w0

    # ------------------------------------------------------------------
    def evaluate(self, weights, returns=None):
        """Evaluate a candidate allocation on the fitted (or supplied) returns."""
        R = self.returns_ if returns is None else pd.DataFrame(returns)
        w = np.asarray(pd.Series(weights).reindex(R.columns).fillna(0.0), dtype=float)
        port = pd.Series(R.to_numpy(dtype=float) @ w, index=R.index)
        vol = float(port.std(ddof=1) * np.sqrt(self.ppy))
        eff_n = float(1.0 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0
        return {
            "ann_return": float(port.mean() * self.ppy),
            "ann_vol": vol,
            "sharpe": float(port.mean() / port.std(ddof=1) * np.sqrt(self.ppy))
            if port.std(ddof=1) > 0 else 0.0,
            "herfindahl": float(np.sum(w ** 2)),
            "effective_n_strategies": eff_n,
            "max_weight": float(np.max(np.abs(w))),
            "returns": port,
        }

    def risk_contributions(self, weights):
        """Share of portfolio variance coming from each strategy (sums to 1)."""
        w = np.asarray(pd.Series(weights).reindex(self.names_).fillna(0.0), dtype=float)
        port_var = float(w @ self.cov_ @ w)
        if port_var <= 0:
            return pd.Series(np.nan, index=self.names_)
        return pd.Series(w * (self.cov_ @ w) / port_var, index=self.names_)

    def get_weights(self):
        return self.weights_

    def all_methods(self, current_weights=None):
        """Solve every method and return a weight matrix for comparison."""
        out = {}
        for m in self.METHODS:
            out[m] = self.optimize(m, current_weights=current_weights)
        return pd.DataFrame(out)

    def report(self):
        if self.weights_ is None:
            return "PortfolioOptimizer has not produced weights yet."
        d = self.diagnostics_
        lines = ["PORTFOLIO OPTIMISATION", "=" * 78]
        lines.append(f"  strategies                 : {d.get('n_strategies')}")
        lines.append(f"  observations               : {d.get('n_observations')}")
        lines.append(f"  observations per parameter : {d.get('observations_per_parameter', 0):.1f}")
        lines.append(f"  mean shrinkage             : {d.get('mean_shrinkage')}")
        lines.append(f"  covariance condition number: {d.get('condition_number', 0):.1f}")
        lines.append(f"  max weight / shorting      : {self.max_weight} / {self.allow_short}")
        lines.append("\n[Expected returns, annualised]")
        raw, shrunk = d.get("raw_mu_annual", {}), d.get("shrunk_mu_annual", {})
        for k in raw:
            lines.append(f"    {k:<28} sample {raw[k]:+7.2%}   shrunk {shrunk[k]:+7.2%}")
        lines.append(f"\n[Weights -- {self.method_}]")
        for k, v in self.weights_.items():
            lines.append(f"    {k:<28} {v:7.2%}")
        return "\n".join(lines)
