"""
FactorModel - how much of a strategy's return is its own, and how much is exposure.

    r_i(t) = alpha_i + beta_i' F(t) + eps_i(t)

There is one instrument and no cross-section, so the usual equity factors
(size, value, momentum across stocks) can't be built. The factors have to come
from the instrument's own time series:

MKT  buy-and-hold return on the same open-to-open convention. Anything that is
     net long picks this up, and alpha_07 is net long most of the time.
VOL  change in trailing 21-day realised vol, scaled to return units. Reversion
     strategies tend to get hurt when a quiet market turns violent; without
     this factor that shows up as (negative) alpha.
REV  a parameter-free contrarian rule: minus the sign of yesterday's move, times
     today's move. This is the part of return every reversion strategy shares.
     Without it, several strategies expressing one idea would each claim alpha.

These use price, which is fine here: this is risk attribution, and no factor
value is ever fed into a trading decision.

OLS with Newey-West standard errors, because daily strategy returns are
autocorrelated and plain OLS t-stats would be too generous.
"""

import numpy as np
import pandas as pd

import config


class FactorModel:
    """Time-series factor decomposition of strategy returns."""

    def __init__(self, ppy=None, nw_lags=None):
        self.ppy = ppy or config.TRADING_DAYS_PER_YEAR
        self.nw_lags = nw_lags
        self.factors_ = None
        self.alpha_ = None
        self.beta_ = None
        self.residuals_ = None
        self.diagnostics_ = None

    # ------------------------------------------------------------------
    @staticmethod
    def build_factors(market_frame, vol_window=21, reversion_lag=1):
        """Construct the systematic factors from the price series.

        Every factor is causal in the sense that matters for attribution: each
        is a *realised return* over the same interval the strategy returns cover,
        so the regression compares like with like.
        """
        m = market_frame.copy()
        idx = pd.DatetimeIndex(m["date"])
        r = pd.Series(m["ret_oo"].fillna(0.0).values, index=idx)

        market = r.copy()

        rv = r.rolling(vol_window, min_periods=vol_window).std()
        vol_change = rv.diff().fillna(0.0)
        # scale to return-like units so the betas are readable
        vol_factor = (vol_change / rv.replace(0.0, np.nan)).fillna(0.0).clip(-1, 1) * r.std()

        # a parameter-free contrarian rule: hold minus yesterday's sign
        reversion = (-np.sign(r.shift(reversion_lag)).fillna(0.0)) * r

        return pd.DataFrame(
            {"MKT": market, "VOL": vol_factor, "REV": reversion}, index=idx
        )

    # ------------------------------------------------------------------
    def fit(self, strategy_returns, factors):
        """Estimate alpha and beta for every strategy column."""
        R = pd.DataFrame(strategy_returns).copy()
        F = pd.DataFrame(factors).copy()
        common = R.index.intersection(F.index)
        R, F = R.loc[common], F.loc[common]

        self.factors_ = F
        X = np.column_stack([np.ones(len(F)), F.to_numpy(dtype=float)])
        names = ["alpha"] + list(F.columns)

        alphas, betas, resids, diags = {}, {}, {}, []
        for col in R.columns:
            y = R[col].to_numpy(dtype=float)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            e = y - X @ coef
            se = self._newey_west_se(X, e)
            t = coef / np.where(se > 0, se, np.nan)

            alphas[col] = float(coef[0])
            betas[col] = dict(zip(F.columns, coef[1:]))
            resids[col] = pd.Series(e, index=common)

            ss_tot = float(((y - y.mean()) ** 2).sum())
            # The stream a desk could actually hold is this strategy with its
            # factor exposures hedged away, and that position earns the intercept.
            # Scoring the bare OLS residual instead would report zero for every
            # strategy, because a residual with a fitted intercept has mean zero
            # by construction.
            hedged = coef[0] + e
            row = {
                "strategy": col,
                "alpha_daily": float(coef[0]),
                "alpha_annual": float(coef[0] * self.ppy),
                "alpha_t": float(t[0]),
                "r_squared": float(1.0 - (e ** 2).sum() / ss_tot) if ss_tot > 0 else 0.0,
                "residual_vol_ann": float(e.std(ddof=1) * np.sqrt(self.ppy)),
                "residual_sharpe": float(hedged.mean() / hedged.std(ddof=1) * np.sqrt(self.ppy))
                if hedged.std(ddof=1) > 0 else 0.0,
            }
            for k, name in enumerate(F.columns, start=1):
                row[f"beta_{name}"] = float(coef[k])
                row[f"t_{name}"] = float(t[k])
            diags.append(row)

        self.alpha_ = pd.Series(alphas)
        self.beta_ = pd.DataFrame(betas).T
        self.residuals_ = pd.DataFrame(resids)
        self.diagnostics_ = pd.DataFrame(diags)
        return self

    def predict(self, strategy_returns=None, factors=None):
        """Systematic component implied by the fitted betas."""
        F = self.factors_ if factors is None else pd.DataFrame(factors)
        if self.beta_ is None:
            raise RuntimeError("FactorModel.fit must be called first.")
        return F.to_numpy(dtype=float) @ self.beta_.T.to_numpy(dtype=float) + self.alpha_.to_numpy()

    def get_alpha(self):
        return self.alpha_

    def get_beta(self):
        return self.beta_

    def get_residuals(self):
        return self.residuals_

    def risk_decomposition(self):
        """Split each strategy's variance into systematic and specific parts."""
        if self.diagnostics_ is None:
            raise RuntimeError("FactorModel.fit must be called first.")
        d = self.diagnostics_.copy()
        d["systematic_share_of_variance"] = d["r_squared"]
        d["specific_share_of_variance"] = 1.0 - d["r_squared"]
        return d[["strategy", "alpha_annual", "alpha_t", "r_squared",
                  "systematic_share_of_variance", "specific_share_of_variance",
                  "residual_vol_ann", "residual_sharpe"]]

    # ------------------------------------------------------------------
    def _newey_west_se(self, X, e):
        """HAC standard errors (Bartlett kernel)."""
        n, k = X.shape
        lags = self.nw_lags
        if lags is None:
            lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        lags = max(0, min(lags, n - k - 1))

        XtX_inv = np.linalg.pinv(X.T @ X)
        u = X * e[:, None]
        S = u.T @ u
        for L in range(1, lags + 1):
            w = 1.0 - L / (lags + 1.0)
            G = u[L:].T @ u[:-L]
            S += w * (G + G.T)
        cov = XtX_inv @ S @ XtX_inv
        return np.sqrt(np.maximum(np.diag(cov), 0.0))

    # ------------------------------------------------------------------
    def report(self):
        if self.diagnostics_ is None:
            return "FactorModel has not been fitted."
        lines = ["FACTOR MODEL -- systematic vs strategy-specific", "=" * 104]
        lines.append("  Factors: MKT (buy-and-hold), VOL (change in realised vol), "
                     "REV (parameter-free contrarian rule)")
        lines.append("  Standard errors are Newey-West; daily strategy returns are autocorrelated.")
        lines.append("")
        lines.append(self.diagnostics_.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
        lines.append("")
        lines.append("[Variance decomposition]")
        lines.append(self.risk_decomposition().to_string(
            index=False, float_format=lambda v: f"{v:8.3f}"))
        return "\n".join(lines)
