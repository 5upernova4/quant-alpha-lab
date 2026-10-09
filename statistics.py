"""
StatisticalTester -- tests whether an alpha is different from noise.

About the filename: the Technical Documentation requires `statistics.py`, which
hides Python's standard-library module of the same name once this project is
on the path. To avoid breaking other code, the stdlib module is loaded from its
own location and its public names are re-exported here, so
`import statistics` still works for third-party code.

What is tested and why
----------------------
Daily strategy returns break most t-test assumptions. They are autocorrelated,
fat-tailed and skewed, and we looked at the data before choosing the
strategies. Each test below handles one of these problems; none handles all of
them, so several are run:

* Plain t-test -- the baseline, for comparison with the others.
* Newey-West t-statistic -- corrects the standard error for autocorrelation and
  heteroskedasticity. Strategies that hold for days have correlated daily
  returns, and ignoring that inflates significance.
* Stationary bootstrap -- resamples blocks of returns instead of single days,
  which keeps the autocorrelation, and gives a confidence interval that does
  not assume normality.
* Signal permutation test -- the key test. It keeps the returns and shuffles
  the strategy's positions in blocks. This breaks the timing but keeps
  turnover and position sizes. It asks: does the performance come from when
  the strategy trades, or just from being in a rising market?
* Probabilistic Sharpe ratio -- the probability the true Sharpe is above zero,
  adjusted for skew, kurtosis and sample length.
* Deflated Sharpe / Benjamini-Hochberg -- we tested several strategies, so the
  best one is partly best by luck. Both correct for this, in different ways.
"""

import importlib.util as _ilu
import sysconfig as _sysconfig
from pathlib import Path as _Path

import numpy as np
import pandas as pd
from scipy import stats as _sps

import config

# ---------------------------------------------------------------------------
# Re-export the standard library's `statistics` so this file can shadow it
# safely. If it cannot be found we carry on; nothing here needs it.
# ---------------------------------------------------------------------------
try:
    _stdlib = _Path(_sysconfig.get_paths()["stdlib"]) / "statistics.py"
    if _stdlib.exists():
        _spec = _ilu.spec_from_file_location("_stdlib_statistics", _stdlib)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _names = getattr(_mod, "__all__", None) or [
            _k for _k in vars(_mod) if not _k.startswith("_")
        ]
        for _n in _names:
            globals().setdefault(_n, getattr(_mod, _n))
except Exception:  # pragma: no cover - best effort only
    pass


class StatisticalTester:
    """Significance and multiple-testing machinery for strategy return series."""

    def __init__(self, n_boot=2000, block_size=10, seed=None, ppy=None):
        self.n_boot = n_boot
        self.block_size = block_size
        self.seed = config.RANDOM_SEED if seed is None else seed
        self.ppy = ppy or config.TRADING_DAYS_PER_YEAR
        self.results_ = {}

    # ------------------------------------------------------------------
    @staticmethod
    def _clean(x):
        return pd.Series(x, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()

    def t_test(self, returns):
        """Plain one-sample t-test that the mean return is zero."""
        r = self._clean(returns)
        if len(r) < 3:
            return {"t_stat": 0.0, "p_value": 1.0, "mean": 0.0, "n": len(r)}
        t, p = _sps.ttest_1samp(r, 0.0)
        return {"t_stat": float(t), "p_value": float(p), "mean": float(r.mean()), "n": int(len(r))}

    def newey_west_t(self, returns, lags=None):
        """Autocorrelation- and heteroskedasticity-robust t-statistic.

        Uses the Bartlett kernel. The default lag length is the usual
        floor(4*(n/100)^(2/9)) rule, which grows slowly with the sample.
        """
        r = self._clean(returns).to_numpy()
        n = len(r)
        if n < 10:
            return {"t_stat": 0.0, "p_value": 1.0, "lags": 0}
        if lags is None:
            lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        lags = max(0, min(lags, n - 2))

        e = r - r.mean()
        gamma0 = float(e @ e) / n
        var = gamma0
        for L in range(1, lags + 1):
            w = 1.0 - L / (lags + 1.0)
            cov = float(e[L:] @ e[:-L]) / n
            var += 2.0 * w * cov
        if var <= 0:
            return {"t_stat": 0.0, "p_value": 1.0, "lags": lags}
        se = np.sqrt(var / n)
        t = float(r.mean() / se)
        p = float(2.0 * (1.0 - _sps.norm.cdf(abs(t))))
        return {"t_stat": t, "p_value": p, "lags": int(lags), "se": float(se)}

    def stationary_bootstrap(self, returns, statistic=None, alpha=0.05):
        """Politis-Romano stationary bootstrap confidence interval.

        Block lengths are geometric with mean `block_size`. This keeps the
        resampled series stationary and keeps short-range dependence.
        """
        r = self._clean(returns).to_numpy()
        n = len(r)
        if n < 30:
            return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}

        statistic = statistic or (lambda x: x.mean() / x.std(ddof=1) * np.sqrt(self.ppy)
                                  if x.std(ddof=1) > 0 else 0.0)
        rng = np.random.default_rng(self.seed)
        p_switch = 1.0 / self.block_size
        draws = np.empty(self.n_boot)

        for b in range(self.n_boot):
            idx = np.empty(n, dtype=int)
            i = rng.integers(n)
            for k in range(n):
                idx[k] = i
                if rng.random() < p_switch:
                    i = rng.integers(n)
                else:
                    i = (i + 1) % n
            draws[b] = statistic(r[idx])

        point = float(statistic(r))
        lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
        # two-sided bootstrap p-value for "statistic is zero"
        # Floor at 1/n_boot: that is the smallest p-value n_boot resamples can
        # resolve, so printing 0.000 would overstate the precision.
        p = float(2.0 * min((draws <= 0).mean(), (draws >= 0).mean()))
        p = max(p, 1.0 / self.n_boot)
        return {
            "point": point,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "p_value": min(1.0, p),
            "n_boot": self.n_boot,
        }

    def permutation_test(self, positions, asset_returns, costs_per_side=None, n_perm=1000):
        """Block-permutation test on the *timing* of the positions.

        The position path is cut into blocks and reordered. Turnover and
        position sizes stay the same; the link between position and return is
        broken. If the real Sharpe does not beat this null, the strategy is not
        timing anything, it is just holding exposure.
        """
        pos = self._clean(positions).to_numpy()
        ret = pd.Series(asset_returns, dtype=float).fillna(0.0).to_numpy()
        n = min(len(pos), len(ret))
        pos, ret = pos[:n], ret[:n]
        if n < 60:
            return {"observed": 0.0, "p_value": 1.0, "null_mean": 0.0}

        c = config.TRANSACTION_COST if costs_per_side is None else costs_per_side

        def sharpe_of(p):
            traded = np.abs(np.diff(np.concatenate([[0.0], p])))
            net = p * ret - c * traded
            sd = net.std(ddof=1)
            return net.mean() / sd * np.sqrt(self.ppy) if sd > 0 else 0.0

        observed = sharpe_of(pos)
        rng = np.random.default_rng(self.seed)
        bs = self.block_size
        n_blocks = int(np.ceil(n / bs))
        null = np.empty(n_perm)
        for i in range(n_perm):
            order = rng.permutation(n_blocks)
            shuffled = np.concatenate([pos[b * bs:(b + 1) * bs] for b in order])[:n]
            null[i] = sharpe_of(shuffled)

        # Always one-sided, upper tail. We ask whether the timing earned more
        # than shuffled timing. Switching tails by sign would give a losing
        # strategy a small p-value for being reliably bad.
        p = float((null >= observed).mean())
        return {
            "observed": float(observed),
            "null_mean": float(null.mean()),
            "null_std": float(null.std(ddof=1)),
            "p_value": float(min(1.0, max(p, 1.0 / n_perm))),
            "n_perm": n_perm,
        }

    def probabilistic_sharpe_ratio(self, returns, benchmark_sr=0.0):
        """Probability the true Sharpe exceeds `benchmark_sr`.

        Adjusts for skew and excess kurtosis. Without this, the Sharpe
        standard error is too small for strategies with uneven returns.
        """
        r = self._clean(returns)
        n = len(r)
        if n < 30 or r.std(ddof=1) == 0:
            return {"psr": 0.5, "sharpe_daily": 0.0, "n": int(n)}
        sr = float(r.mean() / r.std(ddof=1))
        g3 = float(r.skew())
        # pandas .kurtosis() returns excess kurtosis, but the Bailey-Lopez de
        # Prado formula uses raw kurtosis, so add 3 back. Using excess kurtosis
        # would understate the Sharpe variance and overstate the probability.
        g4_raw = float(r.kurtosis()) + 3.0
        sr_bm = benchmark_sr / np.sqrt(self.ppy)
        denom = np.sqrt(max(1e-12, 1.0 - g3 * sr + ((g4_raw - 1.0) / 4.0) * sr ** 2))
        psr = float(_sps.norm.cdf((sr - sr_bm) * np.sqrt(n - 1) / denom))
        return {"psr": psr, "sharpe_daily": sr, "sharpe_annual": sr * np.sqrt(self.ppy),
                "skew": g3, "excess_kurtosis": g4_raw - 3.0, "n": int(n)}

    def deflated_sharpe_ratio(self, returns, n_trials, variance_of_trial_sharpes=None):
        """Sharpe adjusted for the fact that it is the best of `n_trials`.

        Picking the best of several candidates inflates its Sharpe even when
        none has an edge. This computes the Sharpe expected from the best of N
        by luck alone, and checks whether ours beats it.
        """
        r = self._clean(returns)
        if len(r) < 30 or n_trials < 1:
            return {"dsr": 0.5, "expected_max_sharpe": 0.0}
        # The variance of the trial Sharpes drives the correction: candidates
        # with similar scores give a lower "best of N" bar than widely spread
        # ones. The caller should pass the observed spread; 0.5 is only a
        # fallback when a single strategy is tested.
        v = 0.5 if variance_of_trial_sharpes is None else float(variance_of_trial_sharpes)
        e = 0.5772156649015329                      # Euler-Mascheroni
        n = max(2, int(n_trials))
        expected_max = np.sqrt(v) * (
            (1 - e) * _sps.norm.ppf(1 - 1.0 / n) + e * _sps.norm.ppf(1 - 1.0 / (n * np.e))
        )
        psr = self.probabilistic_sharpe_ratio(r, benchmark_sr=expected_max)
        return {
            "dsr": psr["psr"],
            "expected_max_sharpe_under_null": float(expected_max),
            "observed_sharpe": psr.get("sharpe_annual", 0.0),
            "n_trials": n,
        }

    @staticmethod
    def benjamini_hochberg(p_values, alpha=0.05):
        """False-discovery-rate correction across the strategies tested."""
        p = np.asarray(list(p_values), dtype=float)
        n = len(p)
        if n == 0:
            return np.array([]), np.array([], dtype=bool)
        order = np.argsort(p)
        ranked = p[order]
        adj = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
        adj = np.clip(adj, 0, 1)
        out = np.empty(n)
        out[order] = adj
        return out, out <= alpha

    # ------------------------------------------------------------------
    def full_battery(self, returns, positions=None, asset_returns=None, label="",
                     n_trials=1, trial_sharpe_variance=None):
        """Run every applicable test and return a flat dict."""
        r = self._clean(returns)
        res = {"label": label, "n": int(len(r))}
        res.update({f"t_{k}": v for k, v in self.t_test(r).items()})
        res.update({f"nw_{k}": v for k, v in self.newey_west_t(r).items()})
        res.update({f"boot_{k}": v for k, v in self.stationary_bootstrap(r).items()})
        res.update({f"psr_{k}": v for k, v in self.probabilistic_sharpe_ratio(r).items()})
        res.update({f"dsr_{k}": v for k, v in
                    self.deflated_sharpe_ratio(r, n_trials, trial_sharpe_variance).items()})
        if positions is not None and asset_returns is not None:
            res.update({f"perm_{k}": v for k, v in
                        self.permutation_test(positions, asset_returns).items()})
        self.results_[label] = res
        return res

    def report(self, results=None):
        """Printable significance table."""
        rows = results if results is not None else list(self.results_.values())
        if not rows:
            return "No statistical results computed."
        lines = ["STATISTICAL SIGNIFICANCE", "=" * 104]
        lines.append(
            f"{'strategy':<34}{'t':>7}{'NW t':>8}{'NW p':>8}"
            f"{'boot SR':>9}{'boot 95% CI':>20}{'PSR':>7}{'DSR':>7}{'perm p':>8}"
        )
        lines.append("-" * 104)
        for r in rows:
            ci = f"[{r.get('boot_ci_low', 0):.2f}, {r.get('boot_ci_high', 0):.2f}]"
            lines.append(
                f"{str(r.get('label', ''))[:33]:<34}"
                f"{r.get('t_t_stat', 0):>7.2f}{r.get('nw_t_stat', 0):>8.2f}"
                f"{r.get('nw_p_value', 1):>8.3f}{r.get('boot_point', 0):>9.2f}"
                f"{ci:>20}{r.get('psr_psr', 0):>7.2f}{r.get('dsr_dsr', 0):>7.2f}"
                f"{r.get('perm_p_value', float('nan')):>8.3f}"
            )
        return "\n".join(lines)
