"""
PerformanceAnalyzer -- every number that describes a return stream.

One class computes every metric for every strategy, every baseline and every
allocation method, so that a comparison can never be an artefact of two
different people annualising two different ways.

What each metric is actually for
--------------------------------
* Total / annualized return -- how much was made. On its own it says nothing
  about whether the ride was survivable.
* Volatility -- the usual risk proxy. Penalises upside and downside equally,
  which is why it is never used alone here.
* Sharpe -- return per unit of volatility. The default currency of comparison,
  but it punishes a strategy for making money quickly.
* Sortino -- return per unit of *downside* deviation. The honest version of
  Sharpe for anything with asymmetric returns, which mean-reversion strategies
  usually have.
* Max drawdown -- the worst peak-to-trough loss. This is the number that
  decides whether a strategy survives contact with a real risk committee.
* Drawdown duration -- how long the worst one lasted. A 15% drawdown that
  recovers in a month and one that takes two years are not the same product.
* Calmar -- annualized return divided by max drawdown. Return per unit of worst
  pain, rather than per unit of average wobble.
* Turnover and cost drag -- what the strategy pays to exist. A signal with a
  good Sharpe before costs and a negative one after costs is not an alpha.
* Hit rate, profit factor, tail stats -- how the returns are shaped, which is
  what tells you whether a good mean is coming from many small wins or one
  lucky fortnight.
"""

import numpy as np
import pandas as pd

import config


class PerformanceAnalyzer:
    """Computes and reports performance statistics for a return series."""

    def __init__(self, periods_per_year=None, risk_free_rate=None):
        self.ppy = periods_per_year or config.TRADING_DAYS_PER_YEAR
        self.rf = config.RISK_FREE_RATE if risk_free_rate is None else risk_free_rate
        self.metrics_ = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clean(returns):
        s = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        return s

    def _excess(self, returns):
        return self._clean(returns) - self.rf / self.ppy

    @staticmethod
    def equity_from_returns(returns, initial=1.0):
        r = PerformanceAnalyzer._clean(returns)
        return initial * (1.0 + r).cumprod()

    # ------------------------------------------------------------------
    # required interface
    # ------------------------------------------------------------------
    def total_return(self, returns):
        r = self._clean(returns)
        if len(r) == 0:
            return 0.0
        return float((1.0 + r).prod() - 1.0)

    def annualized_return(self, returns):
        """Geometric (CAGR-style) annualisation, not the arithmetic mean.

        Compounding matters: a series that gains 50% then loses 50% has a
        positive arithmetic mean and a negative geometric one. The geometric
        figure is the one an investor actually experiences.
        """
        r = self._clean(returns)
        if len(r) < 2:
            return 0.0
        growth = float((1.0 + r).prod())
        if growth <= 0:
            return -1.0
        return float(growth ** (self.ppy / len(r)) - 1.0)

    def volatility(self, returns, annualize=True):
        r = self._clean(returns)
        if len(r) < 2:
            return 0.0
        v = float(r.std(ddof=1))
        return v * np.sqrt(self.ppy) if annualize else v

    def sharpe_ratio(self, returns):
        e = self._excess(returns)
        if len(e) < 2 or e.std(ddof=1) == 0:
            return 0.0
        return float(e.mean() / e.std(ddof=1) * np.sqrt(self.ppy))

    def sortino_ratio(self, returns, target=0.0):
        """Return per unit of downside deviation.

        Downside deviation is computed against the full sample length, not only
        the losing days. Dividing by the count of losing days alone would make a
        strategy look better simply for having few of them, which is backwards.
        """
        e = self._excess(returns)
        if len(e) < 2:
            return 0.0
        downside = np.minimum(e - target, 0.0)
        dd = float(np.sqrt((downside ** 2).sum() / len(e)))
        if dd == 0:
            return 0.0 if e.mean() <= 0 else float("inf")
        return float(e.mean() / dd * np.sqrt(self.ppy))

    def max_drawdown(self, returns=None, equity=None):
        """Worst peak-to-trough decline, as a negative fraction."""
        eq = self.equity_from_returns(returns) if equity is None else pd.Series(equity, dtype=float)
        if len(eq) < 2:
            return 0.0
        peak = eq.cummax()
        return float((eq / peak - 1.0).min())

    def drawdown_duration(self, returns=None, equity=None):
        """Return (longest_drawdown_periods, current_periods_under_water)."""
        eq = self.equity_from_returns(returns) if equity is None else pd.Series(equity, dtype=float)
        if len(eq) < 2:
            return 0, 0
        peak = eq.cummax()
        under = (eq < peak).to_numpy()
        longest = run = 0
        for flag in under:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        return int(longest), int(run)

    def calmar_ratio(self, returns):
        mdd = abs(self.max_drawdown(returns))
        if mdd < 1e-12:
            return 0.0
        return float(self.annualized_return(returns) / mdd)

    def trade_statistics(self, positions=None, costs=None, returns=None):
        """Turnover, trade count, hit rate and cost drag."""
        stats = {}
        if positions is not None and len(positions) > 1:
            pos = pd.Series(positions, dtype=float).fillna(0.0)
            traded = pos.diff().abs().fillna(pos.abs().iloc[0])
            stats["avg_daily_turnover"] = float(traded.mean())
            stats["annual_turnover"] = float(traded.mean() * self.ppy)
            stats["n_position_changes"] = int((traded > 1e-12).sum())
            stats["time_in_market"] = float((pos.abs() > 1e-12).mean())
            stats["avg_abs_position"] = float(pos.abs().mean())
            stats["pct_long"] = float((pos > 1e-12).mean())
            stats["pct_short"] = float((pos < -1e-12).mean())
        if costs is not None:
            c = pd.Series(costs, dtype=float)
            stats["total_cost"] = float(c.sum())
            stats["annualized_cost_drag"] = float(c.mean() * self.ppy)
        if returns is not None:
            r = self._clean(returns)
            wins, losses = r[r > 0], r[r < 0]
            # Two hit rates, because one of them lies about event strategies. A
            # strategy that is flat four days in five scores a terrible
            # unconditional hit rate purely for sitting on its hands; the
            # conditional figure asks only about the days it actually had risk on.
            stats["hit_rate"] = float((r > 0).mean()) if len(r) else 0.0
            if positions is not None and len(positions) >= len(r):
                active = pd.Series(positions, dtype=float).fillna(0.0).abs().to_numpy()[: len(r)] > 1e-12
                stats["hit_rate_active"] = float((r.to_numpy()[active] > 0).mean()) if active.any() else 0.0
                stats["pct_days_active"] = float(active.mean())
            stats["avg_win"] = float(wins.mean()) if len(wins) else 0.0
            stats["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
            stats["profit_factor"] = (
                float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.inf
            )
            stats["skew"] = float(r.skew()) if len(r) > 2 else 0.0
            stats["kurtosis"] = float(r.kurtosis()) if len(r) > 3 else 0.0
            stats["worst_day"] = float(r.min()) if len(r) else 0.0
            stats["best_day"] = float(r.max()) if len(r) else 0.0
            stats["var_95"] = float(r.quantile(0.05)) if len(r) > 20 else 0.0
            stats["cvar_95"] = float(r[r <= r.quantile(0.05)].mean()) if len(r) > 20 else 0.0
        return stats

    # ------------------------------------------------------------------
    def compute_all(self, returns, positions=None, costs=None, label=""):
        """Return every metric as a flat dict, ready for a results table."""
        r = self._clean(returns)
        longest_dd, current_dd = self.drawdown_duration(r)
        m = {
            "label": label,
            "n_periods": int(len(r)),
            "total_return": self.total_return(r),
            "annualized_return": self.annualized_return(r),
            "volatility": self.volatility(r),
            "sharpe": self.sharpe_ratio(r),
            "sortino": self.sortino_ratio(r),
            "max_drawdown": self.max_drawdown(r),
            "max_dd_duration_days": longest_dd,
            "current_dd_duration_days": current_dd,
            "calmar": self.calmar_ratio(r),
        }
        m.update(self.trade_statistics(positions=positions, costs=costs, returns=r))
        self.metrics_ = m
        return m

    def report(self, returns=None, positions=None, costs=None, label="Strategy", equity_curve=None):
        """Combined performance report as printable text."""
        if returns is None and equity_curve is not None:
            eq = pd.Series(equity_curve, dtype=float)
            returns = eq.pct_change().dropna()
        m = self.compute_all(returns, positions=positions, costs=costs, label=label)
        w = 34
        lines = [f"PERFORMANCE -- {label}", "=" * 62]
        rows = [
            ("Periods", f"{m['n_periods']}"),
            ("Total return", f"{m['total_return']:>10.2%}"),
            ("Annualized return", f"{m['annualized_return']:>10.2%}"),
            ("Annualized volatility", f"{m['volatility']:>10.2%}"),
            ("Sharpe ratio", f"{m['sharpe']:>10.2f}"),
            ("Sortino ratio", f"{m['sortino']:>10.2f}"),
            ("Calmar ratio", f"{m['calmar']:>10.2f}"),
            ("Max drawdown", f"{m['max_drawdown']:>10.2%}"),
            ("Max drawdown duration (days)", f"{m['max_dd_duration_days']:>10d}"),
        ]
        if "hit_rate" in m:
            rows += [
                ("Hit rate (all days)", f"{m['hit_rate']:>10.2%}"),
                ("Hit rate (days with a position)",
                 f"{m.get('hit_rate_active', float('nan')):>10.2%}"),
                ("Profit factor", f"{m['profit_factor']:>10.2f}"),
                ("Return skew", f"{m['skew']:>10.2f}"),
                ("Daily VaR (95%)", f"{m['var_95']:>10.2%}"),
                ("Daily CVaR (95%)", f"{m['cvar_95']:>10.2%}"),
            ]
        if "annual_turnover" in m:
            rows += [
                ("Annual turnover (x notional)", f"{m['annual_turnover']:>10.1f}"),
                ("Time in market", f"{m['time_in_market']:>10.2%}"),
                ("Long / short split", f"{m['pct_long']:>6.1%} / {m['pct_short']:.1%}"),
            ]
        if "annualized_cost_drag" in m:
            rows.append(("Annualized cost drag", f"{m['annualized_cost_drag']:>10.2%}"))
        for k, v in rows:
            lines.append(f"  {k:<{w}}{v}")
        return "\n".join(lines)
