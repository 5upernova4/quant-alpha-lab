"""
Portfolio-level backtest: cost an allocation the way it would actually trade.

The easy way to score an allocation is to weight each strategy's net return
series and add them up. That double-charges costs. If alpha_02 goes from +1 to 0
on the same day alpha_03 goes from 0 to +1, one account holding both doesn't
trade at all, but summing pre-costed returns charges both legs.

So allocations are rebuilt from positions:

    combined_position(t) = sum_i weight_i(t) * position_i(t)

and that one netted position goes through the same ExecutionEngine, Portfolio
and PerformanceAnalyzer as a single strategy. Every method pays 0.05% per side
on what it really traded, through the same code. Weight changes don't need a
separate charge: a rebalance changes the combined position, and that change is
what gets costed.
"""

import numpy as np
import pandas as pd

import config
from backtester import Backtester
from execution_engine import ExecutionEngine
from performance import PerformanceAnalyzer
from portfolio import Portfolio


def combine_positions(positions_frame, weights):
    """Net a set of strategy positions into one book-level position.

    `weights` may be a single allocation (Series) or a weight path indexed like
    the positions (DataFrame).
    """
    P = pd.DataFrame(positions_frame)
    if isinstance(weights, pd.DataFrame):
        W = weights.reindex(index=P.index, columns=P.columns).ffill().fillna(0.0)
    else:
        w = pd.Series(weights).reindex(P.columns).fillna(0.0)
        W = pd.DataFrame(np.tile(w.to_numpy(), (len(P), 1)),
                         index=P.index, columns=P.columns)
    return (P * W).sum(axis=1)


def run_allocation(positions_frame, weights, market_frame, label="allocation",
                   cost=None, slippage_bps=None):
    """Backtest an allocation through the standard execution path.

    Returns the same result dictionary shape as Backtester.run, so allocation
    methods and individual strategies can be handed to the same reporting code.
    """
    market = market_frame.reset_index(drop=True)
    combined = combine_positions(positions_frame, weights)
    combined = combined.reset_index(drop=True).clip(
        -config.MAX_GROSS_POSITION, config.MAX_GROSS_POSITION
    )

    engine = ExecutionEngine(cost_per_side=cost, slippage_bps=slippage_bps)
    bt = Backtester(execution_engine=engine, portfolio=Portfolio(),
                    performance=PerformanceAnalyzer())
    executions = bt.execute_signals(combined, market)
    bt.update_portfolio(executions, market)
    results = bt.analyze(bt.portfolio, label=label)
    results["combined_position"] = pd.Series(
        combined.to_numpy(), index=pd.DatetimeIndex(market["date"])
    )
    return results


def netting_benefit(positions_frame, weights, market_frame):
    """What netting saves per year, versus charging each strategy separately."""
    P = pd.DataFrame(positions_frame)
    if isinstance(weights, pd.DataFrame):
        W = weights.reindex(index=P.index, columns=P.columns).ffill().fillna(0.0)
    else:
        w = pd.Series(weights).reindex(P.columns).fillna(0.0)
        W = pd.DataFrame(np.tile(w.to_numpy(), (len(P), 1)),
                         index=P.index, columns=P.columns)

    gross_turnover = float((P.diff().abs() * W).sum(axis=1).mean())
    netted_turnover = float((P * W).sum(axis=1).diff().abs().mean())
    ppy = config.TRADING_DAYS_PER_YEAR
    return {
        "gross_turnover_annual": gross_turnover * ppy,
        "netted_turnover_annual": netted_turnover * ppy,
        "cost_if_charged_separately": gross_turnover * ppy * config.TRANSACTION_COST,
        "cost_as_one_book": netted_turnover * ppy * config.TRANSACTION_COST,
        "annual_saving_from_netting":
            (gross_turnover - netted_turnover) * ppy * config.TRANSACTION_COST,
    }
