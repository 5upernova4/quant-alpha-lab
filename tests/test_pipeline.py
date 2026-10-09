"""Checks on the Task 1-3 pipeline, run on synthetic data."""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import config
from backtester import Backtester
from feature_engine import FeatureEngine, LookAheadError
from strategies import BaselineStrategy, build_all


def test_uses_synthetic_data_not_the_real_files(tmp_root):
    assert str(config.DATA_DIR).startswith(str(tmp_root))
    assert str(config.RESULTS_DIR).startswith(str(tmp_root))


def test_cleaner_fixes_the_planted_defects(ctx):
    report = ctx.cleaner.get_report()
    assert "mixed date formats" in report
    assert "duplicate price rows" in report
    assert "shuffled" in report
    assert "signals but no price row" in report
    assert ctx.prices["date"].is_monotonic_increasing
    assert ctx.prices["date"].is_unique
    assert len(ctx.prices) == 986


def test_decision_frame_has_no_prices_and_is_lagged(ctx):
    for col in config.FORBIDDEN_STRATEGY_INPUTS:
        assert col not in ctx.decision.columns
    assert ctx.features.validate_no_lookahead(ctx.decision, ctx.signals_clean)
    src = ctx.decision["information_asof"].dropna()
    assert (src < ctx.decision.loc[src.index, "date"]).all()


def test_lookahead_check_catches_a_leak(ctx):
    leaky = ctx.decision.copy()
    leaky["information_asof"] = leaky["date"]       # pretend we used today's signals
    with pytest.raises(LookAheadError):
        FeatureEngine().validate_no_lookahead(leaky, ctx.signals_clean)


def test_strategy_refuses_price_columns(ctx):
    frame = ctx.decision.copy()
    frame["close"] = 1.0
    with pytest.raises(ValueError):
        BaselineStrategy().generate_signal(frame)


def test_vectorised_engine_matches_per_candle_loop(ctx):
    diff = Backtester().assert_matches_reference(ctx.decision, BaselineStrategy(), ctx.market)
    assert diff < 1e-12


def test_round_trip_cost_is_ten_bps():
    from execution_engine import ExecutionEngine
    eng = ExecutionEngine()
    one_side = float(eng.apply_transaction_cost(np.array([1.0]))[0])
    assert one_side == pytest.approx(0.0005)


def test_fitting_is_refused_outside_dev(ctx):
    alphas = build_all()
    for split in ("holdout", "full"):
        with pytest.raises(ValueError):
            ctx.fit_strategies(alphas, split=split)


def test_positions_are_bounded(ctx):
    alphas = ctx.fit_strategies(build_all(), split="dev")
    for s in alphas.values():
        pos = ctx.positions_on_full_history(s)
        assert pos.abs().max() <= config.MAX_GROSS_POSITION + 1e-12
        assert not pos.isna().any()


def test_main_runs_end_to_end():
    """The whole three-task run on synthetic data, with fewer resamples."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "main.py", "--quick"], cwd=root,
                         capture_output=True, text=True, env=os.environ.copy(), timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "RUN COMPLETE" in out.stdout
    assert (config.RESULTS_DIR / "task3_comparison_holdout.csv").exists()
