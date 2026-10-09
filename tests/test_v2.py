"""Checks on the v2 research code, run on synthetic data."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import config
import v2_portfolio as v2
from strategies import ALPHA_REGISTRY, TASK3_ADDITIONS, V2_ADDITIONS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_v1_registries_unchanged():
    # main.py must keep reproducing the submitted results
    assert list(ALPHA_REGISTRY) == [f"alpha_0{i}" for i in range(1, 7)]
    assert list(TASK3_ADDITIONS) == ["alpha_07"]
    assert list(V2_ADDITIONS) == ["alpha_08"]


def test_alpha08_reads_only_its_two_signals(ctx):
    s = v2.new_strategies(["alpha_08"])
    ctx.fit_strategies(s, split="dev")
    a8 = s["alpha_08"]
    base = v2.position_frame(ctx, s)["alpha_08"]
    assert base.abs().max() <= 1.0

    scrambled = ctx.decision.copy()
    rng = np.random.default_rng(0)
    for c in config.ALL_SIGNALS:
        if c not in a8.signals_used:
            scrambled[c] = rng.permutation(scrambled[c].to_numpy())
    again = v2.position_frame(ctx, s, decision=scrambled)["alpha_08"]
    assert np.allclose(base.to_numpy(), again.to_numpy())

    leaky = ctx.decision.copy()
    leaky["open"] = 1.0
    with pytest.raises(ValueError):
        a8.generate_signal(leaky)


def test_fit_until_refuses_the_holdout(ctx):
    with pytest.raises(ValueError):
        v2.fit_until(ctx, v2.new_strategies(["alpha_02"]), "2021-03-01")


def test_fit_until_matches_project_fit_on_dev(ctx):
    keys = v2.V1_KEYS + ["alpha_08"]
    a = v2.fit_until(ctx, v2.new_strategies(keys), config.DEV_END_DATE)
    b = v2.new_strategies(keys)
    ctx.fit_strategies(b, split="dev")
    assert np.array_equal(v2.position_frame(ctx, a).to_numpy(),
                          v2.position_frame(ctx, b).to_numpy())


@pytest.mark.parametrize("overlay", v2.OVERLAYS)
def test_every_overlay_passes_the_truncation_test(ctx, overlay):
    cfg = ("v1_set+alpha_08", "inverse_vol", overlay)
    strategies, P, pos, info = v2.fit_frozen(ctx, cfg)
    assert np.abs(pos).max() <= config.MAX_GROSS_POSITION + 1e-12
    assert v2.causality_check(ctx, cfg, strategies)["status"] == "PASS"


def test_no_trade_band_only_trades_past_the_band():
    out = v2.no_trade_band([0.0, 0.03, 0.2, 0.18, -0.5], 0.05)
    assert out[1] == 0.0                     # inside the band, no trade
    assert out[2] == pytest.approx(0.15)     # moved to the band edge
    assert out[3] == pytest.approx(0.15)
    assert out[4] == pytest.approx(-0.45)


def test_run_v2_both_stages():
    env = os.environ.copy()
    for stage in ("dev", "holdout"):
        out = subprocess.run([sys.executable, "run_v2.py", "--stage", stage], cwd=ROOT,
                             capture_output=True, text=True, env=env, timeout=900)
        assert out.returncode == 0, out.stderr[-3000:]
    frozen = json.loads((config.RESULTS_DIR / "v2" / "frozen_config.json").read_text())
    assert abs(sum(frozen["weights"].values()) - 1.0) < 1e-9
    assert (config.RESULTS_DIR / "v2" / "v1_vs_v2.csv").exists()
