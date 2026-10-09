"""
Test setup: build a synthetic dataset once and point the pipeline at it.

This has to happen before anything imports config, because config reads the
data and results paths when it is first imported. pytest loads conftest.py
first, so setting the environment here is enough.
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

_tmp = Path(tempfile.mkdtemp(prefix="quant_alpha_lab_"))
os.environ["QUANT_DATA_DIR"] = str(_tmp / "data")
os.environ["QUANT_RESULTS_DIR"] = str(_tmp / "results")

import make_synthetic_data  # noqa: E402

make_synthetic_data.write(_tmp / "data", seed=0)

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def ctx():
    from research_context import ResearchContext
    return ResearchContext(verbose=False)


@pytest.fixture(scope="session")
def tmp_root():
    return _tmp
