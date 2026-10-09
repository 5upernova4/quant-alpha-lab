"""Strategy implementations.

baseline_strategy     the naive rule required by Task 1
alpha_01 .. alpha_06  the Task 2 candidates, one hypothesis each
alpha_07              added at Task 3, see the note below
alpha_08              added for v2, after the competition (v2_portfolio.py)

alpha_07 is kept out of ALPHA_REGISTRY on purpose. The Task 2 candidate set and
its selection were frozen on 18 Sep, before the holdout was opened. alpha_07 was
written after that, so letting it into the Task 2 run would rewrite a result that
has already been submitted (it pushes alpha_03 out of the selected set). It is
added to the Task 3 book instead and reported as a late addition.
"""

from strategies.baseline_strategy import BaselineStrategy
from strategies.alpha_01 import Alpha01TrendStretchReversion
from strategies.alpha_02 import Alpha02OscillatorReversal
from strategies.alpha_03 import Alpha03TrendStateFade
from strategies.alpha_04 import Alpha04BandPositionReversion
from strategies.alpha_05 import Alpha05VolumeFlowContinuation
from strategies.alpha_06 import Alpha06VolatilityRegimeReversion
from strategies.alpha_07 import Alpha07DriftAnchoredDualReversion
from strategies.alpha_08 import Alpha08NextDayContinuation


# Task 2 candidates, exactly as submitted
ALPHA_REGISTRY = {
    "alpha_01": Alpha01TrendStretchReversion,
    "alpha_02": Alpha02OscillatorReversal,
    "alpha_03": Alpha03TrendStateFade,
    "alpha_04": Alpha04BandPositionReversion,
    "alpha_05": Alpha05VolumeFlowContinuation,
    "alpha_06": Alpha06VolatilityRegimeReversion,
}

# written after the Task 2 holdout was opened
TASK3_ADDITIONS = {
    "alpha_07": Alpha07DriftAnchoredDualReversion,
}

# v2 only. Kept out of both registries above so main.py still reproduces the
# submitted Task 2 and Task 3 results exactly.
V2_ADDITIONS = {
    "alpha_08": Alpha08NextDayContinuation,
}


def build_all(**kwargs):
    """Instantiate every Task 2 candidate in registry order."""
    return {k: cls(**kwargs) for k, cls in ALPHA_REGISTRY.items()}


def build_task3_additions(**kwargs):
    return {k: cls(**kwargs) for k, cls in TASK3_ADDITIONS.items()}


__all__ = [
    "BaselineStrategy",
    "ALPHA_REGISTRY",
    "TASK3_ADDITIONS",
    "V2_ADDITIONS",
    "build_all",
    "build_task3_additions",
] + [c.__name__ for c in {**ALPHA_REGISTRY, **TASK3_ADDITIONS, **V2_ADDITIONS}.values()]
