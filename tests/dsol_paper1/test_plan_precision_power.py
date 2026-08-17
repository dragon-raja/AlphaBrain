from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts/dsol_paper1/plan_precision_power.py"
SPEC = importlib.util.spec_from_file_location("plan_precision_power", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_power_increases_with_sample_size_and_pairing() -> None:
    low_n = MODULE.paired_binary_power(
        n=100, baseline=0.5, effect=0.1, correlation=0.0
    )
    high_n = MODULE.paired_binary_power(
        n=400, baseline=0.5, effect=0.1, correlation=0.0
    )
    paired = MODULE.paired_binary_power(
        n=400, baseline=0.5, effect=0.1, correlation=0.5
    )
    assert 0 < low_n < high_n < paired < 1


def test_full_population_is_near_target_but_attrition_is_not() -> None:
    full = MODULE.paired_binary_power(
        n=400, baseline=0.5, effect=0.1, correlation=0.0
    )
    retained = MODULE.paired_binary_power(
        n=320, baseline=0.5, effect=0.1, correlation=0.0
    )
    assert full >= 0.8
    assert retained < 0.8
