from __future__ import annotations

from scripts.dsol_paper1.analyze_statewise_view_oracle_v2 import (
    decision,
    stratified_bootstrap,
)


def test_task_stratified_bootstrap_detects_positive_constant_effect() -> None:
    values = {f"state-{task}-{index}": 0.1 for task in range(8) for index in range(2)}
    tasks = {
        f"state-{task}-{index}": f"task-{task}" for task in range(8) for index in range(2)
    }
    result = stratified_bootstrap(values, tasks, label="constant", resamples=100)
    assert result["estimate"] == 0.1
    assert result["ci95"][0] > 0


def test_decision_requires_statistical_and_practical_effects() -> None:
    comparison = {
        "estimate": 0.06,
        "ci95": [0.01, 0.11],
        "ci_halfwidth": 0.05,
        "bootstrap_resamples": 100,
        "P_bootstrap_gt_0": 0.99,
    }
    checkpoint = {
        "comparisons": {
            "statewise_P_top1_minus_canonical": comparison,
            "statewise_P_top1_minus_global_fixed": {**comparison, "estimate": 0.04},
            "geometry_context_ridge_minus_canonical": {**comparison, "estimate": 0.03},
            "image_queryable_ridge_minus_canonical": {**comparison, "estimate": 0.04},
        },
        "population": {
            "geometry_context_ridge": {"harm_vs_canonical": 0.01},
            "image_queryable_ridge": {"harm_vs_canonical": 0.01},
        },
    }
    protocol = {
        "method_selections": {
            "s1": {"statewise_P_top1": "v1"},
            "s2": {"statewise_P_top1": "v2"},
        }
    }
    result = decision(checkpoint, protocol)
    assert result["oracle_existence"]["pass"] is True
    assert result["state_dependence"]["pass"] is True
    assert result["learner_generalization"]["geometry_context_ridge"]["pass"] is True
