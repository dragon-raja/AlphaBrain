from __future__ import annotations

import pytest

from scripts.dsol_paper1.audit_view_value_expectation_completion import (
    EXPECTED_CLAIM_SCOPE,
    validate_final_decision,
    validate_population,
)


def _state(split: str, task: int, index: int) -> dict:
    return {
        "pair_key": f"{split}-pair-{index}",
        "source_group": f"{split}-source-{index}",
        "task_id": f"task-{task}",
    }


def test_population_requires_source_and_state_disjoint_strata() -> None:
    calibration = [_state("calibration", index // 2, index) for index in range(16)]
    heldout = [_state("heldout", index // 6, index) for index in range(48)]
    payload = {
        "schema": "dsol_view_value_expectation_population_v1",
        "status": "PASS",
        "population": {
            "calibration": {"states": calibration},
            "heldout_test": {"states": heldout},
        },
    }

    result = validate_population(payload)

    assert result["source_disjoint"] is True
    assert result["calibration_state_count"] == 16
    assert result["heldout_state_count"] == 48


def test_population_rejects_source_overlap() -> None:
    calibration = [_state("calibration", index // 2, index) for index in range(16)]
    heldout = [_state("heldout", index // 6, index) for index in range(48)]
    heldout[0]["source_group"] = calibration[0]["source_group"]
    payload = {
        "schema": "dsol_view_value_expectation_population_v1",
        "status": "PASS",
        "population": {
            "calibration": {"states": calibration},
            "heldout_test": {"states": heldout},
        },
    }

    with pytest.raises(ValueError, match="sources overlap"):
        validate_population(payload)


def test_final_decision_must_match_both_analysis_gates() -> None:
    calibration = {"status": "VIEW_HEADROOM_NOT_CONFIRMED"}
    heldout = {
        "noise_repeats_per_condition": 32,
        "selector_population_gate": {
            "status": "SELECTOR_GAIN_NOT_CONFIRMED",
            "best_rule_frozen_on_calibration": "calibration_global_fixed_pose",
            "cross_checkpoint_mean_gain_pp": -1.0,
            "cross_checkpoint_mean_harm_probability": 0.1,
            "direction_consistent_positive": False,
        },
    }
    decision = {
        "schema": "dsol_view_value_expectation_final_decision_v1",
        "status": "VIEW_HEADROOM_NOT_CONFIRMED",
        "calibration_headroom_gate": "VIEW_HEADROOM_NOT_CONFIRMED",
        "heldout_selector_gate": "SELECTOR_GAIN_NOT_CONFIRMED",
        "selector_method": "calibration_global_fixed_pose",
        "cross_checkpoint_mean_gain_pp": -1.0,
        "cross_checkpoint_mean_harm_probability": 0.1,
        "direction_consistent_positive": False,
        "noise_repeats_per_condition": 32,
        "claim_scope": EXPECTED_CLAIM_SCOPE,
        "generated_at": "2026-09-03T00:00:00+00:00",
    }

    validate_final_decision(calibration, heldout, decision)

    decision["heldout_selector_gate"] = "SELECTOR_GAIN_CONFIRMED"
    with pytest.raises(ValueError, match="heldout_selector_gate"):
        validate_final_decision(calibration, heldout, decision)
