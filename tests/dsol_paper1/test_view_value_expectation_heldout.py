from __future__ import annotations

from scripts.dsol_paper1.analyze_view_value_expectation_heldout import (
    cross_checkpoint_gate,
    state_method_rows,
)


def _row(bank: str, repeat: int, method: str, success: bool) -> dict:
    return {
        "pair_key": "task::demo_1::frame-10",
        "source_group": "task::demo_1",
        "task_id": "task",
        "selector_method": method,
        "selected_candidate_id": "canonical" if method == "canonical" else "view-1",
        "noise_bank_id": bank,
        "policy_repeat_id": repeat,
        "success": success,
        "normalized_final_progress": float(success),
    }


def test_state_method_rows_keeps_equal_repeat_ids_from_distinct_banks() -> None:
    rows = []
    for bank in ("E", "F"):
        for repeat in range(2):
            rows.append(_row(bank, repeat, "canonical", False))
            rows.append(_row(bank, repeat, "selector", True))

    summaries = state_method_rows(rows, checkpoint_seed=41)
    selected = next(row for row in summaries if row["selector_method"] == "selector")

    assert selected["repeat_count"] == 4
    assert selected["success"] == 1.0
    assert selected["canonical_success"] == 0.0
    assert selected["rescue_count"] == 4


def test_cross_checkpoint_gate_requires_consistent_precise_gain() -> None:
    summaries = {}
    convergence_rows = {}
    best_rules = {}
    for seed in ("41", "42", "43"):
        best_rules[seed] = "selector"
        summaries[seed] = {
            "selector": {
                "success_rate": 0.8,
                "success_gain_pp": 10.0,
                "success_gain_task_stratified_bootstrap_95_pp": [6.0, 14.0],
                "harm_probability": 0.01,
                "harm_wilson_95": [0.0, 0.04],
            }
        }
        rows = []
        for bank in ("E", "F"):
            for repeat in range(32):
                rows.append(_row(bank, repeat, "canonical", False))
                rows.append(_row(bank, repeat, "selector", True))
        convergence_rows[seed] = rows

    result = cross_checkpoint_gate(
        summaries,
        best_rules,
        convergence_rows,
        final_precision=True,
    )

    assert result["status"] == "SELECTOR_GAIN_CONFIRMED"
    assert result["direction_consistent_positive"] is True
    assert result["final_precision_halfwidth_at_most_5pp"] is True
