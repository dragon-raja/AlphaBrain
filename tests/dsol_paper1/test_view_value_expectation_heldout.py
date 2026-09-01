from __future__ import annotations

from scripts.dsol_paper1.analyze_view_value_expectation_heldout import state_method_rows


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
