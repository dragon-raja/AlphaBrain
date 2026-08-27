from __future__ import annotations

import pytest

from scripts.dsol_paper1.summarize_dense_test_selectors import summarize


def _rows(
    pair_key: str,
    method: str,
    outcomes: list[int],
    *,
    task_id: str = "task-a",
) -> list[dict]:
    return [
        {
            "episode_id": f"{pair_key}::{method}",
            "policy_noise_seed": 100 + index,
            "status": "complete",
            "condition": f"selector__{method}",
            "pair_key": pair_key,
            "task_id": task_id,
            "selected_candidate_id": method,
            "success": bool(outcome),
            "completion_steps": 20 + index,
        }
        for index, outcome in enumerate(outcomes)
    ]


def test_summary_reports_independent_test_rescue_and_harm() -> None:
    pair_rescue = "task-a::test::demo_1::stage-01::frame-00010"
    pair_harm = "task-a::test::demo_2::stage-01::frame-00012"
    rows = []
    rows.extend(_rows(pair_rescue, "canonical", [0, 0, 0]))
    rows.extend(_rows(pair_rescue, "visibility_mean", [1, 1, 0]))
    rows.extend(_rows(pair_harm, "canonical", [1, 1, 1]))
    rows.extend(_rows(pair_harm, "visibility_mean", [0, 0, 1]))

    payload, state_rows = summarize(rows, expected_repeats=3)

    assert len(state_rows) == 4
    assert payload["states"] == 2
    assert payload["source_groups"] == 2
    assert payload["selection_uses_test_policy_outcomes"] is False
    visibility = payload["selector_summary"]["visibility_mean"]
    assert visibility["mean_repeat_success_rate"] == pytest.approx(0.5)
    assert visibility["difference_from_canonical_pp"] == pytest.approx(0.0)
    assert visibility["stable_rescue_state_count"] == 1
    assert visibility["stable_harm_state_count"] == 1


def test_summary_uses_majority_of_five_repeats() -> None:
    pair_key = "task-a::test::demo_3::stage-01::frame-00014"
    rows = []
    rows.extend(_rows(pair_key, "canonical", [1, 1, 0, 0, 0]))
    rows.extend(_rows(pair_key, "visibility_mean", [1, 1, 1, 0, 0]))

    payload, _state_rows = summarize(rows, expected_repeats=5)

    assert payload["stable_success_minimum_repeats"] == 3
    visibility = payload["selector_summary"]["visibility_mean"]
    assert visibility["stable_rescue_state_count"] == 1
    assert visibility["stable_harm_state_count"] == 0


def test_summary_reports_task_macro_effect_for_unbalanced_tasks() -> None:
    rows = []
    for demo in ("demo_1", "demo_2"):
        pair_key = f"task-a::test::{demo}::stage-01::frame-00010"
        rows.extend(_rows(pair_key, "canonical", [1, 1, 1], task_id="task-a"))
        rows.extend(
            _rows(pair_key, "visibility_mean", [1, 1, 1], task_id="task-a")
        )
    pair_key = "task-b::test::demo_3::stage-01::frame-00010"
    rows.extend(_rows(pair_key, "canonical", [0, 0, 0], task_id="task-b"))
    rows.extend(_rows(pair_key, "visibility_mean", [1, 1, 1], task_id="task-b"))

    payload, _state_rows = summarize(rows, expected_repeats=3)

    visibility = payload["selector_summary"]["visibility_mean"]
    assert visibility["difference_from_canonical_pp"] == pytest.approx(100 / 3)
    assert visibility["task_macro_difference_from_canonical_pp"] == pytest.approx(50)
    assert payload["task_macro_bootstrap"]["task_count"] == 2
