from __future__ import annotations

from analyze_libero_original_full import compare_rows, summarize_rows


def _rows(success_offset: int = 0) -> list[dict]:
    rows = []
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    for suite_index, suite in enumerate(suites):
        for init_state_index in range(2):
            rows.append(
                {
                    "episode_id": f"{suite}-{init_state_index}-{success_offset}",
                    "suite": suite,
                    "base_task": f"task_{suite_index}",
                    "init_state_index": init_state_index,
                    "condition": "canonical",
                    "success": (suite_index + init_state_index + success_offset) % 2 == 0,
                }
            )
    return rows


def test_summarize_original_full_reports_suite_and_task_units() -> None:
    summary = summarize_rows(_rows(), expected_count=8, expected_trials_per_task=2)
    assert summary["pooled"]["episode_count"] == 8
    assert summary["independent_task_count"] == 4
    assert set(summary["by_suite"]) == {
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    }


def test_compare_original_full_is_strictly_paired() -> None:
    baseline = _rows()
    candidate = _rows(success_offset=1)
    for row in candidate:
        row["episode_id"] += "-candidate"
    comparison = compare_rows(baseline, candidate)
    assert comparison["paired_episode_count"] == 8
    assert comparison["independent_task_count"] == 4


def test_summarize_original_full_accepts_single_suite_smoke() -> None:
    rows = [row for row in _rows() if row["suite"] == "libero_spatial"]
    summary = summarize_rows(rows, expected_count=2, expected_trials_per_task=2)
    assert set(summary["by_suite"]) == {"libero_spatial"}
    assert summary["independent_task_count"] == 1
