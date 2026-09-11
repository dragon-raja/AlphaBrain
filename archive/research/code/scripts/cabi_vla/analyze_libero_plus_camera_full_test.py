from __future__ import annotations

import pytest

from analyze_libero_plus_camera_full import compare_rows, summarize_rows, wilson_interval


def _row(task_id: int, *, success: bool, base_task: str, difficulty: int = 1) -> dict:
    return {
        "episode_id": f"episode-{task_id}-{int(success)}",
        "condition": "official_camera",
        "suite": "libero_goal",
        "official_task_id": task_id,
        "init_state_index": 0,
        "base_task": base_task,
        "difficulty_level": difficulty,
        "perturbation_family": "orbit_yaw",
        "success": success,
    }


def test_summary_reports_official_and_clustered_rates() -> None:
    rows = [
        _row(1, success=True, base_task="a"),
        _row(2, success=False, base_task="a"),
        _row(3, success=True, base_task="b", difficulty=2),
    ]
    summary = summarize_rows(rows, expected_count=3)
    assert summary["official_pooled"]["success_rate"] == pytest.approx(2 / 3)
    assert summary["base_task_macro"]["mean"] == pytest.approx(0.75)
    assert summary["independent_base_task_count"] == 2
    assert summary["by_difficulty"]["1"]["task_count"] == 2


def test_paired_comparison_clusters_by_base_task() -> None:
    baseline = [
        _row(1, success=False, base_task="a"),
        _row(2, success=False, base_task="a"),
        _row(3, success=True, base_task="b"),
    ]
    candidate = [
        _row(1, success=True, base_task="a"),
        _row(2, success=False, base_task="a"),
        _row(3, success=True, base_task="b"),
    ]
    comparison = compare_rows(baseline, candidate)
    assert comparison["candidate_minus_baseline_pooled"] == pytest.approx(1 / 3)
    assert comparison["candidate_minus_baseline_base_task_cluster"]["mean"] == pytest.approx(0.25)


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = wilson_interval(8, 10)
    assert lower < 0.8 < upper
