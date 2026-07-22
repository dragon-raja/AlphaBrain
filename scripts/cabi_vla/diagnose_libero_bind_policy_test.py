import numpy as np

from diagnose_libero_bind_policy import (
    action_chunk,
    factor_sensitivity,
    progress_bin,
    summarize_teacher_rows,
)


def test_factor_sensitivity_separates_source_and_target_interventions() -> None:
    predictions = {}
    source_values = {"red": 0.0, "white": 1.0, "yellow_white": 2.0}
    target_values = {"left": 0.0, "right": 0.25}
    for source, source_value in source_values.items():
        for target, target_value in target_values.items():
            predictions[f"{source}-{target}"] = np.full(
                (10, 7), source_value + target_value, np.float32
            )
    result = factor_sensitivity(predictions)
    assert result["source_interventions"]["pair_count"] == 6
    assert result["target_interventions"]["pair_count"] == 3
    assert result["source_to_target_sensitivity_ratio"] > 10


def test_action_chunk_pads_only_after_available_actions() -> None:
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    chunk = action_chunk(actions, 2, 4)
    np.testing.assert_array_equal(chunk[0], actions[2])
    np.testing.assert_array_equal(chunk[1:], np.zeros((3, 7), np.float32))


def test_progress_bins_cover_episode_quarters() -> None:
    assert [progress_bin(frame, 100) for frame in (0, 25, 50, 75, 100)] == [
        "q1",
        "q2",
        "q3",
        "q4",
        "q4",
    ]


def test_teacher_summary_preserves_edge_phase_and_progress_groups() -> None:
    rows = []
    for edge, phase, quarter, value in (
        ("red-left", "approach", "q1", 1.0),
        ("red-left", "grasp", "q2", 2.0),
        ("white-left", "approach", "q1", 3.0),
    ):
        rows.append(
            {
                "edge_id": edge,
                "phase": phase,
                "progress_bin": quarter,
                "chunk_mse": value,
                "first_step_mse": value,
                "translation_mse": value,
                "rotation_mse": value,
                "gripper_mse": value,
            }
        )
    summary = summarize_teacher_rows(rows)
    assert summary["overall"]["chunk_mse"] == 2.0
    assert summary["by_edge"]["red-left"]["row_count"] == 2
    assert summary["by_phase"]["approach"]["chunk_mse"] == 2.0
    assert summary["by_progress"]["q2"]["row_count"] == 1
