from __future__ import annotations

import json
from pathlib import Path

from summarize_kyc_dual_camera_diagnostics import summarize
from summarize_kyc_dual_camera_screen import METHODS, WRIST_INTERVENTIONS


def _row(*, state: int, pose: str, success: bool) -> dict:
    return {
        "edge_id": "red-left",
        "canonical_state_index": state,
        "execution_horizon": 3,
        "camera_pose": pose,
        "scene_cue_seed": state,
        "initial_agent_sha256": f"agent-{state}-{pose}",
        "initial_wrist_sha256": f"wrist-{state}-{pose}",
        "camera_position": [0.0, 0.0, 1.0],
        "camera_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "camera_intrinsics": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "camera_to_world_opencv": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        "policy_camera_intrinsics": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "policy_camera_to_world_opencv": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        "floor_texture_xy": [0.1, 0.2],
        "floor_texture_yaw_deg": 10.0,
        "visual_table_xy": [0.0, 0.0],
        "visual_table_yaw_deg": 20.0,
        "fixed_room_visuals_hidden": True,
        "robot_base_visual_hidden": True,
        "task_objects_visible": True,
        "task_objects_fully_visible": pose == "baseline",
        "task_centers_in_frame": True,
        "source_touches_border": False,
        "target_touches_border": pose != "baseline",
        "source_visible_fraction": 0.1,
        "target_visible_fraction": 0.2,
        "source_selection_success": True,
        "lift_success": success,
        "transport_success": success,
        "target_placement_success": success,
        "success": success,
        "progress": 1.0 if success else 0.25,
        "completion_steps": 100 if success else 320,
    }


def _evaluation(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps({"status": "complete", "rows": rows}))


def _metrics(path: Path) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "step": step,
                    "examples_seen": 2 * step,
                    "flow_matching_loss": 1.0 / step,
                }
            )
            for step in (10, 20, 30)
        )
        + "\n"
    )


def test_summarize_dual_camera_diagnostics(tmp_path: Path) -> None:
    base_rows = [
        _row(state=state, pose=pose, success=(state == 1))
        for pose in ("baseline", "az_m60")
        for state in (1, 2)
    ]
    evaluations = {}
    metrics = {}
    for method in METHODS:
        path = tmp_path / f"{method}.json"
        rows = [{**row, "success": method == "dual_fla"} for row in base_rows]
        _evaluation(path, rows)
        evaluations[method] = path
        metric_path = tmp_path / f"{method}.jsonl"
        _metrics(metric_path)
        metrics[method] = metric_path

    interventions = {}
    for condition in WRIST_INTERVENTIONS:
        path = tmp_path / f"{condition}.json"
        _evaluation(path, base_rows)
        interventions[condition] = path

    result = summarize(
        evaluation_paths=evaluations,
        intervention_paths=interventions,
        training_metric_paths=metrics,
        bootstrap_resamples=100,
    )

    assert result["pairing_audit"]["status"] == "passed"
    assert result["pairing_audit"]["episode_count_per_condition"] == 4
    assert result["pose_diagnostics"]["baseline"]["visibility"][
        "task_objects_fully_visible"
    ] == 1.0
    assert result["pose_diagnostics"]["az_m60"]["visibility"][
        "target_touches_border"
    ] == 1.0
    assert result["overall"]["dual_fla"]["success"] == 1.0
    assert result["factorial_effects"]["interaction"]["success"]["delta"] == 1.0
    assert result["training"]["dual_fla"]["final_step"] == 30
