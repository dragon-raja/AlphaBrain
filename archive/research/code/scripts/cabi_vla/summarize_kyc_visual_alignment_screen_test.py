from __future__ import annotations

import json

from summarize_kyc_visual_alignment_screen import summarize


def _row(state: int, pose: str, success: float) -> dict:
    return {
        "edge_id": "red-left",
        "canonical_state_index": state,
        "execution_horizon": 3,
        "camera_pose": pose,
        "camera_azimuth_deg": 0.0,
        "camera_elevation_deg": 0.0,
        "camera_radius_scale": 1.0,
        "task_objects_visible": True,
        "task_objects_fully_visible": True,
        "task_centers_in_frame": True,
        "success": success,
        "transport_success": success,
        "progress": success,
        "completion_steps": 100.0 if success else 320.0,
    }


def test_screen_advances_only_with_behavior_gain_and_ray_use(tmp_path) -> None:
    evaluation_paths = {}
    successes = {
        "poseaug_rgb_fla": {40: 1.0, 41: 0.0},
        "poseaug_control_fla": {40: 1.0, 41: 0.0},
        "kyc_fla": {40: 1.0, 41: 1.0},
    }
    for method, by_state in successes.items():
        path = tmp_path / f"{method}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "rows": [
                        _row(state, pose, by_state[state])
                        for state in (40, 41)
                        for pose in ("baseline", "az_m60")
                    ],
                }
            )
        )
        evaluation_paths[method] = path

    ray_path = tmp_path / "ray.json"
    difference = {
        "chunk_rms": 0.01,
        "first_action_rms": 0.01,
        "max_abs": 0.02,
        "cosine_similarity": 0.99,
    }
    ray_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "canonical_vs_correct": difference,
                        "mismatched_vs_correct": difference,
                    }
                ]
            }
        )
    )

    payload = summarize(
        evaluation_paths=evaluation_paths,
        ray_diagnostic=ray_path,
        bootstrap_resamples=100,
    )
    assert payload["gate"]["baseline_valid"]
    assert payload["gate"]["decision"] == "ADVANCE_TO_FULL_CONFIRMATION"
