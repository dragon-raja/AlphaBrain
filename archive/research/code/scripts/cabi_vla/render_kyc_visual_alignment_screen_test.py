from __future__ import annotations

import json
from pathlib import Path

import pytest

from render_kyc_visual_alignment_screen import (
    METHODS,
    _dimension_rows,
    render_pose_response,
    render_screen_summary,
)


def _effect(delta: float) -> dict[str, float | int]:
    return {
        "delta": delta,
        "ci95_low": delta - 0.03,
        "ci95_high": delta + 0.03,
        "snapshot_group_count": 5,
        "bootstrap_resamples": 100,
    }


def _summary() -> dict:
    strata = {}
    for index, stratum in enumerate(
        ("inside_training_support", "objects_visible", "fully_visible")
    ):
        methods = {
            method: {
                "episode_count": 7,
                "snapshot_group_count": 5,
                "success": 0.25 + 0.02 * method_index + 0.01 * index,
                "transport_success": 0.4,
                "progress": 1.5,
                "completion_steps": 100.0,
            }
            for method_index, method in enumerate(METHODS)
        }
        strata[stratum] = {
            "methods": methods,
            "paired_differences": {
                "kyc_fla_minus_poseaug_control_fla": {
                    metric: _effect(0.06 if metric == "success" else 0.01)
                    for metric in (
                        "success",
                        "transport_success",
                        "progress",
                        "completion_steps",
                    )
                },
                "kyc_fla_minus_poseaug_rgb_fla": {},
                "poseaug_control_fla_minus_poseaug_rgb_fla": {},
            },
        }
    return {
        "status": "complete",
        "study": "kyc_pi05_visual_alignment_screen",
        "strata": strata,
        "ray_diagnostic": {
            "canonical_vs_correct": {"chunk_rms": 0.002},
            "mismatched_vs_correct": {"chunk_rms": 0.006},
        },
        "gate": {
            "minimum_causal_ray_rms": 0.005,
            "decision": "ADVANCE_TO_FULL_CONFIRMATION",
        },
    }


def _rows() -> list[dict]:
    poses = (
        ("baseline", 0.0, 0.0, 1.0),
        ("az_m60", -60.0, 0.0, 1.0),
        ("az_p60", 60.0, 0.0, 1.0),
        ("el_m25", 0.0, -25.0, 1.0),
        ("el_p25", 0.0, 25.0, 1.0),
        ("rad_0900", 0.0, 0.0, 0.9),
        ("rad_1250", 0.0, 0.0, 1.25),
    )
    return [
        {
            "edge_id": "red-left",
            "canonical_state_index": 40,
            "execution_horizon": 3,
            "camera_pose": name,
            "camera_azimuth_deg": azimuth,
            "camera_elevation_deg": elevation,
            "camera_radius_scale": radius,
            "success": index % 2 == 0,
            "task_objects_fully_visible": index < 5,
            "task_centers_in_frame": index != 4,
        }
        for index, (name, azimuth, elevation, radius) in enumerate(poses)
    ]


def test_dimension_rows_keeps_only_one_axis_perturbed() -> None:
    rows = _rows()
    selected = _dimension_rows(rows, field="camera_azimuth_deg", value=0.0)
    assert [row["camera_pose"] for row in selected] == ["baseline"]


def test_render_screen_figures(tmp_path: Path) -> None:
    screen_output = tmp_path / "screen.png"
    render_screen_summary(summary=_summary(), output=screen_output)
    assert screen_output.stat().st_size > 10_000

    paths = {}
    for method in METHODS:
        path = tmp_path / f"{method}.json"
        path.write_text(json.dumps({"status": "complete", "rows": _rows()}))
        paths[method] = path
    pose_output = tmp_path / "pose.png"
    render_pose_response(evaluation_paths=paths, output=pose_output)
    assert pose_output.stat().st_size > 10_000

    with pytest.raises(FileExistsError):
        render_screen_summary(summary=_summary(), output=screen_output)
