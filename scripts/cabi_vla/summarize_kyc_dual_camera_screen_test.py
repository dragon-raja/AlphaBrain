from __future__ import annotations

import json
from pathlib import Path

from summarize_kyc_dual_camera_screen import (
    METHODS,
    grouped_linear_contrast,
    summarize,
)


def _row(state: int, pose: str, success: float, *, method: str = "unused") -> dict:
    values = {
        "baseline": (0.0, 0.0, 1.0),
        "az_p60": (60.0, 0.0, 1.0),
    }
    azimuth, elevation, radius = values[pose]
    return {
        "method": method,
        "edge_id": "red-left",
        "canonical_state_index": state,
        "execution_horizon": 3,
        "camera_pose": pose,
        "camera_azimuth_deg": azimuth,
        "camera_elevation_deg": elevation,
        "camera_radius_scale": radius,
        "task_objects_visible": True,
        "task_objects_fully_visible": True,
        "task_centers_in_frame": True,
        "success": success,
        "transport_success": success,
        "progress": success,
        "completion_steps": 100.0 - 10.0 * success,
    }


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"status": "complete", "rows": rows}))
    return path


def test_grouped_linear_contrast_is_paired_by_snapshot() -> None:
    rows = []
    values = {
        "dual_fla": 1.0,
        "external_fla": 0.5,
        "wrist_fla": 0.25,
        "dual_control_fla": 0.0,
    }
    for method, value in values.items():
        rows.extend(
            _row(state, pose, value, method=method)
            for state in (40, 41)
            for pose in ("baseline", "az_p60")
        )
    result = grouped_linear_contrast(
        rows,
        coefficients={
            "dual_fla": 1.0,
            "external_fla": -1.0,
            "wrist_fla": -1.0,
            "dual_control_fla": 1.0,
        },
        metric="success",
        bootstrap_resamples=100,
    )
    assert result["delta"] == 0.25
    assert result["snapshot_group_count"] == 2


def test_summarize_dual_camera_screen(tmp_path: Path) -> None:
    success = {
        "dual_rgb_fla": 0.0,
        "dual_control_fla": 0.25,
        "external_fla": 0.25,
        "wrist_fla": 0.25,
        "dual_fla": 0.5,
    }
    paths = {
        method: _write(
            tmp_path / f"{method}.json",
            [
                _row(state, pose, value)
                for state in (40, 41)
                for pose in ("baseline", "az_p60")
            ],
        )
        for method, value in success.items()
    }
    interventions = {
        "initial": _write(
            tmp_path / "initial.json",
            [
                _row(state, pose, 0.0)
                for state in (40, 41)
                for pose in ("baseline", "az_p60")
            ],
        ),
        "lagged": _write(
            tmp_path / "lagged.json",
            [
                _row(state, pose, 0.25)
                for state in (40, 41)
                for pose in ("baseline", "az_p60")
            ],
        ),
    }
    result = summarize(
        evaluation_paths=paths,
        intervention_paths=interventions,
        bootstrap_resamples=100,
    )
    assert result["status"] == "complete"
    assert set(result["methods"]) == set(METHODS)
    assert result["gate"]["baseline_valid"] is True
    assert result["gate"]["causal_wrist_use"] is True
    assert result["gate"]["decision"] == "ADVANCE_DUAL_CAMERA_CONFIRMATION"
