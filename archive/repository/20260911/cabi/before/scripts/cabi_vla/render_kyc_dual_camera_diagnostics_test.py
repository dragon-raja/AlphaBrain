from __future__ import annotations

from pathlib import Path

from render_kyc_dual_camera_diagnostics import METHOD_LABELS, render


def test_render_dual_camera_diagnostics(tmp_path: Path) -> None:
    poses = ["baseline", "az_m60"]
    effect = {
        "success": {"delta": 0.1, "ci95_low": -0.1, "ci95_high": 0.2}
    }
    payload = {
        "status": "complete",
        "study": "kyc_pi05_dual_camera_diagnostics",
        "seed": 41,
        "training_updates": 2000,
        "execution_horizon": 3,
        "pose_order": poses,
        "overall": {
            method: {
                "source_selection_success": 0.8,
                "lift_success": 0.6,
                "transport_success": 0.4,
                "target_placement_success": 0.3,
                "success": 0.2,
            }
            for method in METHOD_LABELS
        },
        "factorial_effects": {
            key: effect
            for key in (
                "external_at_canonical_wrist",
                "external_at_real_wrist",
                "wrist_at_canonical_external",
                "wrist_at_real_external",
                "external_average_main_effect",
                "wrist_average_main_effect",
                "interaction",
            )
        },
        "pose_diagnostics": {
            pose: {
                "visibility": {
                    "task_objects_visible": 1.0,
                    "task_objects_fully_visible": 0.5,
                    "task_centers_in_frame": 0.75,
                },
                "wrist_ray_intervention": {
                    condition: {"success": 0.2}
                    for condition in ("correct", "initial", "lagged")
                },
            }
            for pose in poses
        },
    }
    output = tmp_path / "diagnostics.png"
    render(payload, output=output)
    assert output.is_file()
    assert output.stat().st_size > 10_000
