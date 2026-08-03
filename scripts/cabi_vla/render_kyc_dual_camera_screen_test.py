from __future__ import annotations

import json
from pathlib import Path

from render_kyc_dual_camera_screen import METHODS, render


def _effect(delta: float) -> dict:
    return {
        "delta": delta,
        "ci95_low": delta - 0.03,
        "ci95_high": delta + 0.03,
        "snapshot_group_count": 2,
        "bootstrap_resamples": 100,
    }


def _rows(value: float) -> list[dict]:
    poses = ("baseline", "az_m60", "az_p60", "el_m25", "el_p25", "rad_0900", "rad_1250")
    return [
        {
            "edge_id": "red-left",
            "canonical_state_index": state,
            "execution_horizon": 3,
            "camera_pose": pose,
            "success": value,
        }
        for state in (40, 41)
        for pose in poses
    ]


def test_render_dual_camera_screen(tmp_path: Path) -> None:
    methods = {
        method: {
            "episode_count": 14,
            "snapshot_group_count": 2,
            "success": 0.20 + 0.03 * index,
        }
        for index, method in enumerate(METHODS)
    }
    summary = {
        "status": "complete",
        "study": "kyc_pi05_dual_camera_screen",
        "strata": {
            "inside_training_support": {
                "methods": methods,
                "paired_differences": {
                    "external_fla_minus_dual_control_fla": {"success": _effect(0.04)},
                    "wrist_fla_minus_dual_control_fla": {"success": _effect(0.03)},
                    "dual_fla_minus_dual_control_fla": {"success": _effect(0.07)},
                    "dual_interaction": {"success": _effect(0.01)},
                },
            }
        },
        "wrist_ray_causal_intervention": {
            "inside_training_support": {
                "conditions": {
                    "correct": {"success": 0.32},
                    "initial": {"success": 0.22},
                    "lagged": {"success": 0.27},
                }
            }
        },
    }
    paths = {}
    for index, method in enumerate(METHODS):
        path = tmp_path / f"{method}.json"
        path.write_text(json.dumps({"status": "complete", "rows": _rows(0.2 + 0.03 * index)}))
        paths[method] = path
    output = tmp_path / "screen.png"
    render(summary=summary, evaluation_paths=paths, output=output)
    assert output.stat().st_size > 20_000
