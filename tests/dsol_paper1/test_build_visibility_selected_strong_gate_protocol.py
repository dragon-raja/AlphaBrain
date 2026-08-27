from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "dsol_paper1"
sys.path.insert(0, str(SCRIPT_ROOT))

from build_visibility_selected_strong_gate_protocol import build  # noqa: E402


def record(pose_id: str, delta: float, pair_id: str, member: str) -> dict:
    return {
        "pose_id": pose_id,
        "group": "constructed_task_orbit",
        "visibility_score": 0.1 + delta,
        "delta_visibility": delta,
        "per_camera_scores": {"agentview": 0.1 + delta, "robot0_eye_in_hand": 0.0},
        "pose": {"pair_id": pair_id, "pair_member": member},
    }


def make_row(tmp_path: Path, split: str, episode: str) -> dict:
    output = tmp_path / f"{split}-{episode}"
    output.mkdir()
    (output / "scan.json").write_text(
        json.dumps(
            {
                "initial_task_success": False,
                "scene_construction": {"sha256": "construction"},
                "records": [
                    {
                        "pose_id": "canonical",
                        "visibility_score": 0.0,
                        "delta_visibility": 0.0,
                        "per_camera_scores": {
                            "agentview": 0.0,
                            "robot0_eye_in_hand": 0.01,
                        },
                        "pose": None,
                    },
                    record("negative", 0.08, "side", "negative"),
                    record("positive", 0.005, "side", "positive"),
                ],
            }
        )
    )
    return {
        "scan_id": f"task::{split}::{episode}",
        "task_id": "task",
        "diagnostic_role": "test",
        "suite": "suite",
        "hdf5": "/data.hdf5",
        "episode_id": episode,
        "demo_name": episode,
        "demo_index": 0,
        "frame": 1,
        "stage_fraction": 0.1,
        "split": split,
        "status": "PASS",
        "output_dir": str(output),
    }


def test_build_selects_visibility_without_policy_outcomes(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}")
    rows = [
        make_row(tmp_path, "val", "v1"),
        make_row(tmp_path, "val", "v2"),
        make_row(tmp_path, "test", "t1"),
        make_row(tmp_path, "test", "t2"),
    ]

    result = build(
        rows,
        catalog=catalog,
        minimum_strong_delta=0.05,
        maximum_control_abs_delta=0.02,
        maximum_canonical_external=0.005,
        maximum_canonical_wrist=0.02,
        minimum_validation_episodes=2,
        minimum_test_episodes=2,
    )

    assert result["status"] == "HOLD_MANUAL_AUDIT"
    assert result["policy_outcomes_used_for_selection"] is False
    assert result["test_passing_state_count"] == 2
    assert result["episode_count"] == 16
    assert {row["strong_pose_id"] for row in result["selected_states"]} == {"negative"}
