from __future__ import annotations

import numpy as np

from audit_pi05_libero_plus_composition_isolation import audit_isolation


def _row(condition: str, pose_x: float = 2.0) -> dict:
    return {
        "pair_key": "pair-0",
        "condition": condition,
        "language": "put the bowl on the plate",
        "initial_metrics": {
            "agent_camera_position": [pose_x, 0.0, 1.0],
            "agent_camera_rotation": np.eye(3).tolist(),
            "physics_state_sha256": "same-state",
        },
    }


def _manifest(training_pose_x: float = 1.0, *, with_background: bool = False) -> dict:
    episode = {
        "split": "train",
        "budget_percentile": 0.1,
        "language_instruction": "put the bowl on the plate",
        "camera_to_world_opencv": [
            [1.0, 0.0, 0.0, training_pose_x],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    if with_background:
        episode["background_texture_id"] = 3
    return {"episodes": [episode]}


def test_audit_identifies_joint_ood_stress_test() -> None:
    result = audit_isolation(
        _manifest(),
        [_row(condition) for condition in sorted(
            {"canonical", "camera_only", "background_only", "camera_background"}
        )],
        training_split="train",
        budget_fraction=0.25,
    )
    assert result["isolation"]["camera_pose_isolation_pass"]
    assert result["isolation"]["paired_physics_pass"]
    assert not result["isolation"]["task_identity_disjoint"]
    assert result["classification"]["label"] == "PAIRED_JOINT_OOD_STRESS_TEST"


def test_audit_detects_camera_pose_overlap() -> None:
    result = audit_isolation(
        _manifest(training_pose_x=2.0),
        [_row(condition) for condition in sorted(
            {"canonical", "camera_only", "background_only", "camera_background"}
        )],
        training_split="train",
        budget_fraction=0.25,
    )
    assert result["isolation"]["camera_pose_overlap_count"] == 1
    assert not result["isolation"]["camera_pose_isolation_pass"]
