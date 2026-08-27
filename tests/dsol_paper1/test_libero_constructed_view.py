from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "dsol_paper1"
CABI_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "cabi_vla"
sys.path[:0] = [str(SCRIPT_ROOT), str(CABI_ROOT)]

from libero_constructed_view import (  # noqa: E402
    CONSTRUCTION_SCHEMA,
    explicit_task_orbit_pose,
    inject_static_visual_occluder,
    paired_task_orbit_poses,
    task_orbit_pose_from_specification,
)


def construction() -> dict:
    return {
        "schema": CONSTRUCTION_SCHEMA,
        "reference_camera_world": [1.0, 0.0, 1.0],
        "camera_pivot_world": [0.0, 0.0, 0.0],
        "occluder": {
            "name": "test_occluder",
            "position_world": [0.2, 0.0, 0.2],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "half_size_xyz": [0.01, 0.1, 0.2],
            "rgba": [0.1, 0.1, 0.1, 1.0],
        },
    }


def test_injected_occluder_is_static_visual_group() -> None:
    xml = '<mujoco><worldbody><body name="existing"/></worldbody></mujoco>'
    transformed = inject_static_visual_occluder(xml, construction())
    assert 'name="test_occluder"' in transformed
    assert 'name="test_occluder_geom"' in transformed
    assert 'contype="0"' in transformed
    assert 'conaffinity="0"' in transformed
    assert 'group="1"' in transformed
    assert "joint" not in transformed


def test_paired_pose_has_equal_position_displacement() -> None:
    poses = paired_task_orbit_poses(
        construction(),
        [{"pair_id": "side", "azimuth_deg": 35.0, "elevation_deg": 10.0}],
    )
    assert len(poses) == 2
    reference = np.asarray(construction()["reference_camera_world"])
    distances = [np.linalg.norm(np.asarray(pose["position_world"]) - reference) for pose in poses]
    assert np.isclose(distances[0], distances[1])
    assert {pose["pair_member"] for pose in poses} == {"negative", "positive"}


def test_explicit_pose_looks_at_declared_pivot() -> None:
    pose = explicit_task_orbit_pose(
        reference_position=[1.0, 0.0, 1.0],
        pivot=[0.0, 0.0, 0.0],
        azimuth_deg=45.0,
        elevation_deg=5.0,
        radius_scale=0.8,
        pose_id="view",
        pair_id="pair",
        pair_member="positive",
    )
    assert pose["orientation_mode"] == "explicit_world_look_at"
    assert np.isclose(np.linalg.norm(pose["quaternion_wxyz"]), 1.0)
    assert pose["pivot_world"] == [0.0, 0.0, 0.0]


def test_task_orbit_pose_selects_frozen_pair_member() -> None:
    specification = {
        "candidate_pairs": [
            {
                "pair_id": "side30_near",
                "azimuth_deg": 30.0,
                "elevation_deg": 5.0,
                "radius_scale": 0.7,
            }
        ]
    }
    pose = task_orbit_pose_from_specification(
        context=construction(),
        specification=specification,
        pair_id="side30_near",
        pair_member="negative",
    )
    assert pose["pose_id"] == "taskcentric_side30_near_negative"
    assert pose["azimuth_deg"] == -30.0
    assert pose["pair_member"] == "negative"
