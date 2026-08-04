from __future__ import annotations

import pytest

from build_libero_plus_view_protocol import (
    build_protocol,
    normalize_degrees,
    parse_camera_task_name,
    synthetic_camera_task_name,
)


def test_parse_camera_task_name_decodes_wrapped_angles() -> None:
    parsed = parse_camera_task_name(
        "put_the_cream_cheese_in_the_bowl_view_330_15_90_352_4_initstate_0"
    )
    assert parsed["base_task"] == "put_the_cream_cheese_in_the_bowl"
    assert parsed["orbit_yaw_deg"] == -30
    assert parsed["orbit_pitch_deg"] == 15
    assert parsed["radius_percent"] == 90
    assert parsed["look_yaw_deg"] == -8
    assert parsed["look_pitch_deg"] == 4
    assert parsed["perturbation_family"] == "combined"
    assert normalize_degrees(180) == 180


def test_synthetic_camera_task_name_encodes_negative_angles() -> None:
    name = synthetic_camera_task_name(
        "task",
        {
            "orbit_yaw_deg": -30,
            "orbit_pitch_deg": 0,
            "radius_percent": 100,
            "look_yaw_deg": -8,
            "look_pitch_deg": 0,
        },
    )
    assert name == "task_view_330_0_100_352_0_initstate_0"


def test_protocol_is_stratified_and_deterministic() -> None:
    classification = {"libero_goal": []}
    task_id = 1
    for difficulty in range(1, 6):
        for index in range(3):
            classification["libero_goal"].append(
                {
                    "id": task_id,
                    "name": f"task_{difficulty}_{index}_view_{index * 10}_0_100_0_0_initstate_0",
                    "category": "Camera Viewpoints",
                    "difficulty_level": difficulty,
                }
            )
            task_id += 1
    first = build_protocol(classification, per_suite_difficulty=2, seed=7)
    second = build_protocol(classification, per_suite_difficulty=2, seed=7)
    assert first == second
    assert first["summary"]["camera_population_count"] == 15
    assert first["summary"]["selected_count"] == 10
    assert first["summary"]["selected_unique_base_task_count"] == 10
    assert {row["difficulty_level"] for row in first["official_camera_tasks"]} == set(range(1, 6))
    assert len(first["candidate_matrix_base_tasks"]) == 15
    assert all("task_index" in row for row in first["candidate_matrix_base_tasks"])


def test_protocol_rejects_underfilled_stratum() -> None:
    classification = {
        "libero_goal": [
            {
                "id": 1,
                "name": "task_view_0_0_100_2_0_initstate_0",
                "category": "Camera Viewpoints",
                "difficulty_level": 1,
            }
        ]
    }
    with pytest.raises(ValueError, match="stratum"):
        build_protocol(classification, per_suite_difficulty=2, seed=7)
