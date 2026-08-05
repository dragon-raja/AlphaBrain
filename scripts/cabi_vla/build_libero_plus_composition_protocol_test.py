from __future__ import annotations

import pytest

from build_libero_plus_composition_protocol import (
    build_composition_protocol,
    compose_camera_background_task_name,
    parse_background_task_name,
)


def test_parse_and_compose_camera_background_task_name() -> None:
    background = parse_background_task_name("task_tb_17")
    assert background == {
        "base_task": "task",
        "background_kind": "tb",
        "background_texture_index": 17,
    }
    assert (
        compose_camera_background_task_name(
            "task_view_330_15_90_352_4_initstate_0",
            "task_tb_17",
        )
        == "task_tb_17_view_330_15_90_352_4_initstate_0"
    )


def test_compose_rejects_different_base_tasks() -> None:
    with pytest.raises(ValueError, match="different bases"):
        compose_camera_background_task_name(
            "task_a_view_0_15_100_0_0_initstate_0",
            "task_b_tb_1",
        )


def test_protocol_selects_same_difficulty_tb_deterministically() -> None:
    view_protocol = {
        "study": "view",
        "official_camera_tasks": [
            {
                "suite": "libero_goal",
                "task_id": 4,
                "task_index": 3,
                "name": "task_view_30_0_100_0_0_initstate_0",
                "base_task": "task",
                "difficulty_level": 3,
                "perturbation_family": "orbit_yaw",
            },
            {
                "suite": "libero_goal",
                "task_id": 5,
                "task_index": 4,
                "name": "missing_view_30_0_100_0_0_initstate_0",
                "base_task": "missing",
                "difficulty_level": 3,
                "perturbation_family": "orbit_yaw",
            },
        ],
    }
    classification = {
        "libero_goal": [
            {
                "id": 1,
                "name": "task_table_1",
                "category": "Background Textures",
                "difficulty_level": 3,
            },
            {
                "id": 2,
                "name": "task_tb_2",
                "category": "Background Textures",
                "difficulty_level": 1,
            },
            {
                "id": 3,
                "name": "task_tb_3",
                "category": "Background Textures",
                "difficulty_level": 3,
            },
        ]
    }
    first = build_composition_protocol(view_protocol, classification, seed=7)
    second = build_composition_protocol(view_protocol, classification, seed=7)
    assert first == second
    row = first["composition_tasks"][0]
    assert row["background_task_name"] == "task_tb_3"
    assert row["background_exact_difficulty_match"] is True
    assert row["camera_background_task_name"] == (
        "task_tb_3_view_30_0_100_0_0_initstate_0"
    )
    assert first["summary"]["exact_difficulty_match_count"] == 1
    assert first["summary"]["excluded_camera_task_count"] == 1
    assert first["excluded_camera_tasks"][0]["base_task"] == "missing"
