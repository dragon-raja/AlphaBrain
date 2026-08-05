from __future__ import annotations

import numpy as np

from evaluate_pi05_libero_plus_views import (
    build_episode_specs,
    clean_task_prompt,
    initial_image_metrics,
    physics_state_sha256,
    quat_to_axis_angle,
    simulator_visibility_metrics,
    stable_seed,
    video_episode_indices,
)


def test_clean_task_prompt_removes_libero_plus_camera_metadata_source() -> None:
    assert clean_task_prompt("put_the_cream_cheese_in_the_bowl") == "put the cream cheese in the bowl"
    assert (
        clean_task_prompt(
            "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
        )
        == "put both the cream cheese box and the butter in the basket"
    )


def _protocol() -> dict:
    return {
        "official_camera_tasks": [
            {
                "suite": "libero_goal",
                "task_id": 3,
                "task_index": 2,
                "name": "task_view_30_0_100_0_0_initstate_0",
                "base_task": "task",
                "difficulty_level": 2,
                "perturbation_family": "orbit_yaw",
            }
        ],
        "candidate_matrix_base_tasks": [{"suite": "libero_goal", "base_task": "task"}],
        "candidate_views": [
            {
                "name": "canonical",
                "orbit_yaw_deg": 0,
                "orbit_pitch_deg": 0,
                "radius_percent": 100,
                "look_yaw_deg": 0,
                "look_pitch_deg": 0,
            },
            {
                "name": "yaw_m30",
                "orbit_yaw_deg": -30,
                "orbit_pitch_deg": 0,
                "radius_percent": 100,
                "look_yaw_deg": 0,
                "look_pitch_deg": 0,
            },
        ],
        "composition_tasks": [
            {
                "suite": "libero_goal",
                "task_id": 3,
                "task_index": 2,
                "name": "task_view_30_0_100_0_0_initstate_0",
                "camera_task_name": "task_view_30_0_100_0_0_initstate_0",
                "base_task": "task",
                "difficulty_level": 2,
                "camera_difficulty_level": 2,
                "perturbation_family": "orbit_yaw",
                "background_task_id": 9,
                "background_task_name": "task_tb_4",
                "background_difficulty_level": 2,
                "background_difficulty_distance": 0,
                "background_kind": "tb",
                "background_texture_index": 4,
                "background_exact_difficulty_match": True,
                "camera_background_task_name": (
                    "task_tb_4_view_30_0_100_0_0_initstate_0"
                ),
            }
        ],
    }


def test_build_episode_specs_pairs_gap_and_candidate_conditions() -> None:
    specs = build_episode_specs(_protocol(), modes=["gap", "candidates"])
    assert [row["condition"] for row in specs] == [
        "canonical",
        "official_camera",
        "candidate:canonical",
        "candidate:yaw_m30",
    ]
    assert specs[0]["pair_key"] == specs[1]["pair_key"]
    assert len({row["episode_id"] for row in specs}) == 4


def test_build_episode_specs_pairs_multiple_initial_states() -> None:
    specs = build_episode_specs(_protocol(), modes=["gap", "candidates"], init_state_count=2)
    assert len(specs) == 8
    assert {row["init_state_index"] for row in specs} == {0, 1}
    for init_state_index in (0, 1):
        rows = [row for row in specs if row["init_state_index"] == init_state_index]
        assert len({row["episode_id"] for row in rows}) == 4


def test_build_episode_specs_composition_has_paired_four_cell_design() -> None:
    specs = build_episode_specs(_protocol(), modes=["composition"], init_state_count=2)
    assert len(specs) == 8
    for init_state_index in (0, 1):
        rows = [row for row in specs if row["init_state_index"] == init_state_index]
        assert {row["condition"] for row in rows} == {
            "canonical",
            "camera_only",
            "background_only",
            "camera_background",
        }
        assert len({row["pair_key"] for row in rows}) == 1
        assert len({row["episode_id"] for row in rows}) == 4


def test_physics_state_sha256_is_stable_and_state_sensitive() -> None:
    class State:
        def __init__(self, values):
            self.values = values

        def flatten(self):
            return self.values

    class Sim:
        def __init__(self):
            self.values = [1.0, 2.0, 3.0]

        def get_state(self):
            return State(self.values)

    class Inner:
        sim = Sim()

    class Env:
        env = Inner()

    first, size = physics_state_sha256(Env())
    second, _ = physics_state_sha256(Env())
    Env.env.sim.values[0] = 2.0
    third, _ = physics_state_sha256(Env())
    assert size == 3
    assert first == second
    assert first != third


def test_video_selection_balances_gap_and_candidate_modes() -> None:
    specs = build_episode_specs(_protocol(), modes=["gap", "candidates"], init_state_count=2)
    selected = video_episode_indices(specs, 4)
    prefixes = [str(specs[index]["pair_key"]).split("::", 1)[0] for index in selected]
    assert prefixes.count("gap") == 2
    assert prefixes.count("candidate") == 2


def test_video_selection_includes_composition_mode() -> None:
    specs = build_episode_specs(_protocol(), modes=["composition"], init_state_count=2)
    selected = video_episode_indices(specs, 4)
    assert len(selected) == 4


def test_stable_seed_is_paired_but_key_sensitive() -> None:
    assert stable_seed("pair", seed=7) == stable_seed("pair", seed=7)
    assert stable_seed("pair", seed=7) != stable_seed("other", seed=7)


def test_image_metrics_and_axis_angle_are_finite() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, 4:] = 255
    metrics = initial_image_metrics(image)
    assert metrics["entropy_32bin_bits"] == 1.0
    assert metrics["mean_edge_strength"] > 0.0
    np.testing.assert_allclose(quat_to_axis_angle(np.asarray([0.0, 0.0, 0.0, 1.0])), 0.0)


def test_simulator_visibility_metrics_tracks_objects_and_frame_boundary() -> None:
    segmentation = np.zeros((4, 4, 2), dtype=np.int32)
    segmentation[..., 1] = -1
    segmentation[..., 0] = 5
    segmentation[0, :2, 1] = 4
    segmentation[2:4, 2:4, 1] = 7

    class Sim:
        def render(self, **_kwargs):
            return segmentation

    class Model:
        geom_ids_to_instances = {4: "object", 7: "goal"}

    class Inner:
        sim = Sim()
        model = Model()

    class Env:
        env = Inner()
        obj_of_interest = ["object", "goal"]

    metrics = simulator_visibility_metrics(Env(), width=4, height=4)
    assert metrics["all_interest_visible"] is True
    assert metrics["all_interest_visible_at_least_16px"] is False
    assert metrics["any_interest_border_touch"] is True
    assert metrics["objects"]["object"]["pixel_count"] == 2
    assert metrics["objects"]["goal"]["pixel_count"] == 4
