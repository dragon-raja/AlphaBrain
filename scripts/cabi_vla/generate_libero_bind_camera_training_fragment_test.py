from __future__ import annotations

from generate_libero_bind_camera_training_fragment import (
    camera_variant_index,
    episode_camera_pool,
    global_camera_catalog,
    sample_training_pose,
    stable_uniform,
    training_camera_pool,
)


def _config(baseline_probability: float = 0.0) -> dict:
    return {
        "seed": 41,
        "baseline_probability": baseline_probability,
        "poses_per_episode": 4,
        "ranges": {
            "azimuth_deg": [-30.0, 30.0],
            "elevation_deg": [-15.0, 15.0],
            "radius_scale": [0.95, 1.15],
        },
    }


def test_stable_uniform_is_reproducible_and_bounded() -> None:
    first = stable_uniform(41, "sample", "azimuth")
    second = stable_uniform(41, "sample", "azimuth")
    assert first == second
    assert 0.0 <= first < 1.0


def test_random_pose_is_deterministic_and_inside_ranges() -> None:
    first = sample_training_pose(sample_id="sample", config=_config())
    second = sample_training_pose(sample_id="sample", config=_config())
    assert first == second
    assert -30.0 <= first["azimuth_deg"] <= 30.0
    assert -15.0 <= first["elevation_deg"] <= 15.0
    assert 0.95 <= first["radius_scale"] <= 1.15


def test_baseline_probability_one_always_selects_baseline() -> None:
    pose = sample_training_pose(
        sample_id="sample",
        config=_config(baseline_probability=1.0),
    )
    assert pose == {
        "name": "baseline",
        "azimuth_deg": 0.0,
        "elevation_deg": 0.0,
        "radius_scale": 1.0,
    }


def test_episode_camera_pool_is_fixed_and_finite() -> None:
    first = episode_camera_pool(
        edge_id="red-left",
        episode_file="episodes/red-left--state-00.npz",
        config=_config(),
    )
    second = episode_camera_pool(
        edge_id="red-left",
        episode_file="episodes/red-left--state-00.npz",
        config=_config(),
    )
    other = episode_camera_pool(
        edge_id="red-left",
        episode_file="episodes/red-left--state-01.npz",
        config=_config(),
    )
    assert first == second
    assert len(first) == 4
    assert first != other


def test_record_selects_only_from_episode_camera_pool() -> None:
    config = _config()
    indices = {
        camera_variant_index(sample_id=f"sample-{index}", config=config)
        for index in range(100)
    }
    assert indices == {0, 1, 2, 3}


def test_global_camera_catalog_is_nested_across_budgets() -> None:
    small = {
        **_config(),
        "sampling_unit": "global_camera_catalog",
        "camera_catalog_size": 10,
    }
    large = {**small, "camera_catalog_size": 45}
    small_catalog = global_camera_catalog(config=small)
    large_catalog = global_camera_catalog(config=large)
    assert len(small_catalog) == 10
    assert len(large_catalog) == 45
    assert small_catalog == large_catalog[:10]


def test_global_catalog_is_shared_across_episodes() -> None:
    config = {
        **_config(),
        "sampling_unit": "global_camera_catalog",
        "camera_catalog_size": 22,
    }
    first = training_camera_pool(
        edge_id="red-left",
        episode_file="episodes/first.npz",
        config=config,
    )
    second = training_camera_pool(
        edge_id="yellow-right",
        episode_file="episodes/second.npz",
        config=config,
    )
    assert first == second


def test_epoch_replicas_choose_deterministic_catalog_members() -> None:
    config = {
        **_config(),
        "sampling_unit": "global_camera_catalog",
        "camera_catalog_size": 45,
    }
    first = [
        camera_variant_index(
            sample_id="sample",
            config=config,
            epoch_replica=replica,
        )
        for replica in range(3)
    ]
    second = [
        camera_variant_index(
            sample_id="sample",
            config=config,
            epoch_replica=replica,
        )
        for replica in range(3)
    ]
    assert first == second
    assert len(set(first)) >= 2
