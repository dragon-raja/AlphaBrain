from __future__ import annotations

import numpy as np

from libero_scene_cues import (
    _quaternion_multiply,
    _yaw_quaternion,
    stable_scene_seed,
)


def test_scene_seed_is_stable_and_sample_specific() -> None:
    assert stable_scene_seed(41, "episode-a") == stable_scene_seed(
        41,
        "episode-a",
    )
    assert stable_scene_seed(41, "episode-a") != stable_scene_seed(
        41,
        "episode-b",
    )


def test_yaw_quaternion_is_unit_length() -> None:
    quaternion = _yaw_quaternion(np.pi / 3)
    assert np.isclose(np.linalg.norm(quaternion), 1.0)


def test_quaternion_product_preserves_unit_length() -> None:
    first = _yaw_quaternion(np.pi / 4)
    second = _yaw_quaternion(-np.pi / 7)
    product = _quaternion_multiply(first, second)
    assert np.isclose(np.linalg.norm(product), 1.0)

