"""Policy admission helpers, using only generated image arrays."""

import numpy as np
import pytest

from benchmarks.LIBERO.search_features import batch_check_inputs, load_observations, policy_example, run
from AlphaBrain.research.dsol.data.artifacts import sha


def test_input_hash_and_exact_image_mapping(tmp_path):
    path = tmp_path / "observations.npz"
    arrays = dict(
        candidate_ids=np.asarray(["a", "b"]),
        external_images=np.ones((2, 3, 3, 3), np.uint8),
        wrist_images=np.full((2, 3, 3, 3), 2, np.uint8),
        robot_states=np.ones((2, 8), np.float32),
        camera_intrinsics=np.tile(np.eye(3), (2, 1, 1)),
        camera_to_world_opencv=np.tile(np.eye(4), (2, 1, 1)))
    np.savez(path, **arrays)
    record = dict(observation_cache=dict(path=str(path), sha256=sha(path)),
                  candidate_ids=["a", "b"], state=dict(prompt="test task"))
    cached = load_observations(record)
    example = policy_example(record, cached, 1)
    np.testing.assert_array_equal(example["image"][0], arrays["external_images"][1])
    np.testing.assert_array_equal(example["image"][1], arrays["wrist_images"][1])
    assert example["lang"] == "test task"
    record["observation_cache"]["sha256"] = "invalid"
    with pytest.raises(ValueError, match="hash"):
        load_observations(record)


def test_no_implicit_model_loading(tmp_path):
    with pytest.raises(ValueError, match="execute"):
        run("absent-config", tmp_path / "never")
    assert not (tmp_path / "never").exists()


def test_batch_check_uses_exact_historical_or_duplicate_axis():
    cached = dict(external_images=np.arange(16), wrist_images=np.arange(16),
                  robot_states=np.arange(16), camera_intrinsics=np.arange(16),
                  camera_to_world_opencv=np.arange(16))
    noise = np.arange(70, dtype=np.float32).reshape(10, 7)
    for duplicate in (False, True):
        examples, actual = batch_check_inputs(
            dict(state=dict(prompt="task")), cached, noise, duplicate=duplicate)
        assert [x["image"][0] for x in examples] == ([0] * 16 if duplicate else list(range(16)))
        np.testing.assert_array_equal(actual, np.repeat(noise[None], 16, 0))
