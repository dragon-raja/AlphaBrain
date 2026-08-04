from __future__ import annotations

import numpy as np
import pytest

from serve_openpi_deterministic import DeterministicNoisePolicy


class FakePolicy:
    metadata = {"name": "fake"}

    def infer(self, observation, *, noise):
        return {"observation": observation, "noise": noise}


def test_deterministic_noise_is_keyed_by_request_seed() -> None:
    policy = DeterministicNoisePolicy(FakePolicy(), action_horizon=3, action_dim=2)
    first = policy.infer({"_eval_seed": 7, "value": 1})
    second = policy.infer({"_eval_seed": 7, "value": 2})
    np.testing.assert_array_equal(first["noise"], second["noise"])
    assert first["noise"].shape == (3, 2)
    assert "_eval_seed" not in first["observation"]
    assert policy.metadata["deterministic_eval_noise"] is True


def test_deterministic_noise_requires_seed() -> None:
    policy = DeterministicNoisePolicy(FakePolicy(), action_horizon=3, action_dim=2)
    with pytest.raises(KeyError, match="_eval_seed"):
        policy.infer({"value": 1})
