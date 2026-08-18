from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from scripts.dsol_paper1.evaluate_dsol_libero_hdf5_views import (
    deployed_camera_names,
    masked_policy_observation,
)


def test_all_blackout_has_no_deployed_camera() -> None:
    assert deployed_camera_names("all_blackout") == ()


def test_all_blackout_zeroes_both_policy_images(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("evaluate_pi05_libero_plus_views")

    def prepare(*_args: object, **_kwargs: object):
        agent = np.full((2, 3, 3), 17, dtype=np.uint8)
        wrist = np.full((2, 3, 3), 29, dtype=np.uint8)
        return {
            "observation/image": agent.copy(),
            "observation/wrist_image": wrist.copy(),
        }, agent, wrist

    module.prepare_policy_observation = prepare
    monkeypatch.setitem(sys.modules, module.__name__, module)
    example, agent, wrist = masked_policy_observation(
        {},
        prompt="test",
        resize_size=2,
        eval_seed=1,
        camera_calibration={},
        sensor_control="all_blackout",
    )
    assert not agent.any()
    assert not wrist.any()
    assert not example["observation/image"].any()
    assert not example["observation/wrist_image"].any()
