from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from benchmarks.Robocasa365.evaluation_installation import (
    check_fixed_installation, prepare_candidate_observation)
from benchmarks.Robocasa365.camera_rig import CAMERA_NAMES


def test_local_mount_cannot_drift_during_rollout():
    positions, quaternions = np.zeros((3, 3)), np.zeros((3, 4))
    env = SimpleNamespace(sim=SimpleNamespace(model=SimpleNamespace(
        cam_pos=positions.copy(), cam_quat=quaternions.copy())))
    candidate = dict(positions=positions, quaternions=quaternions)
    check_fixed_installation(env, candidate)
    env.sim.model.cam_pos[0, 0] = .1
    with pytest.raises(ValueError):
        check_fixed_installation(env, candidate)


def test_candidate_render_gate_is_not_only_metadata():
    module = 'benchmarks.Robocasa365.evaluation_installation.'
    images = [np.zeros((8, 8, 3), np.uint8) for _ in CAMERA_NAMES]
    obs = {'video.' + key: value.copy() for key, value in zip(CAMERA_NAMES, images)}
    candidate = dict(positions=np.zeros((3, 3)), quaternions=np.zeros((3, 4)), images=images)
    env = SimpleNamespace(sim=object())
    with patch(module + 'ReferenceRig'), patch(module + 'refresh_bundle_observation', return_value=(obs, 'task')):
        _, checks = prepare_candidate_observation(env, object(), candidate, .005)
        assert all(row['pass'] for row in checks.values())
        obs['video.' + CAMERA_NAMES[0]][0, 0, 0] = 3
        with pytest.raises(ValueError):
            prepare_candidate_observation(env, object(), candidate, .005)
