"""Install a released camera rig without changing the physical rollout start.

Offsets remain support-local throughout execution; this is not active camera
control and does not add camera coordinates to the policy action head.
"""
import dataclasses

import numpy as np

from .camera_rig import CAMERA_NAMES, ReferenceRig
from .evaluation_render import pixel_check
from .snapshot import inner_env, restore_snapshot
from .snapshot_bundle import (env_numeric_fingerprint, refresh_bundle_observation,
                              snapshot_numeric_fingerprint)


def prepare_candidate_observation(env, bundle, candidate, mean_limit):
    rig = ReferenceRig(inner_env(env).sim)
    rig.install((candidate['positions'], candidate['quaternions']))
    obs, _ = refresh_bundle_observation(env, bundle)
    if len(candidate['images']) != len(CAMERA_NAMES):
        raise ValueError('Expected three candidate images')
    checks = {name: pixel_check(obs['video.' + name], image, mean_limit)
              for name, image in zip(CAMERA_NAMES, candidate['images'])}
    if not all(value['pass'] for value in checks.values()):
        raise ValueError('Installed candidate does not match released RGB cache: ' + str(checks))
    return obs, checks


def restore_candidate_start(env, bundle, candidate):
    """Restore controller/RNG/warm-start, replacing only camera installation."""
    restore_snapshot(env, bundle.snapshot)
    sim = inner_env(env).sim
    # Validate non-exterior cameras using the canonical installation as baseline.
    rig = ReferenceRig(sim)
    rig.install((candidate['positions'], candidate['quaternions']))
    # forward() above can change warm-start. Restore all physical fields again,
    # then write only the already validated local camera arrays. The cached first
    # observation is from precisely this installation and physical state.
    restore_snapshot(env, bundle.snapshot)
    sim.model.cam_pos[:] = candidate['positions']
    sim.model.cam_quat[:] = candidate['quaternions']
    expected = dataclasses.replace(bundle.snapshot, camera_positions=candidate['positions'],
                                   camera_quaternions=candidate['quaternions'])
    actual = env_numeric_fingerprint(env)
    if actual != snapshot_numeric_fingerprint(expected):
        raise ValueError('Candidate installation altered physical rollout start')
    return actual


def check_fixed_installation(env, candidate):
    sim = inner_env(env).sim
    if (not np.array_equal(sim.model.cam_pos, candidate['positions'])
            or not np.array_equal(sim.model.cam_quat, candidate['quaternions'])):
        raise ValueError('Local camera installation changed during rollout')
