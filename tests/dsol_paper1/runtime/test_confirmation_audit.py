import dataclasses
import numpy as np

from benchmarks.Robocasa365.confirmation_audit import candidate_fingerprint
from benchmarks.Robocasa365.snapshot import EnvSnapshot
from benchmarks.Robocasa365.snapshot_bundle import snapshot_numeric_fingerprint


def test_expected_fingerprint_changes_only_installation():
    s=EnvSnapshot(sim_state=np.zeros(5),camera_positions=np.zeros((3,3)),
        camera_quaternions=np.zeros((3,4)),model_runtime={'body_pos':np.zeros((2,3))},
        ctrl=np.zeros(2),qacc_warmstart=np.zeros(2),mocap_positions=None,mocap_quaternions=None,
        rng_state=None,numpy_random_state=(),wrapper_runtime=(),env_runtime={},fixture_runtime={},
        controller_runtime=(),robot_runtime=(),gripper_actions=(),observable_runtime={})
    p=s.camera_positions.copy();p[0,0]=.3
    expected=candidate_fingerprint(s,p,s.camera_quaternions)
    assert expected!=snapshot_numeric_fingerprint(s)
    assert expected==snapshot_numeric_fingerprint(dataclasses.replace(s,camera_positions=p))
    bad=dataclasses.replace(s,ctrl=np.ones(2))
    assert expected!=candidate_fingerprint(bad,p,s.camera_quaternions)
    assert np.all(s.camera_positions==0)
