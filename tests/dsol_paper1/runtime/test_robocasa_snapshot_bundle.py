from copy import deepcopy
import io
import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from benchmarks.Robocasa365.snapshot import EnvSnapshot
from benchmarks.Robocasa365.snapshot_bundle import _LegacySnapshotUnpickler, snapshot_numeric_fingerprint
from benchmarks.Robocasa365.snapshot_bundle import save_snapshot_bundle, load_snapshot_bundle


class SnapshotCompatibilityTests(unittest.TestCase):
    def test_writer_roundtrip_checksums_and_no_overwrite(self):
        snapshot = EnvSnapshot(sim_state=np.zeros(5), camera_positions=np.zeros((3, 3)),
            camera_quaternions=np.zeros((3, 4)), model_runtime={},
            ctrl=np.zeros(2), qacc_warmstart=np.zeros(2), mocap_positions=None, mocap_quaternions=None,
            rng_state=None, numpy_random_state=(), wrapper_runtime=(), env_runtime={}, fixture_runtime={},
            controller_runtime=(), robot_runtime=(), gripper_actions=(), observable_runtime={})
        model=SimpleNamespace(get_xml=lambda:'<mujoco/>')
        base=SimpleNamespace(model=model,sim=SimpleNamespace(model=model),get_ep_meta=lambda:{'layout_id':1})
        with tempfile.TemporaryDirectory() as folder, \
                patch('benchmarks.Robocasa365.snapshot_bundle.inner_env',return_value=base), \
                patch('benchmarks.Robocasa365.snapshot_bundle.model_signature',return_value={'nq':1}):
            path=Path(folder)/'bundle'
            kwargs=dict(env=base,snapshot=snapshot,canonical_images={'rgb':np.zeros((2,2,3),np.uint8)},
                        metadata={'scene_seed':42})
            saved=save_snapshot_bundle(path,**kwargs)
            loaded=load_snapshot_bundle(path)
            self.assertEqual(saved,loaded.manifest)
            np.testing.assert_array_equal(loaded.canonical_images['rgb'],kwargs['canonical_images']['rgb'])
            with self.assertRaises(FileExistsError):save_snapshot_bundle(path,**kwargs)
            with (path/'snapshot.pkl.gz').open('ab') as stream:stream.write(b'bad')
            with self.assertRaises(RuntimeError):load_snapshot_bundle(path)

    def test_legacy_class_resolves_without_archive_import(self):
        unpickler = _LegacySnapshotUnpickler(io.BytesIO())
        self.assertIs(unpickler.find_class('robocasa_snapshot', 'EnvSnapshot'), EnvSnapshot)

    def test_numeric_fingerprint_preserves_and_detects_physical_change(self):
        snapshot = EnvSnapshot(sim_state=np.zeros(5), camera_positions=np.zeros((3, 3)),
            camera_quaternions=np.zeros((3, 4)), model_runtime={'body_pos': np.zeros((2, 3))},
            ctrl=np.zeros(2), qacc_warmstart=np.zeros(2), mocap_positions=None, mocap_quaternions=None,
            rng_state=None, numpy_random_state=(), wrapper_runtime=(), env_runtime={}, fixture_runtime={},
            controller_runtime=(), robot_runtime=(), gripper_actions=(), observable_runtime={})
        copied = deepcopy(snapshot)
        self.assertEqual(snapshot_numeric_fingerprint(snapshot), snapshot_numeric_fingerprint(copied))
        copied.ctrl[0] = 1.
        self.assertNotEqual(snapshot_numeric_fingerprint(snapshot), snapshot_numeric_fingerprint(copied))
