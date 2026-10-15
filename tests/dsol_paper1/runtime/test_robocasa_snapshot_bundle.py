from copy import deepcopy
import io
import gzip
import hashlib
import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from benchmarks.Robocasa365.snapshot import EnvSnapshot
from benchmarks.Robocasa365.snapshot_bundle import _LegacySnapshotUnpickler, snapshot_numeric_fingerprint
from benchmarks.Robocasa365.snapshot_bundle import save_snapshot_bundle, load_snapshot_bundle, load_compiled_model_xml, direct_scene_xml


class SnapshotCompatibilityTests(unittest.TestCase):
    def test_direct_scene_uses_saved_source_and_metadata_without_constructing_env(self):
        class Editor:
            def __init__(self):
                raise AssertionError('Environment constructor must not run')
            def edit_model_xml(self, xml):
                self._cam_configs['left']['pos'][0]=42
                return xml+' original camera transform'
        bundle=SimpleNamespace(model_xml='high precision source',ep_meta={
            'cam_configs':{'left':{'pos':[0,1,2]}},'gen_textures':{}})
        self.assertEqual(direct_scene_xml(bundle,editor_type=Editor),
                         'high precision source original camera transform')
        self.assertEqual(bundle.ep_meta['cam_configs']['left']['pos'],[0,1,2])
        bundle.ep_meta['gen_textures']={'cab_tex':'texture'}
        with self.assertRaises(ValueError):direct_scene_xml(bundle,editor_type=Editor)
        bundle.ep_meta={'cam_configs':{}}
        with self.assertRaises(ValueError):direct_scene_xml(bundle,editor_type=Editor)

    def test_direct_reader_selects_checked_compiled_xml_without_source_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            compiled='<mujoco><worldbody><camera name="runtime_camera"/></worldbody></mujoco>'
            payload=gzip.compress(compiled.encode())
            (path/'compiled_model.xml.gz').write_bytes(payload)
            bundle=SimpleNamespace(path=path,model_xml='<mujoco/>',manifest={'files':{
                'compiled_model.xml.gz':{'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}}})
            self.assertEqual(load_compiled_model_xml(bundle),compiled)
            (path/'compiled_model.xml.gz').write_bytes(payload+b'bad')
            with self.assertRaises(RuntimeError):load_compiled_model_xml(bundle)
            bundle.manifest['files'].clear()
            with self.assertRaises(ValueError):load_compiled_model_xml(bundle)

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
