import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import types
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
NAMES = ['serve_alphabrain_pi05_websocket', 'evaluate_pi05_libero_plus_views',
         'libero_camera_pose', 'build_libero_plus_view_protocol']


@pytest.mark.parametrize('name', NAMES)
def test_implementation_ast_identical_except_import_layout(name):
    manifest = json.loads((ROOT/'archive/paper1/layout_manifest.json').read_text())
    baseline = manifest['shared_extraction']['modules'][name]
    tree = ast.parse((ROOT/'scripts/vla_shared'/f'{name}.py').read_text())
    tree.body = [x for x in tree.body if not isinstance(x, (ast.Import, ast.ImportFrom))
                 and not (isinstance(x, ast.If) and isinstance(x.test, ast.Name) and x.test.id == '__package__')]
    assert hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == baseline['nonimport_ast_sha256']


@pytest.mark.parametrize('name', NAMES)
def test_legacy_import_is_same_module_including_private_symbols(name):
    canonical = importlib.import_module('scripts.vla_shared.'+name)
    legacy = importlib.import_module('scripts.cabi_vla.'+name)
    assert legacy is canonical
    assert Path(legacy.__file__).parent == ROOT/'scripts/vla_shared'
    assert vars(legacy) is vars(canonical)


@pytest.mark.parametrize('seed', [0, 41, 1001])
@pytest.mark.parametrize('explicit', [False, True])
def test_old_and_new_policy_adapter_have_identical_inputs_actions_and_hashes(seed, explicit):
    result = subprocess.run(['git', 'show', '37090c0:scripts/cabi_vla/serve_alphabrain_pi05_websocket.py'],
                            cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        pytest.skip('Historical Git object absent; AST fingerprint and portable unit tests still apply')
    old = types.ModuleType('policy_before_extraction')
    exec(compile(result.stdout, 'policy_before_extraction.py', 'exec'), old.__dict__)
    new = importlib.import_module('scripts.vla_shared.serve_alphabrain_pi05_websocket')
    class FakeModel:
        def predict_action(self, *, examples, noise):
            self.examples, self.noise = examples, noise
            values = np.arange(70, dtype=np.float32).reshape(1,10,7)/20-1.5
            return {'normalized_actions': values}
    policies = []
    for module in [old, new]:
        policy = module.AlphaBrainPi05Policy.__new__(module.AlphaBrainPi05Policy)
        policy._horizon, policy._model = 10, FakeModel()
        policies.append(policy)
    observation = {'observation/image': np.arange(60,dtype=np.uint8).reshape(4,5,3),
                   'observation/wrist_image': np.full((4,5,3),7,np.uint8),
                   'observation/state': np.linspace(-1,1,8), 'prompt':'pick up the mug', '_eval_seed':seed,
                   'camera_intrinsics': np.eye(3), 'camera_to_world_opencv':np.eye(4)}
    if explicit:
        noise=np.arange(70,dtype=np.float32).reshape(10,7)/31
        noise.setflags(write=False)
        observation.update(_eval_noise=noise, _eval_noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest())
    with patch.object(old.time, 'perf_counter', return_value=10.0):
        a,b = [p.infer(observation) for p in policies]
    assert a.keys()==b.keys()
    for key in a:
        if isinstance(a[key],np.ndarray):np.testing.assert_array_equal(a[key],b[key])
        else:assert a[key]==b[key]
    if explicit:np.testing.assert_array_equal(policies[0]._model.noise,policies[1]._model.noise)
    for key in policies[0]._model.examples[0]:
        x,y=[p._model.examples[0][key] for p in policies]
        if isinstance(x,list):
            for lhs,rhs in zip(x,y):np.testing.assert_array_equal(lhs,rhs)
        elif isinstance(x,np.ndarray):np.testing.assert_array_equal(x,y)
        else:assert x==y


def test_shared_path_resolver_only_falls_back_to_verified_frozen_release(tmp_path):
    from scripts.dsol_paper1.shared_runtime_paths import shared_scripts
    repo=tmp_path/'repo';shared=repo/'scripts/vla_shared';shared.mkdir(parents=True)
    assert shared_scripts(repo)==shared
    shared.rmdir()
    legacy=repo/'scripts/cabi_vla';legacy.mkdir()
    server=legacy/'serve_alphabrain_pi05_websocket.py';server.write_text('historical bytes')
    with pytest.raises(FileNotFoundError):shared_scripts(repo)
    (tmp_path/'release.json').write_text(json.dumps({'source_hashes':{str(server):hashlib.sha256(server.read_bytes()).hexdigest()}}))
    assert shared_scripts(repo)==legacy
    server.write_text('modified bytes')
    with pytest.raises(ValueError):shared_scripts(repo)


def test_new_runner_receipts_cover_shared_implementation_not_only_shims():
    from scripts.dsol_paper1.shared_runtime_paths import SHARED_SOURCE_NAMES, shared_code_paths
    paths=shared_code_paths(ROOT)
    assert len(paths)==6
    assert {p.name for p in paths} == {*SHARED_SOURCE_NAMES,'shared_runtime_paths.py'}
    assert all(p.is_file() for p in paths)
    shell=(ROOT/'scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh').read_text()
    assert '"${shared_code_files[@]}"' in shell
    for name in SHARED_SOURCE_NAMES:assert name in shell
