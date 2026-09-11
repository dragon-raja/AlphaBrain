"""CPU-only tests for the opt-in GPU validation harness."""
import copy
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

from tools.paper1.runtime_probe import identity
from tools.paper1.verify_runtime import environment, require_comparison, signature


def test_leaf_imports_do_not_eagerly_load_protocol_dataclasses():
    root = Path(__file__).resolve().parents[2]
    command = (
        "import sys; from AlphaBrain.research.dsol.data import flow_noise; "
        "from AlphaBrain.research.dsol.metrics import visibility; "
        "assert 'AlphaBrain.research.dsol.protocol' not in sys.modules"
    )
    result = subprocess.run([sys.executable, '-c', command], cwd=root,
                            env=dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES=''),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_protocol_public_exports_are_preserved():
    from AlphaBrain.research import dsol
    from AlphaBrain.research.dsol import protocol
    for name in dsol.__all__:
        assert getattr(dsol, name) is getattr(protocol, name)
        assert name in dir(dsol)
    with pytest.raises(AttributeError):
        getattr(dsol, 'not_a_public_export')


def test_trace_gate_checks_more_than_success_rate():
    row = dict(success=True, completion_steps=85, aa0=dict(trace_sha256='trace', counts=dict(calls=17, steps=95)))
    for field, value in [('trace_sha256', 'other'), ('counts', dict(calls=18, steps=95))]:
        changed = copy.deepcopy(row)
        changed['aa0'][field] = value
        assert signature(changed) != signature(row)


@pytest.mark.parametrize('key', ['exact_new_old', 'exact_requests_noise_actions_times',
                                 'frozen_matches_historical', 'current_matches_historical'])
def test_gate_rejects_each_independent_mismatch(key):
    row = {k: True for k in ['exact_new_old', 'exact_requests_noise_actions_times',
                            'frozen_matches_historical', 'current_matches_historical']}
    require_comparison(row)
    row[key] = False
    with pytest.raises(AssertionError):
        require_comparison(row)


def test_probe_identity_preserves_shape_dtype_and_values():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert identity(x) == identity(x.copy())
    assert identity(x) != identity(x.astype(np.float64))
    assert identity(x) != identity(x.reshape(4, 3))
    y = x.copy(); y[0, 0] = 1
    assert identity(x) != identity(y)


def test_environment_prioritizes_explicit_source_and_pinned_dependencies(tmp_path):
    repo = tmp_path / 'current'
    (repo / 'scripts/vla_shared').mkdir(parents=True)
    experiment = tmp_path / 'experiment'
    env = environment(experiment, repo, sim=True)
    assert env['PYTHONPATH'].split(':')[0] == str(repo)
    assert str(experiment / 'runtime/sim/lib/python3.8/site-packages') in env['PYTHONPATH']
    assert env['ALPHABRAIN_DISABLE_AUTO_DOWNLOAD'] == '1'
    assert 'CUDA_VISIBLE_DEVICES' not in env
    policy = environment(experiment, repo, sim=False, gpu=3)
    assert policy['CUDA_VISIBLE_DEVICES'] == '3'
