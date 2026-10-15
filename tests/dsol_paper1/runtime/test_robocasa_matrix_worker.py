from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from benchmarks.Robocasa365.evaluation_matrix import _score_cell, _rollout


class Policy:
    def __init__(self):
        self.noises = []

    def infer(self, request, noise):
        self.noises.append(noise.copy())
        return np.zeros((50, 12), dtype=np.float32)

    def trace(self, request, noise):
        return dict(actions=np.zeros((50, 12), dtype=np.float32), steps=10,
                    velocity=np.broadcast_to(noise, (1, 10, 50, 32)).copy(),
                    times=np.arange(10)/10, scores={'3': {'score': 1., 'degenerate': False}})


def test_scores_keep_all_eight_noises_and_restore_frozen_first_request(tmp_path):
    policy = Policy()
    req = {'prompt': 'task', 'observation/state': np.ones(16, dtype=np.float32)}
    first, result = _score_cell(policy, req, 'initial', tmp_path, {'plan': 'one'})
    assert len(result['rows']) == 8
    assert len({r['noise']['sha256'] for r in result['rows']}) == 8
    changed = {'prompt': 'different', 'observation/state': np.zeros(16, dtype=np.float32)}
    restored, existing = _score_cell(policy, changed, 'initial', tmp_path, {'plan': 'one'})
    np.testing.assert_array_equal(restored['observation/state'], first['observation/state'])
    assert restored['prompt'] == 'task'
    assert existing == result


def test_rollout_preserves_full_horizon_and_addresses_replanning_noise():
    class Env:
        steps = 0
        def step(self, action):
            self.steps += 1
            return {}, 0., False, False, {'success': False}
    env, policy = Env(), Policy()
    req = {'prompt': 'task', 'observation/state': np.ones(16)}
    fake = SimpleNamespace(convert_action=lambda action: action)
    with patch.dict(sys.modules, {'robocasa.utils.env_utils': fake}), \
            patch('benchmarks.Robocasa365.evaluation_matrix.restore_candidate_start', return_value='physical'), \
            patch('benchmarks.Robocasa365.evaluation_matrix.check_fixed_installation'), \
            patch('benchmarks.Robocasa365.evaluation_matrix.env_numeric_fingerprint', return_value='end'), \
            patch('benchmarks.Robocasa365.evaluation_matrix.policy_request', return_value=req):
        result = _rollout(env, None, None, policy, req, 'initial', 2, 55)
    assert env.steps == 55 and result['steps'] == 55
    assert [r['step'] for r in result['plans']] == [0, 25, 50]
    assert len({r['noise']['sha256'] for r in result['plans']}) == 3
    assert not result['success']


def test_worker_cli_is_safe_without_execute():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run([sys.executable, str(root/'scripts/dsol_paper1/runtime/robocasa_matrix_worker.py'),
        '--plan', '/missing', '--worker', 'missing', '--output', '/missing'], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'robosuite' not in result.stderr
