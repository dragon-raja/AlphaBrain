import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from AlphaBrain.research.dsol.data.rollout_replay import ReplayRecorder
from benchmarks.Robocasa365.policy_replay import replay_inputs


class FakePolicy:
    def __init__(self, checkpoint, config):
        pass

    def infer(self, request, noise):
        return noise + request['observation/state'].sum()

    def trace(self, request, noise):
        return dict(actions=self.infer(request, noise), velocity=noise[None])


class PolicyReplayTests(unittest.TestCase):
    def test_frozen_inputs_are_verified_and_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = ReplayRecorder()
            request = {'prompt': 'test', 'observation/state': np.ones(16, dtype=np.float32),
                       'observation/image': np.zeros((4, 4, 3), dtype=np.uint8)}
            noise = np.ones((50, 32), dtype=np.float32)
            recorder.record_plan(0, request, noise, noise, 'physical')
            record = recorder.save(root / 'replay')
            with patch('benchmarks.Robocasa365.policy_replay.ResidentPolicy', FakePolicy):
                result = replay_inputs(record, root / 'model', 'config', root / 'result.json')
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(len(result['rows']), 1)
            with self.assertRaises(FileExistsError):
                replay_inputs(record, root / 'model', 'config', root / 'result.json')

    def test_cli_does_not_load_gpu_without_execute(self):
        root = Path(__file__).resolve().parents[3]
        cli = root / 'scripts/dsol_paper1/diagnostics/robocasa_policy_replay.py'
        result = subprocess.run([sys.executable, str(cli), '--record', '/missing',
            '--checkpoint', '/missing', '--output', '/missing'],
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn('jax', result.stderr)
