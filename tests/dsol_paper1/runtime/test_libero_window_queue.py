import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from benchmarks.LIBERO.window_queue import block_assignment, prepare
from AlphaBrain.research.dsol.execution.manifest import validate


class WindowOwnershipTest(unittest.TestCase):
    def test_full_authorized_scope_without_repeating_admission(self):
        retained, partition = block_assignment()
        blocks = [(m, i) for m, i, _ in retained] + partition['fresh'] + partition['gnu']
        self.assertEqual(len(blocks), 64)
        self.assertEqual(set(blocks), {(m, i) for m in ('canonical', 'broad') for i in range(32)})
        self.assertLessEqual(abs(2 * len(partition['fresh']) - len(partition['gnu'])), 2)

    def test_relative_cli_config_becomes_absolute_manifest_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = dict(worker_config=str(root/'worker.json'), source=str(root/'controller'),
                experiment_id='test', environment={}, gpus={'fresh':[0]},
                control_roots={'fresh':str(root/'control')}, python='python')
            worker = dict(root=tmp, source=str(root/'worker'), inputs=str(root/'inputs.json'), anchors=str(root/'anchors.npz'))
            records = [dict(state=dict(task_id='task', pair_key=f'initial-{i}'), candidate_ids=['view']) for i in range(32)]
            with patch('benchmarks.LIBERO.window_queue.read', side_effect=[cfg,worker,dict(records=records)]), \
                    patch('benchmarks.LIBERO.window_queue.sha', return_value='a'*64), \
                    patch('benchmarks.LIBERO.window_queue.validate_block'), \
                    patch('AlphaBrain.research.dsol.execution.queue.prepare', side_effect=lambda plan, _: validate(plan)):
                self.assertEqual(prepare('relative-config.json', 'fresh'), root/'control')
