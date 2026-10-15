"""CPU-only guards for the one-shot XR-1 phase transition."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.xr1_continuation import audit_complete, supervise


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'canonical'
        write(self.root / 'plan.json', dict(phase='canonical', initial_ids=[str(i) for i in range(40)],
              views=['canonical'], execution_repeats=4, score_repeats=8, rollouts=160))
        workers = [dict(pid=100+i, shard=i, output=str(self.root / 'workers' / str(i))) for i in range(8)]
        write(self.root / 'launch.json', dict(status='LAUNCHED', workers=workers))
        for worker in workers:
            write(Path(worker['output']) / 'status.json', dict(status='RUNNING'))

    @patch('AlphaBrain.research.dsol.data.xr1_continuation.process_running', return_value=True)
    def test_running_does_not_launch(self, _):
        self.assertIsNone(audit_complete(self.root, 'canonical'))

    @patch('AlphaBrain.research.dsol.data.xr1_continuation.process_running', return_value=False)
    def test_dead_worker_stops(self, _):
        with self.assertRaisesRegex(RuntimeError, 'exited without completion'):
            audit_complete(self.root, 'canonical')

    def test_failed_worker_stops(self):
        write(self.root / 'workers/0/status.json', dict(status='FAILED', error='render'))
        with self.assertRaisesRegex(RuntimeError, 'Worker failed'):
            audit_complete(self.root, 'canonical')

    def test_wrong_phase_stops(self):
        with self.assertRaisesRegex(ValueError, 'scope'):
            audit_complete(self.root, 'perturbed')

    @patch('AlphaBrain.research.dsol.data.xr1_continuation.subprocess.run')
    def test_existing_destination_never_relaunches(self, launch):
        target = Path(self.tmp.name) / 'perturbed'
        target.mkdir()
        with self.assertRaises(FileExistsError):
            supervise(self.root, target, Path(self.tmp.name) / 'record')
        launch.assert_not_called()

    @patch('AlphaBrain.research.dsol.data.xr1_continuation.subprocess.run')
    def test_record_prevents_duplicate_controller(self, launch):
        record = Path(self.tmp.name) / 'record'
        write(record / 'status.json', {'status': 'WAITING_CANONICAL'})
        with self.assertRaisesRegex(ValueError, 'Existing continuation'):
            supervise(self.root, Path(self.tmp.name) / 'perturbed', record)
        launch.assert_not_called()

    @patch('AlphaBrain.research.dsol.data.xr1_continuation.process_running', return_value=False)
    def test_complete_negative_panel_passes(self, _):
        from AlphaBrain.research.dsol.data.xr1_continuation import _seed
        plan = read(self.root / 'plan.json')
        plan.update(source=str(self.root / 'source'), panel=str(self.root / 'panel'),
                    checkpoint=str(self.root / 'checkpoint'), noise_version='xr1_paired_v1',
                    jobs=[dict(initial_id=i, task='test', split='target') for i in plan['initial_ids']])
        manifest = {}
        for name in ('xr1_evaluation.py', 'xr1_policy.py'):
            relative = 'benchmarks/Robocasa365/' + name
            write(Path(plan['source']) / relative, {})
            manifest[relative] = sha(Path(plan['source']) / relative)
        write(self.root / 'source-manifest.json', manifest)
        plan['source_manifest_sha256'] = sha(self.root / 'source-manifest.json')
        for directory, filename, key in (('panel', 'plan.json', 'panel_sha256'),
                                         ('checkpoint', 'IMPORT_RECEIPT.json', 'checkpoint_receipt_sha256')):
            write(self.root / directory / filename, {})
            plan[key] = sha(self.root / directory / filename)
        write(self.root / 'plan.json', plan)
        for worker in read(self.root / 'launch.json')['workers']:
            location = Path(worker['output'])
            initials = plan['initial_ids'][worker['shard']::8]
            identity = {k: plan[k] for k in ('panel_sha256', 'checkpoint_receipt_sha256', 'views', 'noise_version', 'execution_repeats')}
            identity.update(initials=initials, evaluator_sha256=manifest['benchmarks/Robocasa365/xr1_evaluation.py'],
                            adapter_sha256=manifest['benchmarks/Robocasa365/xr1_policy.py'])
            write(location / 'identity.json', identity)
            write(location / 'status.json', dict(status='COMPLETE', completed_rollouts=20))
            for initial in initials:
                cell = location / 'cells' / initial / 'canonical'
                write(cell / 'scores.npz', {})
                write(cell / 'scores.json', dict(arrays_sha256=sha(cell / 'scores.npz'),
                      rows=[dict(seed=_seed(initial, 'score', r), accel_p3=.1) for r in range(8)]))
                for r in range(4):
                    write(cell / f'trace_{r}.npz', {})
                    write(cell / f'execution_{r}.json', dict(repeat=r, complete_task_attempt=True,
                          success=False, terminated=False, truncated=False, steps=16, horizon=16,
                          trace_sha256=sha(cell / f'trace_{r}.npz'),
                          plans=[dict(step=0, seed=_seed(initial, 'execution', r))]))
        result = audit_complete(self.root, 'canonical')
        self.assertEqual((result['status'], result['rollouts'], result['successes']), ('PASS', 160, 0))


if __name__ == '__main__':
    unittest.main()
