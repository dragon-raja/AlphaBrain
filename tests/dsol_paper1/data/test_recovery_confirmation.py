from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from AlphaBrain.research.dsol.data.recovery_confirmation import planned_seed, validate_initial_plan, after_probe_preparation
from AlphaBrain.research.dsol.data.artifacts import read, write


class ConfirmationScopeTests(unittest.TestCase):
    def setUp(self):
        self.config={'population':dict(tasks=['Task'],environment_splits=['pretrain','target'],
            new_initials_per_task_split=2,new_physical_initials=4,new_seed_namespace='test')}
        self.plan={'records':[dict(task='Task',split=split,index=index,
            scene_seed=planned_seed('test','Task',split,index))
            for split in ['pretrain','target'] for index in range(2)]}

    def test_exact_scope(self):
        self.assertEqual(len(validate_initial_plan(self.config,self.plan)),4)
        self.assertEqual(planned_seed('robocasa_recovery_confirmation_20261009_v1',
                                    'CloseBlenderLid','pretrain',0),349185954)

    def test_missing_extra_duplicate_or_changed_seed_rejected(self):
        for mutate in [lambda rows:rows.pop(),lambda rows:rows.append(rows[0]),
                       lambda rows:rows[0].update(scene_seed=9),
                       lambda rows:rows[0].update(task='Replacement')]:
            changed=deepcopy(self.plan); mutate(changed['records'])
            with self.assertRaises(ValueError):validate_initial_plan(self.config,changed)

    def test_followthrough_stops_on_engineering_failure_without_retry(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root/'probes/manifest.json', dict(tasks=[dict(output=str(root/'worker'))]))
            write(root/'worker/status.json', dict(status='FAILED'))
            write(root/'config.json', dict(probes={'official': str(root/'probes')},
                  status_file=str(root/'status.json'), timeout_seconds=10))
            with self.assertRaisesRegex(RuntimeError, 'Failed probe retained'):
                after_probe_preparation(root/'config.json')
            self.assertEqual(read(root/'status.json')['gpu_jobs_started'], 0)
            self.assertEqual(read(root/'worker/status.json')['status'], 'FAILED')

    def test_completed_hook_only_prepares_plan_and_never_starts_gpu(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root/'probes/manifest.json', dict(tasks=[dict(output=str(root/'worker'))]))
            write(root/'worker/status.json', dict(status='COMPLETE'))
            write(root/'config.json', dict(probes={'official': str(root/'probes')},
                  selections={'official': str(root/'selections')}, status_file=str(root/'status.json'),
                  release='release', probe_config='probe', execution_plan='output', timeout_seconds=10))
            with patch('AlphaBrain.research.dsol.analysis.recovery_confirmation.freeze_ready_initials',
                       return_value={'status': 'COMPLETE'}) as freeze, patch(
                       'AlphaBrain.research.dsol.data.recovery_confirmation.prepare_execution_plan',
                       return_value={'rollout_count': 264}) as prepare:
                after_probe_preparation(root/'config.json')
            freeze.assert_called_once(); prepare.assert_called_once()
            status = read(root/'status.json')
            self.assertEqual(status['status'], 'READY_FOR_ENDPOINT_RUNTIME_ADMISSION')
            self.assertEqual(status['gpu_jobs_started'], 0)
