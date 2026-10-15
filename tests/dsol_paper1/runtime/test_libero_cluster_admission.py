from copy import deepcopy
import json
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.LIBERO.cluster_admission import compare_rows
from benchmarks.LIBERO.plus_evaluation_queue import PlusEvaluationBackend
from AlphaBrain.research.dsol.data.artifacts import sha


class CrossHostTrajectoryTest(unittest.TestCase):
    def row(self):
        return dict(initial=0, model='canonical', pair_key='a', view='v', repeat=0, success=False,
                    steps=300, max_steps=300, calls=[dict(noise_sha256='n', actions_sha256='a')],
                    initial_input_exact=True, seconds=30)

    def test_timing_can_change_but_science_cannot(self):
        a=self.row(); b=deepcopy(a); b['seconds']=12
        self.assertEqual(compare_rows([a], [b])['episodes'], 1)

    def test_same_success_does_not_excuse_changed_action(self):
        a=self.row(); b=deepcopy(a); b['calls'][0]['actions_sha256']='different'
        with self.assertRaises(ValueError): compare_rows([a], [b])

    def test_missing_episode_refused(self):
        with self.assertRaises(ValueError): compare_rows([self.row()], [])

    def test_host_proof_bound_to_runtime_and_both_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg = root/'config.json'; cfg.write_text('{}')
            wrapper = root/'python'; wrapper.write_text('isolated runtime')
            proof = dict(status='PASS', host=platform.node(), host_label='video',
                production_config_sha256=sha(cfg), python_wrapper=str(wrapper), python_wrapper_sha256=sha(wrapper),
                rows=[dict(model=m, initial_scores_exact=True, action_and_noise_hashes_exact=True,
                    completion_steps_exact=True, successes_exact=True) for m in ('canonical','broad')])
            path = root/'proof.json'; path.write_text(json.dumps(proof))
            plan = dict(runtime=dict(host='video', config=str(cfg)),
                resource_admission=dict(path=str(path), sha256=sha(path)))
            backend = PlusEvaluationBackend(plan, root)
            with patch.object(backend, 'gpu_available', return_value=True):
                self.assertTrue(backend.admit(dict(host='video', id='0')))
                wrapper.write_text('changed runtime')
                with self.assertRaises(ValueError): backend.admit(dict(host='video', id='0'))
