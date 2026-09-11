from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/dsol_paper1'))
from scripts.dsol_paper1.diagnostics.audit_view_speed_equivalence_v1 import compare


class SpeedEquivalenceAuditTest(unittest.TestCase):
    def fixture(self):
        row = dict(task_id='task', environment_seed=1, sensor_control='both', replan_steps=5,
                   wait_steps=0, noise_bank_manifest_sha256='bank', pose=None,
                   initial_metrics={'physics_state_sha256': 'physical'}, success=True,
                   completion_steps=12, policy_calls=[
                       dict(replan_index=i, noise_seed=i, noise_sha256=f'noise-{i}',
                            action_chunk_sha256=f'action-{i}') for i in range(3)])
        reference = {('state', 'view', 0): row}
        return reference, deepcopy(reference)

    def test_late_action_drift_is_not_noise_mismatch(self):
        reference, candidate = self.fixture()
        candidate[('state', 'view', 0)]['policy_calls'][2]['action_chunk_sha256'] = 'different'
        result = compare(reference, candidate)
        self.assertEqual(result['mismatch_counts']['full_action_path_equal'], 1)
        self.assertEqual(result['mismatch_counts']['common_replan_noise_equal'], 0)
        self.assertEqual(result['mismatch_counts']['success_equal'], 0)
        self.assertEqual(result['details'][0]['first_different_action_call'], 2)

    def test_identical_actions_do_not_prove_identical_noise(self):
        reference, candidate = self.fixture()
        candidate[('state', 'view', 0)]['policy_calls'][0]['noise_sha256'] = 'different'
        result = compare(reference, candidate)
        self.assertEqual(result['mismatch_counts']['full_action_path_equal'], 0)
        self.assertEqual(result['mismatch_counts']['common_replan_noise_equal'], 1)

    def test_changed_scientific_controls_rejected(self):
        reference, candidate = self.fixture()
        candidate[('state', 'view', 0)]['replan_steps'] = 10
        with self.assertRaises(ValueError):
            compare(reference, candidate)


if __name__ == '__main__':
    unittest.main()
