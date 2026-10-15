from copy import deepcopy
import unittest

from AlphaBrain.research.dsol.analysis.libero_baseline_panel import validate_pair, validate_native_noise_call


class BaselineAuditTests(unittest.TestCase):
    def test_native_seed_uses_frozen_evaluator_mapping(self):
        common = dict(noise_seed=8775796698384572091, noise_sha256='shared', replan_index=0)
        full = dict(eval_seed=1339418608, shared_noise_sha256='shared',
                    full_noise_sha256='f'*64, call_index=0)
        validate_native_noise_call(full, common)
        for key, wrong in [('eval_seed', common['noise_seed']), ('call_index', 1),
                           ('shared_noise_sha256', 'wrong')]:
            altered = dict(full, **{key: wrong})
            with self.assertRaises(ValueError):
                validate_native_noise_call(altered, common)

    def rows(self):
        row = dict(status='complete', explicit_flow_noise=True, success=False, completion_steps=300,
                   max_steps=300, inference_calls=1,
                   policy_calls=[dict(noise_sha256='abc', replan_index=0)],
                   pair_key='initial', selected_candidate_id='canonical', initialization_receipt={'hash': 'initial'},
                   noise_bank_manifest_sha256='bank', replan_steps=5, wait_steps=0,
                   sensor_control='both', prompt='task', policy_repeat_id=0)
        return [deepcopy(row) for _ in range(3)]

    def test_same_protocol(self):
        validate_pair(self.rows())

    def test_reject_noise_change(self):
        rows = self.rows()
        rows[1]['policy_calls'][0]['noise_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            validate_pair(rows)

    def test_reject_initial_change(self):
        rows = self.rows()
        rows[2]['initialization_receipt']['hash'] = 'changed'
        with self.assertRaises(ValueError):
            validate_pair(rows)

    def test_reject_early_failure(self):
        rows = self.rows()
        rows[0]['completion_steps'] = 10
        with self.assertRaises(ValueError):
            validate_pair(rows)


if __name__ == '__main__':
    unittest.main()
