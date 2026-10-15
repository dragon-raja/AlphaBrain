import unittest

from AlphaBrain.research.dsol.data.training_recovery import compare_replayed_metrics


class TrainingRecoveryTests(unittest.TestCase):
    def test_replay_matches(self):
        row = dict(step=4001, loss=.07, grad_norm=.2, param_norm=1800.)
        self.assertEqual(compare_replayed_metrics(row, row)['status'], 'PASS')

    def test_replay_rejects_wrong_step_or_noise(self):
        row = dict(step=4001, loss=.07, grad_norm=.2, param_norm=1800.)
        for update in (dict(step=4002), dict(loss=.08), dict(loss=float('nan'))):
            with self.assertRaises(ValueError):
                compare_replayed_metrics(dict(row, **update), row)

    def test_numerical_gate_is_explicit_and_bounded(self):
        row = dict(step=4001, loss=.07, grad_norm=.2, param_norm=1800.)
        actual = dict(row, loss=.0705, grad_norm=.203)
        with self.assertRaises(ValueError):
            compare_replayed_metrics(actual, row)
        self.assertEqual(compare_replayed_metrics(actual, row, numerical_diagnostic=True)['status'], 'PASS')
        with self.assertRaises(ValueError):
            compare_replayed_metrics(dict(row, grad_norm=.21), row, numerical_diagnostic=True)
