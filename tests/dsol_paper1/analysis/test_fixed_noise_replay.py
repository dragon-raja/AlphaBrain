import copy
import unittest

from AlphaBrain.research.dsol.analysis.fixed_noise_replay import compare


class FixedNoiseReplayTests(unittest.TestCase):
    def setUp(self):
        self.trial = dict(initial_physical_sha256='initial', final_physical_sha256='final',
                          executed_actions_sha256='actions', success=True, steps=50,
                          plans=[dict(step=i, noise='noise', actions='action',
                                      input_arrays={'observation/image': 'image', 'observation/state': 'state'})
                                 for i in (0, 25)])

    def test_identical(self):
        self.assertIsNone(compare(self.trial, self.trial)['first_recorded_divergence'])

    def test_image_first(self):
        other = copy.deepcopy(self.trial)
        other['plans'][1]['input_arrays']['observation/image'] = 'changed'
        other['plans'][1]['actions'] = 'changed'
        self.assertEqual(compare(self.trial, other)['first_recorded_divergence']['kind'], 'image_only')

    def test_policy_first(self):
        other = copy.deepcopy(self.trial)
        other['plans'][0]['actions'] = 'changed'
        self.assertEqual(compare(self.trial, other)['first_recorded_divergence']['kind'],
                         'actions_with_identical_recorded_input')

    def test_noise_mismatch_rejected(self):
        other = copy.deepcopy(self.trial)
        other['plans'][1]['noise'] = 'different'
        with self.assertRaises(ValueError):
            compare(self.trial, other)


if __name__ == '__main__':
    unittest.main()
