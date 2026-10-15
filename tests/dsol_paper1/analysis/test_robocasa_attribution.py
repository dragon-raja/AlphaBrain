import unittest
import numpy as np
from AlphaBrain.research.dsol.analysis.robocasa_attribution import split_noise_diagnostic


class AttributionTests(unittest.TestCase):
    def test_all_splits_keep_noise_disjoint(self):
        result=split_noise_diagnostic(np.ones((3,5,4)))
        self.assertEqual(len(result['partitions']),6)
        for record in result['partitions']:
            self.assertFalse(set(record['selection_noises'])&set(record['evaluation_noises']))
        np.testing.assert_array_equal(result['selected_by_initial'],np.ones(3))

    def test_does_not_choose_again_on_test_noises(self):
        result=split_noise_diagnostic(np.array([[[1,1,0,0],[0,0,1,1]]]))
        first=result['partitions'][0]
        self.assertEqual(first['selected_success'],0)
        self.assertEqual(first['tie_mean_success'],0)


if __name__=='__main__':
    unittest.main()
