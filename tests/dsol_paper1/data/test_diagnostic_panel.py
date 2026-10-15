import unittest
from itertools import product
import numpy as np

from AlphaBrain.research.dsol.data.diagnostic_panel import frozen_choices, geometry_panel
from AlphaBrain.research.dsol.data.evaluation_matrix import flow_noise


class DiagnosticPanelTests(unittest.TestCase):
    def test_average_before_winner_not_per_noise(self):
        y = np.array([[[1,0,1,0], [1,1,1,0], [0,0,0,1]]])
        c = frozen_choices(y, np.array([[3.,2.,1.]]), np.array([[0.,1.,2.]]))
        self.assertEqual(c['empirical_winner'].tolist(), [1])
        self.assertEqual(c['min_accel'].tolist(), [2])
        self.assertEqual(c['rgb_accel'].tolist(), [0])
        self.assertEqual(c['canonical'].tolist(), [0])

    def test_ties_prefer_catalog_order(self):
        y = np.ones((2,5,4)); x = np.zeros((2,5))
        for choice in frozen_choices(y,x,x).values():
            self.assertEqual(choice.tolist(), [0,0])

    def test_geometry_not_outcomes_and_balanced(self):
        views = [dict(view_id='canonical', role='reference', parameters=[0,0,1])]
        for i, p in enumerate(product((-30,-10,10,30),(-15,-5,5,15),(.8,1-1/15,1+1/15,1.2))):
            views.append(dict(view_id=f'train_{i:03}', role='train', parameters=p))
        rng = np.random.default_rng(41)
        for i, p in enumerate(rng.uniform([-30,-15,.8],[30,15,1.2],(32,3))):
            views.append(dict(view_id=f'extra_{i:03}', role='extra', parameters=p))
        panel = geometry_panel(views)
        self.assertEqual(len(set(panel)),13)
        for group in (panel[1:5],panel[5:9]):
            xyz = [v['parameters'] for v in views if v['view_id'] in group]
            np.testing.assert_allclose(np.mean(xyz,axis=0), [0,0,1])

    def test_new_execution_addresses_do_not_reuse_original(self):
        old = [flow_noise('initial', 'execution',i) for i in range(4)]
        for i in range(4,12):
            new = flow_noise('initial','execution',i)
            self.assertFalse(any(np.array_equal(x,new) for x in old))
            np.testing.assert_array_equal(new,flow_noise('initial','execution',i))


if __name__ == '__main__':
    unittest.main()
