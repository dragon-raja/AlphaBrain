import unittest
import numpy as np
from AlphaBrain.research.dsol.analysis.cross_benchmark_preparation import describe_task


class PreparationTests(unittest.TestCase):
    def test_group_does_not_depend_on_selector_success(self):
        y=np.array([[[1,1],[0,0],[1,1]],[[1,1],[0,0],[1,1]]])
        low_bad=np.array([[1,0,2],[1,0,2]])
        low_good=np.array([[1,2,0],[1,2,0]])
        first=describe_task(y,low_bad,[1,2]);second=describe_task(y,low_good,[1,2])
        self.assertEqual(first['descriptive_group'],'capable_view_sensitive')
        self.assertEqual(first['descriptive_group'],second['descriptive_group'])
        self.assertEqual(first['initial_accel_selected'],0)
        self.assertEqual(second['initial_accel_selected'],1)
