from tests.dsol_paper1.paths import ROOT as REPOSITORY_ROOT
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np

sys.path.insert(0, str(REPOSITORY_ROOT/'scripts/dsol_paper1'))
from scripts.dsol_paper1.trace_view_repeatability_v1 import physical_arrays
from scripts.dsol_paper1.analysis.analyze_view_repeatability_traces_v1 import difference


class RepeatabilityTraceTest(unittest.TestCase):
    def test_rgb_error_does_not_wrap_uint8(self):
        value = difference(np.array([0], dtype=np.uint8), np.array([255], dtype=np.uint8))
        self.assertEqual(value['max_abs'], 255)
        self.assertEqual(value['changed_elements'], 1)

    def test_empty_state_and_shape_mismatch(self):
        self.assertTrue(difference(np.array([]), np.array([]))['equal'])
        self.assertFalse(difference(np.ones(1), np.ones(2))['equal'])

    def test_signed_zero_distinguishes_byte_and_numeric_equality(self):
        value = difference(np.array([0.0]), np.array([-0.0]))
        self.assertFalse(value['equal'])
        self.assertTrue(value['numeric_equal'])
        self.assertEqual(value['max_abs'], 0)

    def test_physical_capture_copies_arrays_without_controller_update(self):
        state = np.arange(4, dtype=float)
        data = SimpleNamespace(qpos=state)
        controller = SimpleNamespace(goal_pos=np.ones(3))
        robot = SimpleNamespace(controller=controller, gripper=SimpleNamespace(current_action=np.array([1.])))
        env = SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(data=data), robots=[robot]), get_sim_state=lambda:state)
        captured = physical_arrays(env)
        captured['qpos'][0] = 99
        captured['controller_goal_pos'][0] = 9
        self.assertEqual(state[0], 0)
        self.assertEqual(controller.goal_pos[0], 1)


if __name__ == '__main__': unittest.main()
