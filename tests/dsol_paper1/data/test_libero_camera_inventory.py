import unittest
from AlphaBrain.research.dsol.data.libero_camera_inventory import parse_camera_task


class CameraInventoryTests(unittest.TestCase):
    def test_wrapper_compatible_parameters(self):
        row=parse_camera_task('pick_up_object_view_11_15_90_2_352_initstate_0')
        self.assertEqual(row['base_task'],'pick_up_object')
        self.assertEqual(row['distance_scale'],.9)
        self.assertEqual(row['endpoint_pitch'],352)
        self.assertEqual(row['robot_initial_variant'],0)

    def test_reject_mixed_noise(self):
        with self.assertRaises(ValueError):
            parse_camera_task('task_view_0_0_100_0_0_initstate_0_noise_2')
