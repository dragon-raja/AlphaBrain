import unittest
import numpy as np

from benchmarks.LIBERO.plus_evaluate import verify_example, policy_example


class PlusInputTest(unittest.TestCase):
    def sample(self):
        example = {'observation/image': np.zeros((2, 2, 3), np.uint8),
                   'observation/wrist_image': np.ones((2, 2, 3), np.uint8),
                   'observation/state': np.array([.125], np.float64),
                   'camera_intrinsics': np.eye(3), 'camera_to_world_opencv': np.eye(4)}
        mapping = {'external_images': 'observation/image', 'wrist_images': 'observation/wrist_image',
                   'robot_states': 'observation/state', 'camera_intrinsics': 'camera_intrinsics',
                   'camera_to_world_opencv': 'camera_to_world_opencv'}
        cache = {k: np.asarray([example[v]]) for k, v in mapping.items()}
        cache['robot_states'] = cache['robot_states'].astype(np.float32)
        return example, cache

    def test_same_policy_normalization(self):
        example, cache = self.sample()
        verify_example(example, cache, 0)
        self.assertEqual(policy_example(example, 'test')['state'].dtype, np.float32)

    def test_changed_rgb_not_tolerated(self):
        example, cache = self.sample()
        example['observation/image'][0, 0, 0] = 1
        with self.assertRaises(ValueError):
            verify_example(example, cache, 0)

    def test_changed_camera_not_tolerated(self):
        example, cache = self.sample()
        example['camera_to_world_opencv'][0, 3] = .01
        with self.assertRaises(ValueError):
            verify_example(example, cache, 0)
