from types import SimpleNamespace
import unittest

import numpy as np

from benchmarks.Robocasa365.openpi_training import audit_batch


class RealTrainingBatchTests(unittest.TestCase):
    def batch(self):
        obs = SimpleNamespace(state=np.zeros((8, 32)), tokenized_prompt=np.zeros((8, 200), dtype=int),
            tokenized_prompt_mask=np.ones((8, 200), dtype=bool), images={key: np.zeros((8, 224, 224, 3), dtype=np.uint8)
            for key in ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')})
        return obs, np.zeros((8, 50, 32))

    def test_canonical_and_broad(self):
        obs, actions = self.batch()
        self.assertEqual(audit_batch(obs, actions, 'canonical_matched')['status'], 'PASS')
        with self.assertRaises(ValueError):
            audit_batch(obs, actions, 'broad_exterior_matched')
        obs.images['base_0_rgb'][1] = 1
        self.assertEqual(audit_batch(obs, actions, 'broad_exterior_matched')['status'], 'PASS')

    def test_wrong_actions_and_wrist_are_rejected(self):
        obs, actions = self.batch()
        actions[1, 0, 0] = 1
        with self.assertRaises(ValueError):
            audit_batch(obs, actions, 'canonical_matched')
        actions[1, 0, 0] = 0
        obs.images['left_wrist_0_rgb'][1] = 1
        with self.assertRaises(ValueError):
            audit_batch(obs, actions, 'broad_exterior_matched')
