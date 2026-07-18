from __future__ import annotations

import unittest

import torch

from AlphaBrain.model.pi05_features import pool_pi05_image_views


class Pi05FeaturePoolingTest(unittest.TestCase):
    def test_view_pooling_does_not_mix_views_or_text(self) -> None:
        hidden = torch.zeros(1, 2 * 256 + 4, 3)
        hidden[:, :256] = 1
        hidden[:, 256:512] = 2
        hidden[:, 512:] = 99
        pooled = pool_pi05_image_views(hidden, num_views=2)
        self.assertEqual(tuple(pooled.shape), (1, 6))
        torch.testing.assert_close(pooled[0, :3], torch.ones(3))
        torch.testing.assert_close(pooled[0, 3:], torch.full((3,), 2.0))


if __name__ == "__main__":
    unittest.main()
