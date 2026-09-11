import unittest

import numpy as np

from probe_feedback_observability import accuracy, block_average, ridge_scores


class FeedbackObservabilityProbeTest(unittest.TestCase):
    def test_block_average_preserves_channel_means(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        pooled = block_average(image, 2).reshape(2, 2, 3)
        np.testing.assert_allclose(pooled[:, :, 0], 1.0)
        np.testing.assert_allclose(pooled[:, :, 1:], 0.0)

    def test_ridge_classifier_separates_simple_groups(self):
        train = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        labels = np.asarray([-1.0, -1.0, 1.0, 1.0])
        query = np.asarray([[-3.0], [3.0]])
        scores = ridge_scores(train, labels, query, 0.1)
        self.assertEqual(accuracy(np.asarray([-1.0, 1.0]), scores), 1.0)


if __name__ == "__main__":
    unittest.main()
