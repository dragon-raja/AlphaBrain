import unittest

import numpy as np

from scripts.branch_vla.gate0_common import (
    block_average,
    fit_ridge,
    grouped_folds,
    masked_mse,
    observation_feature,
)


class Gate0CommonTest(unittest.TestCase):
    def test_block_average_and_feature(self):
        image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        pooled = block_average(image, 2)
        self.assertEqual(pooled.shape, (12,))
        feature = observation_feature(image, image, np.arange(8), lead=2, image_size=2)
        self.assertEqual(feature.shape, (12 + 12 + 8 + 5,))
        self.assertEqual(feature[-4], 1.0)

    def test_ridge_recovers_linear_map(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(100, 4))
        targets = features @ np.asarray([[1.0], [-2.0], [0.5], [3.0]]) + 0.25
        model = fit_ridge(features, targets, 1e-6)
        np.testing.assert_allclose(model.predict(features), targets, atol=1e-5)

    def test_grouped_folds_do_not_split_sources(self):
        sources = np.repeat(np.arange(10), 3)
        folds = grouped_folds(sources, 5)
        self.assertEqual(sum(mask.astype(int) for mask in folds).min(), 1)
        self.assertEqual(sum(mask.astype(int) for mask in folds).max(), 1)
        for source in range(10):
            memberships = [np.unique(mask[sources == source]).tolist() for mask in folds]
            self.assertEqual(sum(values == [True] for values in memberships), 1)

    def test_masked_mse_uses_suffix(self):
        target = np.zeros((4, 2))
        prediction = np.ones((4, 2))
        prediction[:2] = 10.0
        self.assertEqual(masked_mse(prediction, target, 2), 1.0)


if __name__ == "__main__":
    unittest.main()
