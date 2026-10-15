import unittest
import numpy as np
from AlphaBrain.research.dsol.selectors.query_ranker import fit_ranker, development_fit


class QueryRankerTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(71)
        images, wrists = rng.normal(size=(8, 6, 12)), rng.normal(size=(8, 10))
        a = rng.random((8, 6, 8))
        y = .5 + .1 * images[:, :, 0]
        return images, wrists, np.arange(8), a, y

    def test_single_query_matches_batch_no_canonical_context(self):
        im, w, t, a, y = self.data()
        model = fit_ranker(im, w, t, a, y, use_accel=True, alpha=1., image_dim=3, wrist_dim=2)
        batch = model.score(im[0], w[0], t[0], a[0])
        for i in range(6):
            self.assertAlmostEqual(float(model.score(im[0, i], w[0], t[0], a[0, i])), float(batch[i]))

    def test_absolute_task_offset_does_not_change_ranker(self):
        im, w, t, a, y = self.data()
        kw = dict(use_accel=False, alpha=1., image_dim=3, wrist_dim=2)
        first = fit_ranker(im, w, t, a, y, **kw)
        second = fit_ranker(im, w, t, a, y + np.arange(8)[:, None], **kw)
        np.testing.assert_allclose(first.weights, second.weights, atol=1e-10)

    def test_heldout_initials_rejected_at_fit_boundary(self):
        im, w, t, a, y = self.data()
        with self.assertRaises(ValueError):
            development_fit(im, w, t, a, y, np.full(8, 2), {}, use_accel=False)


if __name__ == '__main__':
    unittest.main()
