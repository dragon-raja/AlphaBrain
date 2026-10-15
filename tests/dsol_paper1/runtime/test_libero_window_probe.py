import unittest
import numpy as np

from benchmarks.LIBERO.window_probe import later_noise


class WindowNoiseTest(unittest.TestCase):
    def test_reproducible_distinct_addresses(self):
        a = later_noise('initial-a', 0, 1)
        np.testing.assert_array_equal(a, later_noise('initial-a', 0, 1))
        self.assertEqual(a.shape, (10, 7)); self.assertEqual(a.dtype, np.float32)
        for args in [('initial-b', 0, 1), ('initial-a', 1, 1), ('initial-a', 0, 2)]:
            self.assertFalse(np.array_equal(a, later_noise(*args)))

    def test_first_window_cannot_use_new_noise(self):
        for repeat, window in [(0, 0), (0, 5), (8, 1), (-1, 1)]:
            with self.assertRaises(ValueError):
                later_noise('a', repeat, window)


if __name__ == '__main__':
    unittest.main()
