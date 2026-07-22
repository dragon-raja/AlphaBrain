from __future__ import annotations

import unittest

import numpy as np

from evaluate_libero_bind_offline import action_transport_metrics


class ActionTransportMetricsTest(unittest.TestCase):
    def test_perfect_additive_fourth_corner_has_zero_transport_error(self) -> None:
        base = np.asarray([[0.1, 0.2]], np.float32)
        source = np.asarray([[0.3, 0.2]], np.float32)
        target = np.asarray([[0.1, -0.4]], np.float32)
        fourth = source + target - base
        values = action_transport_metrics(
            {
                "base": base,
                "source_anchor": source,
                "target_anchor": target,
                "fourth_anchor": fourth,
            },
            {"base": base, "source_anchor": source, "target_anchor": target},
        )
        self.assertAlmostEqual(values["observed_corner_mse"], 0.0)
        self.assertAlmostEqual(values["action_free_pseudo_mse"], 0.0)
        self.assertAlmostEqual(values["model_self_closure_mse"], 0.0)
        self.assertAlmostEqual(values["target_effect_transfer_mse"], 0.0)
        self.assertAlmostEqual(values["source_effect_transfer_mse"], 0.0)
        self.assertAlmostEqual(values["target_effect_cosine"], 1.0)
        self.assertAlmostEqual(values["source_effect_cosine"], 1.0)

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            action_transport_metrics(
                {
                    "base": np.zeros((2, 2)),
                    "source_anchor": np.zeros((2, 2)),
                    "target_anchor": np.zeros((2, 2)),
                    "fourth_anchor": np.zeros((2, 3)),
                },
                {
                    "base": np.zeros((2, 2)),
                    "source_anchor": np.zeros((2, 2)),
                    "target_anchor": np.zeros((2, 2)),
                },
            )


if __name__ == "__main__":
    unittest.main()
