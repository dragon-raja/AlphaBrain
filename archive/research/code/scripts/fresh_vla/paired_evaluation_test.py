import unittest

import numpy as np

from paired_evaluation import EvaluationIdentity, bootstrap_summary, paired_delta_summary, per_sample_flow_metrics


class PairedEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = EvaluationIdentity(
            sample_ids=("a", "b"),
            flow_times=np.array([0.2, 0.8], dtype=np.float32),
            noise=np.arange(12, dtype=np.float32).reshape(2, 3, 2),
            action_normalization={"mean": [0.0, 1.0], "std": [2.0, 3.0]},
            sample_content=np.arange(8, dtype=np.float32).reshape(2, 4),
        )

    def test_identity_covers_samples_time_noise_and_normalization(self) -> None:
        fingerprint = self.identity.fingerprint()
        changed = EvaluationIdentity(
            self.identity.sample_ids,
            self.identity.flow_times,
            self.identity.noise.copy(),
            {"mean": [0.0, 1.0], "std": [2.0, 4.0]},
            self.identity.sample_content,
        )
        self.assertNotEqual(fingerprint, changed.fingerprint())

    def test_prefix_and_suffix_metrics_handle_empty_regions(self) -> None:
        rows = per_sample_flow_metrics(np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]), [0, 3], fixed_k=2)
        self.assertIsNone(rows[0]["oracle_prefix"])
        self.assertEqual(rows[0]["suffix"], 2.0)
        self.assertEqual(rows[1]["oracle_prefix"], 2.0)
        self.assertIsNone(rows[1]["suffix"])

    def test_paired_delta_requires_identical_identity_and_samples(self) -> None:
        fingerprint = self.identity.fingerprint()
        baseline = [
            {"sample_id": "a", "evaluation_fingerprint": fingerprint, "fixed_k": 2.0},
            {"sample_id": "b", "evaluation_fingerprint": fingerprint, "fixed_k": 4.0},
        ]
        candidate = [
            {"sample_id": "a", "evaluation_fingerprint": fingerprint, "fixed_k": 1.0},
            {"sample_id": "b", "evaluation_fingerprint": fingerprint, "fixed_k": 3.0},
        ]
        result = paired_delta_summary(
            baseline,
            candidate,
            metric="fixed_k",
            expected_fingerprint=fingerprint,
            bootstrap_samples=200,
        )
        self.assertEqual(result["candidate_better"], 2)
        self.assertEqual(result["candidate_minus_baseline"]["mean"], -1.0)
        candidate[0] = {**candidate[0], "evaluation_fingerprint": "wrong"}
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            paired_delta_summary(baseline, candidate, metric="fixed_k", expected_fingerprint=fingerprint)

    def test_bootstrap_summary_reports_mean_median_se_and_ci(self) -> None:
        summary = bootstrap_summary([1.0, 2.0, 3.0], bootstrap_samples=500, seed=4)
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["median"], 2.0)
        self.assertGreater(summary["standard_error"], 0.0)
        self.assertLessEqual(summary["bootstrap_95_low"], summary["mean"])
        self.assertGreaterEqual(summary["bootstrap_95_high"], summary["mean"])


if __name__ == "__main__":
    unittest.main()
