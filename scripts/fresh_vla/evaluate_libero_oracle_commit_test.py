import unittest

import numpy as np
from evaluate_libero_oracle_commit import (
    policy_sample_seeds,
    sample_policy_chunks,
    should_interrupt_for_runtime_event,
)


class FakePolicy:
    def __init__(self):
        self.seeds = []

    def predict_many(self, observation, seeds):
        self.seeds.extend(seeds)
        return np.stack([np.full((10, 7), seed % 7, dtype=np.float32) for seed in seeds]), 0.25


class OracleCommitEvaluatorTest(unittest.TestCase):
    def test_n8_seed_schedule_is_prefix_of_n16(self):
        n8 = policy_sample_seeds(41, "g0", 2, 8)
        n16 = policy_sample_seeds(41, "g0", 2, 16)
        self.assertEqual(n8, n16[:8])
        self.assertEqual(len(set(n16)), 16)
        reach = policy_sample_seeds(41, "g0", 2, 8, namespace="reach")
        self.assertNotEqual(n8, reach)

    def test_sample_zero_is_anchor_and_server_time_is_preserved(self):
        policy = FakePolicy()
        chunks, client_time, server_time = sample_policy_chunks(
            policy,
            {},
            noise_seed=41,
            pair_id="g0",
            replan_count=0,
            sample_count=8,
        )
        self.assertEqual(chunks.shape, (8, 10, 7))
        self.assertTrue(np.all(chunks[0] == policy.seeds[0] % 7))
        self.assertGreaterEqual(client_time, 0.0)
        self.assertEqual(server_time, 0.25)

    def test_only_oracles_interrupt_at_runtime_event(self):
        self.assertTrue(should_interrupt_for_runtime_event("oracle_branch_safe_commit", "end_to_end", True))
        self.assertTrue(should_interrupt_for_runtime_event("oracle_feedback_reveal_commit", "end_to_end", True))
        self.assertFalse(should_interrupt_for_runtime_event("fixed_k3", "end_to_end", True))
        self.assertFalse(should_interrupt_for_runtime_event("oracle_branch_safe_commit", "isolated", True))


if __name__ == "__main__":
    unittest.main()
