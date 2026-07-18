from __future__ import annotations

import unittest

import torch

from pi05_policy_server import coupled_flow_noise


class CoupledInferenceTest(unittest.TestCase):
    def test_flow_noise_is_identical_across_observations(self) -> None:
        torch.manual_seed(17)
        noise = coupled_flow_noise(
            torch,
            batch_size=4,
            horizon=5,
            action_dim=7,
            device="cpu",
        )
        self.assertEqual(tuple(noise.shape), (4, 5, 7))
        for index in range(1, 4):
            torch.testing.assert_close(noise[0], noise[index], rtol=0, atol=0)

    def test_reseeding_reproduces_coupled_noise(self) -> None:
        torch.manual_seed(23)
        first = coupled_flow_noise(torch, batch_size=2, horizon=3, action_dim=2, device="cpu")
        torch.manual_seed(23)
        second = coupled_flow_noise(torch, batch_size=2, horizon=3, action_dim=2, device="cpu")
        torch.testing.assert_close(first, second, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
