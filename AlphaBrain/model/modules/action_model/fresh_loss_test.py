import unittest

import torch

from AlphaBrain.model.modules.action_model.fresh_loss import feedback_weighted_flow_loss


class FreshLossTest(unittest.TestCase):
    def test_tail_weight_one_is_exact_baseline(self) -> None:
        generator = torch.Generator().manual_seed(7)
        per_dim = torch.randn(3, 7, 5, generator=generator)
        loss, _ = feedback_weighted_flow_loss(per_dim, torch.tensor([2, 5]), tail_weight=1.0)
        self.assertTrue(torch.equal(loss, per_dim.mean()))

    def test_hard_mask_normalizes_each_sample(self) -> None:
        per_dim = torch.tensor([[[1.0], [3.0], [100.0]], [[2.0], [4.0], [6.0]]])
        loss, metrics = feedback_weighted_flow_loss(per_dim, torch.tensor([2, 1]), tail_weight=0.0)
        expected = torch.tensor([2.0, 2.0]).mean()
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.allclose(metrics["mean_feedback_horizon"], torch.tensor(1.5)))

    def test_soft_tail_weight(self) -> None:
        per_dim = torch.tensor([[[1.0], [3.0], [5.0]]])
        loss, _ = feedback_weighted_flow_loss(per_dim, torch.tensor([1]), tail_weight=0.25)
        expected = torch.tensor((1.0 + 0.25 * 3.0 + 0.25 * 5.0) / 1.5)
        self.assertTrue(torch.allclose(loss, expected))

    def test_missing_horizon_rejected_for_fresh(self) -> None:
        with self.assertRaisesRegex(ValueError, "feedback_horizon"):
            feedback_weighted_flow_loss(torch.zeros(2, 4, 3), None, tail_weight=0.1)


if __name__ == "__main__":
    unittest.main()
