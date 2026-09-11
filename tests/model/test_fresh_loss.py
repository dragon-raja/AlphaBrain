import unittest

import torch

from AlphaBrain.model.modules.action_model.fresh_loss import feedback_prefix_weights, feedback_weighted_flow_loss


class FreshLossTest(unittest.TestCase):
    def test_tail_weight_one_is_exact_baseline(self) -> None:
        generator = torch.Generator().manual_seed(7)
        per_dim = torch.randn(3, 7, 5, generator=generator)
        loss, metrics = feedback_weighted_flow_loss(per_dim, torch.tensor([0, 3, 7]), tail_weight=1.0)
        self.assertTrue(torch.equal(loss, per_dim.mean()))
        per_step = per_dim.mean(dim=-1)
        prefix = torch.arange(7)[None, :] < torch.tensor([0, 3, 7])[:, None]
        self.assertTrue(torch.allclose(metrics["prefix_loss"], per_step[prefix].mean()))

    def test_step_metrics_cover_the_horizon_and_fractions_sum_to_one(self) -> None:
        per_dim = torch.tensor([[[1.0], [3.0], [6.0]], [[3.0], [1.0], [6.0]]])
        _, metrics = feedback_weighted_flow_loss(per_dim, torch.tensor([1, 2]), tail_weight=1.0)
        self.assertEqual(metrics["step_loss_00"].item(), 2.0)
        self.assertEqual(metrics["step_loss_01"].item(), 2.0)
        self.assertEqual(metrics["step_loss_02"].item(), 6.0)
        fractions = sum(metrics[f"step_loss_fraction_{step:02d}"] for step in range(3))
        self.assertTrue(torch.allclose(fractions, torch.tensor(1.0)))

    def test_horizon_semantics_are_zero_one_and_full(self) -> None:
        weights, prefix = feedback_prefix_weights(
            torch.tensor([0, 1, 4]),
            action_horizon=4,
            tail_weight=0.0,
            dtype=torch.float32,
            device=torch.device("cpu"),
            batch_size=3,
        )
        expected = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
        )
        self.assertTrue(torch.equal(weights, expected))
        self.assertTrue(torch.equal(prefix, expected.bool()))

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

    def test_mixed_horizons_normalize_each_sample_by_weight_sum(self) -> None:
        per_dim = torch.tensor([[[2.0], [4.0]], [[10.0], [20.0]]])
        loss, _ = feedback_weighted_flow_loss(per_dim, torch.tensor([1, 2]), tail_weight=0.5)
        expected = torch.tensor([(2.0 + 0.5 * 4.0) / 1.5, 15.0]).mean()
        self.assertTrue(torch.allclose(loss, expected))

    def test_prefix_control_downweights_the_same_number_of_front_steps(self) -> None:
        per_dim = torch.tensor([[[10.0], [20.0], [3.0], [5.0]]])
        loss, _ = feedback_weighted_flow_loss(
            per_dim,
            torch.tensor([2]),
            tail_weight=0.1,
            weighting_mode="prefix_control",
        )
        expected = torch.tensor((0.1 * 10.0 + 0.1 * 20.0 + 3.0 + 5.0) / 2.2)
        self.assertTrue(torch.allclose(loss, expected))

    def test_unknown_weighting_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "weighting_mode"):
            feedback_weighted_flow_loss(
                torch.zeros(1, 2, 1),
                torch.tensor([1]),
                tail_weight=0.1,
                weighting_mode="future_magic",
            )

    def test_empty_prefix_hard_mask_is_excluded_from_batch_mean(self) -> None:
        per_dim = torch.tensor([[[100.0], [100.0]], [[2.0], [4.0]]], requires_grad=True)
        loss, metrics = feedback_weighted_flow_loss(per_dim, torch.tensor([0, 2]), tail_weight=0.0)
        self.assertTrue(torch.equal(loss, torch.tensor(3.0)))
        self.assertTrue(torch.equal(metrics["prefix_loss"], torch.tensor(3.0)))
        loss.backward()
        self.assertTrue(torch.equal(per_dim.grad[0], torch.zeros_like(per_dim.grad[0])))

    def test_empty_suffix_at_full_horizon_has_finite_metrics(self) -> None:
        per_dim = torch.ones(1, 3, 2)
        loss, metrics = feedback_weighted_flow_loss(per_dim, torch.tensor([3]), tail_weight=0.0)
        self.assertTrue(torch.equal(loss, torch.tensor(1.0)))
        self.assertTrue(torch.equal(metrics["suffix_loss"], torch.tensor(0.0)))

    def test_all_empty_hard_mask_returns_differentiable_zero(self) -> None:
        per_dim = torch.ones(2, 3, 1, requires_grad=True)
        loss, _ = feedback_weighted_flow_loss(per_dim, torch.tensor([0, 0]), tail_weight=0.0)
        self.assertTrue(torch.equal(loss, torch.tensor(0.0)))
        loss.backward()
        self.assertTrue(torch.equal(per_dim.grad, torch.zeros_like(per_dim)))

    def test_invalid_horizons_are_rejected_instead_of_clamped(self) -> None:
        per_dim = torch.zeros(2, 4, 3)
        for horizons in (torch.tensor([-1, 2]), torch.tensor([2, 5]), torch.tensor([1.5, 2.0])):
            with self.subTest(horizons=horizons.tolist()):
                with self.assertRaisesRegex(ValueError, "feedback_horizon"):
                    feedback_weighted_flow_loss(per_dim, horizons, tail_weight=0.1)

    def test_horizon_batch_size_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 2 feedback horizons"):
            feedback_weighted_flow_loss(torch.zeros(2, 4, 3), torch.tensor([2]), tail_weight=0.1)

    def test_missing_horizon_rejected_for_fresh(self) -> None:
        with self.assertRaisesRegex(ValueError, "feedback_horizon"):
            feedback_weighted_flow_loss(torch.zeros(2, 4, 3), None, tail_weight=0.1)


if __name__ == "__main__":
    unittest.main()
