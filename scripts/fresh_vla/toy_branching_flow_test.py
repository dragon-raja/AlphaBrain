import unittest

import torch

from toy_branching_flow import ToyConfig, TinyBranchingFlow, evaluate, flow_loss, make_generator, sample_batch


class ToyBranchingFlowTest(unittest.TestCase):
    def test_deterministic_control_has_full_oracle_horizon(self) -> None:
        device = torch.device("cpu")
        cfg = ToyConfig(branch_strength=0.0)
        _, _, _, horizons = sample_batch(32, cfg, device, generator=make_generator(device, 4))
        self.assertTrue(torch.equal(horizons, torch.full_like(horizons, cfg.horizon)))

    def test_oracle_soft_is_exact_full_loss_when_horizon_is_full(self) -> None:
        device = torch.device("cpu")
        cfg = ToyConfig(branch_strength=0.0)
        torch.manual_seed(2)
        model = TinyBranchingFlow(cfg)
        context, actions, _, horizons = sample_batch(8, cfg, device, generator=make_generator(device, 7))
        full = flow_loss(
            model,
            context,
            actions,
            horizons,
            mask_mode="oracle",
            tail_weight=1.0,
            generator=make_generator(device, 9),
        )
        oracle = flow_loss(
            model,
            context,
            actions,
            horizons,
            mask_mode="oracle",
            tail_weight=0.1,
            generator=make_generator(device, 9),
        )
        self.assertTrue(torch.equal(full, oracle))

    def test_deterministic_evaluation_marks_empty_suffix(self) -> None:
        torch.manual_seed(3)
        model = TinyBranchingFlow(ToyConfig(branch_strength=0.0))
        metrics, per_sample, _, _ = evaluate(
            model,
            model.cfg,
            seed=41,
            evaluation_size=8,
            multimodal_contexts=2,
            samples_per_context=2,
        )
        self.assertEqual(metrics["flow_suffix_available"], 0.0)
        self.assertEqual(metrics["branch_min_suffix_mse"], 0.0)
        self.assertEqual(per_sample["flow_suffix_mse"], [0.0] * 8)


if __name__ == "__main__":
    unittest.main()
