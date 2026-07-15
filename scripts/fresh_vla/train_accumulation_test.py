from __future__ import annotations

from types import SimpleNamespace
import unittest

from accelerate import Accelerator
import torch

from AlphaBrain.training.train_alphabrain import VLATrainer


class _TinyFlowModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, target: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"action_loss": (self.weight - target).square()}


class TrainAccumulationTest(unittest.TestCase):
    def test_non_deepspeed_updates_only_on_accumulation_boundary(self) -> None:
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        model = _TinyFlowModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        model, optimizer = accelerator.prepare(model, optimizer)
        cfg = SimpleNamespace(
            trainer=SimpleNamespace(gradient_clipping=0.0, ema=None),
            datasets=SimpleNamespace(vla_data=SimpleNamespace(per_device_batch_size=1)),
        )
        trainer = VLATrainer(cfg, model, [], optimizer, scheduler, accelerator)
        trainer.optimizer.zero_grad()

        before = float(accelerator.unwrap_model(model).weight.detach())
        scheduler_before = scheduler.last_epoch
        with trainer._accumulation_context():
            trainer._train_step(torch.tensor(1.0))
            first_boundary = accelerator.sync_gradients
        after_first = float(accelerator.unwrap_model(model).weight.detach())
        scheduler_after_first = scheduler.last_epoch

        with trainer._accumulation_context():
            trainer._train_step(torch.tensor(1.0))
            second_boundary = accelerator.sync_gradients
        after_second = float(accelerator.unwrap_model(model).weight.detach())
        scheduler_after_second = scheduler.last_epoch

        self.assertFalse(first_boundary)
        self.assertEqual(after_first, before)
        self.assertEqual(scheduler_after_first, scheduler_before)
        self.assertTrue(second_boundary)
        self.assertGreater(after_second, after_first)
        self.assertEqual(scheduler_after_second, scheduler_before + 1)


if __name__ == "__main__":
    unittest.main()
