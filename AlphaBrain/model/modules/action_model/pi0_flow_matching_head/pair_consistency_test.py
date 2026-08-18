import unittest

import torch

from AlphaBrain.model.modules.action_model.pi0_flow_matching_head.pair_consistency import (
    dsol_pair_groups,
    paired_prediction_consistency,
    share_pair_noise_time,
    validate_paired_actions,
)


class PairConsistencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = [
            {
                "dsol_pair_id": "state-1",
                "dsol_pair_role": "pair_a",
                "dsol_pair_objective": True,
            },
            {"dsol_pair_objective": False},
            {
                "dsol_pair_id": "state-1",
                "dsol_pair_role": "pair_b",
                "dsol_pair_objective": True,
            },
        ]

    def test_groups_only_explicit_objective_samples(self) -> None:
        self.assertEqual(dsol_pair_groups(self.examples), [(0, 2)])

    def test_groups_support_separate_shared_flow_marker(self) -> None:
        examples = [dict(example) for example in self.examples]
        for example in examples:
            example["dsol_pair_objective"] = False
            example["dsol_pair_shared_flow"] = bool(example.get("dsol_pair_id"))
        self.assertEqual(
            dsol_pair_groups(examples, marker_key="dsol_pair_shared_flow"),
            [(0, 2)],
        )

    def test_incomplete_pair_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly twice"):
            dsol_pair_groups(self.examples[:2])

    def test_duplicate_roles_are_rejected(self) -> None:
        examples = [dict(self.examples[0]), dict(self.examples[2])]
        examples[1]["dsol_pair_role"] = "pair_a"
        with self.assertRaisesRegex(ValueError, "distinct roles"):
            dsol_pair_groups(examples)

    def test_shared_noise_and_time_only_change_pair_partner(self) -> None:
        noise = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
        time = torch.tensor([0.1, 0.2, 0.3])
        shared_noise, shared_time = share_pair_noise_time(noise, time, [(0, 2)])
        self.assertTrue(torch.equal(shared_noise[0], shared_noise[2]))
        self.assertTrue(torch.equal(shared_time[0], shared_time[2]))
        self.assertTrue(torch.equal(shared_noise[1], noise[1]))
        self.assertEqual(shared_time[1].item(), time[1].item())

    def test_action_mismatch_is_rejected(self) -> None:
        actions = torch.zeros(3, 2, 4)
        validate_paired_actions(actions, [(0, 2)])
        actions[2, 1, 3] = 0.1
        with self.assertRaisesRegex(ValueError, "mismatched normalized actions"):
            validate_paired_actions(actions, [(0, 2)])

    def test_consistency_respects_action_mask(self) -> None:
        prediction = torch.zeros(3, 2, 4)
        prediction[2, :, 0] = 2.0
        prediction[2, :, 3] = 100.0
        mse, rmse = paired_prediction_consistency(
            prediction,
            [(0, 2)],
            action_dim_mask=torch.tensor([True, False, False, False]),
        )
        self.assertTrue(torch.equal(mse, torch.tensor(4.0)))
        self.assertTrue(torch.equal(rmse, torch.tensor(2.0)))


if __name__ == "__main__":
    unittest.main()
