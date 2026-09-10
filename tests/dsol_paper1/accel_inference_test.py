from __future__ import annotations

import unittest

import numpy as np

from accel_inference import (
    rank_fixed_state_candidates,
    rank_fixed_state_candidates_chunked,
)


class FakeTraceModel:
    def __init__(self) -> None:
        self.received_noise = None

    def predict_action(self, *, examples, noise, return_flow_trace):
        self.received_noise = np.asarray(noise)
        if not return_flow_trace:
            raise AssertionError("flow trace was not requested")
        count, horizon, action_dim = self.received_noise.shape
        velocity = np.ones((count, 10, horizon, action_dim), dtype=np.float32)
        velocity[1, :, :, :] = np.arange(1, 11, dtype=np.float32)[:, None, None]
        return {
            "normalized_actions": np.zeros(
                (count, horizon, action_dim), dtype=np.float32
            ).tolist(),
            "flow_velocity_trace": velocity.tolist(),
            "flow_times": np.linspace(1.0, 0.1, 10, dtype=np.float32).tolist(),
            "flow_initial_noise": self.received_noise.tolist(),
            "flow_trace_coordinate_system": "normalized_action",
        }


class AccelInferenceTest(unittest.TestCase):
    def test_fixed_state_batch_uses_one_shared_x0_and_ranks_candidates(self) -> None:
        model = FakeTraceModel()
        result = rank_fixed_state_candidates(
            model,
            [{"view": "canonical"}, {"view": "reveal"}],
            ["canonical", "reveal"],
            seed=42,
            action_horizon=4,
            action_dim=2,
        )
        self.assertTrue(
            np.array_equal(model.received_noise[0], model.received_noise[1])
        )
        self.assertEqual(result["selected_candidate_id"], "canonical")
        self.assertTrue(result["shared_flow_noise_audit"]["exactly_shared"])
        self.assertEqual(len(result["flow_times"]), 10)

    def test_rejects_model_that_does_not_return_supplied_x0(self) -> None:
        class BadNoiseModel(FakeTraceModel):
            def predict_action(self, **kwargs):
                output = super().predict_action(**kwargs)
                output["flow_initial_noise"][1][0][0] += 1.0
                return output

        with self.assertRaisesRegex(
            ValueError, "preserve the supplied shared flow noise"
        ):
            rank_fixed_state_candidates(
                BadNoiseModel(),
                [{"view": "a"}, {"view": "b"}],
                ["a", "b"],
                seed=43,
                action_horizon=2,
                action_dim=1,
            )

    def test_rejects_non_ten_step_trace(self) -> None:
        class ShortTraceModel(FakeTraceModel):
            def predict_action(self, **kwargs):
                output = super().predict_action(**kwargs)
                output["flow_velocity_trace"] = [
                    candidate[:9] for candidate in output["flow_velocity_trace"]
                ]
                output["flow_times"] = output["flow_times"][:9]
                return output

        with self.assertRaisesRegex(ValueError, "invalid 10-step velocity trace"):
            rank_fixed_state_candidates(
                ShortTraceModel(),
                [{"view": "a"}, {"view": "b"}],
                ["a", "b"],
                seed=44,
                action_horizon=2,
                action_dim=1,
            )

    def test_chunked_ranking_uses_fixed_batch_shape_and_one_x0(self) -> None:
        class BatchedModel(FakeTraceModel):
            def __init__(self) -> None:
                super().__init__()
                self.batch_shapes = []
                self.noises = []

            def predict_action(self, **kwargs):
                output = super().predict_action(**kwargs)
                self.batch_shapes.append(len(kwargs["examples"]))
                self.noises.append(np.asarray(kwargs["noise"]).copy())
                return output

        model = BatchedModel()
        examples = [{"view": f"v{index}"} for index in range(5)]
        result = rank_fixed_state_candidates_chunked(
            model,
            examples,
            [f"v{index}" for index in range(5)],
            seed=45,
            action_horizon=3,
            action_dim=2,
            batch_size=2,
            include_trace_artifacts=True,
        )
        self.assertEqual(model.batch_shapes, [2, 2, 2])
        self.assertTrue(all(np.array_equal(noise[0], noise[1]) for noise in model.noises))
        self.assertTrue(
            all(np.array_equal(model.noises[0], noise) for noise in model.noises[1:])
        )
        self.assertEqual(result["candidate_count"], 5)
        self.assertEqual(result["batch_count"], 3)
        self.assertEqual(result["batch_audits"][-1]["padded_candidate_count"], 1)
        self.assertEqual(result["flow_velocity_trace"].shape, (5, 10, 3, 2))
        self.assertTrue(result["shared_flow_noise_audit"]["exactly_shared"])

    def test_chunked_ranking_rejects_duplicate_candidate_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_ids must be unique"):
            rank_fixed_state_candidates_chunked(
                FakeTraceModel(),
                [{"view": "a"}, {"view": "b"}],
                ["same", "same"],
                seed=46,
                action_horizon=2,
                action_dim=1,
                batch_size=2,
            )


if __name__ == "__main__":
    unittest.main()
