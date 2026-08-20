from __future__ import annotations

import unittest

import numpy as np

from accel_inference import rank_fixed_state_candidates


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


if __name__ == "__main__":
    unittest.main()
