from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from accel_core import rank_accel_candidates, shared_flow_noise


def rank_fixed_state_candidates(
    model: Any,
    examples: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    *,
    seed: int,
    action_horizon: int,
    action_dim: int,
) -> dict[str, Any]:
    """Run one fixed-state candidate batch with an exactly shared model-space x0."""

    if len(examples) != len(candidate_ids) or not examples:
        raise ValueError(
            "examples and candidate_ids must have the same positive length"
        )
    noise = shared_flow_noise(
        seed=seed,
        candidate_count=len(examples),
        action_horizon=action_horizon,
        action_dim=action_dim,
    )
    output = model.predict_action(
        examples=list(examples),
        noise=noise,
        return_flow_trace=True,
    )
    required = {
        "normalized_actions",
        "flow_velocity_trace",
        "flow_times",
        "flow_initial_noise",
        "flow_trace_coordinate_system",
    }
    missing = required - set(output)
    if missing:
        raise KeyError(f"model flow-trace output is missing {sorted(missing)}")
    if output["flow_trace_coordinate_system"] != "normalized_action":
        raise ValueError("Accel requires a normalized-action velocity trace")

    returned_noise = np.asarray(output["flow_initial_noise"], dtype=np.float32)
    if returned_noise.shape != noise.shape or not np.array_equal(returned_noise, noise):
        raise ValueError("model did not preserve the supplied shared flow noise")
    velocity_trace = np.asarray(output["flow_velocity_trace"], dtype=np.float32)
    expected_trace_shape = (
        len(examples),
        10,
        action_horizon,
        action_dim,
    )
    if velocity_trace.shape != expected_trace_shape or not np.all(
        np.isfinite(velocity_trace)
    ):
        raise ValueError(
            f"invalid 10-step velocity trace: {velocity_trace.shape}"
        )
    flow_times = np.asarray(output["flow_times"], dtype=np.float32)
    if (
        flow_times.shape != (10,)
        or not np.all(np.isfinite(flow_times))
        or not np.all(np.diff(flow_times) < 0)
    ):
        raise ValueError("flow_times must contain 10 finite decreasing values")
    actions = np.asarray(output["normalized_actions"], dtype=np.float32)
    expected_actions_shape = (len(examples), action_horizon, action_dim)
    if actions.shape != expected_actions_shape or not np.all(np.isfinite(actions)):
        raise ValueError(f"invalid normalized actions: {actions.shape}")
    ranking = rank_accel_candidates(
        candidate_ids,
        velocity_trace,
        initial_noise=returned_noise,
    )
    ranking.update(
        {
            "seed": int(seed),
            "flow_times": [float(value) for value in flow_times],
            "actions": actions.tolist(),
        }
    )
    return ranking
