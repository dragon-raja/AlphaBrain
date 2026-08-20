from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from accel_core import rank_accel_candidates, shared_flow_noise


def _validated_trace_output(
    output: Mapping[str, Any],
    supplied_noise: np.ndarray,
    *,
    candidate_count: int,
    action_horizon: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    if returned_noise.shape != supplied_noise.shape or not np.array_equal(
        returned_noise, supplied_noise
    ):
        raise ValueError("model did not preserve the supplied shared flow noise")
    velocity_trace = np.asarray(output["flow_velocity_trace"], dtype=np.float32)
    expected_trace_shape = (candidate_count, 10, action_horizon, action_dim)
    if velocity_trace.shape != expected_trace_shape or not np.all(
        np.isfinite(velocity_trace)
    ):
        raise ValueError(f"invalid 10-step velocity trace: {velocity_trace.shape}")
    flow_times = np.asarray(output["flow_times"], dtype=np.float32)
    if (
        flow_times.shape != (10,)
        or not np.all(np.isfinite(flow_times))
        or not np.all(np.diff(flow_times) < 0)
    ):
        raise ValueError("flow_times must contain 10 finite decreasing values")
    actions = np.asarray(output["normalized_actions"], dtype=np.float32)
    expected_actions_shape = (candidate_count, action_horizon, action_dim)
    if actions.shape != expected_actions_shape or not np.all(np.isfinite(actions)):
        raise ValueError(f"invalid normalized actions: {actions.shape}")
    return velocity_trace, flow_times, returned_noise, actions


def rank_fixed_state_candidates(
    model: Any,
    examples: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    *,
    seed: int,
    action_horizon: int,
    action_dim: int,
    include_trace_artifacts: bool = False,
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
    velocity_trace, flow_times, returned_noise, actions = _validated_trace_output(
        output,
        noise,
        candidate_count=len(examples),
        action_horizon=action_horizon,
        action_dim=action_dim,
    )
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
    if include_trace_artifacts:
        ranking["flow_velocity_trace"] = velocity_trace
        ranking["flow_initial_noise"] = returned_noise
    return ranking


def rank_fixed_state_candidates_chunked(
    model: Any,
    examples: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    *,
    seed: int,
    action_horizon: int,
    action_dim: int,
    batch_size: int,
    include_trace_artifacts: bool = False,
) -> dict[str, Any]:
    """Rank a large candidate bank with fixed-size, shared-noise batches.

    Every batch receives the exact same model-space ``x0``. The final batch is
    padded by repeating its final real example so every real candidate is
    evaluated with the same batch shape; padded outputs are discarded.
    """

    if len(examples) != len(candidate_ids) or not examples:
        raise ValueError(
            "examples and candidate_ids must have the same positive length"
        )
    if len(set(str(value) for value in candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_ids must be unique")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    traces = []
    actions = []
    flow_times_reference: np.ndarray | None = None
    batch_audits = []
    for start in range(0, len(examples), batch_size):
        stop = min(start + batch_size, len(examples))
        real_count = stop - start
        batch_examples = list(examples[start:stop])
        batch_examples.extend([batch_examples[-1]] * (batch_size - real_count))
        noise = shared_flow_noise(
            seed=seed,
            candidate_count=batch_size,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )
        output = model.predict_action(
            examples=batch_examples,
            noise=noise,
            return_flow_trace=True,
        )
        trace, flow_times, returned_noise, batch_actions = _validated_trace_output(
            output,
            noise,
            candidate_count=batch_size,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )
        if flow_times_reference is None:
            flow_times_reference = flow_times
        elif not np.array_equal(flow_times_reference, flow_times):
            raise ValueError("flow_times changed between candidate batches")
        traces.append(trace[:real_count])
        actions.append(batch_actions[:real_count])
        batch_audits.append(
            {
                "start": int(start),
                "stop": int(stop),
                "real_candidate_count": int(real_count),
                "padded_candidate_count": int(batch_size - real_count),
                "shared_flow_noise_exact": bool(
                    np.array_equal(
                        returned_noise,
                        np.repeat(returned_noise[0:1], batch_size, axis=0),
                    )
                ),
            }
        )

    velocity_trace = np.concatenate(traces, axis=0)
    normalized_actions = np.concatenate(actions, axis=0)
    one_noise = shared_flow_noise(
        seed=seed,
        candidate_count=len(examples),
        action_horizon=action_horizon,
        action_dim=action_dim,
    )
    ranking = rank_accel_candidates(
        candidate_ids,
        velocity_trace,
        initial_noise=one_noise,
    )
    ranking.update(
        {
            "seed": int(seed),
            "flow_times": [float(value) for value in flow_times_reference],
            "actions": normalized_actions.tolist(),
            "batch_size": int(batch_size),
            "batch_count": len(batch_audits),
            "fixed_batch_shape": True,
            "batch_audits": batch_audits,
        }
    )
    if include_trace_artifacts:
        ranking["flow_velocity_trace"] = velocity_trace
        ranking["flow_initial_noise"] = one_noise
    return ranking
