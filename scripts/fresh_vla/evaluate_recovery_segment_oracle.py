from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_libero_closed_loop import (
    Pi05Policy,
    RemotePi05Policy,
    _atomic_write_json,
    _load_reference_arrays,
    _policy_observation,
    _restore_recorded_state,
    stable_seed,
)
from evaluate_physical_process_oracle import (
    CONTINUOUS_OUTCOME_FIELDS,
    aggregate_outcomes,
    capture_runtime_snapshot,
    restore_runtime_snapshot,
    summarize_physical_trace,
    _physical_state,
)
from libero_snapshot_collector import DEFAULT_BDDL, _step
from video_io import write_h264_video


METHODS = ("sample0", "random4", "myopic_stage", "receding_oracle")
BINARY_METRICS = (
    "success",
    "next_stage_reached",
    "transport_reached",
    "lift_reached",
    "stable_grasp_at_end",
    "drop",
    "regress",
)
METRICS = BINARY_METRICS + CONTINUOUS_OUTCOME_FIELDS
METRIC_DIRECTIONS = (1, 1, 1, 1, 1, -1, -1, 1, 1, 1)
REPLAY_SEMANTIC_KEYS = (
    "success",
    "regress",
    "next_stage_reached",
    "transport_reached",
    "lift_reached",
    "stable_grasp_at_end",
    "first_regrasp_step",
    "first_transport_step",
)
REPLAY_SIM_TOLERANCE = 1e-8
CANDIDATE_POOL_TOLERANCE = 1e-6
PROGRESS_TIE_EPSILON = 1e-3
DECISION_CONFIG = {
    "split": "val",
    "sample_count": 4,
    "segment_replans": 4,
    "execution_horizon": 3,
    "total_action_budget": 120,
    "lookahead_steps": 30,
    "selection_continuations": 3,
    "decision_heldout_continuations": 5,
    "full_heldout_continuations": 5,
    "random_schedules": 3,
    "stage_dwell_steps": 2,
}
EXPECTED_CHECKPOINT_SHA256 = {
    41: "144a3b3d3dcc8421418564a62059a1038c9a7ef3196ac157f5f9ea1997a31f30",
    42: "98dc52d2ed1983776d218fee7666f3131053d1a55296e93e9f521b1c088ce875",
    43: "5db16350d9835c1f28d01b660dd6e9234bcab3da79abbce1f092e92b08ac9149",
}
# Updated only after the preregistration text is frozen. Decision runs fail
# closed until this value matches the runner-provided document digest.
EXPECTED_PREREGISTRATION_SHA256 = "d3105ba595e3467f2d2cec5642ca052dea2a692a63ad98b026c06a436ecb167c"


def candidate_seed_schedule(
    seed: int,
    pair_id: str,
    replan_index: int,
    sample_count: int,
) -> list[int]:
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")
    seeds = [stable_seed(seed, pair_id, replan_index)]
    seeds.extend(
        stable_seed(seed, pair_id, "receding_candidate", replan_index, candidate_index)
        for candidate_index in range(1, sample_count)
    )
    return seeds


def choose_candidate(candidate_rows: Sequence[Mapping[str, Any]], summary_key: str) -> int:
    if not candidate_rows:
        raise ValueError("candidate_rows must not be empty")
    return max(
        range(len(candidate_rows)),
        key=lambda index: aggregate_recovery_preference_key(
            candidate_rows[index][summary_key]
        ),
    )


def _quantized_progress(value: float) -> int:
    return int(round(float(value) / PROGRESS_TIE_EPSILON))


def recovery_preference_key(outcome: Mapping[str, Any]) -> tuple[float | int, ...]:
    return (
        float(bool(outcome["success"])),
        float(not bool(outcome["regress"])),
        float(bool(outcome["transport_reached"])),
        float(bool(outcome["lift_reached"])),
        float(bool(outcome["next_stage_reached"])),
        float(bool(outcome["stable_grasp_at_end"])),
        _quantized_progress(float(outcome["progress_auc"])),
    )


def aggregate_recovery_preference_key(
    outcome: Mapping[str, Any],
) -> tuple[float | int, ...]:
    return (
        float(outcome["success_rate"]),
        1.0 - float(outcome["regress_rate"]),
        float(outcome["transport_reached_rate"]),
        float(outcome["lift_reached_rate"]),
        float(outcome["next_stage_reached_rate"]),
        float(outcome["stable_grasp_at_end_rate"]),
        _quantized_progress(float(outcome["progress_auc"])),
    )


def semantic_outcome(outcome: Mapping[str, Any]) -> dict[str, Any]:
    return {key: outcome[key] for key in REPLAY_SEMANTIC_KEYS}


def _observation_frame(observation: Mapping[str, Any]) -> np.ndarray:
    policy_input = _policy_observation(observation)
    return np.concatenate(policy_input["image"], axis=1).astype(np.uint8)


def write_method_comparison_video(
    path: Path,
    frame_sets: Sequence[Sequence[np.ndarray]],
) -> None:
    if not frame_sets or any(not frames for frames in frame_sets):
        raise ValueError("comparison video requires at least one frame for every method")
    frame_count = max(len(frames) for frames in frame_sets)
    comparison_frames = [
        np.concatenate(
            [frames[min(index, len(frames) - 1)] for frames in frame_sets],
            axis=1,
        )
        for index in range(frame_count)
    ]
    write_h264_video(path, comparison_frames, fps=10.0)


def _execute_candidate(
    env: Any,
    observation: Mapping[str, Any],
    trace: list[dict[str, Any]],
    candidate: np.ndarray,
    execution_horizon: int,
    frames: list[np.ndarray] | None = None,
) -> tuple[Mapping[str, Any], int]:
    executed = 0
    for action in candidate[:execution_horizon]:
        observation = _step(env, action)
        if frames is not None:
            frames.append(_observation_frame(observation))
        executed += 1
        trace.append(_physical_state(env, observation, bool(env.check_success())))
        if trace[-1]["success"]:
            break
    return observation, executed


def summarize_recovery_trace(
    trace: Sequence[Mapping[str, Any]],
    *,
    stage_dwell_steps: int,
) -> dict[str, Any]:
    outcome = summarize_physical_trace(
        trace,
        stage="feedback",
        dwell_steps=stage_dwell_steps,
    )
    minimum_distance = min(float(state["bowl_xy_distance"]) for state in trace)
    final_distance = float(trace[-1]["bowl_xy_distance"])
    distance_regress = bool(
        outcome["first_regrasp_step"] is not None
        and final_distance > minimum_distance + 0.03
        and not outcome["success"]
    )
    return {
        **outcome,
        "distance_regress": distance_regress,
        "regress": bool(outcome["drop"] or distance_regress),
    }


def generate_candidate_endpoint(
    env: Any,
    snapshot: Mapping[str, Any],
    prefix_trace: Sequence[Mapping[str, Any]],
    *,
    candidate: np.ndarray,
    execution_horizon: int,
    stage_dwell_steps: int,
) -> dict[str, Any]:
    observation = restore_runtime_snapshot(env, snapshot)
    trace = [dict(state) for state in prefix_trace]
    candidate_frames: list[np.ndarray] = []
    observation, candidate_actions = _execute_candidate(
        env,
        observation,
        trace,
        candidate,
        execution_horizon,
        candidate_frames,
    )
    endpoint_snapshot = capture_runtime_snapshot(env)
    direct_policy_input = _policy_observation(observation)
    return {
        "endpoint_snapshot": endpoint_snapshot,
        "trace": trace,
        "direct": summarize_recovery_trace(
            trace,
            stage_dwell_steps=stage_dwell_steps,
        ),
        "candidate_actions": candidate_actions,
        "candidate_frames": candidate_frames,
        "direct_sim_state": np.asarray(endpoint_snapshot["sim_state"], dtype=np.float64),
        "direct_policy_images": np.stack(direct_policy_input["image"]).astype(np.uint8),
        "direct_policy_state": np.asarray(direct_policy_input["state"], dtype=np.float64),
    }


def rollout_endpoint_continuation(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    endpoint: Mapping[str, Any],
    *,
    lookahead_steps: int,
    continuation_seeds: Sequence[int],
    execution_horizon: int,
    stage_dwell_steps: int,
) -> dict[str, Any]:
    observation = restore_runtime_snapshot(env, endpoint["endpoint_snapshot"])
    trace = [dict(state) for state in endpoint["trace"]]
    continuation_actions = 0
    continuation_calls = 0
    while continuation_actions < lookahead_steps and not trace[-1]["success"]:
        if continuation_calls >= len(continuation_seeds):
            raise RuntimeError("continuation seed schedule is shorter than the lookahead rollout")
        chunk = policy.predict(observation, int(continuation_seeds[continuation_calls]))
        continuation_calls += 1
        for action in chunk[:execution_horizon]:
            if continuation_actions >= lookahead_steps:
                break
            observation = _step(env, action)
            continuation_actions += 1
            trace.append(_physical_state(env, observation, bool(env.check_success())))
            if trace[-1]["success"]:
                break

    return {
        "bridge": summarize_recovery_trace(
            trace,
            stage_dwell_steps=stage_dwell_steps,
        ),
        "continuation_policy_calls": continuation_calls,
        "continuation_actions": continuation_actions,
    }


def evaluate_oracle_decision(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    snapshot: Mapping[str, Any],
    observation: Mapping[str, Any],
    prefix_trace: Sequence[Mapping[str, Any]],
    *,
    pair_id: str,
    seed: int,
    source_initial_state_index: int,
    replan_index: int,
    sample_count: int,
    execution_horizon: int,
    lookahead_steps: int,
    selection_continuations: int,
    decision_heldout_continuations: int,
    stage_dwell_steps: int,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    sample_seeds = candidate_seed_schedule(seed, pair_id, replan_index, sample_count)
    inference_started = time.perf_counter()
    candidates, server_wall = policy.predict_many(observation, sample_seeds)
    inference_wall = time.perf_counter() - inference_started

    # Candidate zero follows the natural policy path from the live state. Every
    # other candidate is executed once from the exact same saved state.
    uninterrupted_trace = [dict(state) for state in prefix_trace]
    uninterrupted_candidate0_frames: list[np.ndarray] = []
    uninterrupted_observation, candidate0_actions = _execute_candidate(
        env,
        observation,
        uninterrupted_trace,
        candidates[0],
        execution_horizon,
        uninterrupted_candidate0_frames,
    )
    candidate0_snapshot = capture_runtime_snapshot(env)
    uninterrupted_candidate0_input = _policy_observation(uninterrupted_observation)
    candidate_endpoints = [
        {
            "endpoint_snapshot": candidate0_snapshot,
            "trace": uninterrupted_trace,
            "direct": summarize_recovery_trace(
                uninterrupted_trace,
                stage_dwell_steps=stage_dwell_steps,
            ),
            "candidate_actions": candidate0_actions,
            "candidate_frames": uninterrupted_candidate0_frames,
            "direct_sim_state": np.asarray(
                candidate0_snapshot["sim_state"], dtype=np.float64
            ),
            "direct_policy_images": np.stack(
                uninterrupted_candidate0_input["image"]
            ).astype(np.uint8),
            "direct_policy_state": np.asarray(
                uninterrupted_candidate0_input["state"], dtype=np.float64
            ),
        }
    ]
    candidate_endpoints.extend(
        generate_candidate_endpoint(
            env,
            snapshot,
            prefix_trace,
            candidate=candidate,
            execution_horizon=execution_horizon,
            stage_dwell_steps=stage_dwell_steps,
        )
        for candidate in candidates[1:]
    )

    restored_candidate0_observation = restore_runtime_snapshot(env, candidate0_snapshot)
    restored_candidate0_input = _policy_observation(restored_candidate0_observation)
    candidate0_restore_delta = float(
        np.max(
            np.abs(
                np.asarray(env.get_sim_state(), dtype=np.float64)
                - np.asarray(candidate0_snapshot["sim_state"], dtype=np.float64)
            )
        )
    )
    candidate0_image_delta = int(
        np.max(
            np.abs(
                np.stack(uninterrupted_candidate0_input["image"]).astype(np.int16)
                - np.stack(restored_candidate0_input["image"]).astype(np.int16)
            )
        )
    )
    candidate0_robot_state_delta = float(
        np.max(
            np.abs(
                np.asarray(uninterrupted_candidate0_input["state"], dtype=np.float64)
                - np.asarray(restored_candidate0_input["state"], dtype=np.float64)
            )
        )
    )
    if candidate0_restore_delta > REPLAY_SIM_TOLERANCE:
        raise RuntimeError("capture/restore changed the candidate-0 simulator endpoint")
    if candidate0_image_delta != 0 or candidate0_robot_state_delta > REPLAY_SIM_TOLERANCE:
        raise RuntimeError("capture/restore changed the candidate-0 policy observation")

    continuation_calls = int(np.ceil(lookahead_steps / execution_horizon))
    selection_seed_rows = [
        [
            stable_seed(seed, pair_id, "segment_selection", replan_index, repeat, call_index)
            for call_index in range(continuation_calls)
        ]
        for repeat in range(selection_continuations)
    ]
    decision_heldout_seed_rows = [
        [
            stable_seed(
                seed,
                pair_id,
                "segment_decision_heldout",
                replan_index,
                repeat,
                call_index,
            )
            for call_index in range(continuation_calls)
        ]
        for repeat in range(decision_heldout_continuations)
    ]

    if len(candidates) != len(candidate_endpoints):
        raise RuntimeError("candidate endpoint count does not match sampled candidate count")
    candidate_rows = []
    for candidate_index, (candidate, endpoint) in enumerate(
        zip(candidates, candidate_endpoints)
    ):
        selection_results = [
            rollout_endpoint_continuation(
                env,
                policy,
                endpoint,
                lookahead_steps=lookahead_steps,
                continuation_seeds=continuation_seeds,
                execution_horizon=execution_horizon,
                stage_dwell_steps=stage_dwell_steps,
            )
            for continuation_seeds in selection_seed_rows
        ]
        decision_heldout_results = [
            rollout_endpoint_continuation(
                env,
                policy,
                endpoint,
                lookahead_steps=lookahead_steps,
                continuation_seeds=continuation_seeds,
                execution_horizon=execution_horizon,
                stage_dwell_steps=stage_dwell_steps,
            )
            for continuation_seeds in decision_heldout_seed_rows
        ]
        candidate_rows.append(
            {
                "candidate_index": candidate_index,
                "sample_seed": int(sample_seeds[candidate_index]),
                "action_prefix": np.asarray(candidate[:execution_horizon], dtype=np.float32).round(7).tolist(),
                "executed_action_count": int(endpoint["candidate_actions"]),
                "direct": endpoint["direct"],
                "selection_summary": aggregate_outcomes(
                    [result["bridge"] for result in selection_results]
                ),
                "decision_heldout_summary": aggregate_outcomes(
                    [result["bridge"] for result in decision_heldout_results]
                ),
                "selection_continuations": [
                    {
                        "repeat": repeat,
                        "bridge": result["bridge"],
                        "continuation_policy_calls": result["continuation_policy_calls"],
                        "continuation_actions": result["continuation_actions"],
                    }
                    for repeat, result in enumerate(selection_results)
                ],
                "decision_heldout_continuations": [
                    {
                        "repeat": repeat,
                        "bridge": result["bridge"],
                        "continuation_policy_calls": result["continuation_policy_calls"],
                        "continuation_actions": result["continuation_actions"],
                    }
                    for repeat, result in enumerate(decision_heldout_results)
                ],
            }
        )

    oracle_index = choose_candidate(candidate_rows, "selection_summary")
    decision_heldout_oracle_index = choose_candidate(
        candidate_rows, "decision_heldout_summary"
    )
    policy_input = _policy_observation(observation)
    candidate_execution_counts = np.asarray(
        [endpoint["candidate_actions"] for endpoint in candidate_endpoints],
        dtype=np.int64,
    )
    candidate_action_mask = (
        np.arange(execution_horizon, dtype=np.int64)[None, :]
        < candidate_execution_counts[:, None]
    )
    selection_policy_calls = sum(
        result["continuation_policy_calls"]
        for candidate in candidate_rows
        for result in candidate["selection_continuations"]
    )
    decision_heldout_policy_calls = sum(
        result["continuation_policy_calls"]
        for candidate in candidate_rows
        for result in candidate["decision_heldout_continuations"]
    )
    selection_actions = sum(
        result["continuation_actions"]
        for candidate in candidate_rows
        for result in candidate["selection_continuations"]
    )
    decision_heldout_actions = sum(
        result["continuation_actions"]
        for candidate in candidate_rows
        for result in candidate["decision_heldout_continuations"]
    )
    decision = {
        "replan_index": replan_index,
        "sample_count": sample_count,
        "execution_horizon": execution_horizon,
        "lookahead_steps": lookahead_steps,
        "selection_continuation_count": selection_continuations,
        "decision_heldout_continuation_count": decision_heldout_continuations,
        "candidate_inference_wall_seconds": inference_wall,
        "candidate_server_wall_seconds": server_wall,
        "oracle_index": oracle_index,
        "decision_heldout_oracle_index": decision_heldout_oracle_index,
        "selection_matches_decision_heldout": (
            oracle_index == decision_heldout_oracle_index
        ),
        "candidate0_uninterrupted_restore_max_abs_delta": candidate0_restore_delta,
        "candidate0_uninterrupted_image_max_abs_delta": candidate0_image_delta,
        "candidate0_uninterrupted_robot_state_max_abs_delta": candidate0_robot_state_delta,
        "cost": {
            "candidate_inference_count": sample_count,
            "candidate_policy_batch_calls": 1,
            "candidate_endpoint_simulator_actions": int(candidate_execution_counts.sum()),
            "selection_continuation_policy_calls": int(selection_policy_calls),
            "selection_continuation_simulator_actions": int(selection_actions),
            "decision_heldout_policy_calls": int(decision_heldout_policy_calls),
            "decision_heldout_simulator_actions": int(decision_heldout_actions),
        },
        "candidates": candidate_rows,
    }
    training_record = {
        "images": np.stack(policy_input["image"]).astype(np.uint8),
        "robot_state": np.asarray(policy_input["state"], dtype=np.float32),
        "candidate_action_prefix": np.asarray(
            candidates[:, :execution_horizon],
            dtype=np.float32,
        ),
        "candidate_action_mask": candidate_action_mask,
        "oracle_index": np.int64(oracle_index),
        "replan_index": np.int64(replan_index),
        "decision_uid": np.asarray(
            f"seed{seed}:{pair_id}:replan{replan_index}", dtype=np.str_
        ),
        "source_initial_state_index": np.int64(source_initial_state_index),
        "model_seed": np.int64(seed),
        "candidate_selection_metrics": np.stack(
            [_metric_vector(candidate["selection_summary"]) for candidate in candidate_rows]
        ),
    }
    audit_record = {
        "candidate_actions": np.asarray(candidates, dtype=np.float32),
        "candidate_seeds": np.asarray(sample_seeds, dtype=np.uint32),
        "selection_continuation_seeds": np.asarray(selection_seed_rows, dtype=np.uint32),
        "decision_heldout_continuation_seeds": np.asarray(
            decision_heldout_seed_rows, dtype=np.uint32
        ),
        "decision_heldout_oracle_index": np.int64(decision_heldout_oracle_index),
        "candidate_decision_heldout_metrics": np.stack(
            [
                _metric_vector(candidate["decision_heldout_summary"])
                for candidate in candidate_rows
            ]
        ),
        "snapshot_sim_state": np.asarray(snapshot["sim_state"], dtype=np.float64),
        "snapshot_model_body_pos": np.asarray(
            snapshot["controller_state"]["model_body_pos"],
            dtype=np.float64,
        ),
        "snapshot_object_friction": np.asarray(
            snapshot["controller_state"]["object_friction"],
            dtype=np.float64,
        ),
        "snapshot_gripper_action": np.asarray(
            snapshot["controller_state"]["gripper_action"],
            dtype=np.float64,
        ),
        "candidate_direct_sim_state": np.stack(
            [endpoint["direct_sim_state"] for endpoint in candidate_endpoints]
        ),
    }
    for key, value in snapshot["controller_state"].items():
        audit_record[f"snapshot_controller_state__{key}"] = np.asarray(value)
        audit_record[f"candidate_endpoint_controller_state__{key}"] = np.stack(
            [
                np.asarray(endpoint["endpoint_snapshot"]["controller_state"][key])
                for endpoint in candidate_endpoints
            ]
        )
    return (
        decision,
        np.asarray(candidates),
        training_record,
        audit_record,
        candidate_endpoints[oracle_index],
    )


def _sample_candidates(
    policy: Pi05Policy | RemotePi05Policy,
    observation: Mapping[str, Any],
    *,
    pair_id: str,
    seed: int,
    replan_index: int,
    sample_count: int,
) -> tuple[np.ndarray, list[int]]:
    sample_seeds = candidate_seed_schedule(seed, pair_id, replan_index, sample_count)
    candidates, _ = policy.predict_many(observation, sample_seeds)
    return np.asarray(candidates), sample_seeds


def _choose_myopic_candidate(
    env: Any,
    snapshot: Mapping[str, Any],
    prefix_trace: Sequence[Mapping[str, Any]],
    candidates: np.ndarray,
    *,
    execution_horizon: int,
    stage_dwell_steps: int,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    endpoints = [
        generate_candidate_endpoint(
            env,
            snapshot,
            prefix_trace,
            candidate=candidate,
            execution_horizon=execution_horizon,
            stage_dwell_steps=stage_dwell_steps,
        )
        for candidate in candidates
    ]
    outcomes = [endpoint["direct"] for endpoint in endpoints]
    selected_index = max(
        range(len(outcomes)),
        key=lambda index: recovery_preference_key(outcomes[index]),
    )
    return selected_index, outcomes, endpoints


def _continue_live_policy(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    observation: Mapping[str, Any],
    trace: list[dict[str, Any]],
    *,
    pair_id: str,
    seed: int,
    start_replan_index: int,
    execution_horizon: int,
    total_action_budget: int,
    frames: list[np.ndarray] | None = None,
) -> tuple[Mapping[str, Any], int, int, int]:
    initial_actions = len(trace) - 1
    executed_actions = len(trace) - 1
    replan_index = start_replan_index
    policy_calls = 0
    while executed_actions < total_action_budget and not trace[-1]["success"]:
        chunk = policy.predict(observation, stable_seed(seed, pair_id, replan_index))
        replan_index += 1
        policy_calls += 1
        for action in chunk[:execution_horizon]:
            if executed_actions >= total_action_budget:
                break
            observation = _step(env, action)
            if frames is not None:
                frames.append(_observation_frame(observation))
            executed_actions += 1
            trace.append(_physical_state(env, observation, bool(env.check_success())))
            if trace[-1]["success"]:
                break
    return observation, replan_index, policy_calls, (len(trace) - 1 - initial_actions)


def run_full_heldout_continuations(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    endpoint_snapshot: Mapping[str, Any],
    segment_trace: Sequence[Mapping[str, Any]],
    segment_frames: Sequence[np.ndarray],
    *,
    pair_id: str,
    seed: int,
    start_replan_index: int,
    execution_horizon: int,
    total_action_budget: int,
    continuation_count: int,
    stage_dwell_steps: int,
    video_dir: Path | None = None,
    video_stem: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat in range(continuation_count):
        observation = restore_runtime_snapshot(env, endpoint_snapshot)
        trace = [dict(state) for state in segment_trace]
        frames = [np.asarray(frame) for frame in segment_frames]
        executed_actions = len(trace) - 1
        replan_index = start_replan_index
        policy_calls = 0
        continuation_actions = 0
        while executed_actions < total_action_budget and not trace[-1]["success"]:
            chunk = policy.predict(
                observation,
                stable_seed(
                    seed,
                    pair_id,
                    "segment_full_heldout",
                    repeat,
                    replan_index,
                ),
            )
            replan_index += 1
            policy_calls += 1
            for action in chunk[:execution_horizon]:
                if executed_actions >= total_action_budget:
                    break
                observation = _step(env, action)
                frames.append(_observation_frame(observation))
                executed_actions += 1
                continuation_actions += 1
                trace.append(_physical_state(env, observation, bool(env.check_success())))
                if trace[-1]["success"]:
                    break
        row = {
            "repeat": repeat,
            "policy_calls": policy_calls,
            "simulator_actions": continuation_actions,
            "outcome": summarize_recovery_trace(
                trace,
                stage_dwell_steps=stage_dwell_steps,
            ),
        }
        if video_dir is not None:
            if not video_stem:
                raise ValueError("video_stem is required when video_dir is set")
            video_path = video_dir / f"{video_stem}--full-heldout-{repeat}.mp4"
            write_h264_video(video_path, frames, fps=10.0)
            row["video_file"] = str(video_path)
        rows.append(row)
    return rows, aggregate_outcomes([row["outcome"] for row in rows])


def _sample0_parity_rollout(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    feedback_snapshot: Mapping[str, Any],
    *,
    pair_id: str,
    seed: int,
    execution_horizon: int,
    total_action_budget: int,
    force_restore_each_replan: bool,
) -> list[dict[str, Any]]:
    observation = restore_runtime_snapshot(env, feedback_snapshot)
    records: list[dict[str, Any]] = []
    replan_index = 0
    while len(records) < total_action_budget and not bool(env.check_success()):
        if force_restore_each_replan:
            observation = restore_runtime_snapshot(env, capture_runtime_snapshot(env))
        decision_policy_input = _policy_observation(observation)
        chunk = policy.predict(observation, stable_seed(seed, pair_id, replan_index))
        decision_index = replan_index
        replan_index += 1
        for action in chunk[:execution_horizon]:
            if len(records) >= total_action_budget:
                break
            observation = _step(env, action)
            snapshot = capture_runtime_snapshot(env)
            policy_input = _policy_observation(observation)
            records.append(
                {
                    "action": np.asarray(action, dtype=np.float64),
                    "predicted_prefix": np.asarray(
                        chunk[:execution_horizon], dtype=np.float64
                    ),
                    "decision_images": np.stack(
                        decision_policy_input["image"]
                    ).astype(np.uint8),
                    "decision_robot_state": np.asarray(
                        decision_policy_input["state"], dtype=np.float64
                    ),
                    "decision_index": decision_index,
                    "sim_state": np.asarray(snapshot["sim_state"], dtype=np.float64),
                    "model_body_pos": np.asarray(
                        snapshot["controller_state"]["model_body_pos"], dtype=np.float64
                    ),
                    "object_friction": np.asarray(
                        snapshot["controller_state"]["object_friction"], dtype=np.float64
                    ),
                    "gripper_action": np.asarray(
                        snapshot["controller_state"]["gripper_action"], dtype=np.float64
                    ),
                    "images": np.stack(policy_input["image"]).astype(np.uint8),
                    "robot_state": np.asarray(policy_input["state"], dtype=np.float64),
                    "success": bool(env.check_success()),
                }
            )
            if records[-1]["success"]:
                break
    return records


def run_sample0_restore_parity(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    feedback_snapshot: Mapping[str, Any],
    *,
    pair_id: str,
    seed: int,
    execution_horizon: int,
    total_action_budget: int,
) -> dict[str, Any]:
    natural = _sample0_parity_rollout(
        env,
        policy,
        feedback_snapshot,
        pair_id=pair_id,
        seed=seed,
        execution_horizon=execution_horizon,
        total_action_budget=total_action_budget,
        force_restore_each_replan=False,
    )
    restored = _sample0_parity_rollout(
        env,
        policy,
        feedback_snapshot,
        pair_id=pair_id,
        seed=seed,
        execution_horizon=execution_horizon,
        total_action_budget=total_action_budget,
        force_restore_each_replan=True,
    )
    if len(natural) != len(restored):
        raise RuntimeError("sample0 restore parity changed rollout length")
    if not natural:
        raise RuntimeError("sample0 restore parity executed no actions")

    numeric_keys = (
        "action",
        "sim_state",
        "model_body_pos",
        "object_friction",
        "gripper_action",
        "robot_state",
        "predicted_prefix",
        "decision_robot_state",
    )
    deltas = {
        key: float(
            max(
                np.max(np.abs(left[key] - right[key]))
                for left, right in zip(natural, restored)
            )
        )
        for key in numeric_keys
    }
    image_delta = int(
        max(
            np.max(
                np.abs(
                    left["images"].astype(np.int16) - right["images"].astype(np.int16)
                )
            )
            for left, right in zip(natural, restored)
        )
    )
    decision_image_delta = int(
        max(
            np.max(
                np.abs(
                    left["decision_images"].astype(np.int16)
                    - right["decision_images"].astype(np.int16)
                )
            )
            for left, right in zip(natural, restored)
        )
    )
    success_match = all(
        left["success"] == right["success"]
        for left, right in zip(natural, restored)
    )
    first_image_mismatch = next(
        (
            index
            for index, (left, right) in enumerate(zip(natural, restored), start=1)
            if not np.array_equal(left["images"], right["images"])
        ),
        None,
    )
    first_decision_image_mismatch = next(
        (
            index
            for index, (left, right) in enumerate(zip(natural, restored), start=1)
            if not np.array_equal(left["decision_images"], right["decision_images"])
        ),
        None,
    )
    first_success_mismatch = next(
        (
            index
            for index, (left, right) in enumerate(zip(natural, restored), start=1)
            if left["success"] != right["success"]
        ),
        None,
    )
    first_action_mismatch = next(
        (
            index
            for index, (left, right) in enumerate(zip(natural, restored), start=1)
            if not np.allclose(
                left["action"], right["action"], atol=REPLAY_SIM_TOLERANCE, rtol=0.0
            )
        ),
        None,
    )
    if image_delta != 0 or decision_image_delta != 0 or not success_match:
        raise RuntimeError(
            "sample0 restore parity changed pixels or task success: "
            f"image_delta={image_delta}, first_image_step={first_image_mismatch}, "
            f"decision_image_delta={decision_image_delta}, "
            f"first_decision_image_step={first_decision_image_mismatch}, "
            f"first_action_step={first_action_mismatch}, "
            f"first_success_step={first_success_mismatch}, numeric_deltas={deltas}"
        )
    if max(deltas.values()) > REPLAY_SIM_TOLERANCE:
        raise RuntimeError("sample0 restore parity exceeded numeric tolerance")
    return {
        "actions": len(natural),
        "natural_success": natural[-1]["success"],
        "forced_restore_success": restored[-1]["success"],
        "success_match": success_match,
        "image_max_abs_delta": image_delta,
        "decision_image_max_abs_delta": decision_image_delta,
        "numeric_max_abs_delta": deltas,
        "passed": True,
    }


def run_method(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    feedback_snapshot: Mapping[str, Any],
    *,
    method: str,
    pair_id: str,
    seed: int,
    source_initial_state_index: int,
    sample_count: int,
    segment_replans: int,
    execution_horizon: int,
    total_action_budget: int,
    lookahead_steps: int,
    selection_continuations: int,
    decision_heldout_continuations: int,
    full_heldout_continuations: int,
    random_schedule_index: int = 0,
    stage_dwell_steps: int,
    video_dir: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[np.ndarray]]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    observation = restore_runtime_snapshot(env, feedback_snapshot)
    trace = [_physical_state(env, observation, bool(env.check_success()))]
    frames = [_observation_frame(observation)]
    decisions: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    first_candidate_action_prefixes: np.ndarray | None = None
    cost = {
        "candidate_inference_count": 0,
        "candidate_policy_calls": 0,
        "search_simulator_actions": 0,
        "search_continuation_policy_calls": 0,
        "live_segment_actions": 0,
        "natural_continuation_policy_calls": 0,
        "natural_continuation_simulator_actions": 0,
        "full_heldout_policy_calls": 0,
        "full_heldout_simulator_actions": 0,
    }

    for replan_index in range(segment_replans):
        if trace[-1]["success"]:
            break
        before_steps = len(trace) - 1
        if method == "sample0":
            candidate = policy.predict(observation, stable_seed(seed, pair_id, replan_index))
            cost["candidate_inference_count"] += 1
            cost["candidate_policy_calls"] += 1
            selected_index = 0
            if first_candidate_action_prefixes is None:
                first_candidate_action_prefixes = np.asarray(
                    candidate[None, :execution_horizon], dtype=np.float64
                )
            decision = {
                "replan_index": replan_index,
                "selected_index": 0,
                "action_prefix": np.asarray(candidate[:execution_horizon], dtype=np.float32)
                .round(7)
                .tolist(),
            }
            observation, executed = _execute_candidate(
                env,
                observation,
                trace,
                np.asarray(candidate),
                execution_horizon,
                frames,
            )
        elif method == "receding_oracle":
            snapshot = capture_runtime_snapshot(env)
            (
                decision,
                candidates,
                training_record,
                audit_record,
                selected_endpoint,
            ) = evaluate_oracle_decision(
                env,
                policy,
                snapshot,
                observation,
                trace,
                pair_id=pair_id,
                seed=seed,
                source_initial_state_index=source_initial_state_index,
                replan_index=replan_index,
                sample_count=sample_count,
                execution_horizon=execution_horizon,
                lookahead_steps=lookahead_steps,
                selection_continuations=selection_continuations,
                decision_heldout_continuations=decision_heldout_continuations,
                stage_dwell_steps=stage_dwell_steps,
            )
            selected_index = int(decision["oracle_index"])
            if first_candidate_action_prefixes is None:
                first_candidate_action_prefixes = np.asarray(
                    candidates[:, :execution_horizon], dtype=np.float64
                )
            training_records.append(training_record)
            audit_records.append(audit_record)
            decision["selected_index"] = selected_index
            observation = restore_runtime_snapshot(
                env, selected_endpoint["endpoint_snapshot"]
            )
            trace = [dict(state) for state in selected_endpoint["trace"]]
            frames.extend(selected_endpoint["candidate_frames"])
            executed = int(selected_endpoint["candidate_actions"])
            decision_cost = decision["cost"]
            cost["candidate_inference_count"] += int(
                decision_cost["candidate_inference_count"]
            )
            cost["candidate_policy_calls"] += int(
                decision_cost["candidate_policy_batch_calls"]
            )
            cost["search_simulator_actions"] += int(
                decision_cost["candidate_endpoint_simulator_actions"]
                + decision_cost["selection_continuation_simulator_actions"]
                + decision_cost["decision_heldout_simulator_actions"]
            )
            cost["search_continuation_policy_calls"] += int(
                decision_cost["selection_continuation_policy_calls"]
                + decision_cost["decision_heldout_policy_calls"]
            )
        else:
            candidates, sample_seeds = _sample_candidates(
                policy,
                observation,
                pair_id=pair_id,
                seed=seed,
                replan_index=replan_index,
                sample_count=sample_count,
            )
            cost["candidate_inference_count"] += sample_count
            cost["candidate_policy_calls"] += 1
            if first_candidate_action_prefixes is None:
                first_candidate_action_prefixes = np.asarray(
                    candidates[:, :execution_horizon], dtype=np.float64
                )
            if method == "random4":
                selected_index = int(
                    stable_seed(
                        seed,
                        pair_id,
                        "random_candidate",
                        random_schedule_index,
                        replan_index,
                    )
                    % sample_count
                )
                myopic_outcomes = None
                selected_endpoint = None
            else:
                snapshot = capture_runtime_snapshot(env)
                selected_index, myopic_outcomes, candidate_endpoints = _choose_myopic_candidate(
                    env,
                    snapshot,
                    trace,
                    candidates,
                    execution_horizon=execution_horizon,
                    stage_dwell_steps=stage_dwell_steps,
                )
                selected_endpoint = candidate_endpoints[selected_index]
                cost["search_simulator_actions"] += int(
                    sum(endpoint["candidate_actions"] for endpoint in candidate_endpoints)
                )
            candidate = candidates[selected_index]
            decision = {
                "replan_index": replan_index,
                "selected_index": selected_index,
                "candidate_seeds": sample_seeds,
                "candidate_action_prefixes": np.asarray(
                    candidates[:, :execution_horizon], dtype=np.float32
                )
                .round(7)
                .tolist(),
            }
            if myopic_outcomes is not None:
                decision["candidate_direct_outcomes"] = myopic_outcomes
            if selected_endpoint is None:
                observation, executed = _execute_candidate(
                    env,
                    observation,
                    trace,
                    np.asarray(candidate),
                    execution_horizon,
                    frames,
                )
            else:
                observation = restore_runtime_snapshot(
                    env, selected_endpoint["endpoint_snapshot"]
                )
                trace = [dict(state) for state in selected_endpoint["trace"]]
                frames.extend(selected_endpoint["candidate_frames"])
                executed = int(selected_endpoint["candidate_actions"])

        decision["executed_actions"] = executed
        decision["prefix_steps_after_execution"] = len(trace) - 1
        decision["live_direct_outcome"] = summarize_recovery_trace(
            trace, stage_dwell_steps=stage_dwell_steps
        )
        if method == "receding_oracle":
            expected = decision["candidates"][selected_index]["direct"]
            decision["selected_direct_replay_match"] = semantic_outcome(
                decision["live_direct_outcome"]
            ) == semantic_outcome(expected)
            replay_snapshot = capture_runtime_snapshot(env)
            live_sim_state = np.asarray(replay_snapshot["sim_state"], dtype=np.float64)
            expected_sim_state = np.asarray(
                audit_record["candidate_direct_sim_state"][selected_index],
                dtype=np.float64,
            )
            decision["selected_endpoint_sim_max_abs_delta"] = float(
                np.max(np.abs(live_sim_state - expected_sim_state))
            )
            if not decision["selected_direct_replay_match"]:
                raise RuntimeError("selected action changed physical semantics after exact replay")
            if decision["selected_endpoint_sim_max_abs_delta"] > REPLAY_SIM_TOLERANCE:
                raise RuntimeError("selected action endpoint exceeded simulator replay tolerance")
        if len(trace) - 1 <= before_steps:
            raise RuntimeError("selected candidate executed no actions")
        cost["live_segment_actions"] += executed
        decisions.append(decision)

    segment_snapshot = capture_runtime_snapshot(env)
    segment_trace = [dict(state) for state in trace]
    segment_frames = [np.asarray(frame) for frame in frames]
    observation, final_replan_index, natural_policy_calls, natural_actions = _continue_live_policy(
        env,
        policy,
        observation,
        trace,
        pair_id=pair_id,
        seed=seed,
        start_replan_index=segment_replans,
        execution_horizon=execution_horizon,
        total_action_budget=total_action_budget,
        frames=frames,
    )
    del observation
    natural_outcome = summarize_recovery_trace(
        trace, stage_dwell_steps=stage_dwell_steps
    )
    cost["natural_continuation_policy_calls"] = natural_policy_calls
    cost["natural_continuation_simulator_actions"] = natural_actions

    video_stem = f"{pair_id}--{method}"
    if method == "random4":
        video_stem = f"{video_stem}-schedule{random_schedule_index}"
    video_files: list[str] = []
    if video_dir is not None:
        natural_video = video_dir / f"{video_stem}--natural.mp4"
        write_h264_video(natural_video, frames, fps=10.0)
        video_files.append(str(natural_video))

    full_heldout_rows, full_heldout_summary = run_full_heldout_continuations(
        env,
        policy,
        segment_snapshot,
        segment_trace,
        segment_frames,
        pair_id=pair_id,
        seed=seed,
        start_replan_index=segment_replans,
        execution_horizon=execution_horizon,
        total_action_budget=total_action_budget,
        continuation_count=full_heldout_continuations,
        stage_dwell_steps=stage_dwell_steps,
        video_dir=video_dir,
        video_stem=video_stem,
    )
    video_files.extend(
        row["video_file"] for row in full_heldout_rows if "video_file" in row
    )
    cost["full_heldout_policy_calls"] = int(
        sum(row["policy_calls"] for row in full_heldout_rows)
    )
    cost["full_heldout_simulator_actions"] = int(
        sum(row["simulator_actions"] for row in full_heldout_rows)
    )
    if first_candidate_action_prefixes is None:
        raise RuntimeError("method terminated before producing a candidate pool")
    return (
        {
            "method": method,
            "random_schedule_index": random_schedule_index if method == "random4" else None,
            "segment_replans_requested": segment_replans,
            "segment_replans_executed": len(decisions),
            "segment_action_budget": segment_replans * execution_horizon,
            "total_action_budget": total_action_budget,
            "executed_actions": len(trace) - 1,
            "final_replan_index": final_replan_index,
            "natural_outcome": natural_outcome,
            "full_heldout_outcomes": full_heldout_rows,
            "full_heldout_summary": full_heldout_summary,
            "video_files": video_files,
            "initial_candidate_action_prefixes": first_candidate_action_prefixes.round(7).tolist(),
            "cost": cost,
            "decisions": decisions,
        },
        training_records,
        audit_records,
        frames,
    )


def _metric_vector(summary: Mapping[str, Any]) -> np.ndarray:
    values = []
    for metric in METRICS:
        key = f"{metric}_rate" if metric in BINARY_METRICS else metric
        values.append(float(summary[key]))
    return np.asarray(values, dtype=np.float32)


def build_training_bank(records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError("at least one Oracle decision record is required")
    return {
        "images": np.stack([record["images"] for record in records]).astype(np.uint8),
        "robot_state": np.stack([record["robot_state"] for record in records]).astype(np.float32),
        "candidate_action_prefix": np.stack(
            [record["candidate_action_prefix"] for record in records]
        ).astype(np.float32),
        "candidate_action_mask": np.stack(
            [record["candidate_action_mask"] for record in records]
        ).astype(bool),
        "oracle_index": np.asarray([record["oracle_index"] for record in records], dtype=np.int64),
        "replan_index": np.asarray([record["replan_index"] for record in records], dtype=np.int64),
        "decision_uid": np.asarray([record["decision_uid"] for record in records], dtype=np.str_),
        "source_initial_state_index": np.asarray(
            [record["source_initial_state_index"] for record in records], dtype=np.int64
        ),
        "model_seed": np.asarray([record["model_seed"] for record in records], dtype=np.int64),
        "candidate_selection_metrics": np.stack(
            [record["candidate_selection_metrics"] for record in records]
        ).astype(np.float32),
    }


def aggregate_random_schedule_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not results:
        raise ValueError("at least one random schedule result is required")
    full_heldout_rows = [
        {**row, "random_schedule_index": int(result["random_schedule_index"])}
        for result in results
        for row in result["full_heldout_outcomes"]
    ]
    natural_outcomes = [result["natural_outcome"] for result in results]
    cost_keys = tuple(results[0]["cost"])
    return {
        "method": "random4",
        "random_schedule_count": len(results),
        "schedule_results": list(results),
        "initial_candidate_action_prefixes": results[0][
            "initial_candidate_action_prefixes"
        ],
        "natural_outcome_summary": aggregate_outcomes(natural_outcomes),
        "full_heldout_outcomes": full_heldout_rows,
        "full_heldout_summary": aggregate_outcomes(
            [row["outcome"] for row in full_heldout_rows]
        ),
        "video_files": [
            video_file for result in results for video_file in result.get("video_files", [])
        ],
        "cost": {
            key: int(sum(int(result["cost"][key]) for result in results))
            for key in cost_keys
        },
    }


def build_audit_bank(records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError("at least one Oracle audit record is required")
    return {
        key: np.stack([np.asarray(record[key]) for record in records])
        for key in records[0]
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    result: dict[str, Any] = {
        "pair_group_count": len({row["pair_id"] for row in rows}),
        "source_cluster_count": len({row["source_initial_state_index"] for row in rows}),
        "method_rates": {},
        "initial_candidate_pool_max_abs_delta": float(
            max(row["initial_candidate_pool_max_abs_delta"] for row in rows)
        ),
    }
    for method in METHODS:
        summaries = [row["methods"][method]["full_heldout_summary"] for row in rows]
        result["method_rates"][method] = {
            f"{metric}_rate": float(
                np.mean([float(summary[f"{metric}_rate"]) for summary in summaries])
            )
            for metric in BINARY_METRICS
        }
        result["method_rates"][method].update(
            {
                metric: float(np.mean([float(summary[metric]) for summary in summaries]))
                for metric in CONTINUOUS_OUTCOME_FIELDS
            }
        )
    oracle_decisions = [
        decision
        for row in rows
        for decision in row["methods"]["receding_oracle"]["decisions"]
    ]
    result["oracle_decision_count"] = len(oracle_decisions)
    if oracle_decisions:
        result["selection_matches_decision_heldout_rate"] = float(
            np.mean(
                [
                    decision["selection_matches_decision_heldout"]
                    for decision in oracle_decisions
                ]
            )
        )
        result["selected_direct_replay_match_rate"] = float(
            np.mean([decision["selected_direct_replay_match"] for decision in oracle_decisions])
        )
        result["selected_endpoint_sim_max_abs_delta"] = float(
            max(decision["selected_endpoint_sim_max_abs_delta"] for decision in oracle_decisions)
        )
        result["candidate0_uninterrupted_restore_max_abs_delta"] = float(
            max(
                decision["candidate0_uninterrupted_restore_max_abs_delta"]
                for decision in oracle_decisions
            )
        )
        result["candidate0_uninterrupted_image_max_abs_delta"] = int(
            max(
                decision["candidate0_uninterrupted_image_max_abs_delta"]
                for decision in oracle_decisions
            )
        )
        result["candidate0_uninterrupted_robot_state_max_abs_delta"] = float(
            max(
                decision["candidate0_uninterrupted_robot_state_max_abs_delta"]
                for decision in oracle_decisions
            )
        )
        result["oracle_selected_nonzero_rate"] = float(
            np.mean([decision["oracle_index"] != 0 for decision in oracle_decisions])
        )
    parity_rows = [row["sample0_restore_parity"] for row in rows]
    result["sample0_restore_parity_pass_rate"] = float(
        np.mean([parity["passed"] for parity in parity_rows])
    )
    result["sample0_restore_parity_image_max_abs_delta"] = int(
        max(parity["image_max_abs_delta"] for parity in parity_rows)
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-budget receding sibling-intervention Oracle for Pi0.5 recovery"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-socket", type=Path)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--audit-bank-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--run-kind", choices=("smoke", "decision"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--group-offset", type=int, default=0)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--segment-replans", type=int, default=4)
    parser.add_argument("--execution-horizon", type=int, default=3)
    parser.add_argument("--total-action-budget", type=int, default=120)
    parser.add_argument("--lookahead-steps", type=int, default=30)
    parser.add_argument("--selection-continuations", type=int, default=3)
    parser.add_argument("--decision-heldout-continuations", type=int, default=5)
    parser.add_argument("--full-heldout-continuations", type=int, default=5)
    parser.add_argument("--random-schedules", type=int, default=3)
    parser.add_argument("--stage-dwell-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_run_config(args: argparse.Namespace) -> None:
    if args.run_kind != "decision":
        return
    for field, expected in DECISION_CONFIG.items():
        actual = getattr(args, field)
        if actual != expected:
            raise ValueError(
                f"decision run requires {field}={expected!r}, received {actual!r}"
            )
    if args.seed not in EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("decision run seed must be one of 41, 42, or 43")
    if os.environ.get("FRESH_GIT_DIRTY") != "0":
        raise ValueError("decision run requires an explicitly clean Git worktree")
    if not os.environ.get("FRESH_GIT_SHA"):
        raise ValueError("decision run requires FRESH_GIT_SHA")
    checkpoint_sha = os.environ.get("FRESH_CHECKPOINT_SHA256")
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256[args.seed]:
        raise ValueError("decision run checkpoint SHA256 does not match the frozen seed map")
    preregistration_sha = os.environ.get("FRESH_PREREGISTRATION_SHA256")
    if preregistration_sha != EXPECTED_PREREGISTRATION_SHA256:
        raise ValueError("decision run preregistration SHA256 does not match frozen code")


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    validate_run_config(args)
    if (args.checkpoint is None) == (args.policy_socket is None):
        raise ValueError("provide exactly one of --checkpoint or --policy-socket")
    if args.sample_count < 2 or args.segment_replans < 1:
        raise ValueError("sample-count must be at least two and segment-replans must be positive")
    if args.execution_horizon < 1 or args.lookahead_steps < 0:
        raise ValueError("execution-horizon must be positive and lookahead-steps non-negative")
    segment_budget = args.segment_replans * args.execution_horizon
    if args.total_action_budget < segment_budget:
        raise ValueError("total-action-budget must cover the fixed intervention segment")
    if (
        args.selection_continuations < 1
        or args.decision_heldout_continuations < 1
        or args.full_heldout_continuations < 1
    ):
        raise ValueError("selection and heldout continuations must be positive")
    if args.random_schedules < 1:
        raise ValueError("random-schedules must be positive")
    if args.stage_dwell_steps < 1:
        raise ValueError("stage-dwell-steps must be positive")
    os.environ.setdefault("PRETRAINED_MODELS_DIR", "/share/longjunyu/alphabrain/pretrained_models")

    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    all_split_groups = sorted(
        (group for group in manifest["groups"] if group["split"] == args.split),
        key=lambda row: row["pair_id"],
    )
    if args.group_offset < 0:
        raise ValueError("group-offset must be non-negative")
    groups = all_split_groups[args.group_offset :]
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    if not groups:
        raise ValueError(f"no groups for split={args.split!r}")
    if args.run_kind == "decision":
        source_count = len(
            {int(group["source_initial_state_index"]) for group in all_split_groups}
        )
        if len(all_split_groups) != 13 or source_count != 9:
            raise ValueError(
                "decision run requires the frozen 13-group, 9-source val manifest"
            )

    policy = RemotePi05Policy(args.policy_socket) if args.policy_socket else Pi05Policy(args.checkpoint, args.device)
    if args.execution_horizon > policy.horizon:
        raise ValueError("execution horizon exceeds policy action horizon")
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(manifest.get("bddl", DEFAULT_BDDL))),
        camera_heights=224,
        camera_widths=224,
    )
    env.seed(args.seed)
    args.bank_dir.mkdir(parents=True, exist_ok=True)
    args.audit_bank_dir.mkdir(parents=True, exist_ok=True)
    args.video_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    partial = args.output.with_name(f"{args.output.stem}.partial{args.output.suffix}")

    def payload(status: str) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "status": status,
            "run_kind": args.run_kind,
            "episode_root": str(args.episode_root),
            "video_dir": str(args.video_dir),
            "video_encoding": {
                "container": "mp4",
                "codec": "h264",
                "codec_tag": "avc1",
                "pixel_format": "yuv420p",
                "faststart": True,
                "fps": 10.0,
            },
            "split": args.split,
            "group_offset": args.group_offset,
            "seed": args.seed,
            "sample_count": args.sample_count,
            "segment_replans": args.segment_replans,
            "execution_horizon": args.execution_horizon,
            "segment_action_budget": segment_budget,
            "total_action_budget": args.total_action_budget,
            "lookahead_steps": args.lookahead_steps,
            "selection_continuations": args.selection_continuations,
            "decision_heldout_continuations": args.decision_heldout_continuations,
            "full_heldout_continuations": args.full_heldout_continuations,
            "random_schedules": args.random_schedules,
            "stage_dwell_steps": args.stage_dwell_steps,
            "replay_sim_tolerance": REPLAY_SIM_TOLERANCE,
            "candidate_pool_tolerance": CANDIDATE_POOL_TOLERANCE,
            "progress_tie_epsilon": PROGRESS_TIE_EPSILON,
            "methods": list(METHODS),
            "training_bank_model_input_fields": [
                "images",
                "robot_state",
                "candidate_action_prefix",
                "candidate_action_mask",
            ],
            "training_bank_target_fields": [
                "oracle_index",
                "candidate_selection_metrics",
            ],
            "training_bank_metadata_fields": [
                "decision_uid",
                "replan_index",
                "source_initial_state_index",
                "model_seed",
            ],
            "outcome_metric_order": list(METRICS),
            "outcome_metric_directions": list(METRIC_DIRECTIONS),
            "privileged_audit_bank_is_separate": True,
            "git_sha": os.environ.get("FRESH_GIT_SHA"),
            "git_dirty_at_launch": os.environ.get("FRESH_GIT_DIRTY") == "1",
            "preregistration_sha256": os.environ.get("FRESH_PREREGISTRATION_SHA256"),
            "expected_preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
            "policy_checkpoint_sha256": os.environ.get("FRESH_CHECKPOINT_SHA256"),
            "policy_checkpoint_sha256_source": os.environ.get("FRESH_CHECKPOINT_SHA256_SOURCE"),
            "policy_checkpoint_realpath": getattr(policy, "checkpoint_realpath", None),
            "policy_model_size_bytes": getattr(policy, "model_size_bytes", None),
            "policy_runtime": getattr(policy, "runtime_identity", None),
            "completed_rows": len(rows),
            "expected_rows": len(groups),
            "expected_global_rows": len(all_split_groups),
            "expected_global_pair_source_map": {
                str(group["pair_id"]): int(group["source_initial_state_index"])
                for group in all_split_groups
            },
            "summary": summarize_rows(rows),
            "rows": rows,
        }

    try:
        for group in groups:
            reference = _load_reference_arrays(args.episode_root, group, "slipped")
            feedback_index = int(group["feedback_reveal_time"])
            _restore_recorded_state(env, reference, feedback_index)
            feedback_snapshot = capture_runtime_snapshot(env)
            pair_id = str(group["pair_id"])
            source_initial_state_index = int(group["source_initial_state_index"])
            sample0_restore_parity = run_sample0_restore_parity(
                env,
                policy,
                feedback_snapshot,
                pair_id=pair_id,
                seed=args.seed,
                execution_horizon=args.execution_horizon,
                total_action_budget=args.total_action_budget,
            )
            methods: dict[str, Any] = {}
            common_kwargs = {
                "pair_id": pair_id,
                "seed": args.seed,
                "source_initial_state_index": source_initial_state_index,
                "sample_count": args.sample_count,
                "segment_replans": args.segment_replans,
                "execution_horizon": args.execution_horizon,
                "total_action_budget": args.total_action_budget,
                "lookahead_steps": args.lookahead_steps,
                "selection_continuations": args.selection_continuations,
                "decision_heldout_continuations": args.decision_heldout_continuations,
                "full_heldout_continuations": args.full_heldout_continuations,
                "stage_dwell_steps": args.stage_dwell_steps,
                "video_dir": args.video_dir,
            }
            sample0_result, _, _, sample0_frames = run_method(
                env,
                policy,
                feedback_snapshot,
                method="sample0",
                **common_kwargs,
            )
            methods["sample0"] = sample0_result

            random_results = []
            random_frame_sets = []
            for random_schedule_index in range(args.random_schedules):
                random_result, _, _, random_frames = run_method(
                    env,
                    policy,
                    feedback_snapshot,
                    method="random4",
                    random_schedule_index=random_schedule_index,
                    **common_kwargs,
                )
                random_results.append(random_result)
                random_frame_sets.append(random_frames)
            methods["random4"] = aggregate_random_schedule_results(random_results)

            myopic_result, _, _, myopic_frames = run_method(
                env,
                policy,
                feedback_snapshot,
                method="myopic_stage",
                **common_kwargs,
            )
            methods["myopic_stage"] = myopic_result
            (
                oracle_result,
                oracle_training_records,
                oracle_audit_records,
                oracle_frames,
            ) = run_method(
                env,
                policy,
                feedback_snapshot,
                method="receding_oracle",
                **common_kwargs,
            )
            methods["receding_oracle"] = oracle_result

            comparison_video = args.video_dir / f"{pair_id}--method-comparison.mp4"
            write_method_comparison_video(
                comparison_video,
                [sample0_frames, random_frame_sets[0], myopic_frames, oracle_frames],
            )

            oracle_initial = np.asarray(
                methods["receding_oracle"]["initial_candidate_action_prefixes"],
                dtype=np.float64,
            )
            random_initial = np.asarray(
                methods["random4"]["initial_candidate_action_prefixes"],
                dtype=np.float64,
            )
            myopic_initial = np.asarray(
                methods["myopic_stage"]["initial_candidate_action_prefixes"],
                dtype=np.float64,
            )
            sample0_initial = np.asarray(
                methods["sample0"]["initial_candidate_action_prefixes"][0],
                dtype=np.float64,
            )
            random_schedule_deltas = [
                float(
                    np.max(
                        np.abs(
                            np.asarray(
                                result["initial_candidate_action_prefixes"],
                                dtype=np.float64,
                            )
                            - oracle_initial
                        )
                    )
                )
                for result in random_results
            ]
            initial_pool_delta = float(
                max(
                    np.max(np.abs(random_initial - oracle_initial)),
                    *random_schedule_deltas,
                    np.max(np.abs(myopic_initial - oracle_initial)),
                    np.max(np.abs(sample0_initial - oracle_initial[0])),
                )
            )
            if initial_pool_delta > CANDIDATE_POOL_TOLERANCE:
                raise RuntimeError("methods did not share the same initial candidate pool")

            row = {
                "pair_id": group["pair_id"],
                "split": group["split"],
                "source_initial_state_index": group.get("source_initial_state_index"),
                "feedback_state_index": feedback_index,
                "seed": args.seed,
                "sample0_restore_parity": sample0_restore_parity,
                "comparison_video_file": str(comparison_video),
                "comparison_video_columns": [
                    "sample0",
                    "random4_schedule0",
                    "myopic_stage",
                    "receding_oracle",
                ],
                "initial_candidate_pool_max_abs_delta": initial_pool_delta,
                "methods": methods,
            }
            bank_path = args.bank_dir / f"{group['pair_id']}--seed{args.seed}.npz"
            audit_path = args.audit_bank_dir / f"{group['pair_id']}--seed{args.seed}.npz"
            np.savez_compressed(bank_path, **build_training_bank(oracle_training_records))
            np.savez_compressed(audit_path, **build_audit_bank(oracle_audit_records))
            row["training_bank_file"] = str(bank_path)
            row["privileged_audit_bank_file"] = str(audit_path)
            rows.append(row)
            _atomic_write_json(partial, payload("partial"))
            print(
                json.dumps(
                    {
                        "pair_id": row["pair_id"],
                        "sample0_full_heldout_success_rate": methods["sample0"][
                            "full_heldout_summary"
                        ]["success_rate"],
                        "random4_full_heldout_success_rate": methods["random4"][
                            "full_heldout_summary"
                        ]["success_rate"],
                        "myopic_full_heldout_success_rate": methods["myopic_stage"][
                            "full_heldout_summary"
                        ]["success_rate"],
                        "oracle_full_heldout_success_rate": methods["receding_oracle"][
                            "full_heldout_summary"
                        ]["success_rate"],
                        "oracle_selection_decision_heldout_match_rate": float(
                            np.mean(
                                [
                                    decision["selection_matches_decision_heldout"]
                                    for decision in methods["receding_oracle"]["decisions"]
                                ]
                            )
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        _atomic_write_json(args.output, payload("complete"))
        partial.unlink(missing_ok=True)
    finally:
        env.close()
        policy.close()


if __name__ == "__main__":
    main()
