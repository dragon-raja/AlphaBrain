from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_libero_closed_loop import (
    LIFT_THRESHOLD,
    TRANSPORT_XY_TOLERANCE,
    Pi05Policy,
    RemotePi05Policy,
    _atomic_write_json,
    _load_reference_arrays,
    _policy_observation,
    _restore_recorded_state,
    stable_seed,
)
from libero_full_episode_collector import object_grasped
from libero_snapshot_collector import DEFAULT_BDDL, _capture_controller_state, _restore_snapshot, _step


STAGES = ("feedback", "post_regrasp")
BINARY_OUTCOME_FIELDS = (
    "success",
    "regress",
    "next_stage_reached",
    "transport_reached",
    "lift_reached",
    "stable_grasp_at_end",
    "drop",
)
CONTINUOUS_OUTCOME_FIELDS = (
    "progress_auc",
    "object_to_bowl_progress",
    "object_height_progress",
)


def find_stage_indices(reference: Mapping[str, np.ndarray], feedback_time: int) -> dict[str, int]:
    grasped = np.asarray(reference["grasped"], dtype=bool)
    if feedback_time < 0 or feedback_time >= len(grasped):
        raise ValueError(f"feedback index {feedback_time} outside episode with {len(grasped)} states")
    after_feedback = np.flatnonzero(grasped[feedback_time + 1 :]) + feedback_time + 1
    if len(after_feedback) == 0:
        raise ValueError("slipped reference branch never regrasped after feedback")
    return {
        "feedback": int(feedback_time),
        "post_regrasp": int(after_feedback[0]),
    }


def _physical_state(env: Any, observation: Mapping[str, Any], success: bool) -> dict[str, Any]:
    object_position = np.asarray(observation["cream_cheese_1_pos"], dtype=np.float64)
    bowl_position = np.asarray(observation["akita_black_bowl_1_pos"], dtype=np.float64)
    return {
        "grasped": bool(object_grasped(env)),
        "object_z": float(object_position[2]),
        "bowl_xy_distance": float(np.linalg.norm(object_position[:2] - bowl_position[:2])),
        "success": bool(success),
    }


def capture_runtime_snapshot(env: Any) -> dict[str, Any]:
    return {
        "sim_state": np.asarray(env.get_sim_state(), dtype=np.float64).copy(),
        "controller_state": _capture_controller_state(env),
    }


def restore_runtime_snapshot(env: Any, snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    return _restore_snapshot(
        env,
        np.asarray(snapshot["sim_state"], dtype=np.float64),
        snapshot["controller_state"],
    )


def generate_policy_post_regrasp_state(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    feedback_snapshot: Mapping[str, Any],
    *,
    pair_id: str,
    seed: int,
    execution_horizon: int,
    max_steps: int,
    stable_grasp_steps: int,
) -> tuple[dict[str, Any] | None, Mapping[str, Any] | None, dict[str, Any]]:
    observation = restore_runtime_snapshot(env, feedback_snapshot)
    completion_steps = 0
    replan_count = 0
    consecutive_grasped = 0
    ever_grasped = False
    dropped_after_grasp = False

    while completion_steps < max_steps and not bool(env.check_success()):
        chunk = policy.predict(observation, stable_seed(seed, pair_id, replan_count))
        replan_count += 1
        for action in chunk[:execution_horizon]:
            if completion_steps >= max_steps:
                break
            observation = _step(env, action)
            completion_steps += 1
            grasped = bool(object_grasped(env))
            if ever_grasped and not grasped and not bool(env.check_success()):
                dropped_after_grasp = True
            ever_grasped = ever_grasped or grasped
            consecutive_grasped = consecutive_grasped + 1 if grasped else 0
            if bool(env.check_success()):
                break

        # Capture only at a policy replanning boundary so sample0 can be the
        # exact next baseline policy invocation rather than an extra replan.
        if consecutive_grasped >= stable_grasp_steps:
            return (
                capture_runtime_snapshot(env),
                observation,
                {
                    "eligible": True,
                    "state_source": "policy_generated_post_regrasp",
                    "prefix_completion_steps": completion_steps,
                    "prefix_policy_calls": replan_count,
                    "next_policy_replan_index": replan_count,
                    "stable_grasp_steps": stable_grasp_steps,
                    "dropped_before_capture": dropped_after_grasp,
                },
            )

    return (
        None,
        None,
        {
            "eligible": False,
            "state_source": "policy_generated_post_regrasp",
            "prefix_completion_steps": completion_steps,
            "prefix_policy_calls": replan_count,
            "next_policy_replan_index": replan_count,
            "stable_grasp_steps": stable_grasp_steps,
            "dropped_before_capture": dropped_after_grasp,
            "reason": "baseline_policy_did_not_reach_stable_regrasp",
        },
    )


def summarize_physical_trace(
    trace: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    dwell_steps: int = 2,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    if len(trace) < 2:
        raise ValueError("physical trace must include the restored state and at least one executed action")
    if dwell_steps < 1:
        raise ValueError("dwell_steps must be positive")

    initial = trace[0]
    executed = trace[1:]
    initial_distance = float(initial["bowl_xy_distance"])
    initial_z = float(initial["object_z"])
    starts_stably_grasped = stage == "post_regrasp" and bool(initial["grasped"])
    grasp_run = dwell_steps if starts_stably_grasped else 0
    lift_run = 0
    transport_run = 0
    dropped = False
    first_regrasp = 0 if starts_stably_grasped else None
    first_lift = None
    first_transport = None
    first_success = None
    progress_values = []

    for step, state in enumerate(executed, start=1):
        grasped = bool(state["grasped"])
        success = bool(state["success"])
        near_bowl = float(state["bowl_xy_distance"]) <= TRANSPORT_XY_TOLERANCE
        lifted = float(state["object_z"]) - initial_z >= LIFT_THRESHOLD
        grasp_run = grasp_run + 1 if grasped else 0
        if grasp_run >= dwell_steps and first_regrasp is None:
            first_regrasp = step
        lift_run = lift_run + 1 if first_regrasp is not None and lifted and grasped else 0
        if lift_run >= dwell_steps and first_lift is None:
            first_lift = step
        transport_run = transport_run + 1 if first_lift is not None and near_bowl and (grasped or success) else 0
        if transport_run >= dwell_steps and first_transport is None:
            first_transport = step
        if success and first_success is None:
            first_success = step
        if first_regrasp is not None and not grasped and not success and not near_bowl:
            dropped = True

        if success:
            progress = 1.0
        elif first_transport is not None:
            progress = 0.8
        elif first_lift is not None:
            progress = 0.6
        elif first_regrasp is not None:
            progress = 0.35
        else:
            progress = 0.0
        progress_values.append(progress)

    minimum_distance = min(float(state["bowl_xy_distance"]) for state in trace)
    maximum_z = max(float(state["object_z"]) for state in trace)
    final = trace[-1]
    bowl_progress = initial_distance - minimum_distance
    height_progress = maximum_z - initial_z
    distance_regress = float(final["bowl_xy_distance"]) > initial_distance + 0.03
    regress = bool(dropped or (stage == "post_regrasp" and distance_regress))
    next_stage_reached = first_regrasp is not None if stage == "feedback" else first_lift is not None

    return {
        "steps": len(executed),
        "stage_order": ["stable_regrasp", "lift", "transport", "success"],
        "dwell_steps": dwell_steps,
        "stable_grasp_at_end": bool(first_success is not None or grasp_run >= dwell_steps),
        "first_regrasp_step": first_regrasp,
        "first_lift_step": first_lift,
        "first_transport_step": first_transport,
        "first_success_step": first_success,
        "regrasp_reached": first_regrasp is not None,
        "lift_reached": first_lift is not None,
        "transport_reached": first_transport is not None,
        "success": first_success is not None,
        "next_stage_reached": bool(next_stage_reached),
        "drop": dropped,
        "regress": regress,
        "object_height_progress": float(height_progress),
        "object_to_bowl_progress": float(bowl_progress),
        "progress_auc": float(np.mean(progress_values)),
        "final_object_z_delta": float(final["object_z"]) - initial_z,
        "final_bowl_xy_delta": initial_distance - float(final["bowl_xy_distance"]),
    }


def preference_key(outcome: Mapping[str, Any]) -> tuple[float, ...]:
    first_next = outcome.get("first_regrasp_step")
    if first_next is None or first_next == 0:
        first_next = outcome.get("first_lift_step")
    time_preference = -float(first_next) if first_next is not None else -1e6
    return (
        float(bool(outcome["success"])),
        float(not bool(outcome["regress"])),
        float(bool(outcome["next_stage_reached"])),
        float(bool(outcome["transport_reached"])),
        float(bool(outcome["lift_reached"])),
        float(bool(outcome["stable_grasp_at_end"])),
        float(outcome["progress_auc"]),
        float(outcome["object_to_bowl_progress"]),
        float(outcome["object_height_progress"]),
        time_preference,
    )


def outcome_signature(outcome: Mapping[str, Any]) -> tuple[bool, ...]:
    return tuple(bool(outcome[key]) for key in BINARY_OUTCOME_FIELDS)


def aggregate_outcomes(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not outcomes:
        raise ValueError("at least one outcome is required")
    result: dict[str, Any] = {"repeat_count": len(outcomes)}
    result.update(
        {f"{key}_rate": float(np.mean([bool(outcome[key]) for outcome in outcomes])) for key in BINARY_OUTCOME_FIELDS}
    )
    result.update(
        {key: float(np.mean([float(outcome[key]) for outcome in outcomes])) for key in CONTINUOUS_OUTCOME_FIELDS}
    )
    return result


def aggregate_preference_key(outcome: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(outcome["success_rate"]),
        1.0 - float(outcome["regress_rate"]),
        float(outcome["next_stage_reached_rate"]),
        float(outcome["transport_reached_rate"]),
        float(outcome["lift_reached_rate"]),
        float(outcome["stable_grasp_at_end_rate"]),
        float(outcome["progress_auc"]),
        float(outcome["object_to_bowl_progress"]),
        float(outcome["object_height_progress"]),
    )


def aggregate_outcome_signature(outcome: Mapping[str, Any]) -> tuple[float, ...]:
    keys = tuple(f"{key}_rate" for key in BINARY_OUTCOME_FIELDS)
    return tuple(round(float(outcome[key]), 6) for key in keys)


def _rollout_candidate(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    snapshot: Mapping[str, Any],
    *,
    stage: str,
    candidate: np.ndarray,
    execution_horizon: int,
    bridge_steps: int,
    continuation_seeds: Sequence[int],
    stage_dwell_steps: int,
) -> dict[str, Any]:
    observation = restore_runtime_snapshot(env, snapshot)
    trace = [_physical_state(env, observation, bool(env.check_success()))]

    for action in candidate[:execution_horizon]:
        observation = _step(env, action)
        trace.append(_physical_state(env, observation, bool(env.check_success())))
        if trace[-1]["success"]:
            break
    direct_trace_length = len(trace)

    bridge_actions = 0
    continuation_calls = 0
    while bridge_actions < bridge_steps and not trace[-1]["success"]:
        if continuation_calls >= len(continuation_seeds):
            raise RuntimeError("continuation seed schedule is shorter than requested bridge rollout")
        chunk = policy.predict(observation, int(continuation_seeds[continuation_calls]))
        continuation_calls += 1
        for action in chunk[:execution_horizon]:
            if bridge_actions >= bridge_steps:
                break
            observation = _step(env, action)
            bridge_actions += 1
            trace.append(_physical_state(env, observation, bool(env.check_success())))
            if trace[-1]["success"]:
                break

    return {
        "direct": summarize_physical_trace(trace[:direct_trace_length], stage=stage, dwell_steps=stage_dwell_steps),
        "bridge": summarize_physical_trace(trace, stage=stage, dwell_steps=stage_dwell_steps),
        "continuation_policy_calls": continuation_calls,
        "continuation_actions": bridge_actions,
    }


def replay_semantics(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "success",
        "regress",
        "next_stage_reached",
        "transport_reached",
        "lift_reached",
        "stable_grasp_at_end",
        "first_regrasp_step",
        "first_transport_step",
    )
    return {key: result["bridge"][key] for key in keys}


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, Any] = {}
    for stage in STAGES:
        requested = [row for row in rows if row["stage"] == stage]
        selected = [row for row in requested if row.get("eligible", True)]
        if not requested:
            continue
        if not selected:
            by_stage[stage] = {
                "requested_state_count": len(requested),
                "eligible_state_count": 0,
                "eligibility_rate": 0.0,
            }
            continue
        by_stage[stage] = {
            "requested_state_count": len(requested),
            "eligible_state_count": len(selected),
            "eligibility_rate": len(selected) / len(requested),
            "pair_group_count": len({row["pair_id"] for row in selected}),
            "source_cluster_count": len({row["source_initial_state_index"] for row in selected}),
            "candidate_positive_coverage": float(
                np.mean(
                    [
                        any(
                            candidate["selection_summary"]["next_stage_reached_rate"] > 0.0
                            for candidate in row["candidates"]
                        )
                        for row in selected
                    ]
                )
            ),
            "sample0_next_stage_rate": float(
                np.mean([row["candidates"][0]["selection_summary"]["next_stage_reached_rate"] for row in selected])
            ),
            "oracle_next_stage_rate": float(
                np.mean(
                    [
                        row["candidates"][row["oracle_index"]]["selection_summary"]["next_stage_reached_rate"]
                        for row in selected
                    ]
                )
            ),
            "oracle_minus_sample0_next_stage": float(
                np.mean(
                    [
                        float(
                            row["candidates"][row["oracle_index"]]["selection_summary"][
                                "next_stage_reached_rate"
                            ]
                        )
                        - float(row["candidates"][0]["selection_summary"]["next_stage_reached_rate"])
                        for row in selected
                    ]
                )
            ),
            "sample0_success_rate": float(
                np.mean([row["candidates"][0]["selection_summary"]["success_rate"] for row in selected])
            ),
            "oracle_success_rate": float(
                np.mean(
                    [
                        row["candidates"][row["oracle_index"]]["selection_summary"]["success_rate"]
                        for row in selected
                    ]
                )
            ),
            "oracle_selected_nonzero_rate": float(np.mean([row["oracle_index"] != 0 for row in selected])),
            "mean_unique_outcome_signatures": float(np.mean([row["unique_outcome_signatures"] for row in selected])),
            "oracle_replay_match_rate": float(np.mean([row["oracle_replay_match"] for row in selected])),
        }
        heldout = [entry for row in selected for entry in row.get("heldout_continuations", ())]
        if heldout:
            by_stage[stage].update(
                {
                    "heldout_continuation_count": len(heldout),
                    "heldout_sample0_success_rate": float(
                        np.mean([entry["sample0"]["bridge"]["success"] for entry in heldout])
                    ),
                    "heldout_oracle_success_rate": float(
                        np.mean([entry["oracle"]["bridge"]["success"] for entry in heldout])
                    ),
                    "heldout_oracle_minus_sample0_success": float(
                        np.mean(
                            [
                                float(entry["oracle"]["bridge"]["success"])
                                - float(entry["sample0"]["bridge"]["success"])
                                for entry in heldout
                            ]
                        )
                    ),
                    "heldout_oracle_minus_sample0_next_stage": float(
                        np.mean(
                            [
                                float(entry["oracle"]["bridge"]["next_stage_reached"])
                                - float(entry["sample0"]["bridge"]["next_stage_reached"])
                                for entry in heldout
                            ]
                        )
                    ),
                }
            )
    return by_stage


def evaluate_state(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    snapshot: Mapping[str, Any],
    group: Mapping[str, Any],
    *,
    stage: str,
    state_index: int | None,
    state_metadata: Mapping[str, Any],
    seed: int,
    sample_count: int,
    execution_horizon: int,
    bridge_steps: int,
    selection_continuations: int,
    validation_continuations: int,
    stage_dwell_steps: int,
) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    observation = restore_runtime_snapshot(env, snapshot)
    baseline_replan_index = int(state_metadata.get("next_policy_replan_index", 0))
    sample_seeds = [stable_seed(seed, group["pair_id"], baseline_replan_index)]
    sample_seeds.extend(
        stable_seed(seed, group["pair_id"], stage, "counterfactual_candidate", index)
        for index in range(1, sample_count)
    )
    inference_started = time.perf_counter()
    candidates, server_wall = policy.predict_many(observation, sample_seeds)
    inference_wall = time.perf_counter() - inference_started
    continuation_calls = int(np.ceil(bridge_steps / execution_horizon))
    selection_seed_rows = [
        [
            stable_seed(seed, group["pair_id"], stage, "selection_continuation", repeat, index)
            for index in range(continuation_calls)
        ]
        for repeat in range(selection_continuations)
    ]

    candidate_rows = []
    for candidate_index, candidate in enumerate(candidates):
        repeated_results = [
            _rollout_candidate(
                env,
                policy,
                snapshot,
                stage=stage,
                candidate=candidate,
                execution_horizon=execution_horizon,
                bridge_steps=bridge_steps,
                continuation_seeds=continuation_seeds,
                stage_dwell_steps=stage_dwell_steps,
            )
            for continuation_seeds in selection_seed_rows
        ]
        candidate_rows.append(
            {
                "candidate_index": candidate_index,
                "sample_seed": int(sample_seeds[candidate_index]),
                "action_prefix": np.asarray(candidate[:execution_horizon], dtype=np.float32).round(7).tolist(),
                "direct": repeated_results[0]["direct"],
                "selection_summary": aggregate_outcomes([result["bridge"] for result in repeated_results]),
                "selection_continuations": [
                    {
                        "repeat": repeat,
                        "bridge": result["bridge"],
                        "continuation_policy_calls": result["continuation_policy_calls"],
                        "continuation_actions": result["continuation_actions"],
                    }
                    for repeat, result in enumerate(repeated_results)
                ],
            }
        )

    oracle_index = max(
        range(len(candidate_rows)),
        key=lambda index: aggregate_preference_key(candidate_rows[index]["selection_summary"]),
    )
    oracle_replay = _rollout_candidate(
        env,
        policy,
        snapshot,
        stage=stage,
        candidate=candidates[oracle_index],
        execution_horizon=execution_horizon,
        bridge_steps=bridge_steps,
        continuation_seeds=selection_seed_rows[0],
        stage_dwell_steps=stage_dwell_steps,
    )
    replay_match = replay_semantics(oracle_replay) == replay_semantics(
        candidate_rows[oracle_index]["selection_continuations"][0]
    )
    heldout_continuations = []
    heldout_seed_rows = []
    for repeat in range(validation_continuations):
        heldout_seeds = [
            stable_seed(seed, group["pair_id"], stage, "heldout_continuation", repeat, index)
            for index in range(continuation_calls)
        ]
        sample0_heldout = _rollout_candidate(
            env,
            policy,
            snapshot,
            stage=stage,
            candidate=candidates[0],
            execution_horizon=execution_horizon,
            bridge_steps=bridge_steps,
            continuation_seeds=heldout_seeds,
            stage_dwell_steps=stage_dwell_steps,
        )
        oracle_heldout = _rollout_candidate(
            env,
            policy,
            snapshot,
            stage=stage,
            candidate=candidates[oracle_index],
            execution_horizon=execution_horizon,
            bridge_steps=bridge_steps,
            continuation_seeds=heldout_seeds,
            stage_dwell_steps=stage_dwell_steps,
        )
        heldout_continuations.append(
            {
                "repeat": repeat,
                "sample0": sample0_heldout,
                "oracle": oracle_heldout,
            }
        )
        heldout_seed_rows.append(heldout_seeds)
    policy_input = _policy_observation(observation)
    row = {
        "pair_id": group["pair_id"],
        "split": group["split"],
        "source_initial_state_index": group.get("source_initial_state_index"),
        "stage": stage,
        "state_index": state_index,
        "eligible": True,
        **state_metadata,
        "seed": seed,
        "sample_count": sample_count,
        "execution_horizon": execution_horizon,
        "bridge_steps": bridge_steps,
        "stage_dwell_steps": stage_dwell_steps,
        "selection_continuation_count": selection_continuations,
        "candidate_inference_wall_seconds": inference_wall,
        "candidate_server_wall_seconds": server_wall,
        "oracle_index": oracle_index,
        "oracle_replay_match": replay_match,
        "heldout_continuations": heldout_continuations,
        "unique_outcome_signatures": len(
            {aggregate_outcome_signature(candidate["selection_summary"]) for candidate in candidate_rows}
        ),
        "candidates": candidate_rows,
    }
    training_bank = {
        "images": np.stack(policy_input["image"]).astype(np.uint8),
        "robot_state": np.asarray(policy_input["state"], dtype=np.float32),
        "candidate_action_prefix": np.asarray(candidates[:, :execution_horizon], dtype=np.float32),
    }
    audit_bank = {
        "candidate_actions": np.asarray(candidates, dtype=np.float32),
        "candidate_seeds": np.asarray(sample_seeds, dtype=np.uint32),
        "selection_continuation_seeds": np.asarray(selection_seed_rows, dtype=np.uint32),
        "heldout_continuation_seeds": np.asarray(heldout_seed_rows, dtype=np.uint32),
        "snapshot_sim_state": np.asarray(snapshot["sim_state"], dtype=np.float64),
        "snapshot_model_body_pos": np.asarray(snapshot["controller_state"]["model_body_pos"], dtype=np.float64),
        "snapshot_object_friction": np.asarray(
            snapshot["controller_state"]["object_friction"], dtype=np.float64
        ),
        "snapshot_gripper_action": np.asarray(
            snapshot["controller_state"]["gripper_action"], dtype=np.float64
        ),
    }
    return row, training_bank, audit_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Same-state physical consequence Oracle pilot for Pi0.5 recovery")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-socket", type=Path)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--audit-bank-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--group-offset", type=int, default=0)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--execution-horizon", type=int, default=3)
    parser.add_argument("--bridge-steps", type=int)
    parser.add_argument("--feedback-bridge-steps", type=int, default=120)
    parser.add_argument("--post-regrasp-bridge-steps", type=int, default=60)
    parser.add_argument("--post-regrasp-source", choices=("policy", "expert"), default="policy")
    parser.add_argument("--state-generation-max-steps", type=int, default=200)
    parser.add_argument("--stable-grasp-steps", type=int, default=2)
    parser.add_argument("--stage-dwell-steps", type=int, default=2)
    parser.add_argument("--selection-continuations", type=int, default=3)
    parser.add_argument("--validation-continuations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if (args.checkpoint is None) == (args.policy_socket is None):
        raise ValueError("provide exactly one of --checkpoint or --policy-socket")
    if args.sample_count < 2:
        raise ValueError("sample-count must be at least two")
    if args.execution_horizon < 1:
        raise ValueError("execution-horizon must be positive")
    stage_bridge_steps = {
        "feedback": args.bridge_steps or args.feedback_bridge_steps,
        "post_regrasp": args.bridge_steps or args.post_regrasp_bridge_steps,
    }
    if any(value < 1 for value in stage_bridge_steps.values()):
        raise ValueError("bridge steps must be positive")
    if args.state_generation_max_steps < 1 or args.stable_grasp_steps < 1 or args.stage_dwell_steps < 1:
        raise ValueError("state generation limits must be positive")
    if args.selection_continuations < 1:
        raise ValueError("selection-continuations must be positive")
    if args.validation_continuations < 0:
        raise ValueError("validation-continuations must be non-negative")
    os.environ.setdefault("PRETRAINED_MODELS_DIR", "/share/longjunyu/alphabrain/pretrained_models")

    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    groups = sorted(
        (group for group in manifest["groups"] if group["split"] == args.split),
        key=lambda row: row["pair_id"],
    )
    if args.group_offset < 0:
        raise ValueError("group-offset must be non-negative")
    groups = groups[args.group_offset :]
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    if not groups:
        raise ValueError(f"no groups for split={args.split!r}")

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
    rows = []
    partial = args.output.with_name(f"{args.output.stem}.partial{args.output.suffix}")
    expected_rows = len(groups) * len(args.stages)

    def payload(status: str) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "status": status,
            "episode_root": str(args.episode_root),
            "split": args.split,
            "group_offset": args.group_offset,
            "seed": args.seed,
            "sample_count": args.sample_count,
            "execution_horizon": args.execution_horizon,
            "bridge_steps_by_stage": stage_bridge_steps,
            "post_regrasp_source": args.post_regrasp_source,
            "state_generation_max_steps": args.state_generation_max_steps,
            "stable_grasp_steps": args.stable_grasp_steps,
            "stage_dwell_steps": args.stage_dwell_steps,
            "selection_continuations": args.selection_continuations,
            "validation_continuations": args.validation_continuations,
            "training_bank_policy_input_fields": ["images", "robot_state", "candidate_action_prefix"],
            "privileged_audit_bank_is_separate": True,
            "stages": list(args.stages),
            "git_sha": os.environ.get("FRESH_GIT_SHA"),
            "git_dirty_at_launch": os.environ.get("FRESH_GIT_DIRTY") == "1",
            "policy_checkpoint_sha256": os.environ.get("FRESH_CHECKPOINT_SHA256"),
            "policy_checkpoint_sha256_source": os.environ.get("FRESH_CHECKPOINT_SHA256_SOURCE"),
            "policy_checkpoint_realpath": getattr(policy, "checkpoint_realpath", None),
            "policy_model_size_bytes": getattr(policy, "model_size_bytes", None),
            "policy_runtime": getattr(policy, "runtime_identity", None),
            "completed_rows": len(rows),
            "expected_rows": expected_rows,
            "summary": summarize_rows(rows),
            "rows": rows,
        }

    try:
        for group in groups:
            reference = _load_reference_arrays(args.episode_root, group, "slipped")
            stage_indices = find_stage_indices(reference, int(group["feedback_reveal_time"]))
            _restore_recorded_state(env, reference, stage_indices["feedback"])
            feedback_snapshot = capture_runtime_snapshot(env)
            for stage in args.stages:
                if stage == "feedback":
                    snapshot = feedback_snapshot
                    state_index = stage_indices[stage]
                    state_metadata = {
                        "state_source": "recorded_slip_feedback",
                        "next_policy_replan_index": 0,
                    }
                elif args.post_regrasp_source == "expert":
                    _restore_recorded_state(env, reference, stage_indices[stage])
                    snapshot = capture_runtime_snapshot(env)
                    state_index = stage_indices[stage]
                    state_metadata = {
                        "state_source": "scripted_expert_post_regrasp_control",
                        "next_policy_replan_index": 0,
                    }
                else:
                    snapshot, _, state_metadata = generate_policy_post_regrasp_state(
                        env,
                        policy,
                        feedback_snapshot,
                        pair_id=str(group["pair_id"]),
                        seed=args.seed,
                        execution_horizon=args.execution_horizon,
                        max_steps=args.state_generation_max_steps,
                        stable_grasp_steps=args.stable_grasp_steps,
                    )
                    state_index = None
                    if snapshot is None:
                        rows.append(
                            {
                                "pair_id": group["pair_id"],
                                "split": group["split"],
                                "source_initial_state_index": group.get("source_initial_state_index"),
                                "stage": stage,
                                "state_index": None,
                                **state_metadata,
                            }
                        )
                        _atomic_write_json(partial, payload("partial"))
                        print(
                            json.dumps(
                                {
                                    "pair_id": group["pair_id"],
                                    "stage": stage,
                                    "eligible": False,
                                    "reason": state_metadata["reason"],
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        continue

                row, training_bank, audit_bank = evaluate_state(
                    env,
                    policy,
                    snapshot,
                    group,
                    stage=stage,
                    state_index=state_index,
                    state_metadata=state_metadata,
                    seed=args.seed,
                    sample_count=args.sample_count,
                    execution_horizon=args.execution_horizon,
                    bridge_steps=stage_bridge_steps[stage],
                    selection_continuations=args.selection_continuations,
                    validation_continuations=args.validation_continuations,
                    stage_dwell_steps=args.stage_dwell_steps,
                )
                rows.append(row)
                bank_path = args.bank_dir / f"{group['pair_id']}--{stage}--seed{args.seed}.npz"
                audit_bank_path = args.audit_bank_dir / f"{group['pair_id']}--{stage}--seed{args.seed}.npz"
                np.savez_compressed(bank_path, **training_bank)
                np.savez_compressed(audit_bank_path, **audit_bank)
                row["training_bank_file"] = str(bank_path)
                row["privileged_audit_bank_file"] = str(audit_bank_path)
                _atomic_write_json(partial, payload("partial"))
                print(
                    json.dumps(
                        {
                            "pair_id": row["pair_id"],
                            "stage": stage,
                            "oracle_index": row["oracle_index"],
                            "unique_outcomes": row["unique_outcome_signatures"],
                            "sample0_next_stage_rate": row["candidates"][0]["selection_summary"][
                                "next_stage_reached_rate"
                            ],
                            "oracle_next_stage_rate": row["candidates"][row["oracle_index"]][
                                "selection_summary"
                            ]["next_stage_reached_rate"],
                            "heldout_success_delta": (
                                float(
                                    np.mean(
                                        [
                                            entry["oracle"]["bridge"]["success"]
                                            for entry in row["heldout_continuations"]
                                        ]
                                    )
                                )
                                - float(
                                    np.mean(
                                        [
                                            entry["sample0"]["bridge"]["success"]
                                            for entry in row["heldout_continuations"]
                                        ]
                                    )
                                )
                                if row["heldout_continuations"]
                                else None
                            ),
                            "oracle_replay_match": row["oracle_replay_match"],
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
