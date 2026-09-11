from __future__ import annotations

import argparse
import json
import time
import multiprocessing as mp
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from evaluate_libero_closed_loop import (
    Pi05Policy,
    RemotePi05Policy,
    _atomic_write_json,
    _load_reference_arrays,
    _policy_observation,
    _restore_recorded_state,
    is_failure_continuation,
    is_premature_commitment,
    is_recovery_action,
    stable_seed,
)
from evaluate_physical_process_oracle import (
    _physical_state,
    capture_runtime_snapshot,
    restore_runtime_snapshot,
)
from evaluate_recovery_segment_oracle import (
    _observation_frame,
    generate_candidate_endpoint,
    recovery_preference_key,
    summarize_recovery_trace,
)
from libero_full_episode_collector import FullEpisodeTeacher, object_grasped
from libero_snapshot_collector import DEFAULT_BDDL, _step
from video_io import write_h264_video


METHODS = (
    "single_sample",
    "random_pick_N",
    "self_consistency_pick",
    "oracle_teacher_distance",
    "oracle_short_physical",
    "oracle_policy_continuation",
)
FORMAL_CONFIG = {
    "split": "val",
    "candidate_count": 16,
    "execution_horizon": 2,
    "max_actions": 320,
    "lookahead_actions": 8,
    "continuation_repeats": 2,
}


def _minimal_policy_observation(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(observation[key])
        for key in (
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        )
    }


def _branch_worker(connection: Any, env_kwargs: Mapping[str, Any], seed: int) -> None:
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(**dict(env_kwargs))
    env.seed(seed)
    endpoint_snapshot = None
    endpoint_trace = None
    trace = None
    current_observation = None
    try:
        while True:
            request = connection.recv()
            op = request["op"]
            if op == "close":
                break
            if op == "prepare":
                endpoint = generate_candidate_endpoint(
                    env,
                    request["snapshot"],
                    request["prefix_trace"],
                    candidate=request["candidate"],
                    execution_horizon=request["execution_horizon"],
                    stage_dwell_steps=2,
                )
                endpoint_snapshot = endpoint["endpoint_snapshot"]
                endpoint_trace = endpoint["trace"]
                connection.send(
                    {
                        "direct": endpoint["direct"],
                        "end_grasped": bool(endpoint_trace[-1]["grasped"]),
                        "executed_actions": int(endpoint["candidate_actions"]),
                    }
                )
            elif op == "teacher":
                if endpoint_snapshot is None:
                    raise RuntimeError("worker endpoint is not prepared")
                connection.send(teacher_recoverable(env, endpoint_snapshot, request["max_steps"]))
            elif op == "reset_continuation":
                if endpoint_snapshot is None or endpoint_trace is None:
                    raise RuntimeError("worker endpoint is not prepared")
                current_observation = restore_runtime_snapshot(env, endpoint_snapshot)
                trace = [dict(state) for state in endpoint_trace]
                connection.send(_minimal_policy_observation(current_observation))
            elif op == "advance":
                if trace is None or current_observation is None:
                    raise RuntimeError("worker continuation is not reset")
                executed = 0
                for action in request["chunk"][: request["execution_horizon"]]:
                    if request["actions_done"] + executed >= request["lookahead_actions"]:
                        break
                    if bool(env.check_success()):
                        break
                    current_observation = _step(env, action)
                    executed += 1
                    trace.append(_physical_state(env, current_observation, bool(env.check_success())))
                connection.send(
                    {
                        "observation": _minimal_policy_observation(current_observation),
                        "executed": executed,
                    }
                )
            elif op == "summary":
                if trace is None:
                    raise RuntimeError("worker continuation is not reset")
                connection.send(summarize_recovery_trace(trace, stage_dwell_steps=2))
            else:
                raise ValueError(f"unknown branch worker operation: {op}")
    finally:
        env.close()
        connection.close()


class ParallelBranchPool:
    def __init__(self, count: int, env_kwargs: Mapping[str, Any], seed: int) -> None:
        context = mp.get_context("spawn")
        self.connections = []
        self.processes = []
        for _ in range(count):
            parent, child = context.Pipe()
            process = context.Process(target=_branch_worker, args=(child, dict(env_kwargs), seed))
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)

    def prepare(
        self,
        snapshot: Mapping[str, Any],
        prefix_trace: Sequence[Mapping[str, Any]],
        candidates: np.ndarray,
        execution_horizon: int,
    ) -> list[dict[str, Any]]:
        for connection, candidate in zip(self.connections, candidates):
            connection.send(
                {
                    "op": "prepare",
                    "snapshot": snapshot,
                    "prefix_trace": prefix_trace,
                    "candidate": np.asarray(candidate),
                    "execution_horizon": execution_horizon,
                }
            )
        return [connection.recv() for connection in self.connections[: len(candidates)]]

    def teacher(self, indices: Sequence[int], max_steps: int) -> list[tuple[int, bool, int]]:
        for index in indices:
            self.connections[index].send({"op": "teacher", "max_steps": max_steps})
        return [
            (index, *self.connections[index].recv())
            for index in indices
        ]

    def reset_continuation(self, count: int) -> list[Mapping[str, Any]]:
        for connection in self.connections[:count]:
            connection.send({"op": "reset_continuation"})
        return [connection.recv() for connection in self.connections[:count]]

    def advance(
        self,
        chunks: np.ndarray,
        actions_done: Sequence[int],
        execution_horizon: int,
        lookahead_actions: int,
    ) -> list[dict[str, Any]]:
        for index, (connection, chunk) in enumerate(zip(self.connections, chunks)):
            connection.send(
                {
                    "op": "advance",
                    "chunk": np.asarray(chunk),
                    "actions_done": int(actions_done[index]),
                    "execution_horizon": execution_horizon,
                    "lookahead_actions": lookahead_actions,
                }
            )
        return [connection.recv() for connection in self.connections[: len(chunks)]]

    def summaries(self, count: int) -> list[dict[str, Any]]:
        for connection in self.connections[:count]:
            connection.send({"op": "summary"})
        return [connection.recv() for connection in self.connections[:count]]

    def close(self) -> None:
        for connection in self.connections:
            try:
                connection.send({"op": "close"})
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join()
        for connection in self.connections:
            connection.close()


def candidate_pool_seed(seed: int, pair_id: str, outcome: str, replan: int) -> int:
    return stable_seed("cora-sequential", seed, pair_id, outcome, replan)


def self_consistency_index(candidates: np.ndarray, execution_horizon: int) -> int:
    prefixes = np.asarray(candidates[:, :execution_horizon], dtype=np.float64).reshape(len(candidates), -1)
    distances = np.linalg.norm(prefixes[:, None] - prefixes[None, :], axis=-1)
    return int(np.argmin(distances.mean(axis=1)))


def teacher_actions(
    env: Any,
    snapshot: Mapping[str, Any],
    execution_horizon: int,
) -> np.ndarray:
    observation = restore_runtime_snapshot(env, snapshot)
    teacher = FullEpisodeTeacher(observation)
    actions = []
    for _ in range(execution_horizon):
        decision = teacher.decide(
            observation,
            grasped=object_grasped(env),
            success=bool(env.check_success()),
        )
        action = np.asarray(decision.action, dtype=np.float32)
        actions.append(action)
        observation = _step(env, action)
    return np.stack(actions)


def teacher_recoverable(
    env: Any,
    endpoint_snapshot: Mapping[str, Any],
    max_steps: int,
) -> tuple[bool, int]:
    observation = restore_runtime_snapshot(env, endpoint_snapshot)
    teacher = FullEpisodeTeacher(observation)
    steps = 0
    success = bool(env.check_success())
    while not success and not teacher.done and steps < max_steps:
        decision = teacher.decide(
            observation,
            grasped=object_grasped(env),
            success=success,
        )
        observation = _step(env, decision.action)
        steps += 1
        success = bool(env.check_success())
    return success, steps


def immediate_correct(
    candidate: np.ndarray,
    *,
    starts_grasped: bool,
    end_grasped: bool,
    initial_observation: Mapping[str, Any],
) -> tuple[bool, bool, bool, bool]:
    failure = any(
        is_failure_continuation(action, grasped=starts_grasped)
        for action in candidate
    )
    premature = any(
        is_premature_commitment(
            action,
            grasped=starts_grasped,
            eef_position=initial_observation["robot0_eef_pos"],
            bowl_position=initial_observation["akita_black_bowl_1_pos"],
        )
        for action in candidate
    )
    recovery = any(
        is_recovery_action(
            action,
            grasped=starts_grasped,
            eef_position=initial_observation["robot0_eef_pos"],
            object_position=initial_observation["cream_cheese_1_pos"],
        )
        for action in candidate
    )
    if starts_grasped:
        correct = end_grasped and not any(action[-1] < -0.2 for action in candidate)
    else:
        correct = (not failure) and (not premature) and (recovery or end_grasped)
    return bool(correct), bool(failure), bool(premature), bool(recovery)


def physical_candidate_rows(
    branch_pool: ParallelBranchPool,
    snapshot: Mapping[str, Any],
    observation: Mapping[str, Any],
    prefix_trace: Sequence[Mapping[str, Any]],
    candidates: np.ndarray,
    *,
    execution_horizon: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    endpoints = branch_pool.prepare(snapshot, prefix_trace, candidates, execution_horizon)
    starts_grasped = bool(prefix_trace[-1]["grasped"])
    rows = []
    for candidate, endpoint in zip(candidates, endpoints):
        end_grasped = bool(endpoint["end_grasped"])
        correct, failure, premature, recovery = immediate_correct(
            candidate[:execution_horizon],
            starts_grasped=starts_grasped,
            end_grasped=end_grasped,
            initial_observation=observation,
        )
        rows.append(
            {
                "immediate_correct": correct,
                "failure_continuation": failure,
                "premature_commitment": premature,
                "recovery_action": recovery,
                "teacher_recoverable": None,
                "teacher_completion_steps": None,
                "direct": endpoint["direct"],
            }
        )
    return rows, endpoints


def direct_selection_key(row: Mapping[str, Any]) -> tuple[float | int, ...]:
    return (int(not bool(row["direct"]["regress"])), *recovery_preference_key(row["direct"]))


def lazy_teacher_selection(
    branch_pool: ParallelBranchPool,
    rows: list[dict[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
    teacher_max_steps: int,
) -> tuple[int, int]:
    best_immediate = max(int(bool(row["immediate_correct"])) for row in rows)
    eligible = [index for index, row in enumerate(rows) if int(bool(row["immediate_correct"])) == best_immediate]
    eligible.sort(key=lambda index: direct_selection_key(rows[index]), reverse=True)
    teacher_steps = 0
    cursor = 0
    while cursor < len(eligible):
        tier_key = direct_selection_key(rows[eligible[cursor]])
        tier = []
        while cursor < len(eligible) and direct_selection_key(rows[eligible[cursor]]) == tier_key:
            tier.append(eligible[cursor])
            cursor += 1
        recoverable = []
        for index, success, steps in branch_pool.teacher(tier, teacher_max_steps):
            rows[index]["teacher_recoverable"] = success
            rows[index]["teacher_completion_steps"] = steps
            teacher_steps += steps
            if success:
                recoverable.append(index)
        if recoverable:
            return min(recoverable, key=lambda index: int(rows[index]["teacher_completion_steps"])), teacher_steps
    return eligible[0], teacher_steps


def continuation_selection_key(rows: Sequence[Mapping[str, Any]]) -> tuple[float, ...]:
    return (
        float(np.mean([row["success"] for row in rows])),
        float(np.mean([not row["regress"] for row in rows])),
        float(np.mean([row["transport_reached"] for row in rows])),
        float(np.mean([row["lift_reached"] for row in rows])),
        float(np.mean([row["stable_grasp_at_end"] for row in rows])),
        float(np.mean([row["progress_auc"] for row in rows])),
    )


def batched_policy_continuations(
    branch_pool: ParallelBranchPool,
    policy: Pi05Policy | RemotePi05Policy,
    endpoints: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    pair_id: str,
    outcome: str,
    replan_index: int,
    execution_horizon: int,
    lookahead_actions: int,
    repeats: int,
) -> tuple[list[list[dict[str, Any]]], int, int]:
    all_results: list[list[dict[str, Any]]] = [[] for _ in endpoints]
    batch_calls = 0
    simulator_actions = 0
    for repeat in range(repeats):
        actions_done = [0] * len(endpoints)
        observations = branch_pool.reset_continuation(len(endpoints))
        continuation_replans = int(np.ceil(lookahead_actions / execution_horizon))
        for continuation_replan in range(continuation_replans):
            chunks, _ = policy.predict_observation_batch(
                observations,
                seed=stable_seed(
                    seed,
                    pair_id,
                    outcome,
                    "matched-continuation",
                    replan_index,
                    repeat,
                    continuation_replan,
                ),
            )
            batch_calls += 1
            advanced = branch_pool.advance(
                chunks, actions_done, execution_horizon, lookahead_actions
            )
            observations = [row["observation"] for row in advanced]
            for index, row in enumerate(advanced):
                actions_done[index] += int(row["executed"])
                simulator_actions += int(row["executed"])
        for index, summary in enumerate(branch_pool.summaries(len(endpoints))):
            all_results[index].append(summary)
    return all_results, batch_calls, simulator_actions


def select_candidate(
    method: str,
    teacher_env: Any,
    branch_pool: Optional[ParallelBranchPool],
    policy: Pi05Policy | RemotePi05Policy,
    snapshot: Mapping[str, Any],
    observation: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
    candidates: np.ndarray,
    *,
    seed: int,
    pair_id: str,
    outcome: str,
    replan_index: int,
    execution_horizon: int,
    lookahead_actions: int,
    continuation_repeats: int,
    teacher_max_steps: int,
) -> tuple[int, dict[str, Any]]:
    details: dict[str, Any] = {"search_simulator_actions": 0, "search_policy_batch_calls": 0}
    if method == "single_sample":
        return 0, details
    if method == "random_pick_N":
        return int(stable_seed(seed, pair_id, outcome, "random", replan_index) % len(candidates)), details
    if method == "self_consistency_pick":
        return self_consistency_index(candidates, execution_horizon), details
    if method == "oracle_teacher_distance":
        target = teacher_actions(teacher_env, snapshot, execution_horizon)
        rmse = np.sqrt(np.square(candidates[:, :execution_horizon] - target[None]).mean(axis=(1, 2)))
        details["teacher_rmse"] = rmse.round(7).tolist()
        details["search_simulator_actions"] = execution_horizon
        return int(np.argmin(rmse)), details

    if branch_pool is None:
        raise RuntimeError(f"{method} requires a parallel branch pool")
    rows, endpoints = physical_candidate_rows(
        branch_pool,
        snapshot,
        observation,
        trace,
        candidates,
        execution_horizon=execution_horizon,
    )
    details["candidate_immediate_correct"] = [bool(row["immediate_correct"]) for row in rows]
    details["search_simulator_actions"] = int(len(candidates) * execution_horizon)
    if method == "oracle_short_physical":
        selected, teacher_steps = lazy_teacher_selection(
            branch_pool, rows, endpoints, teacher_max_steps
        )
        details["search_simulator_actions"] += teacher_steps
        details["candidate_teacher_recoverable"] = [row["teacher_recoverable"] for row in rows]
        details["candidate_teacher_completion_steps"] = [
            row["teacher_completion_steps"] for row in rows
        ]
        details["teacher_candidates_evaluated"] = sum(
            row["teacher_recoverable"] is not None for row in rows
        )
        return selected, details
    if method != "oracle_policy_continuation":
        raise ValueError(f"unknown method: {method}")
    continuation_rows, batch_calls, continuation_actions = batched_policy_continuations(
        branch_pool,
        policy,
        endpoints,
        seed=seed,
        pair_id=pair_id,
        outcome=outcome,
        replan_index=replan_index,
        execution_horizon=execution_horizon,
        lookahead_actions=lookahead_actions,
        repeats=continuation_repeats,
    )
    details["search_policy_batch_calls"] = batch_calls
    details["search_simulator_actions"] += continuation_actions
    details["continuation_keys"] = [
        [round(value, 7) for value in continuation_selection_key(candidate_rows)]
        for candidate_rows in continuation_rows
    ]
    return max(
        range(len(continuation_rows)),
        key=lambda index: continuation_selection_key(continuation_rows[index]),
    ), details


def run_method(
    env: Any,
    teacher_env: Any,
    branch_pool: Optional[ParallelBranchPool],
    policy: Pi05Policy | RemotePi05Policy,
    feedback_snapshot: Mapping[str, Any],
    *,
    method: str,
    pair_id: str,
    outcome: str,
    seed: int,
    candidate_count: int,
    execution_horizon: int,
    max_actions: int,
    lookahead_actions: int,
    continuation_repeats: int,
    teacher_max_steps: int,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    observation = restore_runtime_snapshot(env, feedback_snapshot)
    trace = [_physical_state(env, observation, bool(env.check_success()))]
    frames = [_observation_frame(observation)]
    starts_grasped = bool(trace[0]["grasped"])
    decisions = []
    actions_executed = 0
    failure_continuation_seen = False
    premature_commitment_seen = False
    first_recovery_action = None
    cost = {
        "candidate_samples": 0,
        "candidate_batch_calls": 0,
        "search_policy_batch_calls": 0,
        "search_simulator_actions": 0,
        "live_simulator_actions": 0,
    }
    started = time.perf_counter()
    replan_index = 0
    while actions_executed < max_actions and not bool(env.check_success()):
        count = 1 if method == "single_sample" else candidate_count
        pool_seed = candidate_pool_seed(seed, pair_id, outcome, replan_index)
        candidates, inference_seconds = policy.predict_sample_batch(
            observation, count=count, seed=pool_seed
        )
        cost["candidate_samples"] += count
        cost["candidate_batch_calls"] += 1
        snapshot = capture_runtime_snapshot(env)
        starts_grasped_now = bool(trace[-1]["grasped"])
        heuristic_correct = [
            immediate_correct(
                candidate[:execution_horizon],
                starts_grasped=starts_grasped_now,
                end_grasped=starts_grasped_now,
                initial_observation=observation,
            )[0]
            for candidate in candidates
        ]
        selected, details = select_candidate(
            method,
            teacher_env,
            branch_pool,
            policy,
            snapshot,
            observation,
            trace,
            candidates,
            seed=seed,
            pair_id=pair_id,
            outcome=outcome,
            replan_index=replan_index,
            execution_horizon=execution_horizon,
            lookahead_actions=lookahead_actions,
            continuation_repeats=continuation_repeats,
            teacher_max_steps=teacher_max_steps,
        )
        cost["search_policy_batch_calls"] += int(details.pop("search_policy_batch_calls"))
        cost["search_simulator_actions"] += int(details.pop("search_simulator_actions"))
        decision = {
            "replan_index": replan_index,
            "pool_seed": int(pool_seed),
            "candidate_count": count,
            "selected_index": int(selected),
            "selected_immediate_correct_preexecution": bool(heuristic_correct[selected]),
            "pool_immediate_correct_count": int(sum(heuristic_correct)),
            "pool_immediate_correct_rate": float(np.mean(heuristic_correct)),
            "candidate_inference_wall_seconds": inference_seconds,
            **details,
        }
        for action in candidates[selected, :execution_horizon]:
            if actions_executed >= max_actions or bool(env.check_success()):
                break
            grasped = object_grasped(env)
            failure = is_failure_continuation(action, grasped=grasped)
            premature = is_premature_commitment(
                action,
                grasped=grasped,
                eef_position=observation["robot0_eef_pos"],
                bowl_position=observation["akita_black_bowl_1_pos"],
            )
            recovery = is_recovery_action(
                action,
                grasped=grasped,
                eef_position=observation["robot0_eef_pos"],
                object_position=observation["cream_cheese_1_pos"],
            )
            failure_continuation_seen = failure_continuation_seen or failure
            premature_commitment_seen = premature_commitment_seen or premature
            if recovery and first_recovery_action is None:
                first_recovery_action = actions_executed
            observation = _step(env, action)
            actions_executed += 1
            cost["live_simulator_actions"] += 1
            frames.append(_observation_frame(observation))
            trace.append(_physical_state(env, observation, bool(env.check_success())))
        decisions.append(decision)
        replan_index += 1

    summary = summarize_recovery_trace(
        trace,
        stage_dwell_steps=2,
    )
    result = {
        "method": method,
        "outcome": outcome,
        "success": bool(env.check_success()),
        "starts_grasped": starts_grasped,
        "actions": actions_executed,
        "replans": replan_index,
        "failure_continuation": failure_continuation_seen,
        "premature_commitment": premature_commitment_seen,
        "recovery_switch_delay_actions": first_recovery_action,
        "stable_regrasp": bool(summary["regrasp_reached"]),
        "lift_reached": bool(summary["lift_reached"]),
        "transport_reached": bool(summary["transport_reached"]),
        "drop": bool(summary["drop"]),
        "progress_auc": float(summary["progress_auc"]),
        "final_object_z_delta": float(summary["final_object_z_delta"]),
        "final_bowl_xy_delta": float(summary["final_bowl_xy_delta"]),
        "wall_seconds": time.perf_counter() - started,
        "cost": cost,
        "decisions": decisions,
    }
    return result, frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CORA frozen Full-H sequential routing upper bound")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-socket", type=Path)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--outcome", choices=("attached", "slipped"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--group-offset", type=int, default=0)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--execution-horizon", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=320)
    parser.add_argument("--lookahead-actions", type=int, default=8)
    parser.add_argument("--continuation-repeats", type=int, default=2)
    parser.add_argument("--teacher-max-steps", type=int, default=320)
    parser.add_argument("--run-kind", choices=("smoke", "formal"), required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, manifest: Mapping[str, Any]) -> None:
    if "confirmation" in str(args.episode_root).lower():
        raise ValueError("confirmation groups are sealed for the final formal gate")
    if args.run_kind == "formal":
        for field, expected in FORMAL_CONFIG.items():
            if getattr(args, field) != expected:
                raise ValueError(f"formal run requires {field}={expected!r}")
        groups = [group for group in manifest["groups"] if group["split"] == "val"]
        if len(groups) != 13:
            raise ValueError("formal run requires the frozen 13-group validation manifest")
        if args.seed not in {41, 42, 43}:
            raise ValueError("formal seed must be 41, 42, or 43")


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if (args.checkpoint is None) == (args.policy_socket is None):
        raise ValueError("provide exactly one of --checkpoint or --policy-socket")
    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    validate_args(args, manifest)
    selected_groups = sorted(
        [group for group in manifest["groups"] if group["split"] == args.split],
        key=lambda group: group["pair_id"],
    )[args.group_offset :]
    if args.max_groups is not None:
        selected_groups = selected_groups[: args.max_groups]
    if not selected_groups:
        raise ValueError("selected shard contains no groups")

    policy = RemotePi05Policy(args.policy_socket) if args.policy_socket else Pi05Policy(args.checkpoint, args.device)
    env_kwargs = {
        "bddl_file_name": str(Path(manifest.get("bddl", DEFAULT_BDDL))),
        "camera_heights": 224,
        "camera_widths": 224,
    }
    env = OffScreenRenderEnv(**env_kwargs)
    teacher_env = OffScreenRenderEnv(**env_kwargs)
    env.seed(args.seed)
    teacher_env.seed(args.seed)
    branch_pool = (
        ParallelBranchPool(args.candidate_count, env_kwargs, args.seed)
        if args.method in {"oracle_short_physical", "oracle_policy_continuation"}
        else None
    )
    args.video_dir.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(f"{args.output.stem}.partial{args.output.suffix}")
    rows = []
    if partial.exists():
        previous = json.loads(partial.read_text())
        identity = {
            "experiment": "cora_sequential_oracle",
            "run_kind": args.run_kind,
            "seed": args.seed,
            "outcome": args.outcome,
            "method": args.method,
            "split": args.split,
            "candidate_count": args.candidate_count,
            "execution_horizon": args.execution_horizon,
            "max_actions": args.max_actions,
            "lookahead_actions": args.lookahead_actions,
            "continuation_repeats": args.continuation_repeats,
        }
        mismatched = [key for key, value in identity.items() if previous.get(key) != value]
        if mismatched:
            raise ValueError(f"partial resume identity mismatch: {mismatched}")
        rows = list(previous["rows"])
    completed_pair_ids = {str(row["pair_id"]) for row in rows}
    if len(completed_pair_ids) != len(rows):
        raise ValueError("partial resume contains duplicate pair ids")
    groups = [group for group in selected_groups if str(group["pair_id"]) not in completed_pair_ids]

    def payload(status: str) -> dict[str, Any]:
        return {
            "status": status,
            "experiment": "cora_sequential_oracle",
            "run_kind": args.run_kind,
            "seed": args.seed,
            "outcome": args.outcome,
            "method": args.method,
            "split": args.split,
            "candidate_count": args.candidate_count,
            "execution_horizon": args.execution_horizon,
            "max_actions": args.max_actions,
            "lookahead_actions": args.lookahead_actions,
            "continuation_repeats": args.continuation_repeats,
            "confirmation_groups_accessed": False,
            "video_encoding": {
                "container": "mp4",
                "codec": "h264",
                "codec_tag": "avc1",
                "pixel_format": "yuv420p",
                "faststart": True,
            },
            "completed_groups": len(rows),
            "expected_groups": len(selected_groups),
            "rows": rows,
        }

    try:
        for group in groups:
            pair_id = str(group["pair_id"])
            reference = _load_reference_arrays(args.episode_root, group, args.outcome)
            observation = _restore_recorded_state(
                env, reference, int(group["feedback_reveal_time"])
            )
            feedback_snapshot = capture_runtime_snapshot(env)
            result, frames = run_method(
                env,
                teacher_env,
                branch_pool,
                policy,
                feedback_snapshot,
                method=args.method,
                pair_id=pair_id,
                outcome=args.outcome,
                seed=args.seed,
                candidate_count=args.candidate_count,
                execution_horizon=args.execution_horizon,
                max_actions=args.max_actions,
                lookahead_actions=args.lookahead_actions,
                continuation_repeats=args.continuation_repeats,
                teacher_max_steps=args.teacher_max_steps,
            )
            video_path = args.video_dir / f"seed{args.seed}--{pair_id}--{args.outcome}--{args.method}.mp4"
            write_h264_video(video_path, frames, fps=10.0)
            result.update(
                {
                    "pair_id": pair_id,
                    "source_initial_state_index": int(group["source_initial_state_index"]),
                    "feedback_reveal_time": int(group["feedback_reveal_time"]),
                    "video_file": str(video_path),
                }
            )
            rows.append(result)
            _atomic_write_json(partial, payload("partial"))
            print(
                json.dumps(
                    {
                        "seed": args.seed,
                        "pair_id": pair_id,
                        "outcome": args.outcome,
                        "method": args.method,
                        "success": result["success"],
                        "actions": result["actions"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        env.close()
        if branch_pool is not None:
            branch_pool.close()
        teacher_env.close()
        policy.close()
    _atomic_write_json(args.output, payload("complete"))
    partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
