from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from candidate_support import stable_seed
from evaluate_candidate_support import atomic_json, sample_many
from evaluate_libero_closed_loop import (
    Pi05Policy,
    RemotePi05Policy,
    _load_reference_arrays,
    _restore_recorded_state,
    is_failure_continuation,
    is_premature_commitment,
    is_recovery_action,
)
from evaluate_physical_process_oracle import capture_runtime_snapshot, restore_runtime_snapshot
from libero_full_episode_collector import FullEpisodeTeacher, object_grasped
from libero_snapshot_collector import DEFAULT_BDDL, _step
from onpolicy_support import STAGES, classify_boundary_stage, immediate_correct_mode, recall_at_n


def save_state_cache(
    path: Path,
    snapshot: Mapping[str, Any],
    candidates: np.ndarray,
    candidate_seeds: Sequence[int],
) -> None:
    arrays = {
        "sim_state": np.asarray(snapshot["sim_state"], dtype=np.float64),
        "candidates": np.asarray(candidates, dtype=np.float32),
        "candidate_seeds": np.asarray(candidate_seeds, dtype=np.int64),
    }
    arrays.update(
        {f"controller__{key}": np.asarray(value) for key, value in snapshot["controller_state"].items()}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def teacher_reference(
    env: Any,
    snapshot: Mapping[str, Any],
    execution_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    observation = restore_runtime_snapshot(env, snapshot)
    initial_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
    teacher = FullEpisodeTeacher(observation)
    actions = []
    for _ in range(execution_horizon):
        decision = teacher.decide(
            observation,
            grasped=object_grasped(env),
            success=bool(env.check_success()),
        )
        actions.append(np.asarray(decision.action, dtype=np.float32))
        observation = _step(env, decision.action)
    effect = np.asarray(observation["robot0_eef_pos"], dtype=np.float64) - initial_eef
    return np.stack(actions), effect


def evaluate_candidate(
    env: Any,
    snapshot: Mapping[str, Any],
    candidate: np.ndarray,
    teacher_actions: np.ndarray,
    teacher_effect: np.ndarray,
    *,
    execution_horizon: int,
    teacher_max_steps: int,
) -> dict[str, Any]:
    observation = restore_runtime_snapshot(env, snapshot)
    initial_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
    starts_grasped = object_grasped(env)
    grasp_trace = []
    failure_seen = False
    premature_seen = False
    recovery_seen = False
    for action in candidate[:execution_horizon]:
        grasped = object_grasped(env)
        failure_seen = failure_seen or is_failure_continuation(action, grasped=grasped)
        premature_seen = premature_seen or is_premature_commitment(
            action,
            grasped=grasped,
            eef_position=observation["robot0_eef_pos"],
            bowl_position=observation["akita_black_bowl_1_pos"],
        )
        recovery_seen = recovery_seen or is_recovery_action(
            action,
            grasped=grasped,
            eef_position=observation["robot0_eef_pos"],
            object_position=observation["cream_cheese_1_pos"],
        )
        observation = _step(env, action)
        grasp_trace.append(object_grasped(env))
    effect = np.asarray(observation["robot0_eef_pos"], dtype=np.float64) - initial_eef
    correct = immediate_correct_mode(
        starts_grasped=starts_grasped,
        grasp_trace=grasp_trace,
        actions=candidate[:execution_horizon],
        failure_continuation_seen=failure_seen,
        premature_commitment_seen=premature_seen,
        recovery_action_seen=recovery_seen,
    )
    teacher = FullEpisodeTeacher(observation)
    steps = 0
    success = bool(env.check_success())
    while not success and not teacher.done and steps < teacher_max_steps:
        decision = teacher.decide(
            observation,
            grasped=object_grasped(env),
            success=success,
        )
        observation = _step(env, decision.action)
        steps += 1
        success = bool(env.check_success())
    return {
        "immediate_correct_mode": correct,
        "starts_grasped": starts_grasped,
        "grasp_preserved": bool(grasp_trace) and all(grasp_trace),
        "failure_continuation": failure_seen,
        "premature_commitment": premature_seen,
        "recovery_action_seen": recovery_seen,
        "teacher_recoverable": success,
        "teacher_completion_steps": steps,
        "teacher_action_rmse": float(
            np.sqrt(np.square(candidate[:execution_horizon] - teacher_actions).mean())
        ),
        "teacher_effect_rmse": float(np.sqrt(np.square(effect - teacher_effect).mean())),
    }


def evaluate_state(
    branch_env: Any,
    snapshot: Mapping[str, Any],
    candidates: np.ndarray,
    candidate_seeds: Sequence[int],
    *,
    pair_id: str,
    stage: str,
    replan_index: int,
    execution_horizon: int,
    teacher_max_steps: int,
    cache_dir: Path,
) -> dict[str, Any]:
    teacher_actions, teacher_effect = teacher_reference(branch_env, snapshot, execution_horizon)
    candidate_rows = [
        {
            "candidate_index": index,
            "candidate_seed": int(candidate_seeds[index]),
            **evaluate_candidate(
                branch_env,
                snapshot,
                candidate,
                teacher_actions,
                teacher_effect,
                execution_horizon=execution_horizon,
                teacher_max_steps=teacher_max_steps,
            ),
        }
        for index, candidate in enumerate(candidates)
    ]
    cache_file = cache_dir / f"{pair_id}--{stage}--r{replan_index}.npz"
    save_state_cache(cache_file, snapshot, candidates, candidate_seeds)
    return {
        "pair_id": pair_id,
        "stage": stage,
        "replan_index": replan_index,
        "eligible": True,
        "cache_file": str(cache_file),
        "immediate_recall": recall_at_n([row["immediate_correct_mode"] for row in candidate_rows]),
        "teacher_recoverable_recall": recall_at_n([row["teacher_recoverable"] for row in candidate_rows]),
        "candidate_rows": candidate_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CORA on-policy candidate-support Gate")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-socket", type=Path)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--group-offset", type=int, default=0)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--execution-horizon", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=320)
    parser.add_argument("--teacher-max-steps", type=int, default=320)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if args.candidate_count != 16 or args.execution_horizon != 2:
        raise ValueError("formal on-policy Gate requires N=16 and K=2")
    if (args.checkpoint is None) == (args.policy_socket is None):
        raise ValueError("provide exactly one of checkpoint or policy-socket")
    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    groups = sorted(
        [group for group in manifest["groups"] if group["split"] == args.split],
        key=lambda group: group["pair_id"],
    )[args.group_offset :]
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    policy = RemotePi05Policy(args.policy_socket) if args.policy_socket else Pi05Policy(args.checkpoint, args.device)
    env_kwargs = {
        "bddl_file_name": str(Path(manifest.get("bddl", DEFAULT_BDDL))),
        "camera_heights": 224,
        "camera_widths": 224,
    }
    live_env = OffScreenRenderEnv(**env_kwargs)
    branch_env = OffScreenRenderEnv(**env_kwargs)
    live_env.seed(args.seed); branch_env.seed(args.seed)
    rows = []
    episodes = []
    partial = args.output.with_name(f"{args.output.stem}.partial{args.output.suffix}")

    def payload(status: str) -> dict[str, Any]:
        return {
            "status": status,
            "experiment": "cora_onpolicy_candidate_support",
            "seed": args.seed,
            "split": args.split,
            "candidate_count": args.candidate_count,
            "execution_horizon": args.execution_horizon,
            "max_actions": args.max_actions,
            "group_offset": args.group_offset,
            "group_count": len(groups),
            "stages": STAGES,
            "completed_groups": len(episodes),
            "rows": rows,
            "episodes": episodes,
        }

    try:
        for group in groups:
            pair_id = str(group["pair_id"])
            reference = _load_reference_arrays(args.episode_root, group, "slipped")
            observation = _restore_recorded_state(
                live_env, reference, int(group["feedback_reveal_time"])
            )
            initial_distance = float(
                np.linalg.norm(
                    np.asarray(observation["cream_cheese_1_pos"]) - np.asarray(observation["robot0_eef_pos"])
                )
            )
            captured = set()
            recovery_started = False
            previous_failure = False
            actions_executed = 0
            replan_index = 0
            last_state = None
            success = bool(live_env.check_success())
            while actions_executed < args.max_actions and not success:
                seeds = [stable_seed(args.seed, pair_id, "onpolicy", replan_index, index) for index in range(16)]
                started = time.perf_counter()
                candidates, inference_seconds = sample_many(policy, observation, seeds)
                distance = float(
                    np.linalg.norm(
                        np.asarray(observation["cream_cheese_1_pos"]) - np.asarray(observation["robot0_eef_pos"])
                    )
                )
                stage = classify_boundary_stage(
                    replan_index=replan_index,
                    grasped=object_grasped(live_env),
                    previous_failure_continuation=previous_failure,
                    recovery_started=recovery_started,
                    eef_object_distance=distance,
                    initial_eef_object_distance=initial_distance,
                    candidate0_closes=bool(np.any(candidates[0, :2, -1] > 0.2)),
                )
                snapshot = capture_runtime_snapshot(live_env)
                last_state = (snapshot, candidates, seeds, replan_index)
                if stage not in captured:
                    row = evaluate_state(
                        branch_env,
                        snapshot,
                        candidates,
                        seeds,
                        pair_id=pair_id,
                        stage=stage,
                        replan_index=replan_index,
                        execution_horizon=2,
                        teacher_max_steps=args.teacher_max_steps,
                        cache_dir=args.cache_dir,
                    )
                    row["candidate_inference_wall_seconds"] = inference_seconds
                    rows.append(row); captured.add(stage)
                    atomic_json(partial, payload("partial"))
                    print(json.dumps({"pair_id": pair_id, "stage": stage, "recall16": row["immediate_recall"]["16"]}, sort_keys=True), flush=True)

                previous_failure = False
                for action in candidates[0, :2]:
                    grasped = object_grasped(live_env)
                    previous_failure = previous_failure or is_failure_continuation(action, grasped=grasped)
                    recovery_started = recovery_started or is_recovery_action(
                        action,
                        grasped=grasped,
                        eef_position=observation["robot0_eef_pos"],
                        object_position=observation["cream_cheese_1_pos"],
                    )
                    observation = _step(live_env, action)
                    actions_executed += 1
                    success = bool(live_env.check_success())
                    if success or actions_executed >= args.max_actions:
                        break
                replan_index += 1

            if not success and "final_failure" not in captured and last_state is not None:
                snapshot, candidates, seeds, final_replan = last_state
                row = evaluate_state(
                    branch_env,
                    snapshot,
                    candidates,
                    seeds,
                    pair_id=pair_id,
                    stage="final_failure",
                    replan_index=final_replan,
                    execution_horizon=2,
                    teacher_max_steps=args.teacher_max_steps,
                    cache_dir=args.cache_dir,
                )
                rows.append(row); captured.add("final_failure")
            episodes.append(
                {
                    "pair_id": pair_id,
                    "success": success,
                    "actions": actions_executed,
                    "replans": replan_index,
                    "captured_stages": sorted(captured),
                    "missing_stages": sorted(set(STAGES) - captured),
                }
            )
            atomic_json(partial, payload("partial"))
    finally:
        live_env.close(); branch_env.close(); policy.close()
    atomic_json(args.output, payload("complete"))
    partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
