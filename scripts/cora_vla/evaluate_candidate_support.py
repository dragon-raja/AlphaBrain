from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from candidate_support import (
    continuation_compatibility,
    physical_compatibility,
    recall_prefix,
    stable_seed,
    summarize_group_rows,
)
from evaluate_libero_closed_loop import (
    LANGUAGE,
    Pi05Policy,
    RemotePi05Policy,
    _policy_observation,
    _restore_recorded_state,
    is_failure_continuation,
    is_recovery_action,
)
from libero_full_episode_collector import FullEpisodeTeacher, object_grasped
from libero_snapshot_collector import _step


PREFIX_SIZES = (1, 4, 8, 16, 32)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_episode(root: Path, group: Mapping[str, Any], outcome: str) -> dict[str, np.ndarray]:
    with np.load(root / group["episode_files"][outcome], allow_pickle=False) as episode:
        return {key: np.asarray(episode[key]) for key in episode.files}


def sample_many(
    policy: Pi05Policy | RemotePi05Policy,
    observation: Mapping[str, Any],
    seeds: Sequence[int],
) -> tuple[np.ndarray, float]:
    chunks = []
    wall_seconds = 0.0
    for start in range(0, len(seeds), 16):
        batch, elapsed = policy.predict_many(observation, seeds[start : start + 16])
        chunks.append(batch)
        wall_seconds += elapsed
    return np.concatenate(chunks, axis=0), wall_seconds


def observation_signature(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    policy_input = _policy_observation(observation)
    return {
        "agentview": np.asarray(policy_input["image"][0]),
        "wrist": np.asarray(policy_input["image"][1]),
        "robot_state": np.asarray(policy_input["state"]),
    }


def execute_candidate_and_teacher(
    env: Any,
    reference: Mapping[str, np.ndarray],
    frame_index: int,
    actions: np.ndarray,
    *,
    outcome: str,
    execution_horizon: int,
    teacher_max_steps: int,
) -> tuple[dict[str, Any], np.ndarray]:
    observation = _restore_recorded_state(env, reference, frame_index)
    initial_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
    initial_object_distance = float(
        np.linalg.norm(
            np.asarray(observation["cream_cheese_1_pos"], dtype=np.float64) - initial_eef
        )
    )
    grasp_trace = []
    empty_lift = False
    recovery_action_seen = False
    for action in actions[:execution_horizon]:
        grasped_before = object_grasped(env)
        empty_lift = empty_lift or is_failure_continuation(action, grasped=grasped_before)
        recovery_action_seen = recovery_action_seen or is_recovery_action(
            action,
            grasped=grasped_before,
            eef_position=observation["robot0_eef_pos"],
            object_position=observation["cream_cheese_1_pos"],
        )
        observation = _step(env, action)
        grasp_trace.append(object_grasped(env))

    candidate_effect = np.asarray(observation["robot0_eef_pos"], dtype=np.float64) - initial_eef
    final_object_distance = float(
        np.linalg.norm(
            np.asarray(observation["cream_cheese_1_pos"], dtype=np.float64)
            - np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
        )
    )
    teacher = FullEpisodeTeacher(observation)
    teacher_steps = 0
    success = bool(env.check_success())
    while not success and not teacher.done and teacher_steps < teacher_max_steps:
        decision = teacher.decide(
            observation,
            grasped=object_grasped(env),
            success=success,
        )
        observation = _step(env, decision.action)
        teacher_steps += 1
        success = bool(env.check_success())
    compatible = physical_compatibility(
        outcome,
        teacher_success=success,
        grasp_trace=grasp_trace,
        empty_lift=empty_lift,
        recovery_action_seen=recovery_action_seen,
        initial_object_distance=initial_object_distance,
        final_object_distance=final_object_distance,
    )
    return {
        "teacher_success": success,
        "teacher_steps": teacher_steps,
        "grasp_preserved": bool(grasp_trace) and all(grasp_trace),
        "empty_lift": empty_lift,
        "recovery_action_seen": recovery_action_seen,
        "initial_eef_object_distance": initial_object_distance,
        "final_eef_object_distance": final_object_distance,
        "physical_compatible": compatible,
    }, candidate_effect


def evaluate_outcome(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    episode_root: Path,
    group: Mapping[str, Any],
    episodes: Mapping[str, Mapping[str, np.ndarray]],
    *,
    outcome: str,
    checkpoint_seed: int,
    candidate_count: int,
    execution_horizon: int,
    action_margin: float,
    effect_margin: float,
    teacher_max_steps: int,
    candidate_dir: Path,
) -> dict[str, Any]:
    swapped_outcome = "slipped" if outcome == "attached" else "attached"
    frame_index = int(group["feedback_reveal_time"])
    reference = episodes[outcome]
    swapped = episodes[swapped_outcome]
    observation = _restore_recorded_state(env, reference, frame_index)
    seeds = [
        stable_seed("cora-gate1", checkpoint_seed, group["pair_id"], index)
        for index in range(candidate_count)
    ]
    chunks, inference_seconds = sample_many(policy, observation, seeds)
    correct_actions = reference["actions"][frame_index : frame_index + execution_horizon]
    swapped_actions = swapped["actions"][frame_index : frame_index + execution_horizon]
    correct_effect = (
        reference["eef_pose"][frame_index + execution_horizon, :3]
        - reference["eef_pose"][frame_index, :3]
    )
    swapped_effect = (
        swapped["eef_pose"][frame_index + execution_horizon, :3]
        - swapped["eef_pose"][frame_index, :3]
    )

    metrics = []
    effects = []
    for candidate_index, chunk in enumerate(chunks):
        physical, effect = execute_candidate_and_teacher(
            env,
            reference,
            frame_index,
            chunk,
            outcome=outcome,
            execution_horizon=execution_horizon,
            teacher_max_steps=teacher_max_steps,
        )
        compatibility = continuation_compatibility(
            chunk[:execution_horizon],
            effect,
            correct_actions,
            swapped_actions,
            correct_effect,
            swapped_effect,
            action_margin=action_margin,
            effect_margin=effect_margin,
        )
        metrics.append(
            {
                "candidate_index": candidate_index,
                "candidate_seed": seeds[candidate_index],
                **compatibility,
                **physical,
            }
        )
        effects.append(effect)

    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_file = candidate_dir / f"{group['pair_id']}-{outcome}.npz"
    np.savez_compressed(
        candidate_file,
        chunks=np.asarray(chunks, dtype=np.float32),
        seeds=np.asarray(seeds, dtype=np.int64),
        effects=np.asarray(effects, dtype=np.float32),
    )
    row = {
        "pair_id": group["pair_id"],
        "source_initial_state_index": int(group["source_initial_state_index"]),
        "outcome": outcome,
        "frame_index": frame_index,
        "candidate_file": str(candidate_file),
        "inference_wall_seconds": inference_seconds,
        "joint_recall": recall_prefix([m["joint_compatible"] for m in metrics], PREFIX_SIZES),
        "action_recall": recall_prefix([m["action_compatible"] for m in metrics], PREFIX_SIZES),
        "effect_recall": recall_prefix([m["effect_compatible"] for m in metrics], PREFIX_SIZES),
        "physical_recall": recall_prefix([m["physical_compatible"] for m in metrics], PREFIX_SIZES),
        "action_physical_agreement": float(
            np.mean([m["action_compatible"] == m["physical_compatible"] for m in metrics])
        ),
        "joint_physical_agreement": float(
            np.mean([m["joint_compatible"] == m["physical_compatible"] for m in metrics])
        ),
        "candidate_metrics": metrics,
    }
    return row


def audit_pre_feedback(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    group: Mapping[str, Any],
    episodes: Mapping[str, Mapping[str, np.ndarray]],
    *,
    checkpoint_seed: int,
    candidate_count: int,
) -> dict[str, Any]:
    frame_index = int(group["feedback_reveal_time"]) - 1
    signatures = {}
    chunks = {}
    # Exact policy inputs are already compared exhaustively. One paired stochastic
    # draw per group/checkpoint detects branch-dependent runtime routing without
    # spending another 64 serial flow samples on an identical observation.
    audit_candidate_count = min(candidate_count, 1)
    seeds = [
        stable_seed("cora-gate1-pre", checkpoint_seed, group["pair_id"], index)
        for index in range(audit_candidate_count)
    ]
    for outcome in ("attached", "slipped"):
        observation = _restore_recorded_state(env, episodes[outcome], frame_index)
        signatures[outcome] = observation_signature(observation)
        chunks[outcome], _ = sample_many(policy, observation, seeds)
    input_errors = {
        key: float(np.max(np.abs(signatures["attached"][key].astype(np.float64) - signatures["slipped"][key].astype(np.float64))))
        for key in signatures["attached"]
    }
    candidate_max_abs_error = float(
        np.max(np.abs(chunks["attached"].astype(np.float64) - chunks["slipped"].astype(np.float64)))
    )
    return {
        "pair_id": group["pair_id"],
        "frame_index": frame_index,
        "policy_input_max_abs_error": input_errors,
        "candidate_max_abs_error": candidate_max_abs_error,
        "candidate_count": audit_candidate_count,
        "passed": all(value == 0.0 for value in input_errors.values()) and candidate_max_abs_error == 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CORA Gate 1 frozen-policy candidate support evaluation")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-socket", type=Path)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-seed", type=int, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--execution-horizon", type=int, default=2)
    parser.add_argument("--action-margin", type=float, default=0.02)
    parser.add_argument("--effect-margin", type=float, default=0.002)
    parser.add_argument("--teacher-max-steps", type=int, default=320)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--group-start", type=int, default=0)
    parser.add_argument("--group-count", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default=LANGUAGE)
    return parser.parse_args()


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if args.candidate_count != 32:
        raise ValueError("formal Gate 1 requires exactly 32 candidates")
    if args.execution_horizon != 2:
        raise ValueError("formal Gate 1 requires K=2")
    if (args.checkpoint is None) == (args.policy_socket is None):
        raise ValueError("provide exactly one of --checkpoint or --policy-socket")
    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    groups = sorted(
        [group for group in manifest["groups"] if group["split"] == args.split],
        key=lambda group: group["pair_id"],
    )
    if args.group_start < 0:
        raise ValueError("group-start must be non-negative")
    groups = groups[args.group_start :]
    if args.group_count is not None:
        if args.group_count <= 0:
            raise ValueError("group-count must be positive")
        groups = groups[: args.group_count]
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    if not groups:
        raise ValueError("selected split has no groups")
    policy = (
        RemotePi05Policy(args.policy_socket, args.language)
        if args.policy_socket is not None
        else Pi05Policy(args.checkpoint, args.device, args.language)
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(manifest["bddl"])),
        camera_heights=224,
        camera_widths=224,
    )
    env.seed(args.checkpoint_seed)
    partial = args.output.with_name(f"{args.output.stem}.partial{args.output.suffix}")
    rows = []
    pre_feedback = []
    candidate_dir = args.output.parent / f"{args.output.stem}-candidates"
    try:
        for group in groups:
            episodes = {
                outcome: load_episode(args.episode_root, group, outcome)
                for outcome in ("attached", "slipped")
            }
            pre_feedback.append(
                audit_pre_feedback(
                    env,
                    policy,
                    group,
                    episodes,
                    checkpoint_seed=args.checkpoint_seed,
                    candidate_count=args.candidate_count,
                )
            )
            for outcome in ("attached", "slipped"):
                row = evaluate_outcome(
                    env,
                    policy,
                    args.episode_root,
                    group,
                    episodes,
                    outcome=outcome,
                    checkpoint_seed=args.checkpoint_seed,
                    candidate_count=args.candidate_count,
                    execution_horizon=args.execution_horizon,
                    action_margin=args.action_margin,
                    effect_margin=args.effect_margin,
                    teacher_max_steps=args.teacher_max_steps,
                    candidate_dir=candidate_dir,
                )
                rows.append(row)
                atomic_json(
                    partial,
                    {
                        "status": "partial",
                        "completed_rows": len(rows),
                        "expected_rows": 2 * len(groups),
                        "pre_feedback": pre_feedback,
                        "rows": rows,
                    },
                )
                print(
                    json.dumps(
                        {
                            "pair_id": row["pair_id"],
                            "outcome": outcome,
                            "joint_recall@16": row["joint_recall"]["16"],
                            "physical_recall@16": row["physical_recall"]["16"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        env.close()
        policy.close()

    payload = {
        "experiment": "cora_gate1_candidate_support",
        "status": "complete",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(args.checkpoint) if args.checkpoint else policy.checkpoint_realpath,
        "checkpoint_seed": args.checkpoint_seed,
        "split": args.split,
        "group_count": len(groups),
        "candidate_count": args.candidate_count,
        "prefix_sizes": PREFIX_SIZES,
        "execution_horizon": args.execution_horizon,
        "action_margin": args.action_margin,
        "effect_margin": args.effect_margin,
        "teacher_max_steps": args.teacher_max_steps,
        "policy_input_excludes_privileged_outcome": True,
        "pre_feedback_leakage_passed": all(row["passed"] for row in pre_feedback),
        "pre_feedback": pre_feedback,
        "summary": summarize_group_rows(rows, PREFIX_SIZES),
        "rows": rows,
    }
    atomic_json(args.output, payload)
    partial.unlink(missing_ok=True)
    print(json.dumps({"complete": True, "output": str(args.output), "summary": payload["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
