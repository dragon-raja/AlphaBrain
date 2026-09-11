from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from commit_strategies import COMMIT_METHODS, CommitController
from evaluate_libero_closed_loop import (
    LIFT_THRESHOLD,
    SUBGOALS,
    Pi05Policy,
    RemotePi05Policy,
    _atomic_write_json,
    _frame,
    _load_reference_arrays,
    _prepare_isolated_feedback,
    _restore_recorded_state,
    _write_paired_video,
    is_failure_continuation,
    is_premature_commitment,
    is_recovery_action,
    object_near_bowl,
    progress_fraction,
    slip_offset_for_group,
    stable_seed,
    summarize_rows,
    update_subgoals,
)
from libero_full_episode_collector import object_grasped
from libero_snapshot_collector import DEFAULT_BDDL, _set_object_offset, _step

ORACLE_METHODS = {"oracle_branch_safe_commit", "oracle_feedback_reveal_commit"}


def policy_sample_seeds(
    noise_seed: int,
    pair_id: str,
    replan_count: int,
    sample_count: int,
    *,
    namespace: str | None = None,
) -> list[int]:
    prefix = (noise_seed, pair_id, replan_count) if namespace is None else (noise_seed, pair_id, namespace, replan_count)
    return [
        stable_seed(*prefix) if sample_index == 0 else stable_seed(*prefix, "self_consistency", sample_index)
        for sample_index in range(sample_count)
    ]


def should_interrupt_for_runtime_event(method: str, evaluation: str, event_just_triggered: bool) -> bool:
    return bool(event_just_triggered and evaluation == "end_to_end" and method in ORACLE_METHODS)


def sample_policy_chunks(
    policy: Pi05Policy | RemotePi05Policy,
    observation: Mapping[str, Any],
    *,
    noise_seed: int,
    pair_id: str,
    replan_count: int,
    sample_count: int,
    namespace: str | None = None,
) -> tuple[np.ndarray, float, float]:
    seeds = policy_sample_seeds(
        noise_seed,
        pair_id,
        replan_count,
        sample_count,
        namespace=namespace,
    )
    started = time.perf_counter()
    chunks, server_elapsed = policy.predict_many(observation, seeds)
    return chunks, time.perf_counter() - started, server_elapsed


def run_commit_episode(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    controller: CommitController,
    episode_root: Path,
    group: Mapping[str, Any],
    *,
    evaluation: str,
    outcome: str,
    max_steps: int,
    noise_seed: int,
    record_video: bool,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    if evaluation == "isolated":
        observation, scripted_prefix_steps = _prepare_isolated_feedback(
            env,
            episode_root,
            group,
            outcome=outcome,
        )
        event_time = 0
        intervention_triggered = outcome == "slipped"
        reference = _load_reference_arrays(episode_root, group, outcome)
        initial_object_z = float(reference["object_pose"][0, 2])
        prefix_slice = slice(0, scripted_prefix_steps + 1)
        subgoals = {
            "grasp": bool(np.any(reference["grasped"][prefix_slice])),
            "lift": bool(np.max(reference["object_pose"][prefix_slice, 2]) - initial_object_z >= LIFT_THRESHOLD),
            "transport": False,
            "place": bool(env.check_success()),
        }
    elif evaluation == "end_to_end":
        reference = _load_reference_arrays(episode_root, group)
        observation = _restore_recorded_state(env, reference, 0)
        scripted_prefix_steps = 0
        event_time = None
        intervention_triggered = False
        initial_object_z = float(reference["object_pose"][0, 2])
        subgoals = {name: False for name in SUBGOALS}
    else:
        raise ValueError(f"unknown evaluation: {evaluation}")

    frames = [_frame(observation)] if record_video else []
    completion_steps = 0
    replan_count = 0
    policy_forward_calls = 0
    inference_wall_seconds = 0.0
    server_inference_wall_seconds = 0.0
    commit_selector_wall_seconds = 0.0
    episode_started = time.perf_counter()
    first_recovery_step = None
    failure_continuation = False
    premature_commitment = False
    success = bool(env.check_success())
    slip_offset = slip_offset_for_group(group)
    grasped_now = object_grasped(env)
    subgoals = update_subgoals(
        subgoals,
        observation,
        grasped=grasped_now,
        success=success,
        initial_object_z=initial_object_z,
    )
    progress_trace = [progress_fraction(subgoals)]
    regrasp_success = False
    dropped = False
    commit_trace = []
    current_gripper_action = float(np.asarray(env.robots[0].gripper.current_action).reshape(-1)[0])

    while completion_steps < max_steps and not success:
        chunks, elapsed, server_elapsed = sample_policy_chunks(
            policy,
            observation,
            noise_seed=noise_seed,
            pair_id=str(group["pair_id"]),
            replan_count=replan_count,
            sample_count=controller.policy_samples_per_invocation,
        )
        inference_wall_seconds += elapsed
        server_inference_wall_seconds += server_elapsed
        policy_forward_calls += len(chunks)
        global_step = scripted_prefix_steps + completion_steps
        selector_started = time.perf_counter()
        decision = controller.decide(
            str(group["pair_id"]),
            global_step=global_step,
            sampled_chunks=chunks,
            current_gripper_action=current_gripper_action,
        )
        commit_selector_wall_seconds += time.perf_counter() - selector_started
        trace_entry = {
            "replan_index": replan_count,
            "global_step": global_step,
            "planned_commit_length": decision.length,
            **decision.diagnostics,
        }
        commit_trace.append(trace_entry)
        replan_count += 1
        executed_in_commit = 0
        interrupted_by_oracle_event = False
        for action in chunks[0, : decision.length]:
            grasped_before = grasped_now
            if outcome == "slipped" and event_time is not None and first_recovery_step is None:
                if is_failure_continuation(action, grasped=grasped_before):
                    failure_continuation = True
                if is_premature_commitment(
                    action,
                    grasped=grasped_before,
                    eef_position=observation["robot0_eef_pos"],
                    bowl_position=observation["akita_black_bowl_1_pos"],
                ):
                    premature_commitment = True
                if is_recovery_action(
                    action,
                    grasped=grasped_before,
                    eef_position=observation["robot0_eef_pos"],
                    object_position=observation["cream_cheese_1_pos"],
                ):
                    first_recovery_step = completion_steps

            observation = _step(env, action)
            current_gripper_action = float(action[-1])
            completion_steps += 1
            success = bool(env.check_success())
            grasped_after = object_grasped(env)
            forced_slip_this_step = False
            event_just_triggered = False

            if evaluation == "end_to_end" and event_time is None:
                object_lift = float(observation["cream_cheese_1_pos"][2]) - initial_object_z
                if grasped_after and object_lift >= LIFT_THRESHOLD:
                    event_time = completion_steps
                    event_just_triggered = True
                    if outcome == "slipped":
                        observation = _set_object_offset(env, slip_offset)
                        intervention_triggered = True
                        forced_slip_this_step = True
                        grasped_after = object_grasped(env)

            eligible_now = outcome == "slipped" and event_time is not None
            if eligible_now and grasped_after:
                regrasp_success = True
            if (
                grasped_before
                and not grasped_after
                and not success
                and not object_near_bowl(observation)
                and not forced_slip_this_step
            ):
                dropped = True
            subgoals = update_subgoals(
                subgoals,
                observation,
                grasped=grasped_after,
                success=success,
                initial_object_z=initial_object_z,
            )
            progress_trace.append(progress_fraction(subgoals))
            grasped_now = grasped_after
            executed_in_commit += 1
            if record_video:
                frames.append(_frame(observation))
            if should_interrupt_for_runtime_event(controller.method, evaluation, event_just_triggered):
                interrupted_by_oracle_event = True
                break
            if success or completion_steps >= max_steps:
                break
        trace_entry["commit_length"] = executed_in_commit
        trace_entry["interrupted_by_oracle_event"] = interrupted_by_oracle_event

    eligible = outcome == "slipped" and event_time is not None
    switch_latency = None
    if eligible:
        switch_latency = (
            first_recovery_step - event_time if first_recovery_step is not None else completion_steps - event_time
        )
    commit_lengths = [int(row["commit_length"]) for row in commit_trace]
    label_boundary = int(
        group[
            "feedback_reveal_time" if controller.method == "oracle_feedback_reveal_commit" else "action_divergence_time"
        ]
    )
    actual_event_global = None if event_time is None else scripted_prefix_steps + int(event_time)
    return {
        "pair_id": group["pair_id"],
        "split": group["split"],
        "evaluation": evaluation,
        "branch_outcome": outcome,
        "commit_method": controller.method,
        "success": success,
        "recovery_success": bool(outcome == "slipped" and intervention_triggered and success),
        "intervention_triggered": intervention_triggered,
        "event_time": event_time,
        "actual_event_global_step": actual_event_global,
        "label_boundary_global_step": label_boundary,
        "event_boundary_alignment_error": (
            None if actual_event_global is None else int(actual_event_global - label_boundary)
        ),
        "oracle_primary_eligible": bool(group["feedback_reveal_time"] <= group["action_divergence_time"]),
        "scripted_prefix_steps": scripted_prefix_steps,
        "failure_continuation": failure_continuation if eligible else None,
        "premature_commitment": premature_commitment if eligible else None,
        "recovery_switch_latency": switch_latency,
        "recovery_switch_observed": bool(eligible and first_recovery_step is not None),
        "regrasp_success": regrasp_success if eligible else None,
        "drop": dropped,
        "final_progress": progress_fraction(subgoals),
        "progress_auc": float(np.mean(progress_trace)),
        "grasp_subgoal": subgoals["grasp"],
        "lift_subgoal": subgoals["lift"],
        "transport_subgoal": subgoals["transport"],
        "place_subgoal": subgoals["place"],
        "completion_steps": completion_steps,
        "policy_invocation_count": replan_count,
        "policy_forward_calls": policy_forward_calls,
        "inference_wall_seconds": inference_wall_seconds,
        "client_policy_wall_seconds": inference_wall_seconds,
        "server_predict_action_wall_seconds": server_inference_wall_seconds,
        "commit_selector_wall_seconds": commit_selector_wall_seconds,
        "episode_wall_seconds": time.perf_counter() - episode_started,
        "flow_inference_steps_per_call": 10,
        "denoiser_step_count": policy_forward_calls * 10,
        "mean_inference_seconds_per_call": (
            inference_wall_seconds / policy_forward_calls if policy_forward_calls else None
        ),
        "mean_commit_length": float(np.mean(commit_lengths)) if commit_lengths else None,
        "commit_length_histogram": {
            str(length): commit_lengths.count(length) for length in range(1, controller.max_commit + 1)
        },
        "commit_trace": commit_trace,
    }, frames


def evaluation_payload(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int,
    status: str,
) -> dict[str, Any]:
    return {
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "policy_socket": None if args.policy_socket is None else str(args.policy_socket),
        "policy_checkpoint_realpath": args.policy_checkpoint_realpath,
        "policy_model_size_bytes": args.policy_model_size_bytes,
        "policy_checkpoint_sha256": os.environ.get("FRESH_CHECKPOINT_SHA256"),
        "policy_checkpoint_sha256_source": "sha256sum_preflight_verified",
        "git_sha": os.environ.get("FRESH_GIT_SHA"),
        "policy_runtime": args.policy_runtime,
        "episode_root": str(args.episode_root),
        "evaluation": args.evaluation,
        "split": args.split,
        "seed": args.seed,
        "commit_method": args.commit_method,
        "max_commit": args.max_commit,
        "self_consistency_samples": args.self_consistency_samples,
        "self_consistency_threshold": args.self_consistency_threshold,
        "oracle_policy_isolation": (
            "a privileged runtime physical event may interrupt execution only; "
            "policy input and sampled anchor action are unchanged"
        ),
        "status": status,
        "completed_rows": len(rows),
        "expected_rows": expected_rows,
        "summary": summarize_rows(rows),
        "efficiency_summary": {
            "mean_policy_invocations": (
                float(np.mean([row["policy_invocation_count"] for row in rows])) if rows else None
            ),
            "mean_policy_forward_calls": float(np.mean([row["policy_forward_calls"] for row in rows])) if rows else None,
            "mean_inference_wall_seconds": (
                float(np.mean([row["inference_wall_seconds"] for row in rows])) if rows else None
            ),
            "mean_server_predict_action_wall_seconds": (
                float(np.mean([row["server_predict_action_wall_seconds"] for row in rows])) if rows else None
            ),
            "mean_commit_selector_wall_seconds": (
                float(np.mean([row["commit_selector_wall_seconds"] for row in rows])) if rows else None
            ),
        },
        "rows": list(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle and control commit evaluation using frozen Full-H Pi0.5")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-socket", type=Path)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation", choices=("isolated", "end_to_end"), required=True)
    parser.add_argument("--commit-method", choices=COMMIT_METHODS, required=True)
    parser.add_argument("--max-commit", type=int, default=3)
    parser.add_argument("--self-consistency-samples", type=int, default=8)
    parser.add_argument("--self-consistency-threshold", type=float, default=0.15)
    parser.add_argument("--random-boundary-map", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--video-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if (args.checkpoint is None) == (args.policy_socket is None):
        raise ValueError("provide exactly one of --checkpoint or --policy-socket")
    os.environ.setdefault("PRETRAINED_MODELS_DIR", "/share/longjunyu/alphabrain/pretrained_models")
    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    groups = sorted(
        (group for group in manifest["groups"] if group["split"] == args.split),
        key=lambda row: row["pair_id"],
    )
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    if not groups:
        raise ValueError(f"no groups for split={args.split!r}")
    policy = RemotePi05Policy(args.policy_socket) if args.policy_socket else Pi05Policy(args.checkpoint, args.device)
    args.policy_checkpoint_realpath = (
        policy.checkpoint_realpath if isinstance(policy, RemotePi05Policy) else str(args.checkpoint.resolve())
    )
    args.policy_model_size_bytes = (
        policy.model_size_bytes
        if isinstance(policy, RemotePi05Policy)
        else (args.checkpoint / "model.safetensors").stat().st_size
    )
    args.policy_runtime = (
        policy.runtime_identity
        if isinstance(policy, RemotePi05Policy)
        else {
            "torch_version": str(policy.torch.__version__),
            "cuda_version": None if policy.torch.version.cuda is None else str(policy.torch.version.cuda),
            "device_name": (str(policy.torch.cuda.get_device_name(0)) if policy.torch.cuda.is_available() else "cpu"),
        }
    )
    random_boundary_overrides = None
    if args.random_boundary_map is not None:
        configured = json.loads(args.random_boundary_map.read_text())["boundaries"]
        random_boundary_overrides = {str(group["pair_id"]): configured[str(group["pair_id"])] for group in groups}
        if args.evaluation != "end_to_end":
            random_boundary_overrides = {str(group["pair_id"]): None for group in groups}
    if args.commit_method == "random_matched_commit" and random_boundary_overrides is None:
        raise ValueError("random_matched_commit requires --random-boundary-map")
    controller = CommitController(
        args.commit_method,
        groups,
        seed=args.seed,
        max_commit=args.max_commit,
        self_consistency_samples=args.self_consistency_samples,
        self_consistency_threshold=args.self_consistency_threshold,
        random_boundary_overrides=random_boundary_overrides,
    )
    if controller.max_commit > policy.horizon:
        raise ValueError("max commit exceeds policy action horizon")
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(manifest.get("bddl", DEFAULT_BDDL))),
        camera_heights=224,
        camera_widths=224,
    )
    env.seed(args.seed)
    rows = []
    expected_rows = len(groups) * 2
    partial_output = args.output.with_name(f"{args.output.stem}.partial{args.output.suffix}")
    videos: dict[str, dict[str, list[np.ndarray]]] = defaultdict(dict)
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)
    try:
        for group in groups:
            for outcome in ("attached", "slipped"):
                row, frames = run_commit_episode(
                    env,
                    policy,
                    controller,
                    args.episode_root,
                    group,
                    evaluation=args.evaluation,
                    outcome=outcome,
                    max_steps=args.max_steps,
                    noise_seed=args.seed,
                    record_video=args.video_dir is not None,
                )
                rows.append(row)
                if args.video_dir is not None:
                    videos[str(group["pair_id"])][outcome] = frames
                    if videos[str(group["pair_id"])].keys() >= {"attached", "slipped"}:
                        _write_paired_video(
                            args.video_dir / f"{args.evaluation}-{group['pair_id']}.mp4",
                            videos[str(group["pair_id"])]["attached"],
                            videos[str(group["pair_id"])]["slipped"],
                        )
                        del videos[str(group["pair_id"])]
                _atomic_write_json(
                    partial_output,
                    evaluation_payload(args, rows, expected_rows=expected_rows, status="partial"),
                )
                print(
                    json.dumps({key: value for key, value in row.items() if key != "commit_trace"}, sort_keys=True),
                    flush=True,
                )
    finally:
        env.close()
        policy.close()

    result = evaluation_payload(args, rows, expected_rows=expected_rows, status="complete")
    _atomic_write_json(args.output, result)
    partial_output.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "summary": result["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
