from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from commit_strategies import COMMIT_METHODS, CommitController
from evaluate_libero_closed_loop import (
    Pi05Policy,
    RemotePi05Policy,
    _frame,
    _load_reference_arrays,
    _restore_recorded_state,
)
from evaluate_libero_deterministic_reach import reference_target, target_error
from evaluate_libero_oracle_commit import sample_policy_chunks
from libero_snapshot_collector import DEFAULT_BDDL, _step
from video_io import write_h264_video


def run_commit_reach(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    controller: CommitController,
    episode_root: Path,
    group: Mapping[str, Any],
    *,
    max_steps: int,
    threshold: float,
    reference_target_step: int,
    seed: int,
    record_video: bool,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    reference = _load_reference_arrays(episode_root, group)
    observation = _restore_recorded_state(env, reference, 0)
    target, resolved_target_step = reference_target(reference, reference_target_step)
    error = target_error(observation["robot0_eef_pos"], target)
    initial_error = error
    best_error = error
    steps = 0
    replans = 0
    policy_forward_calls = 0
    inference_wall_seconds = 0.0
    server_inference_wall_seconds = 0.0
    commit_selector_wall_seconds = 0.0
    episode_started = time.perf_counter()
    success = error <= threshold
    first_predicted_action = None
    commit_trace = []
    frames = [_frame(observation)] if record_video else []
    current_gripper_action = float(np.asarray(env.robots[0].gripper.current_action).reshape(-1)[0])
    while steps < max_steps and not success:
        chunks, elapsed, server_elapsed = sample_policy_chunks(
            policy,
            observation,
            noise_seed=seed,
            pair_id=str(group["pair_id"]),
            replan_count=replans,
            sample_count=controller.policy_samples_per_invocation,
            namespace="reach",
        )
        inference_wall_seconds += elapsed
        server_inference_wall_seconds += server_elapsed
        policy_forward_calls += len(chunks)
        if first_predicted_action is None:
            first_predicted_action = np.asarray(chunks[0, 0], dtype=np.float64)
        selector_started = time.perf_counter()
        decision = controller.decide(
            str(group["pair_id"]),
            global_step=steps,
            sampled_chunks=chunks,
            current_gripper_action=current_gripper_action,
        )
        commit_selector_wall_seconds += time.perf_counter() - selector_started
        commit_trace.append(
            {
                "replan_index": replans,
                "global_step": steps,
                "commit_length": decision.length,
                **decision.diagnostics,
            }
        )
        replans += 1
        for action in chunks[0, : decision.length]:
            observation = _step(env, action)
            current_gripper_action = float(action[-1])
            steps += 1
            error = target_error(observation["robot0_eef_pos"], target)
            best_error = min(best_error, error)
            success = error <= threshold
            if record_video:
                frames.append(_frame(observation))
            if success or steps >= max_steps:
                break
    if first_predicted_action is None:
        raise RuntimeError("reach evaluation did not produce an action")
    first_reference_action = np.asarray(reference["actions"][0], dtype=np.float64)
    translation_denominator = np.linalg.norm(first_predicted_action[:3]) * np.linalg.norm(first_reference_action[:3])
    first_translation_cosine = (
        float(np.dot(first_predicted_action[:3], first_reference_action[:3]) / translation_denominator)
        if translation_denominator > 1e-12
        else None
    )
    commit_lengths = [int(row["commit_length"]) for row in commit_trace]
    return {
        "pair_id": group["pair_id"],
        "split": group["split"],
        "evaluation": "deterministic_reach",
        "commit_method": controller.method,
        "success": success,
        "reference_target_step": resolved_target_step,
        "initial_target_error": initial_error,
        "best_target_error": best_error,
        "final_target_error": error,
        "best_target_progress": initial_error - best_error,
        "final_target_progress": initial_error - error,
        "first_action_mse": float(np.mean(np.square(first_predicted_action - first_reference_action))),
        "first_translation_cosine": first_translation_cosine,
        "first_predicted_action": first_predicted_action.tolist(),
        "first_reference_action": first_reference_action.tolist(),
        "completion_steps": steps,
        "policy_invocation_count": replans,
        "policy_forward_calls": policy_forward_calls,
        "inference_wall_seconds": inference_wall_seconds,
        "client_policy_wall_seconds": inference_wall_seconds,
        "server_predict_action_wall_seconds": server_inference_wall_seconds,
        "commit_selector_wall_seconds": commit_selector_wall_seconds,
        "episode_wall_seconds": time.perf_counter() - episode_started,
        "flow_inference_steps_per_call": 10,
        "denoiser_step_count": policy_forward_calls * 10,
        "mean_inference_seconds_per_call": inference_wall_seconds / policy_forward_calls,
        "mean_commit_length": float(np.mean(commit_lengths)),
        "commit_length_histogram": {
            str(length): commit_lengths.count(length) for length in range(1, controller.max_commit + 1)
        },
        "oracle_primary_eligible": bool(group["feedback_reveal_time"] <= group["action_divergence_time"]),
        "commit_trace": commit_trace,
    }, frames


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    return {
        "group_count": len({str(row["pair_id"]) for row in rows}),
        "deterministic_reach_success": float(np.mean([row["success"] for row in rows])),
        "mean_best_target_error": float(np.mean([row["best_target_error"] for row in rows])),
        "mean_final_target_error": float(np.mean([row["final_target_error"] for row in rows])),
        "mean_policy_invocations": float(np.mean([row["policy_invocation_count"] for row in rows])),
        "mean_policy_forward_calls": float(np.mean([row["policy_forward_calls"] for row in rows])),
        "mean_inference_wall_seconds": float(np.mean([row["inference_wall_seconds"] for row in rows])),
        "mean_server_predict_action_wall_seconds": float(
            np.mean([row["server_predict_action_wall_seconds"] for row in rows])
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Commit-wrapper deterministic reach control")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-socket", type=Path)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit-method", choices=COMMIT_METHODS, required=True)
    parser.add_argument("--max-commit", type=int, default=3)
    parser.add_argument("--self-consistency-samples", type=int, default=8)
    parser.add_argument("--self-consistency-threshold", type=float, default=0.15)
    parser.add_argument("--random-boundary-map", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--success-threshold", type=float, default=0.04)
    parser.add_argument("--reference-target-step", type=int, default=20)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--video-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if (args.checkpoint is None) == (args.policy_socket is None):
        raise ValueError("provide exactly one of --checkpoint or --policy-socket")
    if args.reference_target_step < 1:
        raise ValueError("reference target step must be positive")
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
    policy_checkpoint_realpath = (
        policy.checkpoint_realpath if isinstance(policy, RemotePi05Policy) else str(args.checkpoint.resolve())
    )
    policy_model_size_bytes = (
        policy.model_size_bytes
        if isinstance(policy, RemotePi05Policy)
        else (args.checkpoint / "model.safetensors").stat().st_size
    )
    policy_runtime = (
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
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(manifest.get("bddl", DEFAULT_BDDL))),
        camera_heights=224,
        camera_widths=224,
    )
    env.seed(args.seed)
    rows = []
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)
    try:
        for group in groups:
            row, frames = run_commit_reach(
                env,
                policy,
                controller,
                args.episode_root,
                group,
                max_steps=args.max_steps,
                threshold=args.success_threshold,
                reference_target_step=args.reference_target_step,
                seed=args.seed,
                record_video=args.video_dir is not None,
            )
            rows.append(row)
            if args.video_dir is not None:
                write_h264_video(args.video_dir / f"reach-{group['pair_id']}.mp4", frames, fps=10.0)
            print(
                json.dumps({key: value for key, value in row.items() if key != "commit_trace"}, sort_keys=True),
                flush=True,
            )
    finally:
        env.close()
        policy.close()
    payload = {
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "policy_socket": None if args.policy_socket is None else str(args.policy_socket),
        "policy_checkpoint_realpath": policy_checkpoint_realpath,
        "policy_model_size_bytes": policy_model_size_bytes,
        "policy_checkpoint_sha256": os.environ.get("FRESH_CHECKPOINT_SHA256"),
        "policy_checkpoint_sha256_source": "sha256sum_preflight_verified",
        "git_sha": os.environ.get("FRESH_GIT_SHA"),
        "policy_runtime": policy_runtime,
        "episode_root": str(args.episode_root),
        "evaluation": "deterministic_reach",
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
        "status": "complete",
        "completed_rows": len(rows),
        "expected_rows": len(groups),
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
