from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from evaluate_libero_closed_loop import (
    Pi05Policy,
    RemotePi05Policy,
    _load_reference_arrays,
    is_failure_continuation,
    is_premature_commitment,
    stable_seed,
)
from evaluate_physical_process_oracle import capture_runtime_snapshot, restore_runtime_snapshot
from evaluate_recovery_expert_handoff import (
    audit_reconstructed_feedback_observation,
    reconstruct_feedback_snapshot,
)
from evaluate_recovery_segment_oracle import _observation_frame
from libero_full_episode_collector import FullEpisodeTeacher, TraceRecorder, object_grasped
from libero_snapshot_collector import DEFAULT_BDDL, _step
from video_io import write_h264_video


LANGUAGE = "put the cream cheese in the bowl"


def intervention_reason(
    action: Sequence[float],
    *,
    grasped: bool,
    eef_position: Sequence[float],
    bowl_position: Sequence[float],
) -> str | None:
    if is_failure_continuation(action, grasped=grasped):
        return "failure_continuation"
    if is_premature_commitment(
        action,
        grasped=grasped,
        eef_position=eef_position,
        bowl_position=bowl_position,
    ):
        return "premature_commitment"
    return None


def serialize_teacher(teacher: FullEpisodeTeacher) -> dict[str, Any]:
    return {
        "phase": teacher.phase,
        "phase_steps": teacher.phase_steps,
        "regrasp_attempts": teacher.regrasp_attempts,
        "place_attempts": teacher.place_attempts,
        "done": teacher.done,
        "initial_eef_xy": np.asarray(teacher.initial_eef_xy, dtype=np.float64).tolist(),
        "initial_object_z": float(teacher.initial_object_z),
    }


def restore_teacher(observation: Mapping[str, Any], state: Mapping[str, Any]) -> FullEpisodeTeacher:
    teacher = FullEpisodeTeacher(observation)
    teacher.phase = str(state["phase"])
    teacher.phase_steps = int(state["phase_steps"])
    teacher.regrasp_attempts = int(state["regrasp_attempts"])
    teacher.place_attempts = int(state["place_attempts"])
    teacher.done = bool(state["done"])
    teacher.initial_eef_xy = np.asarray(state["initial_eef_xy"], dtype=np.float64).copy()
    teacher.initial_object_z = float(state["initial_object_z"])
    return teacher


def induce_policy_deviation(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    feedback_snapshot: Mapping[str, Any],
    *,
    pair_id: str,
    seed: int,
    execution_horizon: int,
    max_actions: int,
    repeats: int,
    dwell_steps: int,
) -> dict[str, Any]:
    attempts = []
    for repeat in range(repeats):
        observation = restore_runtime_snapshot(env, feedback_snapshot)
        frames = [_observation_frame(observation)]
        actions = []
        grasp_run = 0
        replan_index = 0
        trigger = None
        while len(actions) < max_actions:
            chunk = policy.predict(
                observation,
                stable_seed(seed, pair_id, "policy_state_recovery", repeat, replan_index),
            )
            replan_index += 1
            for raw_action in chunk[:execution_horizon]:
                if len(actions) >= max_actions:
                    break
                action = np.asarray(raw_action, dtype=np.float32)
                grasped_before = object_grasped(env)
                candidate = intervention_reason(
                    action,
                    grasped=grasped_before,
                    eef_position=observation["robot0_eef_pos"],
                    bowl_position=observation["akita_black_bowl_1_pos"],
                )
                observation = _step(env, action)
                actions.append(action)
                frames.append(_observation_frame(observation))
                grasped_after = object_grasped(env)
                grasp_run = grasp_run + 1 if grasped_after else 0
                if candidate is not None and not grasped_after:
                    trigger = candidate
                    break
                if grasp_run >= dwell_steps:
                    break
            if trigger is not None or grasp_run >= dwell_steps:
                break
        self_recovered = grasp_run >= dwell_steps
        attempts.append(
            {
                "repeat": repeat,
                "policy_actions": len(actions),
                "replans": replan_index,
                "self_regrasped": self_recovered,
                "trigger_reason": trigger if trigger is not None else "timeout",
            }
        )
        if self_recovered:
            continue
        return {
            "observation": observation,
            "snapshot": capture_runtime_snapshot(env),
            "frames": frames,
            "actions": np.asarray(actions, dtype=np.float32),
            "repeat": repeat,
            "trigger_reason": trigger if trigger is not None else "timeout",
            "attempts": attempts,
            "self_recovered": False,
        }
    return {
        "observation": None,
        "snapshot": None,
        "frames": [],
        "actions": np.empty((0, 7), dtype=np.float32),
        "repeat": None,
        "trigger_reason": "all_repeats_self_regrasped",
        "attempts": attempts,
        "self_recovered": True,
    }


def collect_teacher_correction(
    env: Any,
    observation: Mapping[str, Any],
    *,
    action_horizon: int,
    dwell_steps: int,
    max_teacher_actions: int,
) -> dict[str, Any]:
    teacher = FullEpisodeTeacher(observation)
    recorder = TraceRecorder(env, observation, teacher.phase)
    grasp_run = 0
    correction_action_count = None
    endpoint_snapshot = None
    endpoint_teacher_state = None

    while len(recorder.actions) < max_teacher_actions:
        decision = teacher.decide(
            observation,
            grasped=object_grasped(env),
            success=bool(env.check_success()),
        )
        if teacher.done and not bool(env.check_success()):
            break
        observation = recorder.step(env, decision.action, decision.phase)
        grasp_run = grasp_run + 1 if object_grasped(env) else 0
        if correction_action_count is None and grasp_run >= dwell_steps:
            correction_action_count = len(recorder.actions)
            endpoint_snapshot = capture_runtime_snapshot(env)
            endpoint_teacher_state = serialize_teacher(teacher)
        if (
            correction_action_count is not None
            and len(recorder.actions) >= correction_action_count + action_horizon - 1
        ):
            break

    if correction_action_count is None:
        raise RuntimeError("teacher correction did not reach stable regrasp")
    if len(recorder.actions) < correction_action_count + action_horizon - 1:
        raise RuntimeError("teacher correction did not provide a full action tail")
    return {
        "arrays": recorder.arrays(),
        "correction_action_count": correction_action_count,
        "endpoint_snapshot": endpoint_snapshot,
        "endpoint_teacher_state": endpoint_teacher_state,
        "teacher_actions_with_tail": len(recorder.actions),
    }


def audit_full_teacher_reachability(
    env: Any,
    endpoint_snapshot: Mapping[str, Any],
    teacher_state: Mapping[str, Any],
    *,
    max_actions: int,
) -> dict[str, Any]:
    observation = restore_runtime_snapshot(env, endpoint_snapshot)
    teacher = restore_teacher(observation, teacher_state)
    actions = 0
    success = bool(env.check_success())
    while actions < max_actions and not success:
        decision = teacher.decide(
            observation,
            grasped=object_grasped(env),
            success=success,
        )
        if teacher.done and not success:
            break
        observation = _step(env, decision.action)
        actions += 1
        success = bool(env.check_success())
    return {"success": success, "actions": actions, "teacher_done": bool(teacher.done)}


def audit_policy_reachability(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    endpoint_snapshot: Mapping[str, Any],
    *,
    pair_id: str,
    seed: int,
    execution_horizon: int,
    max_actions: int,
) -> dict[str, Any]:
    observation = restore_runtime_snapshot(env, endpoint_snapshot)
    actions = 0
    replans = 0
    success = bool(env.check_success())
    while actions < max_actions and not success:
        chunk = policy.predict(
            observation,
            stable_seed(seed, pair_id, "policy_state_endpoint_audit", replans),
        )
        replans += 1
        for action in chunk[:execution_horizon]:
            if actions >= max_actions:
                break
            observation = _step(env, action)
            actions += 1
            success = bool(env.check_success())
            if success:
                break
    return {"success": success, "actions": actions, "replans": replans}


def _save_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
        path,
        format="JPEG",
        quality=quality,
        subsampling=0,
    )


def correction_records(
    output_root: Path,
    pair_id: str,
    arrays: Mapping[str, np.ndarray],
    *,
    seed: int,
    correction_action_count: int,
    action_horizon: int,
    jpeg_quality: int,
    trigger_reason: str,
    policy_prefix_actions: int,
) -> list[dict[str, Any]]:
    actions = np.asarray(arrays["actions"], dtype=np.float32)
    if len(actions) < correction_action_count + action_horizon - 1:
        raise ValueError("correction trajectory is too short for requested windows")
    rows = []
    for frame_index in range(correction_action_count):
        frame_dir = output_root / "frames" / pair_id
        agent_path = frame_dir / f"{frame_index:04d}-agent.jpg"
        wrist_path = frame_dir / f"{frame_index:04d}-wrist.jpg"
        _save_jpeg(agent_path, arrays["agentview"][frame_index], jpeg_quality)
        _save_jpeg(wrist_path, arrays["wrist"][frame_index], jpeg_quality)
        rows.append(
            {
                "sample_id": f"{pair_id}::policy-state-seed{seed}::{frame_index:04d}",
                "window_group_id": f"{pair_id}::policy-state::{frame_index:04d}",
                "pair_id": pair_id,
                "branch_id": "policy_state_recovery",
                "branch_outcome": "slipped",
                "task": "policy_state_recovery_correction",
                "split": "train",
                "frame_index": frame_index,
                "observation": {
                    "agentview_path": str(agent_path.relative_to(output_root)),
                    "wrist_path": str(wrist_path.relative_to(output_root)),
                },
                "robot_state": np.asarray(arrays["robot_state"][frame_index], dtype=np.float32).round(8).tolist(),
                "language_instruction": LANGUAGE,
                "action_chunk": actions[frame_index : frame_index + action_horizon].round(8).tolist(),
                "oracle_feedback_horizon": action_horizon,
                "is_policy_state_correction": True,
                "trigger_reason": trigger_reason,
                "policy_prefix_actions": policy_prefix_actions,
            }
        )
    return rows


def build_quality_report(
    group_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    requested_group_count: int,
    minimum_correction_group_rate: float,
) -> dict[str, Any]:
    corrected = [row for row in group_rows if row["retained"]]
    teacher_successes = [bool(row["full_teacher_audit"]["success"]) for row in corrected]
    correction_rate = len(corrected) / requested_group_count if requested_group_count else 0.0
    checks = {
        "requested_groups_processed": len(group_rows) == requested_group_count,
        "train_split_only": all(row["split"] == "train" for row in group_rows),
        "minimum_correction_group_coverage": correction_rate >= minimum_correction_group_rate,
        "stable_regrasp_reached": all(row["stable_regrasp_reached"] for row in corrected),
        "full_teacher_reachability_at_least_90pct": bool(teacher_successes)
        and float(np.mean(teacher_successes)) >= 0.90,
        "feedback_reconstruction_exact": all(
            row["feedback_reconstruction"]["post_injection_sim_max_abs_delta"] <= 1e-10
            and row["feedback_reconstruction"]["prefix_gripper_max_abs_delta"] <= 1e-6
            and row["feedback_reconstruction"]["policy_image_max_abs_delta"] == 0
            and row["feedback_reconstruction"]["policy_robot_state_max_abs_delta"] <= 1e-6
            for row in group_rows
        ),
        "records_are_deployable_inputs_only": all(
            set(row["observation"]) == {"agentview_path", "wrist_path"}
            for row in records
        ),
        "all_records_have_full_chunks": all(
            np.asarray(row["action_chunk"]).shape == (10, 7) for row in records
        ),
        "nonempty_training_windows": bool(records),
    }
    policy_audits = [
        row["frozen_policy_audit"]
        for row in corrected
        if row.get("frozen_policy_audit") is not None
    ]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "requested_group_count": requested_group_count,
            "retained_correction_group_count": len(corrected),
            "correction_group_rate": correction_rate,
            "training_window_count": len(records),
            "source_initial_state_count": len(
                {int(row["source_initial_state_index"]) for row in corrected}
            ),
            "trigger_reason_counts": dict(Counter(row["trigger_reason"] for row in corrected)),
            "mean_policy_prefix_actions": float(
                np.mean([row["policy_prefix_actions"] for row in corrected])
            )
            if corrected
            else None,
            "mean_teacher_correction_actions": float(
                np.mean([row["teacher_correction_actions"] for row in corrected])
            )
            if corrected
            else None,
            "full_teacher_success_rate": float(np.mean(teacher_successes)) if teacher_successes else None,
            "frozen_policy_audit_count": len(policy_audits),
            "frozen_policy_downstream_success_rate": float(
                np.mean([row["success"] for row in policy_audits])
            )
            if policy_audits
            else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect teacher corrections on Pi0.5-induced recovery states")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-socket", type=Path)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--group-offset", type=int, default=0)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--execution-horizon", type=int, default=3)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--policy-prefix-max-actions", type=int, default=12)
    parser.add_argument("--policy-repeats", type=int, default=3)
    parser.add_argument("--stage-dwell-steps", type=int, default=2)
    parser.add_argument("--max-teacher-actions", type=int, default=200)
    parser.add_argument("--downstream-action-budget", type=int, default=320)
    parser.add_argument("--policy-audit-stride", type=int, default=5)
    parser.add_argument("--minimum-correction-group-rate", type=float, default=0.80)
    parser.add_argument("--video-groups", type=int, default=20)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if (args.checkpoint is None) == (args.policy_socket is None):
        raise ValueError("provide exactly one of --checkpoint or --policy-socket")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    if args.execution_horizon < 1 or args.action_horizon < 1:
        raise ValueError("action horizons must be positive")
    if args.policy_prefix_max_actions < 1 or args.policy_repeats < 1:
        raise ValueError("policy collection budgets must be positive")
    if args.stage_dwell_steps < 1 or args.max_teacher_actions < args.action_horizon:
        raise ValueError("teacher budgets are invalid")
    if args.policy_audit_stride < 1 or args.downstream_action_budget < 1:
        raise ValueError("audit budgets must be positive")
    if not 0.0 <= args.minimum_correction_group_rate <= 1.0:
        raise ValueError("minimum correction group rate must be in [0, 1]")
    if args.video_groups < 0 or not 80 <= args.jpeg_quality <= 100:
        raise ValueError("video-groups or jpeg-quality is invalid")

    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    all_groups = sorted(
        (group for group in manifest["groups"] if group["split"] == args.split),
        key=lambda group: group["pair_id"],
    )
    groups = all_groups[args.group_offset :]
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    if not groups:
        raise ValueError("no groups selected")

    policy = (
        RemotePi05Policy(args.policy_socket)
        if args.policy_socket is not None
        else Pi05Policy(args.checkpoint, args.device)
    )
    if args.execution_horizon > policy.horizon:
        raise ValueError("execution horizon exceeds policy action horizon")
    staging = args.output_dir.parent / f".{args.output_dir.name}.staging-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "trajectories").mkdir()
    (staging / "videos").mkdir()
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(manifest.get("bddl", DEFAULT_BDDL))),
        camera_heights=224,
        camera_widths=224,
    )
    env.seed(args.seed)
    group_rows = []
    records = []
    try:
        for group_index, group in enumerate(groups):
            pair_id = str(group["pair_id"])
            reference = _load_reference_arrays(args.episode_root, group, "slipped")
            feedback_index = int(group["feedback_reveal_time"])
            feedback_observation, feedback_snapshot, reconstruction = reconstruct_feedback_snapshot(
                env,
                reference,
                feedback_index,
            )
            reconstruction.update(
                audit_reconstructed_feedback_observation(
                    args.episode_root,
                    group,
                    feedback_index,
                    feedback_observation,
                )
            )
            deviation = induce_policy_deviation(
                env,
                policy,
                feedback_snapshot,
                pair_id=pair_id,
                seed=args.seed,
                execution_horizon=args.execution_horizon,
                max_actions=args.policy_prefix_max_actions,
                repeats=args.policy_repeats,
                dwell_steps=args.stage_dwell_steps,
            )
            row = {
                "pair_id": pair_id,
                "split": args.split,
                "source_initial_state_index": int(group["source_initial_state_index"]),
                "feedback_reveal_time": feedback_index,
                "feedback_reconstruction": reconstruction,
                "retained": not deviation["self_recovered"],
                "trigger_reason": deviation["trigger_reason"],
                "policy_repeat": deviation["repeat"],
                "policy_prefix_actions": int(len(deviation["actions"])),
                "policy_attempts": deviation["attempts"],
                "stable_regrasp_reached": False,
                "full_teacher_audit": None,
                "frozen_policy_audit": None,
            }
            if deviation["self_recovered"]:
                group_rows.append(row)
                continue

            correction = collect_teacher_correction(
                env,
                deviation["observation"],
                action_horizon=args.action_horizon,
                dwell_steps=args.stage_dwell_steps,
                max_teacher_actions=args.max_teacher_actions,
            )
            arrays = correction["arrays"]
            row["stable_regrasp_reached"] = True
            row["teacher_correction_actions"] = int(correction["correction_action_count"])
            row["teacher_actions_with_tail"] = int(correction["teacher_actions_with_tail"])
            row["full_teacher_audit"] = audit_full_teacher_reachability(
                env,
                correction["endpoint_snapshot"],
                correction["endpoint_teacher_state"],
                max_actions=args.downstream_action_budget,
            )
            if group_index % args.policy_audit_stride == 0:
                row["frozen_policy_audit"] = audit_policy_reachability(
                    env,
                    policy,
                    correction["endpoint_snapshot"],
                    pair_id=pair_id,
                    seed=args.seed,
                    execution_horizon=args.execution_horizon,
                    max_actions=args.downstream_action_budget,
                )
            records.extend(
                correction_records(
                    staging,
                    pair_id,
                    arrays,
                    seed=args.seed,
                    correction_action_count=correction["correction_action_count"],
                    action_horizon=args.action_horizon,
                    jpeg_quality=args.jpeg_quality,
                    trigger_reason=deviation["trigger_reason"],
                    policy_prefix_actions=len(deviation["actions"]),
                )
            )
            trajectory_arrays = {
                key: value
                for key, value in arrays.items()
                if key not in {"agentview", "wrist"}
            }
            trajectory_arrays["policy_prefix_actions"] = deviation["actions"]
            np.savez_compressed(staging / "trajectories" / f"{pair_id}.npz", **trajectory_arrays)
            if group_index < args.video_groups:
                correction_frames = [
                    np.concatenate([arrays["agentview"][index], arrays["wrist"][index]], axis=1)
                    for index in range(len(arrays["agentview"]))
                ]
                write_h264_video(
                    staging / "videos" / f"{pair_id}.mp4",
                    [*deviation["frames"], *correction_frames[1:]],
                    fps=10.0,
                )
            group_rows.append(row)
            print(
                json.dumps(
                    {
                        "completed": group_index + 1,
                        "total": len(groups),
                        "pair_id": pair_id,
                        "retained": row["retained"],
                        "trigger": row["trigger_reason"],
                        "correction_actions": row.get("teacher_correction_actions"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        env.close()
        policy.close()

    quality = build_quality_report(
        group_rows,
        records,
        requested_group_count=len(groups),
        minimum_correction_group_rate=args.minimum_correction_group_rate,
    )
    output_manifest = {
        "schema_version": 1,
        "generator": "collect_policy_state_recovery.py",
        "episode_root": str(args.episode_root),
        "split": args.split,
        "seed": args.seed,
        "group_offset": args.group_offset,
        "group_count": len(groups),
        "execution_horizon": args.execution_horizon,
        "action_horizon": args.action_horizon,
        "policy_prefix_max_actions": args.policy_prefix_max_actions,
        "policy_repeats": args.policy_repeats,
        "stage_dwell_steps": args.stage_dwell_steps,
        "correction_endpoint": "first stable regrasp",
        "policy_input_fields": ["agentview", "wrist", "robot_state", "language_instruction"],
        "teacher_privileged_fields": ["object pose", "grasp/contact", "environment success"],
        "frozen_policy_audit_is_diagnostic_not_filter": True,
        "git_sha": os.environ.get("FRESH_GIT_SHA"),
        "git_dirty_at_launch": os.environ.get("FRESH_GIT_DIRTY") == "1",
        "policy_checkpoint_sha256": os.environ.get("FRESH_CHECKPOINT_SHA256"),
        "policy_checkpoint_realpath": getattr(policy, "checkpoint_realpath", None),
        "policy_runtime": getattr(policy, "runtime_identity", None),
        "video_encoding": {
            "container": "mp4",
            "codec": "h264",
            "codec_tag": "avc1",
            "pixel_format": "yuv420p",
            "faststart": True,
        },
        "groups": group_rows,
    }
    (staging / "records.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    (staging / "manifest.json").write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n")
    (staging / "quality_report.json").write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n")
    staging.rename(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), **quality}, sort_keys=True))
    if not quality["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
