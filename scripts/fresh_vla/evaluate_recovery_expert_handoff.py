from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
from evaluate_physical_process_oracle import (
    _physical_state,
    aggregate_outcomes,
    capture_runtime_snapshot,
    restore_runtime_snapshot,
)
from evaluate_recovery_segment_oracle import (
    EXPECTED_CHECKPOINT_SHA256,
    _observation_frame,
    summarize_recovery_trace,
)
from libero_full_episode_collector import FullEpisodeTeacher, object_grasped
from libero_snapshot_collector import DEFAULT_BDDL, _step
from video_io import write_h264_video


HANDOFF_METHODS = (
    "policy_only",
    "teacher_h3",
    "teacher_h12",
    "teacher_to_regrasp",
    "teacher_to_lift",
    "teacher_to_transport",
)
EXPERT_SANITY_METHOD = "teacher_full"
DECISION_TOTAL_ACTION_BUDGET = 320
DECISION_MAX_TEACHER_ACTIONS = 320
TEACHER_ACTION_SOURCE = "reconstructed_original_full_episode_teacher_state"
FEEDBACK_SNAPSHOT_PROTOCOL = (
    "replay_recorded_prefix_then_inject_recorded_post_slip_sim_state"
)
FIXED_TEACHER_ACTIONS = {
    "policy_only": 0,
    "teacher_h3": 3,
    "teacher_h12": 12,
}


def _aligned(action_count: int, execution_horizon: int) -> bool:
    return action_count % execution_horizon == 0


def reconstruct_feedback_snapshot(
    env: Any,
    reference: Mapping[str, np.ndarray],
    feedback_index: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    actions = np.asarray(reference["actions"], dtype=np.float64)
    sim_states = np.asarray(reference["sim_state"], dtype=np.float64)
    gripper_actions = np.asarray(reference["gripper_action"], dtype=np.float64)
    if not 0 <= feedback_index <= len(actions):
        raise ValueError(
            f"feedback index {feedback_index} is outside action trajectory length {len(actions)}"
        )
    if len(sim_states) != len(actions) + 1 or len(gripper_actions) != len(sim_states):
        raise ValueError("recorded feedback trajectory violates T actions / T+1 states")

    observation = _restore_recorded_state(env, reference, 0)
    for action in actions[:feedback_index]:
        observation = _step(env, action)

    expected_gripper = gripper_actions[feedback_index].reshape(-1)
    actual_gripper = np.asarray(
        env.robots[0].gripper.current_action,
        dtype=np.float64,
    ).reshape(-1)
    gripper_delta = float(np.max(np.abs(actual_gripper - expected_gripper)))
    if gripper_delta > 1e-6:
        raise RuntimeError(
            f"feedback prefix replay gripper mismatch: max_abs_delta={gripper_delta}"
        )

    observation = env.regenerate_obs_from_state(sim_states[feedback_index])
    env.robots[0].gripper.current_action = expected_gripper.copy()
    sim_delta = float(
        np.max(
            np.abs(
                np.asarray(env.get_sim_state(), dtype=np.float64)
                - sim_states[feedback_index]
            )
        )
    )
    if sim_delta > 1e-10:
        raise RuntimeError(
            f"recorded post-slip state injection mismatch: max_abs_delta={sim_delta}"
        )
    snapshot = capture_runtime_snapshot(env)
    return observation, snapshot, {
        "protocol": FEEDBACK_SNAPSHOT_PROTOCOL,
        "prefix_actions_replayed": feedback_index,
        "post_injection_sim_max_abs_delta": sim_delta,
        "prefix_gripper_max_abs_delta": gripper_delta,
    }


def reconstruct_teacher_state(
    episode_root: Path,
    group: Mapping[str, Any],
    reference: Mapping[str, np.ndarray],
    feedback_index: int,
) -> dict[str, Any]:
    branch_start = int(group["prefix_steps"])
    if not 0 <= branch_start < feedback_index:
        raise ValueError("teacher branch start must precede feedback")
    if int(group["event_time"]) != feedback_index:
        raise ValueError("teacher reconstruction requires event-aligned feedback")

    episode_path = episode_root / str(group["episode_files"]["slipped"])
    with np.load(episode_path, allow_pickle=False) as episode:
        observation_phases = np.asarray(episode["teacher_phase"])
        action_phases = np.asarray(episode["action_phases"])
    if len(observation_phases) != len(action_phases) + 1:
        raise ValueError("teacher phase trajectory violates T actions / T+1 observations")

    phase = str(observation_phases[feedback_index])
    phase_steps = 0
    for action_phase in action_phases[:feedback_index][::-1]:
        if str(action_phase) != phase:
            break
        phase_steps += 1
    if phase != "lift" or any(
        str(value).startswith("recover_")
        for value in action_phases[branch_start:feedback_index]
    ):
        raise ValueError("feedback must occur before the original teacher enters recovery")

    return {
        "source": TEACHER_ACTION_SOURCE,
        "branch_start_index": branch_start,
        "phase": phase,
        "phase_steps": phase_steps,
        "regrasp_attempts": 0,
        "place_attempts": 0,
        "initial_eef_xy": np.asarray(
            reference["eef_pose"][branch_start, :2],
            dtype=np.float64,
        ).tolist(),
        "initial_object_z": float(reference["object_pose"][branch_start, 2]),
    }


def audit_reconstructed_feedback_observation(
    episode_root: Path,
    group: Mapping[str, Any],
    feedback_index: int,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    episode_path = episode_root / str(group["episode_files"]["slipped"])
    with np.load(episode_path, allow_pickle=False) as episode:
        expected_images = np.stack(
            [episode["agentview"][feedback_index], episode["wrist"][feedback_index]]
        ).astype(np.uint8)
        expected_state = np.asarray(
            episode["robot_state"][feedback_index],
            dtype=np.float32,
        )
    policy_observation = _policy_observation(observation)
    actual_images = np.stack(policy_observation["image"]).astype(np.uint8)
    actual_state = np.asarray(policy_observation["state"], dtype=np.float32)
    image_delta = int(
        np.max(
            np.abs(actual_images.astype(np.int16) - expected_images.astype(np.int16))
        )
    )
    state_delta = float(np.max(np.abs(actual_state - expected_state)))
    if image_delta != 0 or state_delta > 1e-6:
        raise RuntimeError(
            "reconstructed feedback observation disagrees with the recorded policy input: "
            f"image_delta={image_delta}, state_delta={state_delta}"
        )
    return {
        "policy_image_max_abs_delta": image_delta,
        "policy_robot_state_max_abs_delta": state_delta,
    }


def make_reconstructed_teacher(
    observation: Mapping[str, Any],
    state: Mapping[str, Any],
) -> FullEpisodeTeacher:
    teacher = FullEpisodeTeacher(observation)
    teacher.phase = str(state["phase"])
    teacher.phase_steps = int(state["phase_steps"])
    teacher.regrasp_attempts = int(state["regrasp_attempts"])
    teacher.place_attempts = int(state["place_attempts"])
    teacher.initial_eef_xy = np.asarray(state["initial_eef_xy"], dtype=np.float64).copy()
    teacher.initial_object_z = float(state["initial_object_z"])
    teacher.done = False
    return teacher


def _milestone_reached(
    method: str,
    *,
    regrasp_reached: bool,
    lift_reached: bool,
    transport_reached: bool,
) -> bool:
    if method == "teacher_to_regrasp":
        return regrasp_reached
    if method == "teacher_to_lift":
        return lift_reached
    if method == "teacher_to_transport":
        return transport_reached
    raise ValueError(f"method {method!r} has no physical milestone")


def _update_milestones(
    state: Mapping[str, Any],
    *,
    initial_z: float,
    dwell_steps: int,
    grasp_run: int,
    lift_run: int,
    transport_run: int,
    regrasp_reached: bool,
    lift_reached: bool,
    transport_reached: bool,
) -> tuple[int, int, int, bool, bool, bool]:
    grasped = bool(state["grasped"])
    grasp_run = grasp_run + 1 if grasped else 0
    regrasp_reached = regrasp_reached or grasp_run >= dwell_steps

    lifted = float(state["object_z"]) - initial_z >= LIFT_THRESHOLD
    lift_run = lift_run + 1 if regrasp_reached and grasped and lifted else 0
    lift_reached = lift_reached or lift_run >= dwell_steps

    near_bowl = float(state["bowl_xy_distance"]) <= TRANSPORT_XY_TOLERANCE
    transport_run = (
        transport_run + 1
        if lift_reached and near_bowl and (grasped or bool(state["success"]))
        else 0
    )
    transport_reached = transport_reached or transport_run >= dwell_steps
    return (
        grasp_run,
        lift_run,
        transport_run,
        regrasp_reached,
        lift_reached,
        transport_reached,
    )


def generate_teacher_endpoint(
    env: Any,
    feedback_snapshot: Mapping[str, Any],
    teacher_state: Mapping[str, Any],
    *,
    method: str,
    execution_horizon: int,
    total_action_budget: int,
    max_teacher_actions: int,
    stage_dwell_steps: int,
) -> dict[str, Any]:
    if method not in (*HANDOFF_METHODS, EXPERT_SANITY_METHOD):
        raise ValueError(f"unknown handoff method: {method}")
    observation = restore_runtime_snapshot(env, feedback_snapshot)
    teacher = make_reconstructed_teacher(observation, teacher_state)
    trace = [_physical_state(env, observation, bool(env.check_success()))]
    frames = [_observation_frame(observation)]
    initial_z = float(trace[0]["object_z"])
    grasp_run = lift_run = transport_run = 0
    regrasp_reached = lift_reached = transport_reached = False
    teacher_actions = 0
    criterion_reached = method == "policy_only"

    while not criterion_reached and not trace[-1]["success"]:
        if teacher_actions >= min(max_teacher_actions, total_action_budget):
            break
        decision = teacher.decide(
            observation,
            grasped=object_grasped(env),
            success=bool(env.check_success()),
        )
        if teacher.done and not bool(env.check_success()):
            break
        observation = _step(env, decision.action)
        teacher_actions += 1
        state = _physical_state(env, observation, bool(env.check_success()))
        trace.append(state)
        frames.append(_observation_frame(observation))
        (
            grasp_run,
            lift_run,
            transport_run,
            regrasp_reached,
            lift_reached,
            transport_reached,
        ) = _update_milestones(
            state,
            initial_z=initial_z,
            dwell_steps=stage_dwell_steps,
            grasp_run=grasp_run,
            lift_run=lift_run,
            transport_run=transport_run,
            regrasp_reached=regrasp_reached,
            lift_reached=lift_reached,
            transport_reached=transport_reached,
        )

        if method in FIXED_TEACHER_ACTIONS:
            criterion_reached = teacher_actions >= FIXED_TEACHER_ACTIONS[method]
        elif method == EXPERT_SANITY_METHOD:
            criterion_reached = bool(state["success"])
        else:
            criterion_reached = _milestone_reached(
                method,
                regrasp_reached=regrasp_reached,
                lift_reached=lift_reached,
                transport_reached=transport_reached,
            )
        if criterion_reached and method != EXPERT_SANITY_METHOD:
            criterion_reached = _aligned(teacher_actions, execution_horizon)

    return {
        "method": method,
        "endpoint_snapshot": capture_runtime_snapshot(env),
        "observation": observation,
        "trace": trace,
        "frames": frames,
        "teacher_actions": teacher_actions,
        "next_global_replan_index": int(
            np.ceil(teacher_actions / execution_horizon)
        ),
        "criterion_reached": bool(criterion_reached),
        "teacher_done": bool(teacher.done),
        "teacher_success": bool(trace[-1]["success"]),
        "regrasp_reached": regrasp_reached,
        "lift_reached": lift_reached,
        "transport_reached": transport_reached,
    }


def rollout_policy_continuation(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    endpoint: Mapping[str, Any],
    *,
    pair_id: str,
    seed: int,
    repeat: int,
    execution_horizon: int,
    total_action_budget: int,
    stage_dwell_steps: int,
    video_path: Path | None = None,
    restore_endpoint: bool = True,
) -> dict[str, Any]:
    observation = (
        restore_runtime_snapshot(env, endpoint["endpoint_snapshot"])
        if restore_endpoint
        else endpoint["observation"]
    )
    trace = [dict(state) for state in endpoint["trace"]]
    frames = [np.asarray(frame) for frame in endpoint["frames"]]
    executed_actions = len(trace) - 1
    replan_index = int(endpoint["next_global_replan_index"])
    policy_calls = 0

    while executed_actions < total_action_budget and not trace[-1]["success"]:
        chunk = policy.predict(
            observation,
            stable_seed(
                seed,
                pair_id,
                "expert_handoff",
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
            executed_actions += 1
            trace.append(_physical_state(env, observation, bool(env.check_success())))
            frames.append(_observation_frame(observation))
            if trace[-1]["success"]:
                break

    if video_path is not None:
        write_h264_video(video_path, frames, fps=10.0)
    return {
        "repeat": repeat,
        "teacher_actions": int(endpoint["teacher_actions"]),
        "policy_calls": policy_calls,
        "policy_actions": executed_actions - int(endpoint["teacher_actions"]),
        "executed_actions": executed_actions,
        "outcome": summarize_recovery_trace(
            trace,
            stage_dwell_steps=stage_dwell_steps,
        ),
        **({"video_file": str(video_path)} if video_path is not None else {}),
    }


def summarize_method(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    endpoint: Mapping[str, Any],
    *,
    pair_id: str,
    seed: int,
    continuation_count: int,
    execution_horizon: int,
    total_action_budget: int,
    stage_dwell_steps: int,
    video_dir: Path | None,
    video_repeats: int,
    endpoint_factory: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if endpoint["method"] == EXPERT_SANITY_METHOD:
        video_path = None
        if video_dir is not None and video_repeats > 0:
            video_path = video_dir / (
                f"{pair_id}--seed{seed}--{endpoint['method']}--repeat0.mp4"
            )
            write_h264_video(video_path, endpoint["frames"], fps=10.0)
        outcome = summarize_recovery_trace(
            endpoint["trace"],
            stage_dwell_steps=stage_dwell_steps,
        )
        rows = [
            {
                "repeat": 0,
                "teacher_actions": int(endpoint["teacher_actions"]),
                "policy_calls": 0,
                "policy_actions": 0,
                "executed_actions": int(endpoint["teacher_actions"]),
                "outcome": outcome,
                **({"video_file": str(video_path)} if video_path is not None else {}),
            }
        ]
        return {
            "method": endpoint["method"],
            "teacher_actions": int(endpoint["teacher_actions"]),
            "criterion_reached": bool(endpoint["criterion_reached"]),
            "teacher_done": bool(endpoint["teacher_done"]),
            "teacher_success_before_policy": bool(endpoint["teacher_success"]),
            "teacher_prefix_regrasp_reached": bool(endpoint["regrasp_reached"]),
            "teacher_prefix_lift_reached": bool(endpoint["lift_reached"]),
            "teacher_prefix_transport_reached": bool(endpoint["transport_reached"]),
            "continuations": rows,
            "summary": aggregate_outcomes([outcome]),
        }

    rows = []
    for repeat in range(continuation_count):
        current_endpoint = endpoint if repeat == 0 else (
            endpoint_factory() if endpoint_factory is not None else endpoint
        )
        if endpoint_factory is not None:
            if (
                int(current_endpoint["teacher_actions"])
                != int(endpoint["teacher_actions"])
                or bool(current_endpoint["criterion_reached"])
                != bool(endpoint["criterion_reached"])
            ):
                raise RuntimeError("teacher prefix metadata changed across continuations")
            reference_sim = np.asarray(
                endpoint["endpoint_snapshot"]["sim_state"], dtype=np.float64
            )
            current_sim = np.asarray(
                current_endpoint["endpoint_snapshot"]["sim_state"], dtype=np.float64
            )
            sim_delta = float(np.max(np.abs(reference_sim - current_sim)))
            image_delta = int(
                np.max(
                    np.abs(
                        np.asarray(endpoint["frames"][-1]).astype(np.int16)
                        - np.asarray(current_endpoint["frames"][-1]).astype(np.int16)
                    )
                )
            )
            if sim_delta > 1e-8 or image_delta != 0:
                raise RuntimeError(
                    "teacher prefix endpoint changed across matched continuations"
                )
        video_path = None
        if video_dir is not None and repeat < video_repeats:
            video_path = video_dir / (
                f"{pair_id}--seed{seed}--{endpoint['method']}--repeat{repeat}.mp4"
            )
        rows.append(
            rollout_policy_continuation(
                env,
                policy,
                current_endpoint,
                pair_id=pair_id,
                seed=seed,
                repeat=repeat,
                execution_horizon=execution_horizon,
                total_action_budget=total_action_budget,
                stage_dwell_steps=stage_dwell_steps,
                video_path=video_path,
                restore_endpoint=endpoint_factory is None,
            )
        )
    return {
        "method": endpoint["method"],
        "teacher_actions": int(endpoint["teacher_actions"]),
        "criterion_reached": bool(endpoint["criterion_reached"]),
        "teacher_done": bool(endpoint["teacher_done"]),
        "teacher_success_before_policy": bool(endpoint["teacher_success"]),
        "teacher_prefix_regrasp_reached": bool(endpoint["regrasp_reached"]),
        "teacher_prefix_lift_reached": bool(endpoint["lift_reached"]),
        "teacher_prefix_transport_reached": bool(endpoint["transport_reached"]),
        "continuations": rows,
        "summary": aggregate_outcomes([row["outcome"] for row in rows]),
    }


def first_chunk_order_invariance(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    feedback_snapshot: Mapping[str, Any],
    *,
    pair_id: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observation = restore_runtime_snapshot(env, feedback_snapshot)
    policy_input = _policy_observation(observation)
    chunk = policy.predict(
        observation,
        stable_seed(seed, pair_id, "expert_handoff", 0, 0),
    )
    return (
        np.asarray(chunk, dtype=np.float64),
        np.stack(policy_input["image"]).astype(np.uint8),
        np.asarray(policy_input["state"], dtype=np.float64),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-budget expert handoff ladder for slipped Pi0.5 recovery"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-socket", type=Path)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-repeats", type=int, default=1)
    parser.add_argument("--run-kind", choices=("smoke", "decision"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--group-offset", type=int, default=0)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--execution-horizon", type=int, default=3)
    parser.add_argument(
        "--total-action-budget",
        type=int,
        default=DECISION_TOTAL_ACTION_BUDGET,
    )
    parser.add_argument(
        "--max-teacher-actions",
        type=int,
        default=DECISION_MAX_TEACHER_ACTIONS,
    )
    parser.add_argument("--continuations", type=int, default=5)
    parser.add_argument("--stage-dwell-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if (args.checkpoint is None) == (args.policy_socket is None):
        raise ValueError("provide exactly one of --checkpoint or --policy-socket")
    if args.execution_horizon < 1 or args.total_action_budget < 1:
        raise ValueError("action budgets must be positive")
    if args.max_teacher_actions < 12 or args.max_teacher_actions > args.total_action_budget:
        raise ValueError("max-teacher-actions must be in [12, total-action-budget]")
    if args.continuations < 1 or args.stage_dwell_steps < 1:
        raise ValueError("continuations and dwell steps must be positive")
    if args.video_repeats < 0 or args.video_repeats > args.continuations:
        raise ValueError("video-repeats must be between zero and continuations")
    if args.run_kind == "decision":
        expected_config = {
            "split": "val",
            "execution_horizon": 3,
            "total_action_budget": DECISION_TOTAL_ACTION_BUDGET,
            "max_teacher_actions": DECISION_MAX_TEACHER_ACTIONS,
            "continuations": 5,
            "stage_dwell_steps": 2,
        }
        for name, expected in expected_config.items():
            if getattr(args, name) != expected:
                raise ValueError(
                    f"decision run requires {name}={expected!r}, received {getattr(args, name)!r}"
                )
        if args.seed not in EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("decision run seed must be one of 41, 42, or 43")
        if os.environ.get("FRESH_GIT_DIRTY") != "0":
            raise ValueError("decision run requires an explicitly clean Git worktree")
        if os.environ.get("FRESH_CHECKPOINT_SHA256") != EXPECTED_CHECKPOINT_SHA256[
            args.seed
        ]:
            raise ValueError("decision run checkpoint SHA256 does not match seed")
    os.environ.setdefault(
        "PRETRAINED_MODELS_DIR",
        "/share/longjunyu/alphabrain/pretrained_models",
    )

    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    all_groups = sorted(
        (group for group in manifest["groups"] if group["split"] == args.split),
        key=lambda group: group["pair_id"],
    )
    groups = all_groups[args.group_offset :]
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    if not groups:
        raise ValueError(f"no groups for split={args.split!r}")
    if args.run_kind == "decision":
        source_count = len(
            {int(group["source_initial_state_index"]) for group in all_groups}
        )
        if len(all_groups) != 13 or source_count != 9:
            raise ValueError("decision run requires the frozen 13-group, 9-source val manifest")

    policy = (
        RemotePi05Policy(args.policy_socket)
        if args.policy_socket
        else Pi05Policy(args.checkpoint, args.device)
    )
    if args.execution_horizon > policy.horizon:
        raise ValueError("execution horizon exceeds policy action horizon")
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(manifest.get("bddl", DEFAULT_BDDL))),
        camera_heights=224,
        camera_widths=224,
    )
    env.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(f"{args.output.stem}.partial{args.output.suffix}")
    rows: list[dict[str, Any]] = []

    def payload(status: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": status,
            "run_kind": args.run_kind,
            "episode_root": str(args.episode_root),
            "split": args.split,
            "seed": args.seed,
            "group_offset": args.group_offset,
            "expected_rows": len(groups),
            "completed_rows": len(rows),
            "methods": list(HANDOFF_METHODS),
            "expert_sanity_method": EXPERT_SANITY_METHOD,
            "execution_horizon": args.execution_horizon,
            "total_action_budget": args.total_action_budget,
            "max_teacher_actions": args.max_teacher_actions,
            "continuations": args.continuations,
            "stage_dwell_steps": args.stage_dwell_steps,
            "teacher_is_privileged_upper_bound": True,
            "teacher_action_source": TEACHER_ACTION_SOURCE,
            "feedback_snapshot_protocol": FEEDBACK_SNAPSHOT_PROTOCOL,
            "teacher_privileged_inputs": [
                "grasp/contact state",
                "environment success",
                "recorded teacher phase at feedback",
            ],
            "policy_receives_teacher_or_branch_labels": False,
            "continuation_seed_protocol": (
                "same repeat and global replan index use the same policy seed across methods"
            ),
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
                "fps": 10.0,
            },
            "rows": rows,
        }

    try:
        for group in groups:
            reference = _load_reference_arrays(args.episode_root, group, "slipped")
            feedback_index = int(group["feedback_reveal_time"])
            (
                feedback_observation,
                feedback_snapshot,
                feedback_reconstruction,
            ) = reconstruct_feedback_snapshot(env, reference, feedback_index)
            feedback_reconstruction.update(
                audit_reconstructed_feedback_observation(
                    args.episode_root,
                    group,
                    feedback_index,
                    feedback_observation,
                )
            )
            teacher_state = reconstruct_teacher_state(
                args.episode_root,
                group,
                reference,
                feedback_index,
            )
            pair_id = str(group["pair_id"])

            before_chunk, before_images, before_state = first_chunk_order_invariance(
                env,
                policy,
                feedback_snapshot,
                pair_id=pair_id,
                seed=args.seed,
            )
            methods: dict[str, Any] = {}
            for method in (*HANDOFF_METHODS, EXPERT_SANITY_METHOD):
                def endpoint_factory(method_name: str = method) -> Mapping[str, Any]:
                    return generate_teacher_endpoint(
                        env,
                        feedback_snapshot,
                        teacher_state,
                        method=method_name,
                        execution_horizon=args.execution_horizon,
                        total_action_budget=args.total_action_budget,
                        max_teacher_actions=args.max_teacher_actions,
                        stage_dwell_steps=args.stage_dwell_steps,
                    )

                endpoint = endpoint_factory()
                methods[method] = summarize_method(
                    env,
                    policy,
                    endpoint,
                    pair_id=pair_id,
                    seed=args.seed,
                    continuation_count=(
                        1 if method == EXPERT_SANITY_METHOD else args.continuations
                    ),
                    execution_horizon=args.execution_horizon,
                    total_action_budget=args.total_action_budget,
                    stage_dwell_steps=args.stage_dwell_steps,
                    video_dir=args.video_dir,
                    video_repeats=min(
                        args.video_repeats,
                        1 if method == EXPERT_SANITY_METHOD else args.continuations,
                    ),
                    endpoint_factory=(
                        None if method == EXPERT_SANITY_METHOD else endpoint_factory
                    ),
                )

            after_chunk, after_images, after_state = first_chunk_order_invariance(
                env,
                policy,
                feedback_snapshot,
                pair_id=pair_id,
                seed=args.seed,
            )
            order_audit = {
                "first_chunk_max_abs_delta": float(
                    np.max(np.abs(before_chunk - after_chunk))
                ),
                "feedback_image_max_abs_delta": int(
                    np.max(
                        np.abs(before_images.astype(np.int16) - after_images.astype(np.int16))
                    )
                ),
                "feedback_robot_state_max_abs_delta": float(
                    np.max(np.abs(before_state - after_state))
                ),
            }
            if any(value != 0 for value in order_audit.values()):
                raise RuntimeError("method order changed the restored policy input or first chunk")

            row = {
                "pair_id": pair_id,
                "source_initial_state_index": int(group["source_initial_state_index"]),
                "feedback_state_index": feedback_index,
                "feedback_reconstruction": feedback_reconstruction,
                "teacher_state_reconstruction": teacher_state,
                "seed": args.seed,
                "method_order_invariance": order_audit,
                "methods": methods,
            }
            rows.append(row)
            _atomic_write_json(partial, payload("partial"))
            print(
                json.dumps(
                    {
                        "pair_id": pair_id,
                        "policy_only_success": methods["policy_only"]["summary"][
                            "success_rate"
                        ],
                        "teacher_to_regrasp_success": methods[
                            "teacher_to_regrasp"
                        ]["summary"]["success_rate"],
                        "teacher_to_transport_success": methods[
                            "teacher_to_transport"
                        ]["summary"]["success_rate"],
                        "teacher_full_success": methods[EXPERT_SANITY_METHOD]["summary"][
                            "success_rate"
                        ],
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
