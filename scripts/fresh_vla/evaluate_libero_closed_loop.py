from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from libero_full_episode_collector import object_grasped, upright_image
from libero_snapshot_collector import (
    DEFAULT_BDDL,
    _restore_snapshot,
    _set_object_offset,
    _step,
    robot_state_from_observation,
)
from video_io import write_h264_video


LANGUAGE = "put the cream cheese in the bowl"
SUBGOALS = ("grasp", "lift", "transport", "place")
LIFT_THRESHOLD = 0.015
TRANSPORT_XY_TOLERANCE = 0.08


def object_near_bowl(observation: Mapping[str, Any]) -> bool:
    obj = np.asarray(observation["cream_cheese_1_pos"], dtype=np.float64)
    bowl = np.asarray(observation["akita_black_bowl_1_pos"], dtype=np.float64)
    return bool(np.linalg.norm(obj[:2] - bowl[:2]) <= TRANSPORT_XY_TOLERANCE)


def update_subgoals(
    previous: Mapping[str, bool],
    observation: Mapping[str, Any],
    *,
    grasped: bool,
    success: bool,
    initial_object_z: float,
) -> dict[str, bool]:
    obj = np.asarray(observation["cream_cheese_1_pos"], dtype=np.float64)
    result = {name: bool(previous.get(name, False)) for name in SUBGOALS}
    result["grasp"] = result["grasp"] or bool(grasped)
    result["lift"] = result["lift"] or bool(obj[2] - initial_object_z >= LIFT_THRESHOLD)
    result["transport"] = result["transport"] or bool(result["lift"] and object_near_bowl(observation))
    result["place"] = result["place"] or bool(success)
    return result


def progress_fraction(subgoals: Mapping[str, bool]) -> float:
    return float(np.mean([bool(subgoals.get(name, False)) for name in SUBGOALS]))


def slip_offset_for_group(group: Mapping[str, Any]) -> Sequence[float]:
    slipped = group.get("branches", {}).get("slipped", {})
    return slipped.get("applied_slip_offset", group["source_randomization"]["slip_offset"])


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def is_failure_continuation(action: Sequence[float], *, grasped: bool) -> bool:
    value = np.asarray(action, dtype=np.float64)
    return bool(not grasped and value[2] > 0.15 and value[-1] > 0.0)


def is_recovery_action(
    action: Sequence[float],
    *,
    grasped: bool,
    eef_position: Sequence[float],
    object_position: Sequence[float],
) -> bool:
    value = np.asarray(action, dtype=np.float64)
    if grasped:
        return False
    if value[-1] < -0.2:
        return True
    direction = np.asarray(object_position, dtype=np.float64) - np.asarray(eef_position, dtype=np.float64)
    norm = np.linalg.norm(direction)
    return bool(norm > 1e-8 and np.dot(value[:3], direction / norm) > 0.2)


def is_premature_commitment(
    action: Sequence[float],
    *,
    grasped: bool,
    eef_position: Sequence[float],
    bowl_position: Sequence[float],
) -> bool:
    value = np.asarray(action, dtype=np.float64)
    if grasped or value[-1] <= 0.0:
        return False
    direction = np.asarray(bowl_position, dtype=np.float64) - np.asarray(eef_position, dtype=np.float64)
    norm = np.linalg.norm(direction)
    toward_bowl = norm > 1e-8 and np.dot(value[:3], direction / norm) > 0.2
    return bool(value[2] > 0.15 or toward_bowl)


def _policy_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "image": [
            upright_image(observation["agentview_image"]),
            upright_image(observation["robot0_eye_in_hand_image"]),
        ],
        "lang": LANGUAGE,
        "language": LANGUAGE,
        "state": robot_state_from_observation(observation).astype(np.float32),
    }


def _load_reference_arrays(
    episode_root: Path,
    group: Mapping[str, Any],
    outcome: str = "attached",
) -> dict[str, np.ndarray]:
    path = episode_root / group["episode_files"][outcome]
    with np.load(path, allow_pickle=False) as episode:
        return {
            key: np.asarray(episode[key])
            for key in (
                "actions",
                "eef_pose",
                "object_pose",
                "grasped",
                "sim_state",
                "model_body_pos",
                "object_friction",
                "gripper_action",
            )
        }


def _restore_recorded_state(
    env: Any,
    reference: Mapping[str, np.ndarray],
    index: int,
) -> Mapping[str, Any]:
    controller = {
        "model_body_pos": np.asarray(reference["model_body_pos"], dtype=np.float64),
        "object_friction": np.asarray(reference["object_friction"], dtype=np.float64),
        "gripper_action": np.asarray(reference["gripper_action"][index], dtype=np.float64),
    }
    return _restore_snapshot(env, np.asarray(reference["sim_state"][index]), controller)


def _prepare_isolated_feedback(
    env: Any,
    episode_root: Path,
    group: Mapping[str, Any],
    *,
    outcome: str,
) -> tuple[Mapping[str, Any], int]:
    event_time = int(group["feedback_reveal_time"])
    reference = _load_reference_arrays(episode_root, group, outcome)
    observation = _restore_recorded_state(env, reference, event_time)
    return observation, event_time


def _frame(observation: Mapping[str, Any]) -> np.ndarray:
    return np.concatenate(
        [upright_image(observation["agentview_image"]), upright_image(observation["robot0_eye_in_hand_image"])],
        axis=1,
    )


def _write_paired_video(path: Path, attached: Sequence[np.ndarray], slipped: Sequence[np.ndarray], fps: float = 10.0) -> None:
    import cv2

    if not attached or not slipped:
        raise ValueError("paired evaluation video requires both branches")
    height, branch_width = attached[0].shape[:2]

    def frames():
        for index in range(max(len(attached), len(slipped))):
            frame = np.full((height + 26, branch_width * 2, 3), 255, dtype=np.uint8)
            frame[26:, :branch_width] = attached[min(index, len(attached) - 1)]
            frame[26:, branch_width:] = slipped[min(index, len(slipped) - 1)]
            cv2.putText(frame, "attached", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            cv2.putText(frame, "slipped/recovery", (branch_width + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            yield frame

    write_h264_video(path, frames(), fps=fps)


class Pi05Policy:
    def __init__(self, checkpoint: Path, device: str):
        import torch

        from AlphaBrain.model.framework.base_framework import BaseFramework

        self.torch = torch
        self.device = device
        self.model = BaseFramework.from_pretrained(str(checkpoint))
        self.model = self.model.to(torch.bfloat16).to(device).eval()
        self.model.gripper_remap = False
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.horizon = int(self.model.action_horizon)

    def predict(self, observation: Mapping[str, Any], seed: int) -> np.ndarray:
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        with self.torch.inference_mode():
            output = self.model.predict_action(examples=[_policy_observation(observation)])
        actions = np.asarray(output["normalized_actions"][0], dtype=np.float32)
        if actions.shape != (self.horizon, 7):
            raise RuntimeError(f"unexpected Pi0.5 action shape: {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise RuntimeError("Pi0.5 predicted non-finite actions")
        return np.clip(actions, -1.0, 1.0)

    def predict_many(self, observation: Mapping[str, Any], seeds: Sequence[int]) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        actions = [self.predict(observation, int(seed)) for seed in seeds]
        return np.stack(actions), time.perf_counter() - started

    def close(self) -> None:
        pass


class RemotePi05Policy:
    def __init__(self, socket_path: Path):
        self.connection = Client(str(socket_path), family="AF_UNIX", authkey=b"fresh-vla-local")
        handshake = self.connection.recv()
        self.horizon = int(handshake["horizon"])
        self.checkpoint_realpath = str(handshake.get("checkpoint_realpath", ""))
        self.model_size_bytes = int(handshake.get("model_size_bytes", 0))
        self.runtime_identity = {
            "torch_version": handshake.get("torch_version"),
            "cuda_version": handshake.get("cuda_version"),
            "device_name": handshake.get("device_name"),
        }

    def predict(self, observation: Mapping[str, Any], seed: int) -> np.ndarray:
        self.connection.send({"op": "predict", "seed": int(seed), "example": _policy_observation(observation)})
        response = self.connection.recv()
        if "error" in response:
            raise RuntimeError(f"remote Pi0.5 inference failed: {response['error']}")
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.shape != (self.horizon, 7):
            raise RuntimeError(f"remote Pi0.5 action shape changed: {actions.shape}")
        return actions

    def predict_many(self, observation: Mapping[str, Any], seeds: Sequence[int]) -> tuple[np.ndarray, float]:
        self.connection.send(
            {
                "op": "predict_many",
                "seeds": [int(seed) for seed in seeds],
                "example": _policy_observation(observation),
            }
        )
        response = self.connection.recv()
        if "error" in response:
            raise RuntimeError(f"remote Pi0.5 inference failed: {response['error']}")
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 3 or actions.shape[1:] != (self.horizon, 7) or not np.all(np.isfinite(actions)):
            raise RuntimeError(f"remote Pi0.5 sampled action shape changed: {actions.shape}")
        return actions, float(response["predict_action_wall_seconds"])

    def close(self) -> None:
        try:
            self.connection.send({"op": "close"})
        finally:
            self.connection.close()


def run_episode(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    episode_root: Path,
    group: Mapping[str, Any],
    *,
    evaluation: str,
    outcome: str,
    execution_horizon: int,
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

    while completion_steps < max_steps and not success:
        chunk = policy.predict(observation, stable_seed(noise_seed, group["pair_id"], replan_count))
        replan_count += 1
        for action in chunk[:execution_horizon]:
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
            completion_steps += 1
            success = bool(env.check_success())
            grasped_after = object_grasped(env)
            forced_slip_this_step = False

            if evaluation == "end_to_end" and event_time is None:
                object_lift = float(observation["cream_cheese_1_pos"][2]) - initial_object_z
                if grasped_after and object_lift >= LIFT_THRESHOLD:
                    event_time = completion_steps
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

            if record_video:
                frames.append(_frame(observation))
            if success or completion_steps >= max_steps:
                break

    eligible = outcome == "slipped" and event_time is not None
    switch_latency = None
    if eligible:
        switch_latency = (
            first_recovery_step - event_time
            if first_recovery_step is not None
            else completion_steps - event_time
        )
    return {
        "pair_id": group["pair_id"],
        "split": group["split"],
        "evaluation": evaluation,
        "branch_outcome": outcome,
        "execution_horizon": execution_horizon,
        "success": success,
        "recovery_success": bool(outcome == "slipped" and intervention_triggered and success),
        "intervention_triggered": intervention_triggered,
        "event_time": event_time,
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
        "replan_count": replan_count,
    }, frames


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attached = [row for row in rows if row["branch_outcome"] == "attached"]
    slipped = [row for row in rows if row["branch_outcome"] == "slipped"]
    return {
        "episode_count": len(rows),
        "group_count": len({row["pair_id"] for row in rows}),
        "overall_task_success": _mean(rows, "success"),
        "attached_task_success": _mean(attached, "success"),
        "slip_final_recovery_success": _mean(slipped, "recovery_success"),
        "failure_continuation_rate": _mean(slipped, "failure_continuation"),
        "premature_commitment_rate": _mean(slipped, "premature_commitment"),
        "mean_recovery_switch_latency": _mean(slipped, "recovery_switch_latency"),
        "recovery_switch_observed_rate": _mean(slipped, "recovery_switch_observed"),
        "slip_regrasp_success_rate": _mean(slipped, "regrasp_success"),
        "overall_drop_rate": _mean(rows, "drop"),
        "attached_drop_rate": _mean(attached, "drop"),
        "slip_drop_rate": _mean(slipped, "drop"),
        "mean_final_progress": _mean(rows, "final_progress"),
        "attached_final_progress": _mean(attached, "final_progress"),
        "slip_final_progress": _mean(slipped, "final_progress"),
        "mean_progress_auc": _mean(rows, "progress_auc"),
        "grasp_subgoal_rate": _mean(rows, "grasp_subgoal"),
        "lift_subgoal_rate": _mean(rows, "lift_subgoal"),
        "transport_subgoal_rate": _mean(rows, "transport_subgoal"),
        "place_subgoal_rate": _mean(rows, "place_subgoal"),
        "event_trigger_rate": _mean(
            [{**row, "triggered": row.get("event_time") is not None} for row in rows], "triggered"
        ),
        "mean_completion_steps": _mean(rows, "completion_steps"),
        "normal_no_intervention_success": _mean(attached, "success"),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _evaluation_payload(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int,
    status: str,
) -> dict[str, Any]:
    by_k = {
        str(execution_horizon): summarize_rows(
            [row for row in rows if row["execution_horizon"] == execution_horizon]
        )
        for execution_horizon in args.execution_horizons
    }
    return {
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "policy_socket": None if args.policy_socket is None else str(args.policy_socket),
        "episode_root": str(args.episode_root),
        "evaluation": args.evaluation,
        "split": args.split,
        "seed": args.seed,
        "status": status,
        "completed_rows": len(rows),
        "expected_rows": expected_rows,
        "metric_definitions": {
            "failure_continuation": "after slip feedback, ungrasped action keeps closed gripper and positive lift > 0.15",
            "premature_commitment": "after slip feedback, ungrasped closed-gripper action lifts or moves toward bowl before recovery",
            "recovery_switch_latency": "executed actions from feedback reveal to first open-gripper or move-toward-object action",
            "regrasp_success": "slip branch regains a valid object grasp after feedback reveal",
            "drop": "a held object is lost outside the bowl, excluding the forced-slip intervention",
            "final_progress": "fraction of cumulative grasp/lift/transport/place subgoals reached at episode end",
            "progress_auc": "mean cumulative subgoal fraction over the closed-loop episode",
            "transport_subgoal": "lift has occurred and object XY distance to bowl is at most 0.08 m",
        },
        "summary": by_k,
        "rows": list(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-K closed-loop Pi0.5 evaluation on complete LIBERO branches")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-socket", type=Path)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation", choices=("isolated", "end_to_end"), required=True)
    parser.add_argument("--execution-horizons", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-groups", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if any(value not in {1, 2, 3} for value in args.execution_horizons):
        raise ValueError("execution horizons must be selected from 1, 2, 3")
    os.environ.setdefault("PRETRAINED_MODELS_DIR", "/share/longjunyu/alphabrain/pretrained_models")
    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    groups = [group for group in manifest["groups"] if group["split"] == args.split]
    groups = sorted(groups, key=lambda row: row["pair_id"])
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    if not groups:
        raise ValueError(f"no groups for split={args.split!r}")
    if (args.checkpoint is None) == (args.policy_socket is None):
        raise ValueError("provide exactly one of --checkpoint or --policy-socket")
    policy = (
        RemotePi05Policy(args.policy_socket)
        if args.policy_socket is not None
        else Pi05Policy(args.checkpoint, args.device)
    )
    if max(args.execution_horizons) > policy.horizon:
        raise ValueError("execution horizon exceeds policy chunk")

    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(manifest.get("bddl", DEFAULT_BDDL))),
        camera_heights=224,
        camera_widths=224,
    )
    env.seed(args.seed)
    rows = []
    expected_rows = len(args.execution_horizons) * len(groups) * 2
    partial_output = args.output.with_name(f"{args.output.stem}.partial{args.output.suffix}")
    videos: dict[tuple[int, str], dict[str, list[np.ndarray]]] = defaultdict(dict)
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)
    try:
        for execution_horizon in args.execution_horizons:
            for group_index, group in enumerate(groups):
                for outcome in ("attached", "slipped"):
                    record_video = args.video_dir is not None and group_index < args.video_groups
                    row, frames = run_episode(
                        env,
                        policy,
                        args.episode_root,
                        group,
                        evaluation=args.evaluation,
                        outcome=outcome,
                        execution_horizon=execution_horizon,
                        max_steps=args.max_steps,
                        noise_seed=args.seed,
                        record_video=record_video,
                    )
                    rows.append(row)
                    if record_video:
                        video_key = (execution_horizon, group["pair_id"])
                        videos[video_key][outcome] = frames
                        if videos[video_key].keys() >= {"attached", "slipped"}:
                            _write_paired_video(
                                args.video_dir / f"{args.evaluation}-k{execution_horizon}-{group['pair_id']}.mp4",
                                videos[video_key]["attached"],
                                videos[video_key]["slipped"],
                            )
                            del videos[video_key]
                    _atomic_write_json(
                        partial_output,
                        _evaluation_payload(args, rows, expected_rows=expected_rows, status="partial"),
                    )
                    print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        env.close()
        policy.close()

    result = _evaluation_payload(args, rows, expected_rows=expected_rows, status="complete")
    _atomic_write_json(args.output, result)
    partial_output.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "summary": result["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
