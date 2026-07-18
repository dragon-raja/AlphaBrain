from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.fresh_vla.evaluate_libero_closed_loop import _restore_recorded_state
from scripts.fresh_vla.libero_full_episode_collector import (
    FullEpisodeTeacher,
    object_grasped,
    upright_image,
)
from scripts.fresh_vla.libero_snapshot_collector import DEFAULT_BDDL, _set_object_offset, _step, robot_state_from_observation
from scripts.fresh_vla.video_io import write_h264_video
from scripts.verify_vla.probe_gate_common import OUTCOMES, PROBE_NAMES, detach_offsets, pixel_mae, probe_actions


DEFAULT_EPISODE_ROOT = Path("/share/longjunyu/fresh-vla/libero-full-episode-v2-128")
DEFAULT_OUTPUT_ROOT = Path("/share/longjunyu/verify-vla/gate0-probe-v1")
MAX_INITIAL_PIXEL_MAE = 2.0
MAX_INITIAL_STATE_ERROR = 1e-7


def _assert_unsealed_path(path: Path) -> None:
    forbidden = {"test", "tests", "confirmation", "confirm", "sealed"}
    lowered = {part.lower() for part in path.parts}
    if lowered & forbidden or any("confirmation" in part for part in lowered):
        raise ValueError(f"refusing to access sealed path: {path}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _load_reference_arrays(episode_root: Path, group: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = episode_root / str(group["episode_files"]["attached"])
    with np.load(path, allow_pickle=False) as episode:
        return {
            key: np.asarray(episode[key])
            for key in (
                "sim_state",
                "model_body_pos",
                "object_friction",
                "gripper_action",
            )
        }


def _frame(observation: Mapping[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            upright_image(observation["agentview_image"]),
            upright_image(observation["robot0_eye_in_hand_image"]),
        ),
        axis=1,
    )


def _paired_frames(
    attached: Sequence[np.ndarray],
    detached: Sequence[np.ndarray],
) -> Sequence[np.ndarray]:
    import cv2

    if not attached or not detached:
        raise ValueError("paired video needs both outcome traces")
    branch_height, branch_width = attached[0].shape[:2]
    frames = []
    for index in range(max(len(attached), len(detached))):
        frame = np.full((branch_height + 28, branch_width * 2, 3), 255, dtype=np.uint8)
        frame[28:, :branch_width] = attached[min(index, len(attached) - 1)]
        frame[28:, branch_width:] = detached[min(index, len(detached) - 1)]
        cv2.putText(frame, "attached", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.putText(
            frame,
            "latent detached",
            (branch_width + 8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
        )
        frames.append(frame)
    return frames


def _restore_outcome(
    env: Any,
    reference: Mapping[str, np.ndarray],
    index: int,
    outcome: str,
    detached_offset: Sequence[float],
) -> Mapping[str, Any]:
    observation = _restore_recorded_state(env, reference, index)
    if outcome == "detached":
        observation = _set_object_offset(env, detached_offset)
    grasped = object_grasped(env)
    expected = outcome == "attached"
    if grasped != expected:
        raise RuntimeError(f"restored {outcome} grasp predicate is {grasped}, expected {expected}")
    return observation


def _find_detached_twin(
    env: Any,
    reference: Mapping[str, np.ndarray],
    index: int,
    direction_xy: Sequence[float],
) -> tuple[np.ndarray, dict[str, float]]:
    attached = _restore_recorded_state(env, reference, index)
    if not object_grasped(env):
        raise RuntimeError("attached source state is not grasped")
    attached_agent = upright_image(attached["agentview_image"])
    attached_wrist = upright_image(attached["robot0_eye_in_hand_image"])
    attached_state = robot_state_from_observation(attached)
    for offset in detach_offsets(direction_xy):
        _restore_recorded_state(env, reference, index)
        detached = _set_object_offset(env, offset)
        if object_grasped(env):
            continue
        agent_mae = pixel_mae(attached_agent, upright_image(detached["agentview_image"]))
        wrist_mae = pixel_mae(attached_wrist, upright_image(detached["robot0_eye_in_hand_image"]))
        state_error = float(
            np.max(np.abs(attached_state - robot_state_from_observation(detached)))
        )
        metrics = {
            "agentview_pixel_mae_raw_255": agent_mae,
            "wrist_pixel_mae_raw_255": wrist_mae,
            "double_view_pixel_mae_raw_255": 0.5 * (agent_mae + wrist_mae),
            "robot_state_max_abs_error": state_error,
            "offset_magnitude_m": float(np.linalg.norm(offset)),
        }
        if metrics["double_view_pixel_mae_raw_255"] > MAX_INITIAL_PIXEL_MAE:
            raise RuntimeError(
                "minimum detached twin exceeds frozen visual tolerance: "
                f"{metrics['double_view_pixel_mae_raw_255']:.6f} > {MAX_INITIAL_PIXEL_MAE}"
            )
        if state_error > MAX_INITIAL_STATE_ERROR:
            raise RuntimeError(
                f"detached twin changes robot state: {state_error:.3e} > {MAX_INITIAL_STATE_ERROR:.3e}"
            )
        return offset, metrics
    raise RuntimeError("no detached twin found on the frozen 0.25 mm grid through 3 mm")


def _run_teacher(
    env: Any,
    observation: Mapping[str, Any],
    *,
    max_steps: int,
    video_frames: list[np.ndarray] | None,
) -> dict[str, Any]:
    teacher = FullEpisodeTeacher(observation)
    completion_steps = 0
    phases: list[str] = []
    error = None
    try:
        for _ in range(max_steps):
            success = bool(env.check_success())
            decision = teacher.decide(
                observation,
                grasped=object_grasped(env),
                success=success,
            )
            phases.append(decision.phase)
            if teacher.done:
                break
            observation = _step(env, decision.action)
            completion_steps += 1
            if video_frames is not None:
                video_frames.append(_frame(observation))
        else:
            error = f"teacher exceeded max_steps={max_steps} in phase={teacher.phase}"
    except Exception as exception:  # Preserve a failed viability endpoint for group-level analysis.
        error = f"{type(exception).__name__}: {exception}"
    return {
        "success": bool(env.check_success()) if error is None else False,
        "completion_steps": completion_steps,
        "final_phase": teacher.phase,
        "regrasp_attempts": teacher.regrasp_attempts,
        "error": error,
        "phase_trace": phases,
    }


def _run_condition(
    env: Any,
    reference: Mapping[str, np.ndarray],
    index: int,
    outcome: str,
    detached_offset: Sequence[float],
    actions: np.ndarray,
    *,
    evaluate_viability: bool,
    max_teacher_steps: int,
    record_video: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[np.ndarray]]:
    observation = _restore_outcome(env, reference, index, outcome, detached_offset)
    initial_object = np.asarray(observation["cream_cheese_1_pos"], dtype=np.float64).copy()
    arrays: dict[str, list[np.ndarray | bool]] = {
        "agentview": [],
        "wrist": [],
        "robot_state": [],
        "eef_position": [],
        "object_position": [],
        "grasped": [],
        "success": [],
    }
    video_frames = [_frame(observation)] if record_video else []

    def append() -> None:
        arrays["agentview"].append(upright_image(observation["agentview_image"]))
        arrays["wrist"].append(upright_image(observation["robot0_eye_in_hand_image"]))
        arrays["robot_state"].append(robot_state_from_observation(observation).astype(np.float32))
        arrays["eef_position"].append(np.asarray(observation["robot0_eef_pos"], dtype=np.float32))
        arrays["object_position"].append(
            np.asarray(observation["cream_cheese_1_pos"], dtype=np.float32)
        )
        arrays["grasped"].append(object_grasped(env))
        arrays["success"].append(bool(env.check_success()))

    append()
    for action in actions:
        observation = _step(env, action)
        append()
        if record_video:
            video_frames.append(_frame(observation))

    endpoint_object = np.asarray(observation["cream_cheese_1_pos"], dtype=np.float64)
    endpoint = {
        "grasped": object_grasped(env),
        "success": bool(env.check_success()),
        "object_displacement_m": float(np.linalg.norm(endpoint_object - initial_object)),
        "eef_position": np.asarray(observation["robot0_eef_pos"]).round(8).tolist(),
        "object_position": endpoint_object.round(8).tolist(),
    }
    viability = None
    if evaluate_viability:
        viability = _run_teacher(
            env,
            observation,
            max_steps=max_teacher_steps,
            video_frames=video_frames if record_video else None,
        )
    endpoint["teacher_viability"] = viability
    return (
        {key: np.asarray(values) for key, values in arrays.items()},
        endpoint,
        video_frames,
    )


def collect_group(
    env: Any,
    episode_root: Path,
    output_root: Path,
    group: Mapping[str, Any],
    *,
    max_teacher_steps: int,
    write_videos: bool,
) -> dict[str, Any]:
    pair_id = str(group["pair_id"])
    reference = _load_reference_arrays(episode_root, group)
    prefix_steps = int(group["prefix_steps"])
    slip_offset = np.asarray(group["source_randomization"]["slip_offset"], dtype=np.float64)
    detached_offset, twin_metrics = _find_detached_twin(
        env,
        reference,
        prefix_steps,
        slip_offset[:2],
    )
    evaluate_viability = str(group["split"]) == "val"
    record_video = bool(write_videos and evaluate_viability)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "valid",
        "pair_id": pair_id,
        "split": str(group["split"]),
        "source_initial_state_index": int(group["source_initial_state_index"]),
        "prefix_steps": prefix_steps,
        "detached_offset_m": detached_offset.tolist(),
        "twin_metrics": twin_metrics,
        "probe_steps": 4,
        "teacher_max_steps": max_teacher_steps,
        "teacher_is_privileged_viability_upper_bound": True,
        "policy_inputs_include_outcome_or_privileged_state": False,
        "conditions": {},
    }

    no_probe_traces: dict[str, list[np.ndarray]] = {}
    metadata["conditions"]["no_probe"] = {}
    for outcome in OUTCOMES:
        _, endpoint, frames = _run_condition(
            env,
            reference,
            prefix_steps,
            outcome,
            detached_offset,
            np.empty((0, 7), dtype=np.float32),
            evaluate_viability=evaluate_viability,
            max_teacher_steps=max_teacher_steps,
            record_video=record_video,
        )
        metadata["conditions"]["no_probe"][outcome] = endpoint
        no_probe_traces[outcome] = frames
    if record_video:
        write_h264_video(
            output_root / "videos" / pair_id / "no_probe.mp4",
            _paired_frames(no_probe_traces["attached"], no_probe_traces["detached"]),
            fps=10.0,
        )

    probe_arrays: dict[str, list[np.ndarray]] = {
        "agentview": [],
        "wrist": [],
        "robot_state": [],
        "eef_position": [],
        "object_position": [],
        "grasped": [],
        "success": [],
    }
    for probe_name in PROBE_NAMES:
        outcome_arrays: dict[str, list[np.ndarray]] = {key: [] for key in probe_arrays}
        traces: dict[str, list[np.ndarray]] = {}
        metadata["conditions"][probe_name] = {}
        for outcome in OUTCOMES:
            arrays, endpoint, frames = _run_condition(
                env,
                reference,
                prefix_steps,
                outcome,
                detached_offset,
                probe_actions(probe_name),
                evaluate_viability=evaluate_viability,
                max_teacher_steps=max_teacher_steps,
                record_video=record_video,
            )
            for key, value in arrays.items():
                outcome_arrays[key].append(value)
            metadata["conditions"][probe_name][outcome] = endpoint
            traces[outcome] = frames
        for key in probe_arrays:
            probe_arrays[key].append(np.stack(outcome_arrays[key], axis=0))
        if record_video:
            write_h264_video(
                output_root / "videos" / pair_id / f"{probe_name}.mp4",
                _paired_frames(traces["attached"], traces["detached"]),
                fps=10.0,
            )

    arrays_payload = {
        "probe_names": np.asarray(PROBE_NAMES, dtype="U24"),
        "outcome_names": np.asarray(OUTCOMES, dtype="U16"),
        **{key: np.stack(values, axis=0) for key, values in probe_arrays.items()},
    }
    _atomic_npz(output_root / "records" / f"{pair_id}.npz", arrays_payload)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect preregistered latent-contact VERIFY-VLA Gate 0 probes")
    parser.add_argument("--episode-root", type=Path, default=DEFAULT_EPISODE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--max-teacher-steps", type=int, default=320)
    parser.add_argument("--seed", type=int, default=260718)
    parser.add_argument("--write-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _assert_unsealed_path(args.episode_root)
    _assert_unsealed_path(args.output_root)
    if set(args.splits) - {"train", "val"}:
        raise ValueError("Gate 0 may only collect train and val splits")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    if args.max_teacher_steps < 1:
        raise ValueError("max teacher steps must be positive")

    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    groups = sorted(
        (group for group in manifest["groups"] if group["split"] in set(args.splits)),
        key=lambda group: str(group["pair_id"]),
    )
    groups = groups[args.shard_index :: args.shard_count]
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    if not groups:
        raise ValueError("no groups selected")

    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(manifest.get("bddl", DEFAULT_BDDL))),
        camera_heights=224,
        camera_widths=224,
    )
    env.seed(args.seed + args.shard_index)
    args.output_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    invalid = 0
    try:
        for group in groups:
            pair_id = str(group["pair_id"])
            metadata_path = args.output_root / "records" / f"{pair_id}.json"
            arrays_path = args.output_root / "records" / f"{pair_id}.npz"
            if metadata_path.exists() and not args.overwrite:
                previous = json.loads(metadata_path.read_text())
                if previous.get("status") == "invalid" or arrays_path.exists():
                    completed += 1
                    continue
            try:
                payload = collect_group(
                    env,
                    args.episode_root,
                    args.output_root,
                    group,
                    max_teacher_steps=args.max_teacher_steps,
                    write_videos=args.write_videos,
                )
            except Exception as exception:
                invalid += 1
                payload = {
                    "schema_version": 1,
                    "status": "invalid",
                    "pair_id": pair_id,
                    "split": str(group["split"]),
                    "source_initial_state_index": int(group["source_initial_state_index"]),
                    "error": f"{type(exception).__name__}: {exception}",
                    "traceback": traceback.format_exc(),
                }
                _atomic_json(metadata_path, payload)
                print(json.dumps({"pair_id": pair_id, "status": "invalid", "error": payload["error"]}), flush=True)
                if args.fail_fast:
                    raise
            else:
                _atomic_json(metadata_path, payload)
                print(json.dumps({"pair_id": pair_id, "status": "valid"}), flush=True)
            completed += 1
    finally:
        env.close()

    shard_payload = {
        "schema_version": 1,
        "status": "complete",
        "episode_root": str(args.episode_root),
        "output_root": str(args.output_root),
        "splits": list(args.splits),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "selected_groups": len(groups),
        "completed_groups": completed,
        "invalid_groups_in_this_invocation": invalid,
        "write_videos": args.write_videos,
    }
    _atomic_json(args.output_root / "shards" / f"shard-{args.shard_index:02d}.json", shard_payload)
    print(json.dumps(shard_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
