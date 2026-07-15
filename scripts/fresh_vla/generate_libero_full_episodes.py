from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from libero_full_episode_collector import (
    collect_branch_continuation,
    collect_grasp_prefix,
    first_persistent_action_divergence,
    first_visual_reveal,
    merge_prefix_and_continuation,
)
from libero_snapshot_collector import DEFAULT_BDDL, DEFAULT_INIT_STATES
from video_io import write_h264_video


def slip_offset_candidates(offset: Sequence[float], count: int = 8) -> list[np.ndarray]:
    value = np.asarray(offset, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError("slip offset must be a three-vector")
    if count <= 0:
        raise ValueError("candidate count must be positive")
    radius = float(np.linalg.norm(value[:2]))
    angle = float(np.arctan2(value[1], value[0]))
    return [
        np.asarray(
            [radius * np.cos(angle + 2 * np.pi * index / count), radius * np.sin(angle + 2 * np.pi * index / count), value[2]],
            dtype=np.float64,
        )
        for index in range(count)
    ]


def is_recoverability_failure(error: RuntimeError) -> bool:
    message = str(error)
    return message.startswith("slipped teacher ended") or message.startswith("slipped teacher exceeded")


def _write_video(path: Path, frames: np.ndarray, fps: float = 10.0) -> None:
    if len(frames) == 0:
        raise ValueError(f"cannot write empty video: {path}")
    write_h264_video(path, frames, fps=fps)


def _branch_frames(episode: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([episode["agentview"], episode["wrist"]], axis=2)


def _write_paired_video(
    path: Path,
    attached: Mapping[str, np.ndarray],
    slipped: Mapping[str, np.ndarray],
    fps: float = 10.0,
) -> None:
    import cv2

    left = _branch_frames(attached)
    right = _branch_frames(slipped)
    frame_count = max(len(left), len(right))
    height, branch_width = left[0].shape[:2]
    width = branch_width * 2

    def frames():
        for index in range(frame_count):
            frame = np.full((height + 26, width, 3), 255, dtype=np.uint8)
            frame[26:, :branch_width] = left[min(index, len(left) - 1)]
            frame[26:, branch_width:] = right[min(index, len(right) - 1)]
            cv2.putText(frame, "attached | agent + wrist", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            cv2.putText(
                frame,
                "slipped/recovery | agent + wrist",
                (branch_width + 8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
            )
            yield frame

    write_h264_video(path, frames(), fps=fps)


def _recovery_index(episode: Mapping[str, np.ndarray], fallback: int) -> int:
    phases = np.asarray(episode["teacher_phase"])
    indices = np.flatnonzero(np.char.startswith(phases.astype(str), "recover_"))
    return int(indices[min(8, len(indices) - 1)]) if len(indices) else fallback


def _contact_row(
    pair_id: str,
    attached: Mapping[str, np.ndarray],
    slipped: Mapping[str, np.ndarray],
    event_time: int,
    feedback_reveal_time: int,
) -> tuple[str, list[tuple[str, np.ndarray]]]:
    pre = max(0, event_time - 1)
    recovery = _recovery_index(slipped, feedback_reveal_time)
    return pair_id, [
        ("pre agent", attached["agentview"][pre]),
        ("pre wrist", attached["wrist"][pre]),
        ("attached reveal", attached["agentview"][feedback_reveal_time]),
        ("slipped reveal", slipped["agentview"][feedback_reveal_time]),
        ("attached wrist", attached["wrist"][feedback_reveal_time]),
        ("slipped wrist", slipped["wrist"][feedback_reveal_time]),
        ("recovery", slipped["agentview"][recovery]),
        ("attached final", attached["agentview"][-1]),
        ("slipped final", slipped["agentview"][-1]),
    ]


def _write_contact_sheet(path: Path, rows: Sequence[tuple[str, list[tuple[str, np.ndarray]]]]) -> None:
    if not rows:
        raise ValueError("contact sheet requires at least one group")
    cell = 224
    header = 26
    columns = len(rows[0][1])
    canvas = Image.new("RGB", (cell * columns, (cell + header) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (pair_id, images) in enumerate(rows):
        if len(images) != columns:
            raise ValueError("contact sheet column count changed")
        y = row_index * (cell + header)
        for column, (label, value) in enumerate(images):
            x = column * cell
            canvas.paste(Image.fromarray(np.asarray(value)).resize((cell, cell)), (x, y + header))
            title = f"{pair_id} | {label}" if column == 0 else label
            draw.text((x + 3, y + 5), title, fill="black")
    canvas.save(path)


def _prefix_max_abs_error(
    attached: Mapping[str, np.ndarray], slipped: Mapping[str, np.ndarray], event_time: int
) -> dict[str, float]:
    return {
        "agentview": float(
            np.max(np.abs(attached["agentview"][:event_time].astype(np.int16) - slipped["agentview"][:event_time].astype(np.int16)))
        ),
        "wrist": float(
            np.max(np.abs(attached["wrist"][:event_time].astype(np.int16) - slipped["wrist"][:event_time].astype(np.int16)))
        ),
        "robot_state": float(np.max(np.abs(attached["robot_state"][:event_time] - slipped["robot_state"][:event_time]))),
        "actions": float(np.max(np.abs(attached["actions"][:event_time] - slipped["actions"][:event_time]))),
    }


def _validate_episode_arrays(episode: Mapping[str, np.ndarray], resolution: int) -> None:
    steps = len(episode["actions"])
    observation_keys = (
        "agentview",
        "wrist",
        "robot_state",
        "eef_pose",
        "object_pose",
        "gripper_qpos",
        "gripper_action",
        "grasped",
        "contact",
        "success",
        "sim_state",
        "teacher_phase",
    )
    if any(len(episode[key]) != steps + 1 for key in observation_keys):
        raise RuntimeError("episode observation arrays must have T+1 rows")
    if episode["agentview"].shape[1:] != (resolution, resolution, 3):
        raise RuntimeError(f"invalid agent-view shape: {episode['agentview'].shape}")
    if episode["wrist"].shape[1:] != (resolution, resolution, 3):
        raise RuntimeError(f"invalid wrist-view shape: {episode['wrist'].shape}")
    if episode["actions"].shape[1:] != (7,):
        raise RuntimeError(f"invalid action shape: {episode['actions'].shape}")


def collect_group(
    env: Any,
    initial_states: np.ndarray,
    source: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    randomization = source["randomization"]
    prefix, _, snapshot, controller = collect_grasp_prefix(
        env,
        initial_states[int(source["initial_state_index"])],
        settle_steps=args.settle_steps,
        object_offset=randomization["object_offset"],
        grasp_offset=randomization["grasp_offset"],
        friction_scale=float(randomization["friction_scale"]),
    )
    episodes = {}
    branch_metadata = {}
    continuation, metadata = collect_branch_continuation(
        env,
        snapshot,
        controller,
        outcome="attached",
        slip_offset=randomization["slip_offset"],
        max_steps=args.max_steps,
    )
    branch_results = [("attached", continuation, metadata)]

    candidate_errors = []
    slipped_result = None
    for candidate_index, candidate in enumerate(slip_offset_candidates(randomization["slip_offset"])):
        try:
            continuation, metadata = collect_branch_continuation(
                env,
                snapshot,
                controller,
                outcome="slipped",
                slip_offset=candidate,
                max_steps=args.max_steps,
            )
        except RuntimeError as error:
            if not is_recoverability_failure(error):
                raise
            candidate_errors.append(str(error))
            continue
        metadata["requested_slip_offset"] = list(randomization["slip_offset"])
        metadata["applied_slip_offset"] = candidate.round(8).tolist()
        metadata["intervention_candidate_index"] = candidate_index
        metadata["failed_intervention_candidates"] = len(candidate_errors)
        slipped_result = ("slipped", continuation, metadata)
        break
    if slipped_result is None:
        summaries = [error.split(":", 1)[0] for error in candidate_errors]
        raise RuntimeError(
            f"all slip intervention candidates failed for {source['pair_id']}: {summaries}"
        )
    branch_results.append(slipped_result)

    for outcome, continuation, metadata in branch_results:
        episode = merge_prefix_and_continuation(prefix, continuation)
        episode["model_body_pos"] = np.asarray(controller["model_body_pos"], dtype=np.float64)
        episode["object_friction"] = np.asarray(controller["object_friction"], dtype=np.float64)
        _validate_episode_arrays(episode, args.resolution)
        episodes[outcome] = episode
        branch_metadata[outcome] = metadata

    prefix_steps = len(prefix["actions"])
    attached_event = prefix_steps + int(branch_metadata["attached"]["event_time"])
    slipped_event = prefix_steps + int(branch_metadata["slipped"]["event_time"])
    if attached_event != slipped_event:
        raise RuntimeError(f"contact-triggered event mismatch: {attached_event} != {slipped_event}")
    event_time = attached_event
    action_divergence_time = first_persistent_action_divergence(
        episodes["attached"]["actions"], episodes["slipped"]["actions"]
    )
    feedback_reveal_time = first_visual_reveal(
        episodes["attached"], episodes["slipped"], start=event_time
    )
    prefix_errors = _prefix_max_abs_error(episodes["attached"], episodes["slipped"], event_time)
    if max(prefix_errors.values()) != 0.0:
        raise RuntimeError(f"pre-event conditioning leaked for {source['pair_id']}: {prefix_errors}")
    if feedback_reveal_time != event_time:
        raise RuntimeError(
            f"visual feedback must reveal at intervention: {feedback_reveal_time} != {event_time}"
        )
    if action_divergence_time != event_time:
        raise RuntimeError(
            f"teacher action divergence must follow feedback: {action_divergence_time} != {event_time}"
        )

    slipped_recovery = branch_metadata["slipped"]["recovery_action_time"]
    recovery_time = None if slipped_recovery is None else prefix_steps + int(slipped_recovery)
    metadata = {
        "pair_id": source["pair_id"],
        "task": "grasp_slip_full_episode",
        "source_initial_state_index": int(source["initial_state_index"]),
        "source_randomization": randomization,
        "prefix_steps": prefix_steps,
        "event_trigger": "grasped object lifted by at least 0.015 m",
        "event_time": event_time,
        "feedback_reveal_time": feedback_reveal_time,
        "action_divergence_time": action_divergence_time,
        "recovery_action_time": recovery_time,
        "recovery_latency": None if recovery_time is None else recovery_time - feedback_reveal_time,
        "pre_event_max_abs_error": prefix_errors,
        "branches": {
            outcome: {
                **branch_metadata[outcome],
                "total_steps": int(len(episodes[outcome]["actions"])),
                "final_success": bool(episodes[outcome]["success"][-1]),
                "final_object_position": episodes[outcome]["object_pose"][-1, :3].round(8).tolist(),
            }
            for outcome in ("attached", "slipped")
        },
    }
    return episodes, metadata


def build_quality_report(groups: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    slipped_rows = [row["branches"]["slipped"] for row in groups]
    checks = {
        "requested_group_count": len(groups) == args.group_count,
        "pre_event_conditioning_exact": all(
            max(row["pre_event_max_abs_error"].values()) == 0 for row in groups
        ),
        "no_early_visual_leak": all(row["feedback_reveal_time"] >= row["event_time"] for row in groups),
        "feedback_visible_at_event": all(row["feedback_reveal_time"] == row["event_time"] for row in groups),
        "teacher_diverges_at_feedback": all(row["action_divergence_time"] == row["feedback_reveal_time"] for row in groups),
        "recovery_starts_without_delay": all(row["recovery_latency"] == 0 for row in groups),
        "attached_teacher_success": all(row["branches"]["attached"]["final_success"] for row in groups),
        "slipped_teacher_recovery_success": all(row["branches"]["slipped"]["final_success"] for row in groups),
        "slipped_branch_regrasped": all(row["branches"]["slipped"]["regrasp_attempts"] >= 1 for row in groups),
        "slip_candidate_magnitude_preserved": all(
            np.isclose(
                np.linalg.norm(np.asarray(row.get("requested_slip_offset", (0.0, 0.0)))[:2]),
                np.linalg.norm(np.asarray(row.get("applied_slip_offset", (0.0, 0.0)))[:2]),
            )
            for row in slipped_rows
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "group_count": len(groups),
            "mean_attached_steps": float(np.mean([row["branches"]["attached"]["total_steps"] for row in groups])),
            "mean_slipped_steps": float(np.mean([row["branches"]["slipped"]["total_steps"] for row in groups])),
            "mean_recovery_latency": float(np.mean([row["recovery_latency"] for row in groups])),
            "attached_success_rate": float(np.mean([row["branches"]["attached"]["final_success"] for row in groups])),
            "slipped_success_rate": float(np.mean([row["branches"]["slipped"]["final_success"] for row in groups])),
            "rotated_intervention_group_count": sum(
                int(row.get("intervention_candidate_index", 0) > 0) for row in slipped_rows
            ),
            "max_intervention_candidate_index": max(
                (int(row.get("intervention_candidate_index", 0)) for row in slipped_rows),
                default=0,
            ),
        },
    }


def collect(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    from libero.libero.envs import OffScreenRenderEnv

    source_manifest = json.loads((args.source_root / "manifest.json").read_text())
    source_splits = json.loads((args.source_root / "splits.json").read_text())["pair_splits"]
    source_groups = sorted(
        (row for row in source_manifest["pairs"] if row["task"] == "grasp_slip"),
        key=lambda row: row["pair_id"],
    )
    selected = source_groups[args.group_start : args.group_start + args.group_count]
    if len(selected) != args.group_count:
        raise ValueError(
            f"requested groups [{args.group_start}, {args.group_start + args.group_count}) but only {len(selected)} exist"
        )
    initial_states = np.load(args.init_states)
    episodes_dir = output_dir / "episodes"
    videos_dir = output_dir / "videos"
    paired_dir = output_dir / "paired_videos"
    episodes_dir.mkdir(parents=True)
    videos_dir.mkdir(parents=True)
    paired_dir.mkdir(parents=True)

    env = OffScreenRenderEnv(
        bddl_file_name=str(args.bddl),
        camera_heights=args.resolution,
        camera_widths=args.resolution,
        horizon=args.max_steps + 500,
        ignore_done=True,
    )
    env.seed(args.seed)
    groups = []
    contact_rows = []
    try:
        for source in selected:
            episodes, metadata = collect_group(env, initial_states, source, args)
            pair_id = metadata["pair_id"]
            pair_dir = episodes_dir / pair_id
            pair_dir.mkdir()
            for outcome, episode in episodes.items():
                np.savez_compressed(pair_dir / f"{outcome}.npz", **episode)
                _write_video(videos_dir / f"{pair_id}-{outcome}.mp4", _branch_frames(episode))
            _write_paired_video(paired_dir / f"{pair_id}-attached-vs-slipped.mp4", episodes["attached"], episodes["slipped"])
            contact_rows.append(
                _contact_row(
                    pair_id,
                    episodes["attached"],
                    episodes["slipped"],
                    metadata["event_time"],
                    metadata["feedback_reveal_time"],
                )
            )
            metadata["split"] = source_splits[pair_id]
            metadata["episode_files"] = {
                outcome: f"episodes/{pair_id}/{outcome}.npz" for outcome in ("attached", "slipped")
            }
            groups.append(metadata)
            print(json.dumps({"collected": pair_id, "split": metadata["split"], **metadata["branches"]}), flush=True)
    finally:
        env.close()

    _write_contact_sheet(output_dir / "contact_sheet.png", contact_rows)
    quality = build_quality_report(groups, args)
    manifest = {
        "schema_version": 1,
        "generator": "generate_libero_full_episodes.py",
        "source_root": str(args.source_root),
        "bddl": str(args.bddl),
        "init_states": str(args.init_states),
        "seed": args.seed,
        "image_transform": "rotate_180_to_match_openpi_libero",
        "image_resolution": [args.resolution, args.resolution],
        "group_start": args.group_start,
        "group_count": len(groups),
        "policy_input_fields": ["agentview", "wrist", "robot_state", "language_instruction"],
        "teacher_privileged_fields": ["grasped", "contact", "environment_success"],
        "groups": groups,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "quality_report.json").write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n")
    return quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate complete paired LIBERO grasp/slip episodes")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/libero-counterfactual-v1-128"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/libero-full-episode-qa20"),
    )
    parser.add_argument("--bddl", type=Path, default=DEFAULT_BDDL)
    parser.add_argument("--init-states", type=Path, default=DEFAULT_INIT_STATES)
    parser.add_argument("--group-start", type=int, default=0)
    parser.add_argument("--group-count", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=640)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.group_start < 0 or args.group_count <= 0:
        raise ValueError("group_start must be non-negative and group_count must be positive")
    if args.resolution != 224:
        raise ValueError("full-episode policy observations must be 224x224")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    staging = args.output_dir.parent / f".{args.output_dir.name}.staging-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    quality = collect(args, staging)
    staging.rename(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), **quality}, sort_keys=True))
    if not quality["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
