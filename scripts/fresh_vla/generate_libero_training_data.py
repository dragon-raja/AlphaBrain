from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from counterfactual_data import (
    CounterfactualRecord,
    build_policy_inputs,
    estimate_branch_divergence,
    threshold_sensitivity,
    validate_record,
)
from libero_snapshot_collector import (
    DEFAULT_BDDL,
    DEFAULT_INIT_STATES,
    LANGUAGE,
    _conditioning_hash,
    _prepare_grasp_snapshot,
    _prepare_reach_snapshot,
    _restore_snapshot,
    _run_grasp_branch,
    _run_reach_branch,
    compact_policy_observation,
    gripper_transition_horizon,
    robot_state_from_observation,
    validate_physical_branches,
)
from video_io import write_h264_video


BRANCHES = {
    "grasp_slip": ("attached", "slipped"),
    "deterministic_reach": ("repeat_a", "repeat_b"),
}


def upright_image(image: np.ndarray) -> np.ndarray:
    """Match the 180-degree LIBERO image transform used by OpenPI evaluation."""
    return np.asarray(image)[::-1, ::-1].copy()


def upright_pair_frame(frame: np.ndarray) -> np.ndarray:
    midpoint = frame.shape[1] // 2
    return np.concatenate((upright_image(frame[:, :midpoint]), upright_image(frame[:, midpoint:])), axis=1)


def assign_group_splits(pair_tasks: Mapping[str, str], seed: int) -> dict[str, str]:
    """Create deterministic task-stratified splits without breaking groups."""
    rng = np.random.default_rng(seed)
    result = {}
    for task in sorted(set(pair_tasks.values())):
        pair_ids = sorted(pair_id for pair_id, value in pair_tasks.items() if value == task)
        rng.shuffle(pair_ids)
        count = len(pair_ids)
        if count < 3:
            val_count = 0
            test_count = 0
        else:
            val_count = max(1, int(round(count * 0.1)))
            test_count = max(1, int(round(count * 0.1)))
            while val_count + test_count >= count:
                if test_count >= val_count:
                    test_count -= 1
                else:
                    val_count -= 1
        test_ids = set(pair_ids[:test_count])
        val_ids = set(pair_ids[test_count : test_count + val_count])
        for pair_id in pair_ids:
            result[pair_id] = "test" if pair_id in test_ids else "val" if pair_id in val_ids else "train"
    return result


def build_training_labels(
    records: Sequence[CounterfactualRecord],
    pair_splits: Mapping[str, str],
    *,
    horizon: int,
    seed: int,
) -> dict[str, Any]:
    """Build paired controls while keeping random labels independent of branch outcome."""
    grouped = {}
    for record in records:
        grouped.setdefault(record.pair_id, []).append(record)

    rng = np.random.default_rng(seed)
    random_by_pair = {}
    shuffled_by_pair = {}
    strata = {}
    for pair_id, pair_records in grouped.items():
        key = (pair_splits[pair_id], bool(pair_records[0].is_deterministic_control))
        strata.setdefault(key, []).append(pair_id)
    for pair_ids in strata.values():
        ordered = sorted(pair_ids)
        oracle_values = np.asarray([grouped[pair_id][0].oracle_feedback_horizon for pair_id in ordered])
        shuffled = oracle_values.copy()
        if len(shuffled) > 1:
            for _ in range(8):
                rng.shuffle(shuffled)
                if not np.array_equal(shuffled, oracle_values):
                    break
        random_values = rng.choice(oracle_values, size=len(ordered), replace=True)
        for index, pair_id in enumerate(ordered):
            shuffled_by_pair[pair_id] = int(shuffled[index])
            random_by_pair[pair_id] = int(random_values[index])

    labels = {}
    for record in records:
        oracle = int(record.oracle_feedback_horizon)
        record_id = f"{record.pair_id}::{record.branch_id}"
        labels[record_id] = {
            "full_h": horizon,
            "random_feedback_horizon": random_by_pair[record.pair_id],
            "shuffled_oracle_horizon": shuffled_by_pair[record.pair_id],
            "early_feedback_horizon": max(0, oracle - 2),
            "late_feedback_horizon": min(horizon, oracle + 2),
            "gripper_transition_horizon": int(record.gripper_transition_horizon),
            "oracle_feedback_horizon": oracle,
            "short_h": min(5, horizon),
        }
    return {
        "schema_version": 1,
        "horizon": horizon,
        "label_semantics": "h means supervise action steps [0, h); suffix weighting is configured by the method",
        "records": labels,
    }


def _write_video(path: Path, frames: Sequence[np.ndarray], fps: float = 6.0) -> None:
    if not frames:
        raise ValueError(f"cannot write empty video: {path}")
    transformed = [upright_pair_frame(frame) for frame in frames]
    write_h264_video(path, transformed, fps=fps)


def _write_paired_video(
    path: Path,
    left_frames: Sequence[np.ndarray],
    right_frames: Sequence[np.ndarray],
    left_label: str,
    right_label: str,
    fps: float = 6.0,
) -> None:
    import cv2

    if not left_frames or len(left_frames) != len(right_frames):
        raise ValueError(f"paired video requires equal non-empty branches: {path}")
    def paired_frames():
        for left, right in zip(left_frames, right_frames):
            left_view = upright_pair_frame(left)
            right_view = upright_pair_frame(right)
            frame = np.full((left_view.shape[0] + 26, left_view.shape[1] * 2, 3), 255, dtype=np.uint8)
            frame[26:, : left_view.shape[1]] = left_view
            frame[26:, left_view.shape[1] :] = right_view
            cv2.putText(frame, left_label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                right_label,
                (left_view.shape[1] + 8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            yield frame

    write_h264_video(path, paired_frames(), fps=fps)


def _rollout_array_payload(
    branch_rollouts: Mapping[str, np.ndarray],
    branch_physics: Mapping[str, Sequence[Mapping[str, Any]]],
    branch_frames: Mapping[str, Sequence[Sequence[np.ndarray]]],
    *,
    effect_key: str,
) -> dict[str, np.ndarray]:
    payload = {}
    for branch_name, repeat_frames in branch_frames.items():
        for repeat_index, frames in enumerate(repeat_frames):
            transformed = np.stack([upright_pair_frame(frame) for frame in frames])
            midpoint = transformed.shape[2] // 2
            prefix = f"{branch_name}_repeat_{repeat_index:02d}"
            payload[f"{prefix}_agentview"] = transformed[:, :, :midpoint]
            payload[f"{prefix}_wrist"] = transformed[:, :, midpoint:]
            payload[f"{prefix}_robot_state"] = np.asarray(
                branch_physics[branch_name][repeat_index]["robot_state_trajectory"], dtype=np.float32
            )
            payload[f"{prefix}_actions"] = np.asarray(
                branch_rollouts[branch_name][repeat_index], dtype=np.float32
            )
            payload[f"{prefix}_effect_position"] = np.asarray(
                branch_physics[branch_name][repeat_index][effect_key], dtype=np.float32
            )
    return payload


def _validate_rollout_array_payload(payload: Mapping[str, np.ndarray], horizon: int) -> None:
    for key, value in payload.items():
        array = np.asarray(value)
        if key.endswith(("_agentview", "_wrist")):
            expected = (horizon + 1, 224, 224, 3)
            if array.shape != expected or array.dtype != np.uint8:
                raise ValueError(f"invalid frame array {key}: {array.shape} {array.dtype}, expected {expected} uint8")
        elif key.endswith("_robot_state"):
            if array.shape != (horizon + 1, 8):
                raise ValueError(f"invalid robot-state array {key}: {array.shape}")
        elif key.endswith("_actions"):
            if array.shape != (horizon, 7):
                raise ValueError(f"invalid action array {key}: {array.shape}")
        elif key.endswith("_effect_position"):
            if array.shape != (horizon + 1, 3):
                raise ValueError(f"invalid effect-position array {key}: {array.shape}")
        else:
            raise ValueError(f"unknown rollout observation array: {key}")


def _write_contact_sheet(path: Path, rows: Sequence[tuple[str, Sequence[np.ndarray]]]) -> None:
    if not rows:
        raise ValueError("contact sheet needs at least one row")
    cell_size = 224
    header = 24
    labels = ("pre agent", "pre wrist", "attached agent", "attached wrist", "slipped agent", "slipped wrist")
    canvas = Image.new("RGB", (cell_size * len(labels), (cell_size + header) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (pair_id, images) in enumerate(rows):
        if len(images) != len(labels):
            raise ValueError(f"expected six contact-sheet images for {pair_id}")
        y = row_index * (cell_size + header)
        for column, (label, array) in enumerate(zip(labels, images)):
            x = column * cell_size
            image = Image.fromarray(np.asarray(array)).resize((cell_size, cell_size), Image.BILINEAR)
            canvas.paste(image, (x, y + header))
            title = f"{pair_id} | {label}" if column == 0 else label
            draw.text((x + 3, y + 4), title, fill="black")
    canvas.save(path)


def _record_pair(
    *,
    pair_id: str,
    branch_names: Sequence[str],
    branch_rollouts: Mapping[str, np.ndarray],
    observation: Mapping[str, Any],
    snapshot_key: str,
    task: str,
    event_time: int,
    feedback_reveal_time: int,
    estimate: Any,
) -> list[CounterfactualRecord]:
    transformed_observation = dict(observation)
    transformed_observation["agentview_image"] = upright_image(observation["agentview_image"])
    transformed_observation["robot0_eye_in_hand_image"] = upright_image(
        observation["robot0_eye_in_hand_image"]
    )
    policy_observation = compact_policy_observation(transformed_observation, snapshot_key)
    robot_state = robot_state_from_observation(observation).round(8).tolist()
    deterministic = task == "deterministic_reach"
    records = []
    for branch_name in branch_names:
        mean_actions = branch_rollouts[branch_name].mean(axis=0)
        record = CounterfactualRecord(
            pair_id=pair_id,
            branch_id=branch_name,
            branch_outcome=branch_name,
            observation=policy_observation,
            robot_state=robot_state,
            language_instruction=LANGUAGE,
            action_chunk=mean_actions.round(7).tolist(),
            event_time=event_time,
            feedback_reveal_time=feedback_reveal_time,
            action_divergence_time=estimate.action_divergence_time,
            gripper_transition_horizon=gripper_transition_horizon(mean_actions),
            oracle_feedback_horizon=estimate.oracle_feedback_horizon,
            per_step_branch_divergence=estimate.per_step_branch_divergence,
            is_deterministic_control=deterministic,
        )
        validate_record(record)
        build_policy_inputs(record)
        records.append(record)
    return records


def _pre_branch_errors(branch_physics: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[float, float]:
    rows = [row for branch_rows in branch_physics.values() for row in branch_rows]
    return (
        max(float(row["pre_branch_image_max_abs_error"]) for row in rows),
        max(float(row["pre_branch_state_max_abs_error"]) for row in rows),
    )


def _collect_grasp_group(
    env: Any,
    initial_states: np.ndarray,
    pair_index: int,
    args: argparse.Namespace,
) -> tuple[
    list[CounterfactualRecord],
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, list[np.ndarray]],
    dict[str, np.ndarray],
]:
    pair_id = f"libero-grasp-slip-{pair_index:04d}"
    group_rng = np.random.default_rng(args.seed + pair_index * 7919)
    last_error = None
    for attempt in range(args.max_retries):
        state_index = int(group_rng.integers(0, len(initial_states)))
        object_offset = np.asarray([*group_rng.uniform(-0.010, 0.010, size=2), 0.0])
        grasp_offset = np.asarray(
            [*group_rng.uniform(-0.0025, 0.0025, size=2), group_rng.uniform(-0.001, 0.001)]
        )
        friction_scale = float(group_rng.uniform(0.8, 1.2))
        event_time = int(group_rng.integers(2, min(5, args.horizon - 2)))
        slip_angle = float(group_rng.uniform(0.0, 2.0 * np.pi))
        slip_radius = float(group_rng.uniform(0.052, 0.068))
        slip_offset = np.asarray(
            [slip_radius * np.cos(slip_angle), slip_radius * np.sin(slip_angle), group_rng.uniform(-0.008, -0.003)]
        )
        try:
            observation, snapshot, controller_state = _prepare_grasp_snapshot(
                env,
                initial_states[state_index],
                args.settle_steps,
                object_offset=object_offset,
                grasp_offset=grasp_offset,
                friction_scale=friction_scale,
            )
            observation = _restore_snapshot(env, snapshot, controller_state)
            branch_rollouts = {}
            branch_physics = {}
            branch_frames = {}
            max_restore_error = 0.0
            for branch_index, branch_name in enumerate(BRANCHES["grasp_slip"]):
                rollouts = []
                physics_rows = []
                repeat_frames = []
                for repeat_index in range(args.repeats):
                    frames = []
                    rollout_rng = np.random.default_rng(
                        args.seed + pair_index * 10007 + branch_index * 101 + repeat_index
                    )
                    actions, physics, restore_error = _run_grasp_branch(
                        env,
                        snapshot,
                        controller_state,
                        branch_name,
                        horizon=args.horizon,
                        event_time=event_time,
                        rng=rollout_rng,
                        slip_offset=slip_offset,
                        reference_observation=observation,
                        video_frames=frames,
                    )
                    max_restore_error = max(max_restore_error, restore_error)
                    rollouts.append(actions)
                    physics_rows.append(physics)
                    repeat_frames.append(frames)
                branch_rollouts[branch_name] = np.stack(rollouts)
                branch_physics[branch_name] = physics_rows
                branch_frames[branch_name] = repeat_frames

            estimate = estimate_branch_divergence(branch_rollouts, persistence=2)
            sensitivity = threshold_sensitivity(branch_rollouts, persistence=2)
            if estimate.oracle_feedback_horizon != event_time:
                raise RuntimeError(f"oracle boundary {estimate.oracle_feedback_horizon} != event {event_time}")
            if any(value.oracle_feedback_horizon != event_time for value in sensitivity.values()):
                raise RuntimeError("oracle boundary changed in the threshold sensitivity sweep")
            physical_validation = validate_physical_branches("grasp_slip", branch_physics)
            effect_rollouts = {
                name: np.stack(
                    [np.asarray(row["object_trajectory"], dtype=np.float64)[1:] for row in branch_physics[name]]
                )
                for name in BRANCHES["grasp_slip"]
            }
            effect_estimate = estimate_branch_divergence(effect_rollouts, persistence=2)
            pre_image_error, pre_state_error = _pre_branch_errors(branch_physics)
            common_prefix_error = float(
                np.max(
                    np.abs(
                        branch_rollouts["attached"][:, :event_time]
                        - branch_rollouts["slipped"][:, :event_time]
                    )
                )
            )
            if max(max_restore_error, pre_image_error, pre_state_error, common_prefix_error) != 0.0:
                raise RuntimeError(
                    "pre-branch equality check failed: "
                    f"snapshot={max_restore_error}, image={pre_image_error}, "
                    f"state={pre_state_error}, action_prefix={common_prefix_error}"
                )

            records = _record_pair(
                pair_id=pair_id,
                branch_names=BRANCHES["grasp_slip"],
                branch_rollouts=branch_rollouts,
                observation=observation,
                snapshot_key=pair_id,
                task="grasp_slip",
                event_time=event_time,
                feedback_reveal_time=effect_estimate.action_divergence_time,
                estimate=estimate,
            )
            transformed_observation = dict(observation)
            transformed_observation["agentview_image"] = upright_image(observation["agentview_image"])
            transformed_observation["robot0_eye_in_hand_image"] = upright_image(
                observation["robot0_eye_in_hand_image"]
            )
            policy_observation = compact_policy_observation(transformed_observation, pair_id)
            metadata = {
                "pair_id": pair_id,
                "task": "grasp_slip",
                "conditioning_sha256": _conditioning_hash(
                    policy_observation, robot_state_from_observation(observation).round(8).tolist()
                ),
                "initial_state_index": state_index,
                "attempt": attempt + 1,
                "randomization": {
                    "object_offset": object_offset.round(8).tolist(),
                    "grasp_offset": grasp_offset.round(8).tolist(),
                    "friction_scale": friction_scale,
                    "slip_offset": slip_offset.round(8).tolist(),
                },
                "snapshot_state_dim": int(snapshot.size),
                "max_snapshot_restore_abs_error": max_restore_error,
                "pre_branch_image_max_abs_error": pre_image_error,
                "pre_branch_state_max_abs_error": pre_state_error,
                "common_action_prefix_max_abs_error": common_prefix_error,
                "event_time": event_time,
                "feedback_reveal_time": effect_estimate.action_divergence_time,
                "action_divergence_time": estimate.action_divergence_time,
                "oracle_feedback_horizon": estimate.oracle_feedback_horizon,
                "within_branch_threshold": estimate.within_branch_threshold,
                "threshold_sensitivity": {key: asdict(value) for key, value in sensitivity.items()},
                "physical_validation": physical_validation,
                "gripper_transition_horizons": {
                    name: gripper_transition_horizon(branch_rollouts[name].mean(axis=0))
                    for name in BRANCHES["grasp_slip"]
                },
                "physics": branch_physics,
            }
            snapshots = {
                f"{pair_id}_agentview": upright_image(observation["agentview_image"]),
                f"{pair_id}_wrist": upright_image(observation["robot0_eye_in_hand_image"]),
            }
            videos = {name: branch_frames[name][0] for name in BRANCHES["grasp_slip"]}
            rollout_arrays = _rollout_array_payload(
                branch_rollouts,
                branch_physics,
                branch_frames,
                effect_key="object_trajectory",
            )
            return records, metadata, snapshots, videos, rollout_arrays
        except RuntimeError as exc:
            last_error = exc
    raise RuntimeError(f"failed to collect {pair_id} after {args.max_retries} attempts: {last_error}")


def _collect_reach_group(
    env: Any,
    initial_states: np.ndarray,
    pair_index: int,
    args: argparse.Namespace,
) -> tuple[
    list[CounterfactualRecord],
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, list[np.ndarray]],
    dict[str, np.ndarray],
]:
    pair_id = f"libero-deterministic-reach-{pair_index:04d}"
    state_index = (pair_index * 13 + 7) % len(initial_states)
    observation, snapshot, controller_state = _prepare_reach_snapshot(
        env, initial_states[state_index], args.settle_steps
    )
    observation = _restore_snapshot(env, snapshot, controller_state)
    branch_rollouts = {}
    branch_physics = {}
    branch_frames = {}
    max_restore_error = 0.0
    for branch_name in BRANCHES["deterministic_reach"]:
        rollouts = []
        physics_rows = []
        repeat_frames = []
        for repeat_index in range(args.repeats):
            frames = []
            actions, physics, restore_error = _run_reach_branch(
                env,
                snapshot,
                controller_state,
                horizon=args.horizon,
                reference_observation=observation,
                video_frames=frames,
            )
            max_restore_error = max(max_restore_error, restore_error)
            rollouts.append(actions)
            physics_rows.append(physics)
            repeat_frames.append(frames)
        branch_rollouts[branch_name] = np.stack(rollouts)
        branch_physics[branch_name] = physics_rows
        branch_frames[branch_name] = repeat_frames

    estimate = estimate_branch_divergence(branch_rollouts, persistence=2)
    sensitivity = threshold_sensitivity(branch_rollouts, persistence=2)
    if estimate.oracle_feedback_horizon != args.horizon:
        raise RuntimeError(f"deterministic control diverged for {pair_id}")
    physical_validation = validate_physical_branches("deterministic_reach", branch_physics)
    pre_image_error, pre_state_error = _pre_branch_errors(branch_physics)
    common_prefix_error = float(
        np.max(np.abs(branch_rollouts["repeat_a"] - branch_rollouts["repeat_b"]))
    )
    if max(max_restore_error, pre_image_error, pre_state_error, common_prefix_error) != 0.0:
        raise RuntimeError(f"deterministic replay equality check failed for {pair_id}")

    records = _record_pair(
        pair_id=pair_id,
        branch_names=BRANCHES["deterministic_reach"],
        branch_rollouts=branch_rollouts,
        observation=observation,
        snapshot_key=pair_id,
        task="deterministic_reach",
        event_time=args.horizon,
        feedback_reveal_time=args.horizon,
        estimate=estimate,
    )
    transformed_observation = dict(observation)
    transformed_observation["agentview_image"] = upright_image(observation["agentview_image"])
    transformed_observation["robot0_eye_in_hand_image"] = upright_image(
        observation["robot0_eye_in_hand_image"]
    )
    policy_observation = compact_policy_observation(transformed_observation, pair_id)
    metadata = {
        "pair_id": pair_id,
        "task": "deterministic_reach",
        "conditioning_sha256": _conditioning_hash(
            policy_observation, robot_state_from_observation(observation).round(8).tolist()
        ),
        "initial_state_index": state_index,
        "snapshot_state_dim": int(snapshot.size),
        "max_snapshot_restore_abs_error": max_restore_error,
        "pre_branch_image_max_abs_error": pre_image_error,
        "pre_branch_state_max_abs_error": pre_state_error,
        "common_action_prefix_max_abs_error": common_prefix_error,
        "event_time": args.horizon,
        "feedback_reveal_time": args.horizon,
        "action_divergence_time": estimate.action_divergence_time,
        "oracle_feedback_horizon": estimate.oracle_feedback_horizon,
        "within_branch_threshold": estimate.within_branch_threshold,
        "threshold_sensitivity": {key: asdict(value) for key, value in sensitivity.items()},
        "physical_validation": physical_validation,
        "physics": branch_physics,
    }
    snapshots = {
        f"{pair_id}_agentview": upright_image(observation["agentview_image"]),
        f"{pair_id}_wrist": upright_image(observation["robot0_eye_in_hand_image"]),
    }
    videos = {name: branch_frames[name][0] for name in BRANCHES["deterministic_reach"]}
    rollout_arrays = _rollout_array_payload(
        branch_rollouts,
        branch_physics,
        branch_frames,
        effect_key="eef_trajectory",
    )
    return records, metadata, snapshots, videos, rollout_arrays


def build_quality_report(
    manifest: Mapping[str, Any],
    pair_splits: Mapping[str, str],
    labels: Mapping[str, Any],
    *,
    branch_video_count: int,
    paired_video_count: int,
    rollout_shard_count: int,
    rollout_array_count: int,
) -> dict[str, Any]:
    pairs = manifest["pairs"]
    grasp_pairs = [row for row in pairs if row["task"] == "grasp_slip"]
    deterministic_pairs = [row for row in pairs if row["task"] == "deterministic_reach"]
    split_counts = {name: sum(value == name for value in pair_splits.values()) for name in ("train", "val", "test")}
    label_values = [value for row in labels["records"].values() for value in row.values()]
    checks = {
        "requested_grasp_group_count": len(grasp_pairs) == manifest["requested_grasp_groups"],
        "requested_deterministic_group_count": (
            len(deterministic_pairs) == manifest["requested_deterministic_groups"]
        ),
        "snapshot_restore_exact": all(row["max_snapshot_restore_abs_error"] == 0.0 for row in pairs),
        "pre_branch_images_exact": all(row["pre_branch_image_max_abs_error"] == 0.0 for row in pairs),
        "pre_branch_state_exact": all(row["pre_branch_state_max_abs_error"] == 0.0 for row in pairs),
        "common_action_prefix_exact": all(row["common_action_prefix_max_abs_error"] == 0.0 for row in pairs),
        "threshold_sweep_stable": all(
            all(value["oracle_feedback_horizon"] == row["oracle_feedback_horizon"] for value in row["threshold_sensitivity"].values())
            for row in pairs
        ),
        "attached_grasp_rate_one": all(row["physical_validation"]["attached_grasp_rate"] == 1.0 for row in grasp_pairs),
        "slipped_grasp_rate_zero": all(row["physical_validation"]["slipped_grasp_rate"] == 0.0 for row in grasp_pairs),
        "deterministic_effect_exact": all(row["physical_validation"]["max_effect_distance"] == 0.0 for row in deterministic_pairs),
        "group_split_total": len(pair_splits) == len(pairs) == len(set(pair_splits)),
        "all_splits_nonempty": all(value > 0 for value in split_counts.values()),
        "labels_in_range": all(0 <= int(value) <= manifest["horizon"] for value in label_values),
        "one_video_per_branch": branch_video_count == len(pairs) * 2,
        "one_synchronized_paired_video_per_group": paired_video_count == len(pairs),
        "one_rollout_observation_shard_per_group": rollout_shard_count == len(pairs),
        "all_repeat_observation_arrays_present": (
            rollout_array_count == len(pairs) * 2 * manifest["repeats"] * 5
        ),
        "policy_input_leakage_guard": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "grasp_groups": len(grasp_pairs),
            "deterministic_groups": len(deterministic_pairs),
            "records": manifest["record_count"],
            "branch_videos": branch_video_count,
            "paired_videos": paired_video_count,
            "rollout_observation_shards": rollout_shard_count,
            "rollout_arrays": rollout_array_count,
            "split_counts": split_counts,
            "oracle_horizon_histogram": {
                str(value): sum(row["oracle_feedback_horizon"] == value for row in pairs)
                for value in sorted({row["oracle_feedback_horizon"] for row in pairs})
            },
            "max_attempts": max(int(row.get("attempt", 1)) for row in pairs),
        },
    }


def _quality_markdown(report: Mapping[str, Any]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    lines = [f"# LIBERO Counterfactual QA: {status}", "", "## Checks", ""]
    lines.extend(
        f"- [{'x' if passed else ' '}] `{name}`" for name, passed in report["checks"].items()
    )
    lines.extend(("", "## Metrics", "", "```json", json.dumps(report["metrics"], indent=2, sort_keys=True), "```", ""))
    return "\n".join(lines)


def collect(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    from libero.libero.envs import OffScreenRenderEnv

    initial_states = np.load(args.init_states)
    if args.grasp_groups < 3 or args.deterministic_groups < 3:
        raise ValueError("at least three groups per task are required for group-aware train/val/test splits")
    if args.repeats < 2:
        raise ValueError("repeats must be at least two")
    if args.horizon < 8:
        raise ValueError("horizon must be at least eight")
    if args.resolution != 224:
        raise ValueError("training snapshots must be captured at 224x224")

    videos_dir = output_dir / "videos"
    paired_videos_dir = output_dir / "paired_videos"
    rollout_observations_dir = output_dir / "rollout_observations"
    videos_dir.mkdir(parents=True)
    paired_videos_dir.mkdir(parents=True)
    rollout_observations_dir.mkdir(parents=True)
    env = OffScreenRenderEnv(
        bddl_file_name=str(args.bddl),
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.seed(args.seed)
    records = []
    pair_metadata = []
    snapshots = {}
    contact_rows = []
    branch_video_count = 0
    paired_video_count = 0
    rollout_shard_count = 0
    rollout_array_count = 0
    try:
        for pair_index in range(args.grasp_groups):
            pair_records, metadata, pair_snapshots, videos, rollout_arrays = _collect_grasp_group(
                env, initial_states, pair_index, args
            )
            records.extend(pair_records)
            pair_metadata.append(metadata)
            snapshots.update(pair_snapshots)
            for branch_name, frames in videos.items():
                _write_video(videos_dir / f"{metadata['pair_id']}-{branch_name}.mp4", frames)
                branch_video_count += 1
            _write_paired_video(
                paired_videos_dir / f"{metadata['pair_id']}-attached-vs-slipped.mp4",
                videos["attached"],
                videos["slipped"],
                "attached | agent + wrist",
                "slipped | agent + wrist",
            )
            paired_video_count += 1
            _validate_rollout_array_payload(rollout_arrays, args.horizon)
            np.savez_compressed(
                rollout_observations_dir / f"{metadata['pair_id']}.npz", **rollout_arrays
            )
            rollout_shard_count += 1
            rollout_array_count += len(rollout_arrays)
            if len(contact_rows) < args.contact_sheet_groups:
                pre = [pair_snapshots[f"{metadata['pair_id']}_agentview"], pair_snapshots[f"{metadata['pair_id']}_wrist"]]
                attached_final = upright_pair_frame(videos["attached"][-1])
                slipped_final = upright_pair_frame(videos["slipped"][-1])
                midpoint = attached_final.shape[1] // 2
                contact_rows.append(
                    (
                        metadata["pair_id"],
                        [*pre, attached_final[:, :midpoint], attached_final[:, midpoint:], slipped_final[:, :midpoint], slipped_final[:, midpoint:]],
                    )
                )
            print(json.dumps({"collected": metadata["pair_id"], "attempt": metadata["attempt"]}), flush=True)

        for pair_index in range(args.deterministic_groups):
            pair_records, metadata, pair_snapshots, videos, rollout_arrays = _collect_reach_group(
                env, initial_states, pair_index, args
            )
            records.extend(pair_records)
            pair_metadata.append(metadata)
            snapshots.update(pair_snapshots)
            for branch_name, frames in videos.items():
                _write_video(videos_dir / f"{metadata['pair_id']}-{branch_name}.mp4", frames)
                branch_video_count += 1
            _write_paired_video(
                paired_videos_dir / f"{metadata['pair_id']}-repeat-a-vs-b.mp4",
                videos["repeat_a"],
                videos["repeat_b"],
                "repeat A | agent + wrist",
                "repeat B | agent + wrist",
            )
            paired_video_count += 1
            _validate_rollout_array_payload(rollout_arrays, args.horizon)
            np.savez_compressed(
                rollout_observations_dir / f"{metadata['pair_id']}.npz", **rollout_arrays
            )
            rollout_shard_count += 1
            rollout_array_count += len(rollout_arrays)
            print(json.dumps({"collected": metadata["pair_id"]}), flush=True)
    finally:
        env.close()

    pair_tasks = {row["pair_id"]: row["task"] for row in pair_metadata}
    pair_splits = assign_group_splits(pair_tasks, args.seed + 17)
    labels = build_training_labels(records, pair_splits, horizon=args.horizon, seed=args.seed + 31)
    manifest = {
        "schema_version": 1,
        "generator": "generate_libero_training_data.py",
        "seed": args.seed,
        "bddl": str(args.bddl),
        "image_transform": "rotate_180_to_match_openpi_libero",
        "image_resolution": [224, 224],
        "horizon": args.horizon,
        "repeats": args.repeats,
        "requested_grasp_groups": args.grasp_groups,
        "requested_deterministic_groups": args.deterministic_groups,
        "pair_count": len(pair_metadata),
        "record_count": len(records),
        "policy_input_fields": ["observation", "robot_state", "language_instruction"],
        "training_tasks": ["grasp_slip"],
        "control_tasks": ["deterministic_reach"],
        "rollout_observation_schema": {
            "storage": "one compressed npz shard per pair",
            "per_branch_repeat_arrays": {
                "agentview": "[H+1, 224, 224, 3] uint8",
                "wrist": "[H+1, 224, 224, 3] uint8",
                "robot_state": "[H+1, 8] float32",
                "actions": "[H, 7] float32",
                "effect_position": "[H+1, 3] float32; object for grasp/slip, EEF for reach",
            },
        },
        "pairs": pair_metadata,
    }
    report = build_quality_report(
        manifest,
        pair_splits,
        labels,
        branch_video_count=branch_video_count,
        paired_video_count=paired_video_count,
        rollout_shard_count=rollout_shard_count,
        rollout_array_count=rollout_array_count,
    )

    (output_dir / "records.jsonl").write_text(
        "".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in records)
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "splits.json").write_text(
        json.dumps({"schema_version": 1, "pair_splits": pair_splits}, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "training_labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(output_dir / "policy_observation_snapshots.npz", **snapshots)
    _write_contact_sheet(output_dir / "contact_sheet.png", contact_rows)
    (output_dir / "quality_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output_dir / "quality_report.md").write_text(_quality_markdown(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grouped LIBERO counterfactual Pi0.5 samples")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/libero-counterfactual-qa30"),
    )
    parser.add_argument("--bddl", type=Path, default=DEFAULT_BDDL)
    parser.add_argument("--init-states", type=Path, default=DEFAULT_INIT_STATES)
    parser.add_argument("--grasp-groups", type=int, default=30)
    parser.add_argument("--deterministic-groups", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--contact-sheet-groups", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    staging = args.output_dir.parent / f".{args.output_dir.name}.staging-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    report = collect(args, staging)
    staging.rename(args.output_dir)
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "quality_passed": report["passed"], **report["metrics"]},
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
