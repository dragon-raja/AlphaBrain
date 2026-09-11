from __future__ import annotations

import argparse
import json
import os
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_libero_closed_loop import is_failure_continuation, is_recovery_action, stable_seed


LANGUAGE = "put the cream cheese in the bowl"
EPISODE_KEYS = (
    "actions",
    "agentview",
    "wrist",
    "robot_state",
    "eef_pose",
    "object_pose",
    "grasped",
)


class StoredObservationPolicy:
    def __init__(self, socket_path: Path) -> None:
        self.connection = Client(str(socket_path), family="AF_UNIX", authkey=b"fresh-vla-local")
        handshake = self.connection.recv()
        self.horizon = int(handshake["horizon"])
        self.identity = {
            "checkpoint_realpath": str(handshake.get("checkpoint_realpath", "")),
            "model_size_bytes": int(handshake.get("model_size_bytes", 0)),
            "torch_version": handshake.get("torch_version"),
            "cuda_version": handshake.get("cuda_version"),
            "device_name": handshake.get("device_name"),
        }

    def predict(self, episode: Mapping[str, np.ndarray], index: int, seed: int) -> np.ndarray:
        example = {
            "image": [
                np.asarray(episode["agentview"][index], dtype=np.uint8),
                np.asarray(episode["wrist"][index], dtype=np.uint8),
            ],
            "lang": LANGUAGE,
            "language": LANGUAGE,
            "state": np.asarray(episode["robot_state"][index], dtype=np.float32),
        }
        self.connection.send({"op": "predict", "seed": int(seed), "example": example})
        response = self.connection.recv()
        if "error" in response:
            raise RuntimeError(f"remote Pi0.5 inference failed: {response['error']}")
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.shape != (self.horizon, 7) or not np.all(np.isfinite(actions)):
            raise RuntimeError(f"invalid remote Pi0.5 actions: {actions.shape}")
        return actions

    def close(self) -> None:
        try:
            self.connection.send({"op": "close"})
        finally:
            self.connection.close()


def load_episode(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as episode:
        return {key: np.asarray(episode[key]) for key in EPISODE_KEYS}


def mse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.square(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)).mean())


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left_flat = np.asarray(left, dtype=np.float64).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    if denominator <= 1e-12:
        return None
    return float(np.dot(left_flat, right_flat) / denominator)


def revision_metrics(
    stale: np.ndarray,
    fresh_attached: np.ndarray,
    fresh_slipped: np.ndarray,
    teacher_attached: np.ndarray,
    teacher_slipped: np.ndarray,
) -> dict[str, float | bool | None]:
    arrays = [stale, fresh_attached, fresh_slipped, teacher_attached, teacher_slipped]
    if len({np.asarray(value).shape for value in arrays}) != 1:
        raise ValueError("all action prefixes must have the same shape")
    stale_attached = mse(stale, teacher_attached)
    stale_slipped = mse(stale, teacher_slipped)
    fresh_attached_error = mse(fresh_attached, teacher_attached)
    fresh_slipped_error = mse(fresh_slipped, teacher_slipped)
    direct = fresh_attached_error + fresh_slipped_error
    swapped = mse(fresh_attached, teacher_slipped) + mse(fresh_slipped, teacher_attached)
    teacher_delta = np.asarray(teacher_slipped) - np.asarray(teacher_attached)
    fresh_delta = np.asarray(fresh_slipped) - np.asarray(fresh_attached)
    return {
        "teacher_branch_mse": mse(teacher_attached, teacher_slipped),
        "stale_attached_mse": stale_attached,
        "stale_slipped_mse": stale_slipped,
        "fresh_attached_mse": fresh_attached_error,
        "fresh_slipped_mse": fresh_slipped_error,
        "stale_joint_mse": stale_attached + stale_slipped,
        "fresh_joint_mse": direct,
        "attached_fresh_minus_stale_mse": fresh_attached_error - stale_attached,
        "slipped_stale_minus_fresh_mse": stale_slipped - fresh_slipped_error,
        "fresh_revision_attached_mse": mse(fresh_attached, stale),
        "fresh_revision_slipped_mse": mse(fresh_slipped, stale),
        "fresh_predicted_branch_mse": mse(fresh_attached, fresh_slipped),
        "fresh_pair_assignment_margin": swapped - direct,
        "fresh_pair_assignment_correct": bool(direct < swapped),
        "fresh_attached_prefers_attached": bool(
            fresh_attached_error < mse(fresh_attached, teacher_slipped)
        ),
        "fresh_slipped_prefers_slipped": bool(
            fresh_slipped_error < mse(fresh_slipped, teacher_attached)
        ),
        "fresh_revision_alignment": cosine(fresh_delta, teacher_delta),
    }


def _observation_deltas(
    attached: Mapping[str, np.ndarray],
    slipped: Mapping[str, np.ndarray],
    index: int,
) -> tuple[float, float]:
    image_delta = max(
        int(
            np.abs(
                np.asarray(attached[key][index], dtype=np.int16)
                - np.asarray(slipped[key][index], dtype=np.int16)
            ).max()
        )
        for key in ("agentview", "wrist")
    )
    state_delta = float(
        np.abs(
            np.asarray(attached["robot_state"][index], dtype=np.float64)
            - np.asarray(slipped["robot_state"][index], dtype=np.float64)
        ).max()
    )
    return float(image_delta), state_delta


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = {}
    identity_keys = {"pair_id", "source_initial_state_index", "stale_age", "horizon"}
    numeric_keys = sorted(
        key
        for key in set().union(*(row.keys() for row in rows)) - identity_keys
        if any(isinstance(row.get(key), (bool, int, float)) for row in rows)
    )
    for stale_age in sorted({int(row["stale_age"]) for row in rows}):
        for horizon in sorted({int(row["horizon"]) for row in rows}):
            selected = [
                row
                for row in rows
                if int(row["stale_age"]) == stale_age and int(row["horizon"]) == horizon
            ]
            values = {
                key: float(np.mean([float(row[key]) for row in selected]))
                for key in numeric_keys
                if all(row.get(key) is not None for row in selected)
            }
            stale_slip = values.get("stale_slipped_mse")
            fresh_slip = values.get("fresh_slipped_mse")
            values["relative_slipped_mse_reduction"] = (
                None
                if stale_slip is None or stale_slip <= 1e-12 or fresh_slip is None
                else float((stale_slip - fresh_slip) / stale_slip)
            )
            summaries[f"age{stale_age}_h{horizon}"] = {
                "group_count": len(selected),
                "source_initial_state_count": len(
                    {int(row["source_initial_state_index"]) for row in selected}
                ),
                "means": values,
            }
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose stale-tail versus fresh-feedback Pi0.5 actions")
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--stale-ages", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--horizons", type=int, nargs="+", default=(1, 2, 3))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if args.max_groups is not None and args.max_groups < 1:
        raise ValueError("max-groups must be positive")
    if any(value < 1 for value in (*args.stale_ages, *args.horizons)):
        raise ValueError("stale ages and horizons must be positive")

    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    groups = sorted(
        (group for group in manifest["groups"] if group["split"] == args.split),
        key=lambda group: str(group["pair_id"]),
    )
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    if not groups:
        raise ValueError("no groups selected")

    policy = StoredObservationPolicy(args.policy_socket)
    rows = []
    try:
        if max(args.stale_ages) + max(args.horizons) > policy.horizon:
            raise ValueError("requested stale tail exceeds policy horizon")
        for group_index, group in enumerate(groups):
            pair_id = str(group["pair_id"])
            feedback = int(group["feedback_reveal_time"])
            attached = load_episode(args.episode_root / str(group["episode_files"]["attached"]))
            slipped = load_episode(args.episode_root / str(group["episode_files"]["slipped"]))
            if feedback <= max(args.stale_ages):
                raise ValueError(f"feedback index is too early for {pair_id}: {feedback}")
            for age in args.stale_ages:
                pre_image_delta, pre_state_delta = _observation_deltas(
                    attached, slipped, feedback - age
                )
                if pre_image_delta != 0.0 or pre_state_delta != 0.0:
                    raise RuntimeError(f"pre-feedback twin leakage for {pair_id} at age {age}")
            post_image_delta, post_state_delta = _observation_deltas(attached, slipped, feedback)
            if post_image_delta <= 0.0:
                raise RuntimeError(f"feedback is not visually observable for {pair_id}")

            inference_seed = stable_seed(
                args.policy_seed,
                pair_id,
                "counterfactual_feedback_revision",
            )
            stale_chunks = {
                age: policy.predict(attached, feedback - age, inference_seed)
                for age in args.stale_ages
            }
            fresh_attached = policy.predict(attached, feedback, inference_seed)
            fresh_slipped = policy.predict(slipped, feedback, inference_seed)
            teacher_attached_all = np.asarray(attached["actions"][feedback:], dtype=np.float32)
            teacher_slipped_all = np.asarray(slipped["actions"][feedback:], dtype=np.float32)
            if len(teacher_attached_all) < max(args.horizons) or len(teacher_slipped_all) < max(args.horizons):
                raise RuntimeError(f"teacher tail is too short for {pair_id}")

            slipped_grasped = bool(slipped["grasped"][feedback])
            eef_position = np.asarray(slipped["eef_pose"][feedback][:3], dtype=np.float32)
            object_position = np.asarray(slipped["object_pose"][feedback][:3], dtype=np.float32)
            for age in args.stale_ages:
                for horizon in args.horizons:
                    stale = stale_chunks[age][age : age + horizon]
                    teacher_attached = teacher_attached_all[:horizon]
                    teacher_slipped = teacher_slipped_all[:horizon]
                    row = {
                        "pair_id": pair_id,
                        "source_initial_state_index": int(group["source_initial_state_index"]),
                        "stale_age": int(age),
                        "horizon": int(horizon),
                        "post_feedback_image_max_delta": post_image_delta,
                        "post_feedback_robot_state_max_delta": post_state_delta,
                        **revision_metrics(
                            stale,
                            fresh_attached[:horizon],
                            fresh_slipped[:horizon],
                            teacher_attached,
                            teacher_slipped,
                        ),
                        "stale_slipped_failure_continuation": is_failure_continuation(
                            stale[0], grasped=slipped_grasped
                        ),
                        "fresh_slipped_failure_continuation": is_failure_continuation(
                            fresh_slipped[0], grasped=slipped_grasped
                        ),
                        "teacher_slipped_failure_continuation": is_failure_continuation(
                            teacher_slipped[0], grasped=slipped_grasped
                        ),
                        "stale_slipped_recovery_action": is_recovery_action(
                            stale[0],
                            grasped=slipped_grasped,
                            eef_position=eef_position,
                            object_position=object_position,
                        ),
                        "fresh_slipped_recovery_action": is_recovery_action(
                            fresh_slipped[0],
                            grasped=slipped_grasped,
                            eef_position=eef_position,
                            object_position=object_position,
                        ),
                        "teacher_slipped_recovery_action": is_recovery_action(
                            teacher_slipped[0],
                            grasped=slipped_grasped,
                            eef_position=eef_position,
                            object_position=object_position,
                        ),
                    }
                    rows.append(row)
            print(json.dumps({"completed": group_index + 1, "total": len(groups), "pair_id": pair_id}))
    finally:
        policy.close()

    result = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "mechanism_diagnostic_only",
        "split": args.split,
        "test_split_opened": False,
        "policy_seed": args.policy_seed,
        "git_sha": os.environ.get("FRESH_GIT_SHA"),
        "git_dirty_at_launch": os.environ.get("FRESH_GIT_DIRTY") != "0",
        "episode_root": str(args.episode_root.resolve()),
        "policy_identity": policy.identity,
        "group_count": len(groups),
        "source_initial_state_count": len({int(group["source_initial_state_index"]) for group in groups}),
        "stale_ages": list(args.stale_ages),
        "horizons": list(args.horizons),
        "rows": rows,
        "summaries": summarize_rows(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "group_count": len(groups), "status": "complete"}))


if __name__ == "__main__":
    main()
