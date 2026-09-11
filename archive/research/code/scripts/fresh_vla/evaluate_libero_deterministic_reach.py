from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_libero_closed_loop import (
    Pi05Policy,
    RemotePi05Policy,
    _load_reference_arrays,
    _restore_recorded_state,
    stable_seed,
)
from libero_snapshot_collector import DEFAULT_BDDL, _step


def target_error(position: Sequence[float], target: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(position, dtype=np.float64) - np.asarray(target, dtype=np.float64)))


def reference_target(reference: Mapping[str, np.ndarray], target_step: int) -> tuple[np.ndarray, int]:
    eef_pose = np.asarray(reference["eef_pose"], dtype=np.float64)
    if eef_pose.ndim != 2 or eef_pose.shape[1] < 3 or len(eef_pose) == 0:
        raise ValueError("reference eef_pose must have shape [steps, >=3]")
    resolved_step = min(target_step, len(eef_pose) - 1)
    return eef_pose[resolved_step, :3], resolved_step


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    return {
        "group_count": len({str(row["pair_id"]) for row in rows}),
        "deterministic_reach_success": float(np.mean([row["success"] for row in rows])),
        "mean_best_target_error": float(np.mean([row["best_target_error"] for row in rows])),
        "mean_final_target_error": float(np.mean([row["final_target_error"] for row in rows])),
    }


def run_reach(
    env: Any,
    policy: Pi05Policy | RemotePi05Policy,
    episode_root: Path,
    group: Mapping[str, Any],
    *,
    execution_horizon: int,
    max_steps: int,
    threshold: float,
    reference_target_step: int,
    seed: int,
) -> dict[str, Any]:
    reference = _load_reference_arrays(episode_root, group)
    observation = _restore_recorded_state(env, reference, 0)
    target, resolved_target_step = reference_target(reference, reference_target_step)
    error = target_error(observation["robot0_eef_pos"], target)
    initial_error = error
    best_error = error
    steps = 0
    replans = 0
    success = error <= threshold
    first_predicted_action = None
    while steps < max_steps and not success:
        chunk = policy.predict(observation, stable_seed(seed, group["pair_id"], "reach", replans))
        if first_predicted_action is None:
            first_predicted_action = np.asarray(chunk[0], dtype=np.float64)
        replans += 1
        for action in chunk[:execution_horizon]:
            observation = _step(env, action)
            steps += 1
            error = target_error(observation["robot0_eef_pos"], target)
            best_error = min(best_error, error)
            success = error <= threshold
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
    return {
        "pair_id": group["pair_id"],
        "split": group["split"],
        "execution_horizon": execution_horizon,
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
        "replan_count": replans,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-K deterministic reach control")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-socket", type=Path)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-horizons", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--success-threshold", type=float, default=0.04)
    parser.add_argument("--reference-target-step", type=int, default=20)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--seed", type=int, default=271828)
    return parser.parse_args()


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if any(value not in {1, 2, 3} for value in args.execution_horizons):
        raise ValueError("execution horizons must be selected from 1, 2, 3")
    if args.reference_target_step < 1:
        raise ValueError("reference target step must be positive")
    if (args.checkpoint is None) == (args.policy_socket is None):
        raise ValueError("provide exactly one of --checkpoint or --policy-socket")
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
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(manifest.get("bddl", DEFAULT_BDDL))),
        camera_heights=224,
        camera_widths=224,
    )
    env.seed(args.seed)
    rows = []
    try:
        for execution_horizon in args.execution_horizons:
            for group in groups:
                row = run_reach(
                    env,
                    policy,
                    args.episode_root,
                    group,
                    execution_horizon=execution_horizon,
                    max_steps=args.max_steps,
                    threshold=args.success_threshold,
                    reference_target_step=args.reference_target_step,
                    seed=args.seed,
                )
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        env.close()
        policy.close()
    summary = {
        str(k): summarize_rows([row for row in rows if row["execution_horizon"] == k])
        for k in args.execution_horizons
    }
    payload = {
        "evaluation": "deterministic_reach",
        "split": args.split,
        "seed": args.seed,
        "target_definition": "recorded_expert_eef_position",
        "reference_target_step": args.reference_target_step,
        "success_threshold": args.success_threshold,
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
