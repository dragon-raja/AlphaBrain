from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from counterfactual_data import (
    CounterfactualRecord,
    build_policy_inputs,
    estimate_branch_divergence,
    threshold_sensitivity,
    validate_record,
)


TASKS = ("grasp", "blocked_push", "deterministic_reach", "intent_control")


def _common_actions(rng: np.random.Generator, horizon: int) -> np.ndarray:
    start = rng.normal(0.0, 0.2, size=2)
    progress = np.linspace(0.0, 1.0, horizon)
    return np.stack(
        [start[0] + 0.2 * progress, start[1] - 0.15 * progress + 0.03 * np.sin(np.pi * progress)],
        axis=-1,
    )


def _task_branches(
    task: str,
    common: np.ndarray,
    pair_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, str], int, int, int, int, str]:
    horizon = len(common)
    event_time = 3 + pair_index % 2
    reveal_time = event_time + 1
    divergence_time = reveal_time + 1
    suffix = np.linspace(0.2, 1.0, horizon - divergence_time)[:, None]

    if task == "grasp":
        success = common.copy()
        failure = common.copy()
        success[divergence_time:] += suffix * np.array([0.9, 0.3])
        failure[divergence_time:] += suffix * np.array([-0.5, 0.8])
        return (
            {"success": success, "failure_recovery": failure},
            {"success": "attached", "failure_recovery": "slipped"},
            event_time,
            reveal_time,
            divergence_time,
            event_time,
            "grasp and lift the object",
        )
    if task == "blocked_push":
        free = common.copy()
        blocked = common.copy()
        free[divergence_time:] += suffix * np.array([1.0, 0.0])
        blocked[divergence_time:] += suffix * np.array([-0.2, 1.0])
        return (
            {"free_slide": free, "blocked_recovery": blocked},
            {"free_slide": "sliding", "blocked_recovery": "blocked"},
            event_time,
            reveal_time,
            divergence_time,
            horizon,
            "push the object to the target",
        )
    if task == "deterministic_reach":
        return (
            {"repeat_a": common.copy(), "repeat_b": common.copy()},
            {"repeat_a": "reached", "repeat_b": "reached"},
            horizon,
            horizon,
            horizon,
            horizon,
            "reach the marked point",
        )
    if task == "intent_control":
        # The language is part of the conditioning, so each intent forms its own pair.
        direction = -1.0 if pair_index % 2 == 0 else 1.0
        target = common.copy()
        target += np.linspace(0.0, direction, horizon)[:, None] * np.array([1.0, 0.0])
        language = "reach the left cup" if direction < 0 else "reach the right cup"
        return (
            {"repeat_a": target.copy(), "repeat_b": target.copy()},
            {"repeat_a": "intent_known", "repeat_b": "intent_known"},
            horizon,
            horizon,
            horizon,
            horizon,
            language,
        )
    raise ValueError(f"unknown task: {task}")


def generate_records(
    *,
    pairs_per_task: int,
    repeats: int,
    horizon: int,
    seed: int,
) -> tuple[list[CounterfactualRecord], list[dict[str, object]]]:
    if horizon < 8:
        raise ValueError("horizon must be at least 8")
    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    rng = np.random.default_rng(seed)
    records = []
    pair_metadata = []

    for task in TASKS:
        for pair_index in range(pairs_per_task):
            pair_id = f"{task}-{pair_index:05d}"
            common = _common_actions(rng, horizon)
            branches, outcomes, event, reveal, expected_divergence, gripper_horizon, language = _task_branches(
                task,
                common,
                pair_index,
            )
            repeated = {
                name: np.stack([actions + rng.normal(0.0, 0.005, actions.shape) for _ in range(repeats)])
                for name, actions in branches.items()
            }
            estimate = estimate_branch_divergence(repeated, persistence=2)
            sensitivity = threshold_sensitivity(repeated, persistence=2)
            if task in {"grasp", "blocked_push"} and estimate.oracle_feedback_horizon != expected_divergence:
                raise RuntimeError(
                    f"unstable synthetic boundary for {pair_id}: expected {expected_divergence}, "
                    f"got {estimate.oracle_feedback_horizon}"
                )

            observation = {
                "task": task,
                "visible_features": rng.normal(size=4).round(6).tolist(),
            }
            robot_state = rng.normal(size=4).round(6).tolist()
            for branch_id, actions in branches.items():
                record = CounterfactualRecord(
                    pair_id=pair_id,
                    branch_id=branch_id,
                    branch_outcome=outcomes[branch_id],
                    observation=observation,
                    robot_state=robot_state,
                    language_instruction=language,
                    action_chunk=actions.round(7).tolist(),
                    event_time=event,
                    feedback_reveal_time=reveal,
                    action_divergence_time=estimate.action_divergence_time,
                    gripper_transition_horizon=gripper_horizon,
                    oracle_feedback_horizon=estimate.oracle_feedback_horizon,
                    per_step_branch_divergence=estimate.per_step_branch_divergence,
                    is_deterministic_control=task in {"deterministic_reach", "intent_control"},
                )
                validate_record(record)
                build_policy_inputs(record)
                records.append(record)

            conditioning = json.dumps(
                {
                    "observation": observation,
                    "robot_state": robot_state,
                    "language_instruction": language,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            pair_metadata.append(
                {
                    "pair_id": pair_id,
                    "task": task,
                    "conditioning_sha256": hashlib.sha256(conditioning.encode()).hexdigest(),
                    "within_branch_threshold": estimate.within_branch_threshold,
                    "oracle_feedback_horizon": estimate.oracle_feedback_horizon,
                    "threshold_sensitivity": {
                        multiplier: asdict(value) for multiplier, value in sensitivity.items()
                    },
                }
            )
    return records, pair_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate zero-download FRESH counterfactual pair fixtures")
    parser.add_argument("--output-dir", type=Path, default=Path("/share/longjunyu/fresh-vla/counterfactual-toy"))
    parser.add_argument("--pairs-per-task", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, metadata = generate_records(
        pairs_per_task=args.pairs_per_task,
        repeats=args.repeats,
        horizon=args.horizon,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    records_path.write_text("".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in records))
    manifest = {
        "generator": "generate_counterfactual_pairs.py",
        "seed": args.seed,
        "horizon": args.horizon,
        "repeats": args.repeats,
        "pairs_per_task": args.pairs_per_task,
        "record_count": len(records),
        "pair_count": len(metadata),
        "policy_input_fields": ["observation", "robot_state", "language_instruction"],
        "pairs": metadata,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "pairs"}, sort_keys=True))


if __name__ == "__main__":
    main()
