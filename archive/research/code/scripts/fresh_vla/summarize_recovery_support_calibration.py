from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_libero_closed_loop import summarize_rows


SEEDS = (41, 42, 43)
EXECUTION_HORIZONS = (1, 2, 3)
METRICS = (
    "attached_task_success",
    "normal_no_intervention_success",
    "overall_task_success",
    "slip_final_recovery_success",
    "slip_regrasp_success_rate",
    "grasp_subgoal_rate",
    "lift_subgoal_rate",
    "transport_subgoal_rate",
    "place_subgoal_rate",
    "failure_continuation_rate",
    "premature_commitment_rate",
)


def validate_evaluation_payload(
    payload: Mapping[str, Any],
    *,
    expected_groups: int = 13,
) -> None:
    expected_rows = expected_groups * 2 * len(EXECUTION_HORIZONS)
    if payload.get("status") != "complete" or payload.get("split") != "val":
        raise ValueError("calibration evaluation must be a complete validation run")
    if payload.get("evaluation") != "end_to_end":
        raise ValueError("calibration evaluation must be end_to_end")
    if int(payload.get("completed_rows", -1)) != expected_rows:
        raise ValueError(f"calibration evaluation must contain {expected_rows} completed rows")
    rows = list(payload.get("rows", ()))
    if len(rows) != expected_rows:
        raise ValueError(f"calibration row list must contain {expected_rows} rows")

    identities = [
        (int(row["execution_horizon"]), str(row["pair_id"]), str(row["branch_outcome"]))
        for row in rows
    ]
    if len(set(identities)) != expected_rows:
        raise ValueError("calibration evaluation contains duplicate row identities")
    for horizon in EXECUTION_HORIZONS:
        selected = [row for row in rows if int(row["execution_horizon"]) == horizon]
        pair_ids = {str(row["pair_id"]) for row in selected}
        if len(pair_ids) != expected_groups:
            raise ValueError(f"K={horizon} does not contain {expected_groups} groups")
        for pair_id in pair_ids:
            outcomes = {
                str(row["branch_outcome"])
                for row in selected
                if str(row["pair_id"]) == pair_id
            }
            if outcomes != {"attached", "slipped"}:
                raise ValueError(f"K={horizon} group {pair_id} does not contain both branches")
        recomputed = summarize_rows(selected)
        declared = payload.get("summary", {}).get(str(horizon))
        if declared is None:
            raise ValueError(f"calibration summary is missing K={horizon}")
        if set(recomputed) != set(declared):
            raise ValueError(f"calibration summary keys changed for K={horizon}")
        for key, expected in recomputed.items():
            actual = declared[key]
            if expected is None or actual is None:
                if expected is not actual:
                    raise ValueError(f"calibration summary null mismatch for K={horizon} {key}")
            elif not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12):
                raise ValueError(f"calibration summary mismatch for K={horizon} {key}")


def _means(per_seed: Mapping[str, Mapping[str, float | None]]) -> dict[str, float | None]:
    result = {}
    for metric in METRICS:
        values = [
            float(row[metric])
            for row in per_seed.values()
            if row.get(metric) is not None
        ]
        result[metric] = float(np.mean(values)) if values else None
    return result


def calibration_gate(
    per_seed_k3: Mapping[str, Mapping[str, float | None]],
) -> dict[str, Any]:
    attached = {
        seed: float(row["attached_task_success"])
        for seed, row in per_seed_k3.items()
        if row.get("attached_task_success") is not None
    }
    finite = len(attached) == len(SEEDS) and all(np.isfinite(value) for value in attached.values())
    mean = float(np.mean(list(attached.values()))) if attached else None
    seeds_at_threshold = sum(value >= 0.20 for value in attached.values())
    passed = bool(finite and mean is not None and mean >= 0.30 and seeds_at_threshold >= 2)
    return {
        "execution_horizon": 3,
        "minimum_cross_seed_attached_success": 0.30,
        "minimum_seeds_at_or_above_0_20": 2,
        "all_seed_metrics_finite": finite,
        "observed_cross_seed_attached_success": mean,
        "observed_seeds_at_or_above_0_20": seeds_at_threshold,
        "passed": passed,
    }


def summarize_calibration(
    output_root: Path,
    baseline_root: Path,
    *,
    steps: int,
) -> dict[str, Any]:
    per_seed = {}
    prior = {}
    source_files = {}
    for seed in SEEDS:
        candidate_path = (
            output_root
            / f"recovery_support_base_continuation_seed{seed}_steps{steps}"
            / f"closed_loop_end_to_end_val_calibration_steps{steps}.json"
        )
        prior_path = (
            baseline_root
            / f"fresh_closed_loop_full_h_seed{seed}"
            / "closed_loop_end_to_end_val_budget27607.json"
        )
        candidate = json.loads(candidate_path.read_text())
        baseline = json.loads(prior_path.read_text())
        validate_evaluation_payload(candidate)
        validate_evaluation_payload(baseline)
        per_seed[str(seed)] = {
            metric: candidate["summary"]["3"].get(metric) for metric in METRICS
        }
        prior[str(seed)] = {
            metric: baseline["summary"]["3"].get(metric) for metric in METRICS
        }
        source_files[str(seed)] = {
            "candidate": str(candidate_path),
            "original_full_h": str(prior_path),
        }

    mean = _means(per_seed)
    prior_mean = _means(prior)
    delta = {
        metric: (
            mean[metric] - prior_mean[metric]
            if mean[metric] is not None and prior_mean[metric] is not None
            else None
        )
        for metric in METRICS
    }
    gate = calibration_gate(per_seed)
    if gate["passed"]:
        decision = f"FREEZE_{steps}_UPDATES"
    elif steps < 13_804:
        decision = "RETRY_FROM_ORIGINAL_CHECKPOINT_AT_13804_UPDATES"
    else:
        decision = "BASELINE_INVALID_OR_DATA_INSUFFICIENT"
    video_audit = output_root / f"video_artifact_audit_steps{steps}.json"
    return {
        "schema_version": 1,
        "arm": "base_continuation",
        "calibration_updates": steps,
        "split": "val",
        "test_opened": False,
        "status": "complete",
        "source_files": source_files,
        "per_seed_k3": per_seed,
        "cross_seed_mean_k3": mean,
        "original_full_h_per_seed_k3": prior,
        "original_full_h_cross_seed_mean_k3": prior_mean,
        "delta_from_original_full_h_cross_seed_mean_k3": delta,
        "gate": gate,
        "decision": decision,
        "video_audit": str(video_audit) if video_audit.is_file() else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and gate recovery-support Base calibration")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/recovery-support-v1"),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2"),
    )
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    payload = summarize_calibration(
        args.output_root,
        args.baseline_root,
        steps=args.steps,
    )
    output = args.output or args.output_root / f"base_continuation_calibration_steps{args.steps}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "gate": payload["gate"], "decision": payload["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
