from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from scripts.fresh_vla.summarize_libero_baseline_validation import (
        METRICS,
        group_metrics,
        summarize_across_seeds,
    )
except ModuleNotFoundError:
    from summarize_libero_baseline_validation import (
        METRICS,
        group_metrics,
        summarize_across_seeds,
    )


EXPECTED_HORIZONS = (1, 2, 3)
EXPECTED_GROUPS = 13
EXPECTED_MODEL_SIZE = 17_591_583_484


def baseline_gate(
    attached_by_seed: Mapping[int, float],
    *,
    minimum_cross_seed_mean: float,
    minimum_per_seed: float,
    minimum_seed_count: int,
) -> dict[str, Any]:
    if not attached_by_seed:
        raise ValueError("attached success is required for every seed")
    cross_seed_mean = float(np.mean(list(attached_by_seed.values())))
    passing_seeds = sorted(
        seed for seed, value in attached_by_seed.items() if value >= minimum_per_seed
    )
    conditions = {
        "cross_seed_attached_mean": cross_seed_mean >= minimum_cross_seed_mean,
        "minimum_seed_count": len(passing_seeds) >= minimum_seed_count,
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "cross_seed_attached_mean": cross_seed_mean,
        "per_seed_attached_success": {
            str(seed): attached_by_seed[seed] for seed in sorted(attached_by_seed)
        },
        "minimum_cross_seed_mean": minimum_cross_seed_mean,
        "minimum_per_seed": minimum_per_seed,
        "minimum_seed_count": minimum_seed_count,
        "passing_seeds": passing_seeds,
    }


def load_seed_result(
    eval_root: Path,
    *,
    seed: int,
    steps: int,
    tag: str,
) -> dict[str, Any]:
    run_dir = eval_root / f"fresh_closed_loop_repair_step{steps}_seed{seed}"
    result_path = run_dir / f"closed_loop_end_to_end_{tag}.json"
    payload = json.loads(result_path.read_text())
    if payload.get("status") != "complete" or payload.get("split") != "val":
        raise ValueError(f"incomplete or non-validation result: {result_path}")
    if payload.get("evaluation") != "end_to_end" or len(payload.get("rows", ())) != 78:
        raise ValueError(f"invalid end-to-end row set: {result_path}")
    if int(payload.get("completed_rows", -1)) != 78 or int(payload.get("expected_rows", -1)) != 78:
        raise ValueError(f"row-count metadata mismatch: {result_path}")
    expected_policy_seed = 314159 + seed
    if int(payload.get("seed", -1)) != expected_policy_seed:
        raise ValueError(f"policy evaluation seed mismatch: {result_path}")

    groups_by_horizon = {}
    for horizon in EXPECTED_HORIZONS:
        groups = group_metrics(payload["rows"], horizon)
        if len(groups) != EXPECTED_GROUPS:
            raise ValueError(f"expected {EXPECTED_GROUPS} groups for K={horizon}: {result_path}")
        groups_by_horizon[horizon] = groups

    identity_path = run_dir / "training_run_identity.json"
    identity = json.loads(identity_path.read_text())
    if int(identity.get("seed", -1)) != seed:
        raise ValueError(f"training seed mismatch: {identity_path}")
    if identity.get("git_dirty_at_launch") is not False:
        raise ValueError(f"training run was dirty at launch: {identity_path}")
    if identity.get("test_split_opened") is not False:
        raise ValueError(f"training identity says test was opened: {identity_path}")
    if int(identity.get("effective_batch_size", -1)) != 8:
        raise ValueError(f"effective batch mismatch: {identity_path}")
    if int(identity.get("optimizer_steps", -1)) < steps:
        raise ValueError(f"training run ended before selected checkpoint: {identity_path}")

    training_run_dir = identity_path.resolve().parent
    resume_meta_path = training_run_dir / "checkpoints" / f"steps_{steps}" / "resume_meta.json"
    resume_meta = json.loads(resume_meta_path.read_text())
    if int(resume_meta.get("completed_steps", -1)) != steps:
        raise ValueError(f"checkpoint step mismatch: {resume_meta_path}")
    if int(resume_meta.get("effective_batch_size", -1)) != 8:
        raise ValueError(f"checkpoint batch mismatch: {resume_meta_path}")

    model_path = run_dir / "final_model" / "model.safetensors"
    if model_path.stat().st_size != EXPECTED_MODEL_SIZE:
        raise ValueError(f"model size mismatch: {model_path}")
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "result_path": str(result_path),
        "training_identity_path": str(identity_path.resolve()),
        "checkpoint_resume_meta_path": str(resume_meta_path),
        "checkpoint_model_path": str(model_path.resolve()),
        "checkpoint_model_size_bytes": model_path.stat().st_size,
        "git_sha": str(identity["git_sha"]),
        "policy_evaluation_seed": expected_policy_seed,
        "groups_by_horizon": groups_by_horizon,
        "payload": payload,
    }


def seed41_repeat_equivalent(eval_root: Path, *, steps: int, final_tag: str) -> bool:
    run_dir = eval_root / f"fresh_closed_loop_repair_step{steps}_seed41"
    prior = json.loads((run_dir / "closed_loop_end_to_end_val_gate.json").read_text())
    repeated = json.loads((run_dir / f"closed_loop_end_to_end_{final_tag}.json").read_text())
    return prior.get("rows") == repeated.get("rows") and prior.get("summary") == repeated.get(
        "summary"
    )


def summarize_gate(
    runs: Mapping[int, Mapping[str, Any]],
    *,
    steps: int,
    bootstrap_samples: int,
    minimum_cross_seed_mean: float,
    minimum_per_seed: float,
    minimum_seed_count: int,
    seed41_repeat_match: bool,
) -> dict[str, Any]:
    git_shas = {str(run["git_sha"]) for run in runs.values()}
    if len(git_shas) != 1:
        raise ValueError(f"training Git SHA differs across seeds: {git_shas}")
    groups_by_seed = {seed: run["groups_by_horizon"][3] for seed, run in runs.items()}
    reference_groups = None
    for seed, groups in groups_by_seed.items():
        if reference_groups is None:
            reference_groups = set(groups)
        elif set(groups) != reference_groups:
            raise ValueError(f"validation groups differ for seed {seed}")

    per_seed_k3 = {
        str(seed): dict(run["payload"]["summary"]["3"])
        for seed, run in sorted(runs.items())
    }
    attached_by_seed = {
        seed: float(per_seed_k3[str(seed)]["attached_task_success"]) for seed in runs
    }
    gate = baseline_gate(
        attached_by_seed,
        minimum_cross_seed_mean=minimum_cross_seed_mean,
        minimum_per_seed=minimum_per_seed,
        minimum_seed_count=minimum_seed_count,
    )
    aggregate = summarize_across_seeds(
        groups_by_seed,
        bootstrap_samples=bootstrap_samples,
    )
    cross_seed_mean_k3 = {}
    for metric in sorted(per_seed_k3[str(next(iter(runs)))].keys()):
        values = [per_seed_k3[str(seed)].get(metric) for seed in runs]
        finite = [float(value) for value in values if value is not None]
        cross_seed_mean_k3[metric] = float(np.mean(finite)) if finite else None

    decision = (
        "BASELINE_VALID_PROCEED_TO_RECOVERY_CONTROLS"
        if gate["passed"]
        else "BASELINE_INVALID_OR_DATA_INSUFFICIENT"
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "split": "val",
        "test_split_opened": False,
        "method": "full_h_baseline_repair",
        "uniform_training_budget_steps": steps,
        "seeds": sorted(runs),
        "execution_horizon_for_gate": 3,
        "statistical_unit": "snapshot group after averaging seeds",
        "git_sha": next(iter(git_shas)),
        "seed41_repeat_equivalent": seed41_repeat_match,
        "gate": gate,
        "decision": decision,
        "per_seed_k3": per_seed_k3,
        "cross_seed_mean_k3": cross_seed_mean_k3,
        "group_bootstrap_k3": aggregate,
        "source_runs": {
            str(seed): {
                key: run[key]
                for key in (
                    "run_dir",
                    "result_path",
                    "training_identity_path",
                    "checkpoint_resume_meta_path",
                    "checkpoint_model_path",
                    "checkpoint_model_size_bytes",
                    "policy_evaluation_seed",
                )
            }
            for seed, run in sorted(runs.items())
        },
    }


def markdown_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Full-H Baseline Repair Gate",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Uniform optimizer steps: {result['uniform_training_budget_steps']}",
        f"- Fixed K: {result['execution_horizon_for_gate']}",
        f"- Seed-41 deterministic repeat: `{str(result['seed41_repeat_equivalent']).lower()}`",
        f"- Test split opened: `{str(result['test_split_opened']).lower()}`",
        "",
        "| Seed | Attached | Overall | Slip recovery | Failure continuation | Premature commitment |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in result["seeds"]:
        row = result["per_seed_k3"][str(seed)]
        lines.append(
            f"| {seed} | {row['attached_task_success']:.3f} | "
            f"{row['overall_task_success']:.3f} | {row['slip_final_recovery_success']:.3f} | "
            f"{row['failure_continuation_rate']:.3f} | {row['premature_commitment_rate']:.3f} |"
        )
    lines.extend(("", "| Group-level metric | Mean [bootstrap 95% CI] |", "| --- | ---: |"))
    for metric in METRICS:
        row = result["group_bootstrap_k3"][metric]
        lines.append(
            f"| `{metric}` | {row['mean']:.3f} "
            f"[{row['bootstrap_95_low']:.3f}, {row['bootstrap_95_high']:.3f}] |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the repaired three-seed Full-H gate")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/baseline-repair-v1/eval_views"),
    )
    parser.add_argument("--steps", type=int, default=3451)
    parser.add_argument("--seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--tag", default="val_gate_final")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--minimum-cross-seed-mean", type=float, default=0.30)
    parser.add_argument("--minimum-per-seed", type=float, default=0.20)
    parser.add_argument("--minimum-seed-count", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = {
        seed: load_seed_result(args.eval_root, seed=seed, steps=args.steps, tag=args.tag)
        for seed in args.seeds
    }
    repeat_match = seed41_repeat_equivalent(
        args.eval_root,
        steps=args.steps,
        final_tag=args.tag,
    )
    if not repeat_match:
        raise ValueError("seed-41 deterministic repeated evaluation changed")
    result = summarize_gate(
        runs,
        steps=args.steps,
        bootstrap_samples=args.bootstrap_samples,
        minimum_cross_seed_mean=args.minimum_cross_seed_mean,
        minimum_per_seed=args.minimum_per_seed,
        minimum_seed_count=args.minimum_seed_count,
        seed41_repeat_match=repeat_match,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.report.write_text(markdown_report(result))
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "gate_passed": result["gate"]["passed"],
                "output": str(args.output),
                "report": str(args.report),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
