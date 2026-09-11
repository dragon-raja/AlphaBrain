from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scripts.fresh_vla.summarize_libero_baseline_validation import group_metrics
except ModuleNotFoundError:
    from summarize_libero_baseline_validation import group_metrics


DEFAULT_STEPS = (3451, 6902, 10353, 13804)
EXPECTED_HORIZONS = (1, 2, 3)


def _mean_metric(groups: Mapping[str, Mapping[str, float]], metric: str) -> float:
    return float(np.mean([row[metric] for row in groups.values()]))


def load_validation_result(
    path: Path,
    *,
    expected_groups: int,
) -> tuple[dict[int, dict[str, dict[str, float]]], dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"validation result is not complete: {path}")
    if payload.get("split") != "val" or payload.get("evaluation") != "end_to_end":
        raise ValueError(f"result is not validation end-to-end evaluation: {path}")

    rows = payload.get("rows", [])
    expected_rows = expected_groups * 2 * len(EXPECTED_HORIZONS)
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows in {path}, found {len(rows)}")
    if int(payload.get("completed_rows", len(rows))) != expected_rows:
        raise ValueError(f"completed_rows mismatch in {path}")
    if int(payload.get("expected_rows", expected_rows)) != expected_rows:
        raise ValueError(f"expected_rows mismatch in {path}")

    groups_by_horizon = {}
    for horizon in EXPECTED_HORIZONS:
        groups = group_metrics(rows, horizon)
        if len(groups) != expected_groups:
            raise ValueError(
                f"expected {expected_groups} paired groups for K={horizon} in {path}, "
                f"found {len(groups)}"
            )
        groups_by_horizon[horizon] = groups
    return groups_by_horizon, payload


def select_checkpoint(
    results: Mapping[int, Mapping[str, Mapping[str, float]]],
    *,
    minimum_attached_success: float,
    minimum_overall_success: float,
    maximum_attached_regression: float,
    fallback_steps: int,
) -> dict[str, Any]:
    steps = sorted(results)
    if fallback_steps not in results:
        raise ValueError("fallback checkpoint must be present in results")
    if not steps:
        raise ValueError("at least one checkpoint result is required")

    reference_groups = set(results[steps[0]])
    summaries = []
    selected_steps = None
    previous_attached = None
    for step in steps:
        groups = results[step]
        if set(groups) != reference_groups:
            raise ValueError(f"validation group set changed at step {step}")
        attached = _mean_metric(groups, "attached_task_success")
        overall = _mean_metric(groups, "overall_task_success")
        slipped = _mean_metric(groups, "slip_task_success")
        attached_change = None if previous_attached is None else attached - previous_attached
        no_regression = (
            previous_attached is None
            or attached_change is not None
            and attached_change >= -maximum_attached_regression - 1e-12
        )
        conditions = {
            "attached_success": attached >= minimum_attached_success,
            "overall_success": overall >= minimum_overall_success,
            "no_attached_regression": no_regression,
        }
        passed = all(conditions.values())
        summaries.append(
            {
                "optimizer_steps": step,
                "group_count": len(groups),
                "attached_task_success": attached,
                "overall_task_success": overall,
                "slip_task_success": slipped,
                "attached_change_from_previous": attached_change,
                "conditions": conditions,
                "passed": passed,
            }
        )
        if selected_steps is None and passed:
            selected_steps = step
        previous_attached = attached

    selected_by_gate = selected_steps is not None
    if selected_steps is None:
        selected_steps = fallback_steps
    return {
        "schema_version": 1,
        "status": "complete",
        "split": "val",
        "execution_horizon": 3,
        "statistical_unit": "snapshot group",
        "test_split_opened": False,
        "offline_diagnostics_used_for_selection": False,
        "criteria": {
            "minimum_attached_success": minimum_attached_success,
            "minimum_overall_success": minimum_overall_success,
            "maximum_attached_regression": maximum_attached_regression,
            "selection_rule": "earliest checkpoint satisfying all criteria",
            "fallback_steps": fallback_steps,
        },
        "checkpoint_summaries": summaries,
        "selected_by_gate": selected_by_gate,
        "uniform_training_budget_steps": selected_steps,
        "decision": (
            "SELECT_EARLIEST_PASSING_CHECKPOINT"
            if selected_by_gate
            else "FALLBACK_TO_PREREGISTERED_MAX_BUDGET"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the frozen Full-H baseline-repair budget")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/baseline-repair-v1/eval_views"),
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--tag", default="val_gate")
    parser.add_argument("--expected-groups", type=int, default=13)
    parser.add_argument("--minimum-attached-success", type=float, default=0.30)
    parser.add_argument("--minimum-overall-success", type=float, default=0.25)
    parser.add_argument("--maximum-attached-regression", type=float, default=0.10)
    parser.add_argument("--fallback-steps", type=int, default=13804)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_groups < 1:
        raise ValueError("expected-groups must be positive")
    for name in (
        "minimum_attached_success",
        "minimum_overall_success",
        "maximum_attached_regression",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name.replace('_', '-')} must be in [0, 1]")

    k3_results = {}
    sources = {}
    checkpoint_realpaths = {}
    training_identity_realpaths = {}
    policy_seeds = set()
    for step in sorted(set(args.steps)):
        run_dir = args.eval_root / f"fresh_closed_loop_repair_step{step}_seed{args.seed}"
        path = run_dir / f"closed_loop_end_to_end_{args.tag}.json"
        checkpoint_dir = run_dir / "final_model"
        training_identity = run_dir / "training_run_identity.json"
        if not (checkpoint_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"missing checkpoint model for step {step}: {checkpoint_dir}")
        if not training_identity.is_file():
            raise FileNotFoundError(
                f"missing training run identity for step {step}: {training_identity}"
            )
        groups_by_horizon, payload = load_validation_result(
            path,
            expected_groups=args.expected_groups,
        )
        k3_results[step] = groups_by_horizon[3]
        sources[str(step)] = str(path)
        checkpoint_realpaths[str(step)] = str(checkpoint_dir.resolve())
        training_identity_realpaths[str(step)] = str(training_identity.resolve())
        policy_seeds.add(int(payload["seed"]))
    if len(policy_seeds) != 1:
        raise ValueError("policy evaluation seed changed across checkpoints")

    result = select_checkpoint(
        k3_results,
        minimum_attached_success=args.minimum_attached_success,
        minimum_overall_success=args.minimum_overall_success,
        maximum_attached_regression=args.maximum_attached_regression,
        fallback_steps=args.fallback_steps,
    )
    result.update(
        {
            "training_seed": args.seed,
            "policy_evaluation_seed": next(iter(policy_seeds)),
            "source_files": sources,
            "checkpoint_realpaths": checkpoint_realpaths,
            "training_identity_realpaths": training_identity_realpaths,
        }
    )
    output = args.output or args.eval_root.parent / (
        f"baseline_repair_seed{args.seed}_checkpoint_selection.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "uniform_training_budget_steps": result["uniform_training_budget_steps"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
