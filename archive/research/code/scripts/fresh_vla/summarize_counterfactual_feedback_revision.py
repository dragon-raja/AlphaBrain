from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SEEDS = (41, 42, 43)
PRIMARY_KEY = "age1_h1"
BOOTSTRAP_METRICS = (
    "slipped_stale_minus_fresh_mse",
    "failure_continuation_reduction",
    "recovery_action_gain",
    "fresh_pair_assignment_correct",
    "fresh_slipped_failure_continuation",
    "fresh_slipped_recovery_action",
    "attached_fresh_minus_stale_mse",
)
SEED_REPORT_METRICS = (
    "slipped_stale_minus_fresh_mse",
    "relative_slipped_mse_reduction",
    "stale_slipped_recovery_action",
    "fresh_slipped_recovery_action",
    "fresh_pair_assignment_correct",
    "attached_fresh_minus_stale_mse",
)


def _row_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(row["pair_id"]), int(row["stale_age"]), int(row["horizon"])


def load_seed(path: Path, expected_seed: int) -> Mapping[str, Any]:
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "complete"
        or payload.get("purpose") != "mechanism_diagnostic_only"
        or payload.get("split") != "train"
        or payload.get("test_split_opened") is not False
        or payload.get("policy_seed") != expected_seed
        or payload.get("git_dirty_at_launch") is not False
    ):
        raise ValueError(f"invalid feedback-revision diagnostic: {path}")
    rows = payload.get("rows", [])
    if len(rows) != int(payload["group_count"]) * len(payload["stale_ages"]) * len(payload["horizons"]):
        raise ValueError(f"feedback-revision row count mismatch: {path}")
    if len({_row_key(row) for row in rows}) != len(rows):
        raise ValueError(f"duplicate feedback-revision row identity: {path}")
    return payload


def average_seed_rows(payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not payloads:
        raise ValueError("at least one payload is required")
    indexed = [{_row_key(row): row for row in payload["rows"]} for payload in payloads]
    identities = set(indexed[0])
    if any(set(rows) != identities for rows in indexed[1:]):
        raise ValueError("seed diagnostics do not contain identical paired groups")
    result = []
    identity_fields = {"pair_id", "source_initial_state_index", "stale_age", "horizon"}
    for identity in sorted(identities):
        seed_rows = [rows[identity] for rows in indexed]
        if len({int(row["source_initial_state_index"]) for row in seed_rows}) != 1:
            raise ValueError(f"source identity changed across seeds: {identity}")
        row = {key: seed_rows[0][key] for key in identity_fields}
        value_keys = set().union(*(seed_row.keys() for seed_row in seed_rows)) - identity_fields
        for key in sorted(value_keys):
            values = [seed_row.get(key) for seed_row in seed_rows]
            numeric = [float(value) for value in values if isinstance(value, (bool, int, float))]
            row[key] = float(np.mean(numeric)) if len(numeric) == len(values) else None
        row["failure_continuation_reduction"] = (
            row["stale_slipped_failure_continuation"]
            - row["fresh_slipped_failure_continuation"]
        )
        row["recovery_action_gain"] = (
            row["fresh_slipped_recovery_action"] - row["stale_slipped_recovery_action"]
        )
        result.append(row)
    return result


def bootstrap_interval(values: Sequence[float], *, samples: int, seed: int) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a nonempty finite vector")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[draws].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "bootstrap_95_low": float(np.quantile(means, 0.025)),
        "bootstrap_95_high": float(np.quantile(means, 0.975)),
        "group_count": int(len(array)),
    }


def seed_primary_metrics(payload: Mapping[str, Any]) -> dict[str, float | int]:
    try:
        means = payload["summaries"][PRIMARY_KEY]["means"]
        return {
            "policy_seed": int(payload["policy_seed"]),
            **{metric: float(means[metric]) for metric in SEED_REPORT_METRICS},
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("diagnostic is missing primary per-seed metrics") from exc


def mechanism_interpretation(primary: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    slip_gain = primary["slipped_stale_minus_fresh_mse"]
    failure_gain = primary["failure_continuation_reduction"]
    recovery_gain = primary["recovery_action_gain"]
    assignment = primary["fresh_pair_assignment_correct"]
    attached_change = primary["attached_fresh_minus_stale_mse"]
    fresh_failure = primary["fresh_slipped_failure_continuation"]
    fresh_recovery = primary["fresh_slipped_recovery_action"]

    execution_staleness = (
        slip_gain["bootstrap_95_low"] > 0.0
        and (
            (
                failure_gain["mean"] >= 0.10
                and failure_gain["bootstrap_95_low"] > 0.0
            )
            or (
                recovery_gain["mean"] >= 0.10
                and recovery_gain["bootstrap_95_low"] > 0.0
            )
        )
    )
    feedback_specificity = (
        assignment["mean"] >= 0.75 and attached_change["mean"] <= 0.05
    )
    immediate_policy_gap = (
        fresh_failure["mean"] >= 0.20
        or fresh_recovery["mean"] < 0.80
        or assignment["mean"] < 0.75
    )
    if execution_staleness and immediate_policy_gap:
        label = "STALE_TAIL_AND_IMMEDIATE_POLICY_GAP"
    elif execution_staleness:
        label = "STALE_TAIL_CONFIRMED_IMMEDIATE_REPLAN_CAPABLE"
    elif immediate_policy_gap:
        label = "IMMEDIATE_POLICY_SUPPORT_GAP"
    else:
        label = "NO_STRONG_IMMEDIATE_GAP"
    return {
        "label": label,
        "execution_staleness_confirmed": bool(execution_staleness),
        "feedback_specificity_supported": bool(feedback_specificity),
        "immediate_policy_gap": bool(immediate_policy_gap),
        "scope": (
            "This diagnoses only the first post-feedback revision. Existing multi-step handoff "
            "results remain necessary to assess downstream recovery support."
        ),
    }


def summarize(
    payloads: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    git_shas = {payload.get("git_sha") for payload in payloads}
    checkpoints = {payload["policy_identity"]["checkpoint_realpath"] for payload in payloads}
    if len(git_shas) != 1 or None in git_shas:
        raise ValueError("diagnostics must share one nonempty Git SHA")
    if len(checkpoints) != len(payloads):
        raise ValueError("each policy seed must use a distinct checkpoint")
    rows = average_seed_rows(payloads)
    summaries = {}
    for stale_age in sorted({int(row["stale_age"]) for row in rows}):
        for horizon in sorted({int(row["horizon"]) for row in rows}):
            key = f"age{stale_age}_h{horizon}"
            selected = [
                row
                for row in rows
                if int(row["stale_age"]) == stale_age and int(row["horizon"]) == horizon
            ]
            metrics = {
                metric: bootstrap_interval(
                    [float(row[metric]) for row in selected],
                    samples=bootstrap_samples,
                    seed=81_000 + stale_age * 1_000 + horizon * 100 + metric_index,
                )
                for metric_index, metric in enumerate(BOOTSTRAP_METRICS)
            }
            stale_mse = metrics["slipped_stale_minus_fresh_mse"]["mean"] + float(
                np.mean([row["fresh_slipped_mse"] for row in selected])
            )
            metrics["relative_slipped_mse_reduction"] = {
                "mean": None if stale_mse <= 1e-12 else metrics["slipped_stale_minus_fresh_mse"]["mean"] / stale_mse
            }
            summaries[key] = {
                "group_count": len(selected),
                "source_initial_state_count": len(
                    {int(row["source_initial_state_index"]) for row in selected}
                ),
                "metrics": metrics,
            }
    return {
        "schema_version": 1,
        "status": "complete",
        "purpose": "mechanism_diagnostic_only",
        "split": "train",
        "test_split_opened": False,
        "policy_seeds": [int(payload["policy_seed"]) for payload in payloads],
        "git_sha": next(iter(git_shas)),
        "statistical_unit": "snapshot group after averaging policy seeds",
        "bootstrap_samples": bootstrap_samples,
        "group_count": int(payloads[0]["group_count"]),
        "per_seed_primary": [seed_primary_metrics(payload) for payload in payloads],
        "summaries": summaries,
        "mechanism_interpretation": mechanism_interpretation(summaries[PRIMARY_KEY]["metrics"]),
    }


def _percent(interval: Mapping[str, float]) -> str:
    return (
        f"{100 * interval['mean']:.1f}% "
        f"[{100 * interval['bootstrap_95_low']:.1f}, {100 * interval['bootstrap_95_high']:.1f}]"
    )


def report_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Counterfactual Feedback Revision Diagnostic",
        "",
        "Train-split mechanism diagnostic only; no validation or test result was opened.",
        "",
        "## Per-seed primary result (stale age 1, horizon 1)",
        "",
        "| Policy seed | Slip MSE gain | Relative slip MSE reduction | Stale recovery action | Fresh recovery action | Fresh twin assignment | Attached MSE change |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["per_seed_primary"]:
        lines.append(
            f"| {row['policy_seed']} | {row['slipped_stale_minus_fresh_mse']:.4f} | "
            f"{100 * row['relative_slipped_mse_reduction']:.1f}% | "
            f"{100 * row['stale_slipped_recovery_action']:.1f}% | "
            f"{100 * row['fresh_slipped_recovery_action']:.1f}% | "
            f"{100 * row['fresh_pair_assignment_correct']:.1f}% | "
            f"{row['attached_fresh_minus_stale_mse']:.4f} |"
        )
    lines.extend(
        [
        "",
        "## Snapshot-group aggregate",
        "",
        "| Stale age | Horizon | Slip MSE gain | Failure-continuation reduction | Recovery-action gain | Fresh twin assignment |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, summary in payload["summaries"].items():
        age, horizon = key.removeprefix("age").split("_h")
        metrics = summary["metrics"]
        mse = metrics["slipped_stale_minus_fresh_mse"]
        lines.append(
            f"| {age} | {horizon} | {mse['mean']:.4f} "
            f"[{mse['bootstrap_95_low']:.4f}, {mse['bootstrap_95_high']:.4f}] | "
            f"{_percent(metrics['failure_continuation_reduction'])} | "
            f"{_percent(metrics['recovery_action_gain'])} | "
            f"{_percent(metrics['fresh_pair_assignment_correct'])} |"
        )
    interpretation = payload["mechanism_interpretation"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"`{interpretation['label']}`",
            "",
            interpretation["scope"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate paired feedback-revision diagnostics")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/mechanism-diagnostics-v1"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if tuple(args.seeds) != SEEDS or args.bootstrap_samples < 1:
        raise ValueError(f"formal diagnostic requires seeds={SEEDS} and positive bootstrap samples")
    payloads = [
        load_seed(args.root / f"counterfactual_feedback_revision_train_seed{seed}.json", seed)
        for seed in args.seeds
    ]
    result = summarize(payloads, bootstrap_samples=args.bootstrap_samples)
    output = args.output or args.root / "counterfactual_feedback_revision_summary.json"
    report = args.report or args.root / "counterfactual_feedback_revision_summary.md"
    if output.exists() or report.exists():
        raise FileExistsError("refusing to overwrite feedback-revision summary")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report.write_text(report_markdown(result))
    print(json.dumps({"output": str(output), **result["mechanism_interpretation"]}, sort_keys=True))


if __name__ == "__main__":
    main()
