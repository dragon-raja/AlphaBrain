from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DECISIONS = (
    "CONTINUE_FRESH",
    "PIVOT_TO_PREDICTABILITY_WEIGHTING",
    "STOP_TRAINING_WEIGHTING_ROUTE",
    "BASELINE_INVALID_OR_DATA_INSUFFICIENT",
)
METHODS = (
    "full_h",
    "random_soft010",
    "shuffled_oracle_soft010",
    "gripper_soft010",
    "oracle_soft010",
    "short_h",
)
PRIMARY_K = 3
SUPPORTING_K = 2
NEGATIVE_CONTROL_K = 1
PRIMARY_METRICS = ("slip_recovery_success", "overall_task_success")


def _optional_mean(payload: Mapping[str, Any], k: int, method: str, metric: str) -> float | None:
    value = payload["aggregate"][str(k)][method][metric]["mean"]
    return None if value is None else float(value)


def _mean(payload: Mapping[str, Any], k: int, method: str, metric: str) -> float:
    value = _optional_mean(payload, k, method, metric)
    if value is None:
        raise ValueError(f"missing aggregate value for K={k} {method} {metric}")
    return value


def _optional_delta(
    payload: Mapping[str, Any], k: int, baseline: str, metric: str
) -> dict[str, float] | None:
    summary = payload["paired_comparisons"][str(k)][f"oracle_vs_{baseline}"][metric][
        "candidate_minus_baseline"
    ]
    required = ("mean", "bootstrap_95_low", "bootstrap_95_high")
    if any(summary.get(key) is None for key in required):
        return None
    return {key: float(summary[key]) for key in required}


def _delta(payload: Mapping[str, Any], k: int, baseline: str, metric: str) -> dict[str, float]:
    summary = _optional_delta(payload, k, baseline, metric)
    if summary is None:
        raise ValueError(f"missing paired comparison for K={k} oracle_vs_{baseline} {metric}")
    return summary


def _effect_passes(summary: Mapping[str, float], threshold: float) -> bool:
    return bool(summary["mean"] >= threshold or summary["bootstrap_95_low"] > 0.0)


def _error_reduction_passes(summary: Mapping[str, float]) -> bool:
    return bool(summary["mean"] <= -0.05 or summary["bootstrap_95_high"] < 0.0)


def _select_primary_metric(payload: Mapping[str, Any]) -> str | None:
    for metric in PRIMARY_METRICS:
        if _effect_passes(_delta(payload, PRIMARY_K, "full_h", metric), 0.10):
            return metric
    return None


def _baseline_is_valid(payload: Mapping[str, Any]) -> tuple[bool, dict[str, float]]:
    full_attached = _mean(payload, PRIMARY_K, "full_h", "attached_task_success")
    full_overall = _mean(payload, PRIMARY_K, "full_h", "overall_task_success")
    best_overall = max(_mean(payload, PRIMARY_K, method, "overall_task_success") for method in METHODS)
    full_event = _mean(payload, PRIMARY_K, "full_h", "event_trigger_rate")
    valid = not (full_attached < 0.20 and full_overall < 0.20 and best_overall < 0.20)
    return valid, {
        "full_attached_success": full_attached,
        "full_overall_success": full_overall,
        "best_method_overall_success": best_overall,
        "full_event_trigger_rate": full_event,
    }


def _continue_checks(payload: Mapping[str, Any], metric: str | None) -> dict[str, bool]:
    if metric is None:
        return {"oracle_vs_full_primary": False}
    full = _delta(payload, PRIMARY_K, "full_h", metric)
    random = _delta(payload, PRIMARY_K, "random_soft010", metric)
    shuffled = _delta(payload, PRIMARY_K, "shuffled_oracle_soft010", metric)
    gripper = _delta(payload, PRIMARY_K, "gripper_soft010", metric)
    short = _delta(payload, PRIMARY_K, "short_h", metric)
    attached = _delta(payload, PRIMARY_K, "full_h", "attached_task_success")
    failure = _optional_delta(payload, PRIMARY_K, "full_h", "failure_continuation_rate")
    premature = _optional_delta(payload, PRIMARY_K, "full_h", "premature_commitment_rate")
    supporting = _delta(payload, SUPPORTING_K, "full_h", metric)
    return {
        "oracle_vs_full_primary": _effect_passes(full, 0.10),
        "oracle_vs_random": _effect_passes(random, 0.05),
        "oracle_vs_shuffled": _effect_passes(shuffled, 0.05),
        "oracle_vs_gripper": gripper["mean"] > 0.0,
        "oracle_not_matched_by_short": _effect_passes(short, 0.05),
        "attached_degradation_within_5pp": attached["mean"] >= -0.05,
        "failure_continuation_reduced": failure is not None and _error_reduction_passes(failure),
        "premature_commitment_reduced": premature is not None and _error_reduction_passes(premature),
        "positive_supporting_k2_effect": supporting["mean"] > 0.0,
        "oracle_event_trigger_rate_at_least_50pct": (
            _mean(payload, PRIMARY_K, "oracle_soft010", "event_trigger_rate") >= 0.50
        ),
    }


def _pivot_checks(payload: Mapping[str, Any]) -> dict[str, bool]:
    candidate_metric = None
    for metric in PRIMARY_METRICS:
        full = _mean(payload, PRIMARY_K, "full_h", metric)
        if all(
            _mean(payload, PRIMARY_K, method, metric) - full >= 0.05
            for method in ("random_soft010", "shuffled_oracle_soft010", "oracle_soft010")
        ):
            candidate_metric = metric
            break
    if candidate_metric is None:
        return {"all_soft_methods_improve_over_full": False}
    random = _delta(payload, PRIMARY_K, "random_soft010", candidate_metric)
    shuffled = _delta(payload, PRIMARY_K, "shuffled_oracle_soft010", candidate_metric)
    return {
        "all_soft_methods_improve_over_full": True,
        "oracle_not_distinct_from_random": (
            abs(random["mean"]) < 0.05
            and random["bootstrap_95_low"] <= 0.0 <= random["bootstrap_95_high"]
        ),
        "oracle_not_distinct_from_shuffled": (
            abs(shuffled["mean"]) < 0.05
            and shuffled["bootstrap_95_low"] <= 0.0 <= shuffled["bootstrap_95_high"]
        ),
    }


def decide(payload: Mapping[str, Any]) -> dict[str, Any]:
    if tuple(payload.get("execution_horizons", ())) != (1, 2, 3):
        raise ValueError("final decision requires fixed execution horizons K=1,2,3")
    if tuple(payload.get("methods", ())) != METHODS:
        raise ValueError("final decision requires exactly the six preregistered methods")
    if tuple(payload.get("seeds", ())) != (41, 42, 43):
        raise ValueError("final decision requires seeds 41,42,43")

    baseline_valid, baseline_evidence = _baseline_is_valid(payload)
    primary_metric = _select_primary_metric(payload)
    continue_checks = _continue_checks(payload, primary_metric)
    pivot_checks = _pivot_checks(payload)
    if not baseline_valid:
        decision = "BASELINE_INVALID_OR_DATA_INSUFFICIENT"
    elif continue_checks and all(continue_checks.values()):
        decision = "CONTINUE_FRESH"
    elif pivot_checks and all(pivot_checks.values()):
        decision = "PIVOT_TO_PREDICTABILITY_WEIGHTING"
    else:
        decision = "STOP_TRAINING_WEIGHTING_ROUTE"

    evidence = {}
    for k in (NEGATIVE_CONTROL_K, SUPPORTING_K, PRIMARY_K):
        evidence[str(k)] = {
            "methods": {
                method: {
                    metric: _optional_mean(payload, k, method, metric)
                    for metric in (
                        "overall_task_success",
                        "attached_task_success",
                        "slip_recovery_success",
                        "isolated_recovery_success",
                        "failure_continuation_rate",
                        "premature_commitment_rate",
                        "event_trigger_rate",
                        "final_progress",
                    )
                }
                for method in METHODS
            },
            "oracle_vs_full": {
                metric: _optional_delta(payload, k, "full_h", metric)
                for metric in (
                    "overall_task_success",
                    "attached_task_success",
                    "slip_recovery_success",
                    "isolated_recovery_success",
                    "failure_continuation_rate",
                    "premature_commitment_rate",
                )
            },
        }
    return {
        "schema_version": 1,
        "decision": decision,
        "allowed_decisions": list(DECISIONS),
        "preregistered_analysis": {
            "primary_execution_horizon": PRIMARY_K,
            "supporting_execution_horizon": SUPPORTING_K,
            "negative_control_execution_horizon": NEGATIVE_CONTROL_K,
            "primary_metric_selected_by_fixed_rule": primary_metric,
            "statistical_unit": payload.get("statistical_unit"),
        },
        "baseline_valid": baseline_valid,
        "baseline_evidence": baseline_evidence,
        "continue_checks": continue_checks,
        "pivot_checks": pivot_checks,
        "evidence": evidence,
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_pp(summary: Mapping[str, float] | None) -> str:
    if summary is None:
        return "n/a"
    return (
        f"{100.0 * summary['mean']:+.1f} pp "
        f"[{100.0 * summary['bootstrap_95_low']:+.1f}, "
        f"{100.0 * summary['bootstrap_95_high']:+.1f}]"
    )


def render_markdown(result: Mapping[str, Any], offline: Mapping[str, Any] | None = None) -> str:
    decision = str(result["decision"])
    evidence = result["evidence"]
    primary = evidence[str(PRIMARY_K)]
    lines = [
        "# FRESH-VLA LIBERO Final Decision",
        "",
        f"**Decision: `{decision}`**",
        "",
        "The final comparison uses held-out snapshot groups, three fixed seeds, and fixed execution horizons. "
        "K=3 is the preregistered primary commitment setting, K=2 is supporting evidence, and K=1 is a negative control.",
        "",
        "## Baseline Gate",
        "",
        f"Baseline valid: `{str(result['baseline_valid']).lower()}`.",
        "",
        "| Full-H attached | Full-H overall | Best overall | Full-H event trigger |",
        "| ---: | ---: | ---: | ---: |",
        "| "
        + " | ".join(_fmt(float(result["baseline_evidence"][key])) for key in (
            "full_attached_success",
            "full_overall_success",
            "best_method_overall_success",
            "full_event_trigger_rate",
        ))
        + " |",
        "",
        "## Primary Closed-Loop Results (K=3)",
        "",
        "| Method | Overall | Attached | Slip recovery | Isolated recovery | Failure continuation | Premature commitment | Final progress |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = primary["methods"][method]
        lines.append(
            f"| `{method}` | {_fmt(row['overall_task_success'])} | {_fmt(row['attached_task_success'])} | "
            f"{_fmt(row['slip_recovery_success'])} | {_fmt(row['isolated_recovery_success'])} | "
            f"{_fmt(row['failure_continuation_rate'])} | {_fmt(row['premature_commitment_rate'])} | "
            f"{_fmt(row['final_progress'])} |"
        )
    lines.extend((
        "",
        "## Oracle vs Full-H Paired Deltas",
        "",
        "Success deltas above zero favor Oracle; behavior-error deltas below zero favor Oracle.",
        "",
        "| K | Overall | Slip recovery | Isolated recovery | Failure continuation | Premature commitment |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ))
    for k in (1, 2, 3):
        row = evidence[str(k)]["oracle_vs_full"]
        lines.append(
            f"| {k} | {_fmt_pp(row['overall_task_success'])} | {_fmt_pp(row['slip_recovery_success'])} | "
            f"{_fmt_pp(row['isolated_recovery_success'])} | {_fmt_pp(row['failure_continuation_rate'])} | "
            f"{_fmt_pp(row['premature_commitment_rate'])} |"
        )
    lines.extend(("", "## Rule Checks", "", "### Continue FRESH", ""))
    for name, passed in result["continue_checks"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(("", "### Predictability-Weighting Pivot", ""))
    for name, passed in result["pivot_checks"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    if offline is not None:
        lines.extend((
            "",
            "## Auxiliary Offline Check",
            "",
            "Offline MSE and suffix mode coverage are mechanism diagnostics only and did not determine the decision.",
            "",
            "| Method | K=1 MSE | K=2 MSE | K=3 MSE | Suffix mode coverage |",
            "| --- | ---: | ---: | ---: | ---: |",
        ))
        for method in METHODS:
            off = offline["aggregate"]["offline"][method]
            mode = offline["aggregate"]["mode"][method]
            lines.append(
                f"| `{method}` | {off['fixed_k_1']['mean']:.4f} | {off['fixed_k_2']['mean']:.4f} | "
                f"{off['fixed_k_3']['mean']:.4f} | {mode['suffix_mode_coverage']['mean']:.3f} |"
            )
    lines.extend((
        "",
        "## Interpretation",
        "",
        "The decision follows the preregistered behavioral gate. Offline prediction errors, deterministic reach, "
        "and individual frames were not used as substitutes for full-task and recovery behavior.",
        "",
        decision,
    ))
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_runs = Path("/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2")
    parser = argparse.ArgumentParser(description="Apply the preregistered FRESH-VLA final decision gate")
    parser.add_argument("--closed-loop-results", type=Path, default=default_runs / "closed_loop_summary/results.json")
    parser.add_argument("--offline-results", type=Path, default=default_runs / "episode_offline_summary/results.json")
    parser.add_argument("--output-json", type=Path, default=default_runs / "final_decision.json")
    parser.add_argument("--output-md", type=Path, default=repo_root / "docs/fresh_vla_final_decision.md")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    closed_loop = json.loads(args.closed_loop_results.read_text())
    offline = json.loads(args.offline_results.read_text()) if args.offline_results.is_file() else None
    result = decide(closed_loop)
    result["inputs"] = {
        "closed_loop_results": str(args.closed_loop_results),
        "offline_results": str(args.offline_results) if offline is not None else None,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(render_markdown(result, offline))
    print(json.dumps({"decision": result["decision"], "output_json": str(args.output_json), "output_md": str(args.output_md)}, sort_keys=True))


if __name__ == "__main__":
    main()
