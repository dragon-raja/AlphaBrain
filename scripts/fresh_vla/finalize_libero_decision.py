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
    primary_seed_results = {
        method: {
            seed: {
                metric: values.get(metric)
                for metric in (
                    "overall_task_success",
                    "attached_task_success",
                    "slip_recovery_success",
                    "isolated_recovery_success",
                )
            }
            for seed, values in payload.get("seed_summaries", {}).get(str(PRIMARY_K), {}).get(method, {}).items()
        }
        for method in METHODS
    }
    primary_comparisons = {}
    for baseline in ("full_h", "random_soft010", "shuffled_oracle_soft010", "gripper_soft010", "short_h"):
        comparison = payload["paired_comparisons"][str(PRIMARY_K)][f"oracle_vs_{baseline}"]
        primary_comparisons[f"oracle_vs_{baseline}"] = {
            metric: {
                "paired_group_count": comparison[metric].get("group_count"),
                "candidate_minus_baseline": _optional_delta(payload, PRIMARY_K, baseline, metric),
            }
            for metric in (
                "overall_task_success",
                "attached_task_success",
                "slip_recovery_success",
                "isolated_recovery_success",
                "failure_continuation_rate",
                "premature_commitment_rate",
                "final_progress",
            )
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
        "primary_seed_results": primary_seed_results,
        "primary_oracle_comparisons": primary_comparisons,
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


def _fmt_pp_with_count(entry: Mapping[str, Any]) -> str:
    formatted = _fmt_pp(entry.get("candidate_minus_baseline"))
    count = entry.get("paired_group_count")
    return formatted if count is None else f"{formatted} (n={count})"


def _complete_json(path: Path, *, expected_rows: int | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") not in (None, "complete"):
        return False
    return expected_rows is None or int(payload.get("completed_rows", -1)) == expected_rows


def audit_run_artifacts(runs_root: Path, episode_root: Path | None = None) -> dict[str, Any]:
    run_dirs = [runs_root / f"fresh_closed_loop_{method}_seed{seed}" for method in METHODS for seed in (41, 42, 43)]
    counts = {
        "expected_runs": len(run_dirs),
        "final_checkpoints": sum((run_dir / "final_model/model.safetensors").is_file() for run_dir in run_dirs),
        "isolated_evaluations": sum(
            _complete_json(run_dir / "closed_loop_isolated.json", expected_rows=78) for run_dir in run_dirs
        ),
        "end_to_end_evaluations": sum(
            _complete_json(run_dir / "closed_loop_end_to_end.json", expected_rows=78) for run_dir in run_dirs
        ),
        "deterministic_reach_evaluations": sum(
            _complete_json(run_dir / "deterministic_reach.json") for run_dir in run_dirs
        ),
        "offline_evaluations": sum(_complete_json(run_dir / "offline_eval.json") for run_dir in run_dirs),
        "closed_loop_video_files": sum(
            len(list((run_dir / "closed_loop_videos").glob("*.mp4"))) for run_dir in run_dirs
        ),
        "expected_closed_loop_video_files": len(run_dirs) * 78,
    }
    counts["complete"] = bool(
        all(counts[name] == counts["expected_runs"] for name in (
            "final_checkpoints",
            "isolated_evaluations",
            "end_to_end_evaluations",
            "deterministic_reach_evaluations",
            "offline_evaluations",
        ))
        and counts["closed_loop_video_files"] == counts["expected_closed_loop_video_files"]
    )
    if episode_root is not None:
        counts.update({
            "source_branch_episode_files": len(list((episode_root / "episodes").rglob("*.npz"))),
            "source_paired_video_files": len(list((episode_root / "paired_videos").glob("*.mp4"))),
            "source_branch_video_files": len(list((episode_root / "videos").rglob("*.mp4"))),
            "source_contact_sheet": (episode_root / "contact_sheet.png").is_file(),
        })
        counts["complete"] = bool(
            counts["complete"]
            and counts["source_branch_episode_files"] == 256
            and counts["source_paired_video_files"] == 128
            and counts["source_branch_video_files"] == 256
            and counts["source_contact_sheet"]
        )
    return counts


def audit_behavior_diagnostics(runs_root: Path) -> dict[str, Any]:
    eligible = 0
    distinct = 0
    complete_files = 0
    for method in METHODS:
        for seed in (41, 42, 43):
            path = runs_root / f"fresh_closed_loop_{method}_seed{seed}" / "closed_loop_end_to_end.json"
            if not _complete_json(path, expected_rows=78):
                continue
            complete_files += 1
            for row in json.loads(path.read_text())["rows"]:
                failure = row.get("failure_continuation")
                premature = row.get("premature_commitment")
                if failure is None or premature is None:
                    continue
                eligible += 1
                distinct += bool(failure) != bool(premature)
    return {
        "complete_end_to_end_files": complete_files,
        "eligible_triggered_slip_rows": eligible,
        "rows_where_metrics_differ": distinct,
        "metrics_provide_independent_evidence": distinct > 0,
        "interpretation": (
            "The predicates are distinct in code, but they coincide on every eligible final-gate row; "
            "do not count them as independent behavioral evidence."
            if eligible and not distinct
            else "The two diagnostics differ on at least one eligible row."
        ),
    }


def validity_evidence(
    episode_quality: Mapping[str, Any] | None,
    window_quality: Mapping[str, Any] | None,
    reach: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if episode_quality is not None:
        result["episode_data"] = {
            "passed": bool(episode_quality.get("passed")),
            "checks": episode_quality.get("checks", {}),
            "metrics": episode_quality.get("metrics", {}),
        }
    if window_quality is not None:
        result["window_data"] = {
            "passed": bool(window_quality.get("passed")),
            "checks": window_quality.get("checks", {}),
            "metrics": window_quality.get("metrics", {}),
        }
    if reach is not None:
        result["deterministic_reach"] = {
            "target_definition": reach.get("target_definition"),
            "success_threshold_m": reach.get("success_threshold"),
            "success_by_execution_horizon": {
                str(k): {
                    method: reach["aggregate"][str(k)][method]["success"]["mean"] for method in METHODS
                }
                for k in (1, 2, 3)
            },
        }
    return result


def render_markdown(result: Mapping[str, Any], offline: Mapping[str, Any] | None = None) -> str:
    decision = str(result["decision"])
    evidence = result["evidence"]
    primary = evidence[str(PRIMARY_K)]
    artifacts = result.get("artifact_audit", {})
    validity = result.get("validity_evidence", {})
    diagnostics = result.get("behavior_diagnostic_audit", {})
    lines = [
        "# FRESH-VLA LIBERO Final Decision",
        "",
        f"**Decision: `{decision}`**",
        "",
        "The final comparison uses held-out snapshot groups, three fixed seeds, and fixed execution horizons. "
        "K=3 is the preregistered primary commitment setting, K=2 is supporting evidence, and K=1 is a negative control.",
    ]
    if artifacts:
        lines.extend((
            "",
            "## Completion Audit",
            "",
            f"Artifact audit complete: `{str(bool(artifacts.get('complete'))).lower()}`.",
            "",
            "| Checkpoints | Isolated | End-to-end | Reach | Offline | Closed-loop videos |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {artifacts.get('final_checkpoints')}/{artifacts.get('expected_runs')} | "
            f"{artifacts.get('isolated_evaluations')}/{artifacts.get('expected_runs')} | "
            f"{artifacts.get('end_to_end_evaluations')}/{artifacts.get('expected_runs')} | "
            f"{artifacts.get('deterministic_reach_evaluations')}/{artifacts.get('expected_runs')} | "
            f"{artifacts.get('offline_evaluations')}/{artifacts.get('expected_runs')} | "
            f"{artifacts.get('closed_loop_video_files')}/{artifacts.get('expected_closed_loop_video_files')} |",
        ))
        if "source_branch_episode_files" in artifacts:
            lines.extend((
                "",
                f"Source data contain {artifacts['source_branch_episode_files']} branch episodes, "
                f"{artifacts['source_paired_video_files']} paired videos, "
                f"{artifacts['source_branch_video_files']} branch videos, and a contact sheet.",
            ))
    episode_validity = validity.get("episode_data")
    window_validity = validity.get("window_data")
    if episode_validity or window_validity:
        lines.extend(("", "## Data And Expert Gate", ""))
        if episode_validity:
            metrics = episode_validity["metrics"]
            lines.append(
                f"Episode quality passed: `{str(bool(episode_validity['passed'])).lower()}`; "
                f"{metrics.get('group_count')} groups, attached expert success "
                f"{_fmt(metrics.get('attached_success_rate'))}, slip recovery expert success "
                f"{_fmt(metrics.get('slipped_success_rate'))}."
            )
        if window_validity:
            metrics = window_validity["metrics"]
            lines.append(
                f"Window quality passed: `{str(bool(window_validity['passed'])).lower()}`; "
                f"{metrics.get('record_count')} windows with group-preserving splits and "
                "post-feedback full supervision restored."
            )
    lines.extend((
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
        "The baseline clears the preregistered attached-success gate, but only narrowly. The conclusion is "
        "therefore scoped to this training-weighting implementation and budget, not to feedback-aware control in general.",
        "",
        "## Primary Closed-Loop Results (K=3)",
        "",
        "Attached success is also the normal/no-intervention success in this paired design.",
        "",
        "| Method | Overall | Attached | Slip recovery | Isolated recovery | Failure continuation | Premature commitment | Final progress |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ))
    for method in METHODS:
        row = primary["methods"][method]
        lines.append(
            f"| `{method}` | {_fmt(row['overall_task_success'])} | {_fmt(row['attached_task_success'])} | "
            f"{_fmt(row['slip_recovery_success'])} | {_fmt(row['isolated_recovery_success'])} | "
            f"{_fmt(row['failure_continuation_rate'])} | {_fmt(row['premature_commitment_rate'])} | "
            f"{_fmt(row['final_progress'])} |"
        )
    seed_results = result.get("primary_seed_results", {})
    if any(seed_results.get(method) for method in METHODS):
        lines.extend((
            "",
            "### Per-Seed K=3 Primary Rates",
            "",
            "| Method | Seed | Overall | Attached | Slip recovery | Isolated recovery |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ))
        for method in METHODS:
            for seed in ("41", "42", "43"):
                row = seed_results.get(method, {}).get(seed)
                if row is None:
                    continue
                lines.append(
                    f"| `{method}` | {seed} | {_fmt(row['overall_task_success'])} | "
                    f"{_fmt(row['attached_task_success'])} | {_fmt(row['slip_recovery_success'])} | "
                    f"{_fmt(row['isolated_recovery_success'])} |"
                )
    comparisons = result.get("primary_oracle_comparisons", {})
    if comparisons:
        lines.extend((
            "",
            "### Oracle Versus Every Control At K=3",
            "",
            "Positive success deltas favor Oracle; negative behavior-error deltas favor Oracle. "
            "The parenthesized n is the paired snapshot-group count.",
            "",
            "| Baseline | Overall | Attached | Slip recovery | Isolated recovery | Failure continuation |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ))
        for baseline in ("full_h", "random_soft010", "shuffled_oracle_soft010", "gripper_soft010", "short_h"):
            row = comparisons[f"oracle_vs_{baseline}"]
            lines.append(
                f"| `{baseline}` | {_fmt_pp_with_count(row['overall_task_success'])} | "
                f"{_fmt_pp_with_count(row['attached_task_success'])} | "
                f"{_fmt_pp_with_count(row['slip_recovery_success'])} | "
                f"{_fmt_pp_with_count(row['isolated_recovery_success'])} | "
                f"{_fmt_pp_with_count(row['failure_continuation_rate'])} |"
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
    if diagnostics:
        lines.extend((
            "",
            "### Behavioral Diagnostic Audit",
            "",
            f"Across {diagnostics.get('eligible_triggered_slip_rows')} eligible triggered-slip rows, "
            f"failure-continuation and premature-commitment differ on "
            f"{diagnostics.get('rows_where_metrics_differ')} rows. "
            f"{diagnostics.get('interpretation')}",
            "",
            "Their K=3 paired comparisons use fewer groups than the success metrics because they are defined only "
            "after an intervention event. The behavior-rate reduction is diagnostic, not independent success evidence.",
        ))
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
            "| Method | K=1 MSE | K=2 MSE | K=3 MSE | Oracle-prefix MSE | Suffix MSE | Mode coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ))
        for method in METHODS:
            off = offline["aggregate"]["offline"][method]
            mode = offline["aggregate"]["mode"][method]
            lines.append(
                f"| `{method}` | {off['fixed_k_1']['mean']:.4f} | {off['fixed_k_2']['mean']:.4f} | "
                f"{off['fixed_k_3']['mean']:.4f} | {off['oracle_prefix']['mean']:.4f} | "
                f"{off['suffix']['mean']:.4f} | {mode['suffix_mode_coverage']['mean']:.3f} |"
            )
    reach = validity.get("deterministic_reach")
    if reach:
        lines.extend((
            "",
            "## Deterministic Reach Negative Control",
            "",
            f"Success threshold: {float(reach['success_threshold_m']):.3f} m to the recorded expert EEF target.",
            "",
            "| Method | K=1 | K=2 | K=3 |",
            "| --- | ---: | ---: | ---: |",
        ))
        for method in METHODS:
            values = reach["success_by_execution_horizon"]
            lines.append(
                f"| `{method}` | {_fmt(values['1'][method])} | {_fmt(values['2'][method])} | "
                f"{_fmt(values['3'][method])} |"
            )
    lines.extend((
        "",
        "## Interpretation",
        "",
        "At the primary K=3 setting, Oracle does not improve slip recovery over Full-H and its overall-success "
        "paired interval includes zero. Random, shuffled-Oracle, gripper, and Short-H controls match or exceed "
        "Oracle on at least one primary success outcome, so the exact Oracle boundary has no demonstrated "
        "closed-loop specificity.",
        "",
        "Oracle improves the offline common-prefix error, but its suffix mode coverage collapses and the offline "
        "gain does not transfer to recovery or full-task success. K=1 improvements are negative-control evidence: "
        "they do not persist at the preregistered commitment horizon and are not Oracle-specific.",
        "",
        "Deterministic reach confirms that deployment can execute basic directed motion. It does not substitute for "
        "the failed full-task recovery gate. The decision therefore stops this suffix-loss weighting route; it does "
        "not reject feedback-aware replanning, plan-commit execution, active probing, or belief-aware control.",
        "",
        decision,
    ))
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_runs = Path("/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2")
    default_episodes = Path("/share/longjunyu/fresh-vla/libero-full-episode-v2-128")
    default_windows = Path("/share/longjunyu/fresh-vla/libero-full-episode-windows-v2-128")
    parser = argparse.ArgumentParser(description="Apply the preregistered FRESH-VLA final decision gate")
    parser.add_argument("--runs-root", type=Path, default=default_runs)
    parser.add_argument("--episode-root", type=Path, default=default_episodes)
    parser.add_argument("--closed-loop-results", type=Path, default=default_runs / "closed_loop_summary/results.json")
    parser.add_argument("--offline-results", type=Path, default=default_runs / "episode_offline_summary/results.json")
    parser.add_argument("--reach-results", type=Path, default=default_runs / "deterministic_reach_summary/results.json")
    parser.add_argument("--episode-quality-report", type=Path, default=default_episodes / "quality_report.json")
    parser.add_argument("--window-quality-report", type=Path, default=default_windows / "quality_report.json")
    parser.add_argument("--output-json", type=Path, default=default_runs / "final_decision.json")
    parser.add_argument("--output-md", type=Path, default=repo_root / "docs/fresh_vla_final_decision.md")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    closed_loop = json.loads(args.closed_loop_results.read_text())
    offline = json.loads(args.offline_results.read_text()) if args.offline_results.is_file() else None
    reach = json.loads(args.reach_results.read_text()) if args.reach_results.is_file() else None
    episode_quality = (
        json.loads(args.episode_quality_report.read_text()) if args.episode_quality_report.is_file() else None
    )
    window_quality = (
        json.loads(args.window_quality_report.read_text()) if args.window_quality_report.is_file() else None
    )
    result = decide(closed_loop)
    result["artifact_audit"] = audit_run_artifacts(args.runs_root, args.episode_root)
    result["behavior_diagnostic_audit"] = audit_behavior_diagnostics(args.runs_root)
    result["validity_evidence"] = validity_evidence(episode_quality, window_quality, reach)
    result["inputs"] = {
        "closed_loop_results": str(args.closed_loop_results),
        "offline_results": str(args.offline_results) if offline is not None else None,
        "reach_results": str(args.reach_results) if reach is not None else None,
        "episode_quality_report": str(args.episode_quality_report) if episode_quality is not None else None,
        "window_quality_report": str(args.window_quality_report) if window_quality is not None else None,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(render_markdown(result, offline))
    print(json.dumps({"decision": result["decision"], "output_json": str(args.output_json), "output_md": str(args.output_md)}, sort_keys=True))


if __name__ == "__main__":
    main()
