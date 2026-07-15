from __future__ import annotations

# ruff: noqa: RUF001
import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scripts.fresh_vla.paired_evaluation import bootstrap_summary
except ModuleNotFoundError:
    from paired_evaluation import bootstrap_summary


METHODS = (
    "fixed_k1",
    "fixed_k2",
    "fixed_k3",
    "oracle_branch_safe_commit",
    "oracle_feedback_reveal_commit",
    "gripper_commit",
    "random_matched_commit",
    "self_consistency_commit",
)
NEW_METHODS = METHODS[3:]
SEEDS = (41, 42, 43)
METRICS = (
    "overall_task_success",
    "attached_task_success",
    "slip_recovery_success",
    "isolated_recovery_success",
    "slip_regrasp_success",
    "isolated_regrasp_success",
    "failure_continuation_rate",
    "premature_commitment_rate",
    "recovery_switch_latency",
    "drop_rate",
    "attached_drop_rate",
    "slip_drop_rate",
    "final_progress",
    "attached_final_progress",
    "slip_final_progress",
    "progress_auc",
    "grasp_subgoal_rate",
    "lift_subgoal_rate",
    "transport_subgoal_rate",
    "place_subgoal_rate",
    "event_trigger_rate",
    "completion_steps",
    "normal_no_intervention_success",
    "deterministic_reach_success",
    "policy_invocation_count",
    "policy_forward_calls",
    "inference_wall_seconds",
    "isolated_policy_invocation_count",
    "isolated_policy_forward_calls",
    "isolated_inference_wall_seconds",
    "mean_commit_length",
    "event_boundary_alignment_error",
)
CORE_METRICS = (
    "overall_task_success",
    "slip_recovery_success",
    "isolated_recovery_success",
    "slip_regrasp_success",
)
CHECKPOINT_SHA256 = {
    41: "144a3b3d3dcc8421418564a62059a1038c9a7ef3196ac157f5f9ea1997a31f30",
    42: "98dc52d2ed1983776d218fee7666f3131053d1a55296e93e9f521b1c088ce875",
    43: "5db16350d9835c1f28d01b660dd6e9234bcab3da79abbce1f092e92b08ac9149",
}


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _rate(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    return None if value is None else float(value)


def _paired_branches(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        result[str(row["pair_id"])][str(row["branch_outcome"])] = row
    return result


def _fixed_rows(rows: Sequence[Mapping[str, Any]], method: str, latency: float) -> list[dict[str, Any]]:
    horizon = int(method[-1])
    selected = []
    for row in rows:
        if int(row["execution_horizon"]) != horizon:
            continue
        calls = int(row["replan_count"])
        selected.append(
            {
                **row,
                "commit_method": method,
                "policy_invocation_count": calls,
                "policy_forward_calls": calls,
                "inference_wall_seconds": calls * latency,
                "inference_wall_time_source": "estimated_from_same_seed_single-sample_commit_calls",
                "mean_commit_length": (float(row["completion_steps"]) / calls if calls else None),
            }
        )
    return selected


def _load_json(path: Path, *, expected_rows: int) -> Mapping[str, Any]:
    payload = json.loads(path.read_text())
    if len(payload.get("rows", ())) != expected_rows:
        raise ValueError(f"unexpected row count in {path}: {len(payload.get('rows', ()))}/{expected_rows}")
    if "status" in payload and payload["status"] != "complete":
        raise ValueError(f"incomplete evaluation: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_fixed_payload(path: Path, *, expected_rows: int, evaluation: str, seed: int) -> Mapping[str, Any]:
    payload = _load_json(path, expected_rows=expected_rows)
    expected_seed = (271828 if evaluation == "deterministic_reach" else 314159) + seed
    if payload.get("seed") != expected_seed:
        raise ValueError(f"fixed-K seed mismatch in {path}: {payload.get('seed')} != {expected_seed}")
    if payload.get("split") != "test" or payload.get("evaluation") != evaluation:
        raise ValueError(f"fixed-K protocol mismatch in {path}")
    horizons = {int(row["execution_horizon"]) for row in payload["rows"]}
    if horizons != {1, 2, 3}:
        raise ValueError(f"fixed-K horizon set changed in {path}: {horizons}")
    return payload


def _assert_frozen_checkpoint(payload: Mapping[str, Any], expected: Path, seed: int) -> None:
    actual = str(payload.get("policy_checkpoint_realpath", ""))
    if actual != str(expected.resolve()):
        raise ValueError(f"evaluation used unexpected checkpoint: {actual!r} != {str(expected.resolve())!r}")
    if payload.get("policy_checkpoint_sha256") != CHECKPOINT_SHA256[seed]:
        raise ValueError(f"evaluation checkpoint SHA256 mismatch for seed {seed}")
    if not payload.get("git_sha"):
        raise ValueError("evaluation is missing the frozen Git SHA")


def measured_latency_by_seed(
    new_root: Path,
    baseline_root: Path,
    seeds: Sequence[int],
) -> dict[int, float]:
    result = {}
    for seed in seeds:
        values = []
        expected_checkpoint = baseline_root / f"fresh_closed_loop_full_h_seed{seed}" / "final_model"
        for method in (
            "oracle_branch_safe_commit",
            "oracle_feedback_reveal_commit",
            "gripper_commit",
            "random_matched_commit",
        ):
            run = new_root / f"oracle_commit_{method}_seed{seed}"
            for name, count in (
                ("closed_loop_isolated.json", 26),
                ("closed_loop_end_to_end.json", 26),
                ("deterministic_reach.json", 13),
            ):
                payload = _load_json(run / name, expected_rows=count)
                _assert_frozen_checkpoint(payload, expected_checkpoint, seed)
                for row in payload["rows"]:
                    calls = int(row["policy_forward_calls"])
                    if calls:
                        values.append(float(row["inference_wall_seconds"]) / calls)
        if not values:
            raise ValueError(f"no measured single-sample inference latency for seed {seed}")
        result[seed] = float(statistics.median(values))
    return result


def frozen_git_sha(new_root: Path, seeds: Sequence[int]) -> str:
    values = set()
    for method in NEW_METHODS:
        for seed in seeds:
            run = new_root / f"oracle_commit_{method}_seed{seed}"
            for name in ("closed_loop_isolated.json", "closed_loop_end_to_end.json", "deterministic_reach.json"):
                values.add(json.loads((run / name).read_text()).get("git_sha"))
    if None in values or len(values) != 1:
        raise ValueError(f"new commit evaluations do not share one frozen Git SHA: {sorted(map(str, values))}")
    return str(next(iter(values)))


def fixed_baseline_provenance(baseline_root: Path, seeds: Sequence[int]) -> dict[str, Any]:
    artifacts = {}
    for seed in seeds:
        run = baseline_root / f"fresh_closed_loop_full_h_seed{seed}"
        artifacts[str(seed)] = {
            name: _file_sha256(run / name)
            for name in ("closed_loop_isolated.json", "closed_loop_end_to_end.json", "deterministic_reach.json")
        }
    return {
        "root": str(baseline_root.resolve()),
        "result_artifact_sha256": artifacts,
        "checkpoint_sha256_confirmed_by_new_frozen_policy_runs": {str(seed): CHECKPOINT_SHA256[seed] for seed in seeds},
        "historical_binding_limit": (
            "fixed-K JSON recorded the exact evaluation seed, split, protocol, rows, and run-directory convention, "
            "but its remote policy socket payload did not embed a checkpoint hash; one-group parity and the original "
            "runner path convention provide the remaining binding evidence"
        ),
    }


def load_method_rows(
    method: str,
    seed: int,
    *,
    baseline_root: Path,
    new_root: Path,
    fixed_latency: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if method.startswith("fixed_k"):
        run = baseline_root / f"fresh_closed_loop_full_h_seed{seed}"
        isolated = _fixed_rows(
            _load_fixed_payload(
                run / "closed_loop_isolated.json",
                expected_rows=78,
                evaluation="isolated",
                seed=seed,
            )["rows"],
            method,
            fixed_latency,
        )
        end_to_end = _fixed_rows(
            _load_fixed_payload(
                run / "closed_loop_end_to_end.json",
                expected_rows=78,
                evaluation="end_to_end",
                seed=seed,
            )["rows"],
            method,
            fixed_latency,
        )
        reach = _fixed_rows(
            _load_fixed_payload(
                run / "deterministic_reach.json",
                expected_rows=39,
                evaluation="deterministic_reach",
                seed=seed,
            )["rows"],
            method,
            fixed_latency,
        )
        return isolated, end_to_end, reach

    run = new_root / f"oracle_commit_{method}_seed{seed}"
    expected_checkpoint = baseline_root / f"fresh_closed_loop_full_h_seed{seed}" / "final_model"
    isolated_payload = _load_json(run / "closed_loop_isolated.json", expected_rows=26)
    end_to_end_payload = _load_json(run / "closed_loop_end_to_end.json", expected_rows=26)
    reach_payload = _load_json(run / "deterministic_reach.json", expected_rows=13)
    for payload in (isolated_payload, end_to_end_payload, reach_payload):
        _assert_frozen_checkpoint(payload, expected_checkpoint, seed)
    isolated = [dict(row) for row in isolated_payload["rows"]]
    end_to_end = [dict(row) for row in end_to_end_payload["rows"]]
    reach = [dict(row) for row in reach_payload["rows"]]
    for row in isolated + end_to_end + reach:
        row["inference_wall_time_source"] = "measured"
    return isolated, end_to_end, reach


def group_metrics(
    isolated_rows: Sequence[Mapping[str, Any]],
    end_to_end_rows: Sequence[Mapping[str, Any]],
    reach_rows: Sequence[Mapping[str, Any]],
    eligibility: Mapping[str, bool],
) -> dict[str, dict[str, float | None]]:
    isolated = _paired_branches(isolated_rows)
    end_to_end = _paired_branches(end_to_end_rows)
    reach = {str(row["pair_id"]): row for row in reach_rows}
    if isolated.keys() != end_to_end.keys() or end_to_end.keys() != reach.keys():
        raise ValueError("isolated, end-to-end, and reach snapshot groups do not match")
    result = {}
    for pair_id in sorted(end_to_end):
        if isolated[pair_id].keys() < {"attached", "slipped"} or end_to_end[pair_id].keys() < {"attached", "slipped"}:
            raise ValueError(f"missing paired branch for {pair_id}")
        iso = isolated[pair_id]
        e2e = end_to_end[pair_id]
        attached = e2e["attached"]
        slipped = e2e["slipped"]
        invocation_count = 0.5 * (float(attached["policy_invocation_count"]) + float(slipped["policy_invocation_count"]))
        forward_calls = 0.5 * (float(attached["policy_forward_calls"]) + float(slipped["policy_forward_calls"]))
        inference_time = 0.5 * (float(attached["inference_wall_seconds"]) + float(slipped["inference_wall_seconds"]))
        result[pair_id] = {
            "oracle_primary_eligible": float(bool(eligibility[pair_id])),
            "overall_task_success": 0.5 * (float(attached["success"]) + float(slipped["success"])),
            "attached_task_success": float(attached["success"]),
            "slip_recovery_success": float(slipped["recovery_success"]),
            "isolated_recovery_success": float(iso["slipped"]["recovery_success"]),
            "slip_regrasp_success": _rate(slipped, "regrasp_success"),
            "isolated_regrasp_success": _rate(iso["slipped"], "regrasp_success"),
            "failure_continuation_rate": _rate(slipped, "failure_continuation"),
            "premature_commitment_rate": _rate(slipped, "premature_commitment"),
            "recovery_switch_latency": _rate(slipped, "recovery_switch_latency"),
            "drop_rate": 0.5 * (float(attached["drop"]) + float(slipped["drop"])),
            "attached_drop_rate": float(attached["drop"]),
            "slip_drop_rate": float(slipped["drop"]),
            "final_progress": 0.5 * (float(attached["final_progress"]) + float(slipped["final_progress"])),
            "attached_final_progress": float(attached["final_progress"]),
            "slip_final_progress": float(slipped["final_progress"]),
            "progress_auc": 0.5 * (float(attached["progress_auc"]) + float(slipped["progress_auc"])),
            "grasp_subgoal_rate": 0.5 * (float(attached["grasp_subgoal"]) + float(slipped["grasp_subgoal"])),
            "lift_subgoal_rate": 0.5 * (float(attached["lift_subgoal"]) + float(slipped["lift_subgoal"])),
            "transport_subgoal_rate": 0.5 * (float(attached["transport_subgoal"]) + float(slipped["transport_subgoal"])),
            "place_subgoal_rate": 0.5 * (float(attached["place_subgoal"]) + float(slipped["place_subgoal"])),
            "event_trigger_rate": 0.5
            * (float(attached.get("event_time") is not None) + float(slipped.get("event_time") is not None)),
            "completion_steps": 0.5 * (float(attached["completion_steps"]) + float(slipped["completion_steps"])),
            "normal_no_intervention_success": float(attached["success"]),
            "deterministic_reach_success": float(reach[pair_id]["success"]),
            "policy_invocation_count": invocation_count,
            "policy_forward_calls": forward_calls,
            "inference_wall_seconds": inference_time,
            "isolated_policy_invocation_count": 0.5
            * (float(iso["attached"]["policy_invocation_count"]) + float(iso["slipped"]["policy_invocation_count"])),
            "isolated_policy_forward_calls": 0.5
            * (float(iso["attached"]["policy_forward_calls"]) + float(iso["slipped"]["policy_forward_calls"])),
            "isolated_inference_wall_seconds": 0.5
            * (float(iso["attached"]["inference_wall_seconds"]) + float(iso["slipped"]["inference_wall_seconds"])),
            "mean_commit_length": 0.5 * (float(attached["mean_commit_length"]) + float(slipped["mean_commit_length"])),
            "event_boundary_alignment_error": _rate(slipped, "event_boundary_alignment_error"),
        }
    return result


def summarize_groups(
    groups: Mapping[str, Mapping[str, float | None]], group_ids: Sequence[str]
) -> dict[str, float | None]:
    selected = [groups[pair_id] for pair_id in group_ids]
    return {metric: _mean([float(row[metric]) for row in selected if row.get(metric) is not None]) for metric in METRICS}


def bootstrap_groups(
    groups_by_seed: Mapping[int, Mapping[str, Mapping[str, float | None]]],
    group_ids: Sequence[str],
    metric: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    values = []
    for pair_id in group_ids:
        per_seed = [
            float(groups_by_seed[run_seed][pair_id][metric])
            for run_seed in sorted(groups_by_seed)
            if groups_by_seed[run_seed][pair_id].get(metric) is not None
        ]
        if per_seed:
            values.append(float(np.mean(per_seed)))
    if not values:
        return {"count": 0, "mean": None, "bootstrap_95_low": None, "bootstrap_95_high": None}
    return bootstrap_summary(values, bootstrap_samples=bootstrap_samples, seed=seed)


def paired_delta(
    baseline: Mapping[int, Mapping[str, Mapping[str, float | None]]],
    candidate: Mapping[int, Mapping[str, Mapping[str, float | None]]],
    group_ids: Sequence[str],
    metric: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    group_deltas = []
    seed_deltas = {}
    for run_seed in sorted(baseline):
        values = []
        for pair_id in group_ids:
            base = baseline[run_seed][pair_id].get(metric)
            value = candidate[run_seed][pair_id].get(metric)
            if base is not None and value is not None:
                values.append(float(value) - float(base))
        seed_deltas[str(run_seed)] = _mean(values)
    for pair_id in group_ids:
        values = []
        for run_seed in sorted(baseline):
            base = baseline[run_seed][pair_id].get(metric)
            value = candidate[run_seed][pair_id].get(metric)
            if base is not None and value is not None:
                values.append(float(value) - float(base))
        if values:
            group_deltas.append(float(np.mean(values)))
    summary = (
        bootstrap_summary(group_deltas, bootstrap_samples=bootstrap_samples, seed=seed)
        if group_deltas
        else {"count": 0, "mean": None, "bootstrap_95_low": None, "bootstrap_95_high": None}
    )
    return {
        "unit": "snapshot_group_after_averaging_seeds",
        "candidate_minus_baseline": summary,
        "seed_deltas": seed_deltas,
        "group_deltas": group_deltas,
    }


def bootstrap_clusters(
    groups_by_seed: Mapping[int, Mapping[str, Mapping[str, float | None]]],
    clusters: Mapping[str, Sequence[str]],
    metric: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    values = []
    for pair_ids in clusters.values():
        observations = [
            float(groups_by_seed[run_seed][pair_id][metric])
            for pair_id in pair_ids
            for run_seed in sorted(groups_by_seed)
            if groups_by_seed[run_seed][pair_id].get(metric) is not None
        ]
        if observations:
            values.append(float(np.mean(observations)))
    if not values:
        return {"count": 0, "mean": None, "bootstrap_95_low": None, "bootstrap_95_high": None}
    return bootstrap_summary(values, bootstrap_samples=bootstrap_samples, seed=seed)


def paired_delta_clusters(
    baseline: Mapping[int, Mapping[str, Mapping[str, float | None]]],
    candidate: Mapping[int, Mapping[str, Mapping[str, float | None]]],
    clusters: Mapping[str, Sequence[str]],
    metric: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    cluster_deltas = []
    all_pair_ids = sorted({pair_id for pair_ids in clusters.values() for pair_id in pair_ids})
    seed_deltas = {}
    for run_seed in sorted(baseline):
        values = [
            float(candidate[run_seed][pair_id][metric]) - float(baseline[run_seed][pair_id][metric])
            for pair_id in all_pair_ids
            if baseline[run_seed][pair_id].get(metric) is not None
            and candidate[run_seed][pair_id].get(metric) is not None
        ]
        seed_deltas[str(run_seed)] = _mean(values)
    for pair_ids in clusters.values():
        values = [
            float(candidate[run_seed][pair_id][metric]) - float(baseline[run_seed][pair_id][metric])
            for pair_id in pair_ids
            for run_seed in sorted(baseline)
            if baseline[run_seed][pair_id].get(metric) is not None
            and candidate[run_seed][pair_id].get(metric) is not None
        ]
        if values:
            cluster_deltas.append(float(np.mean(values)))
    summary = (
        bootstrap_summary(cluster_deltas, bootstrap_samples=bootstrap_samples, seed=seed)
        if cluster_deltas
        else {"count": 0, "mean": None, "bootstrap_95_low": None, "bootstrap_95_high": None}
    )
    return {
        "unit": "source_initial_state_after_averaging_snapshot_groups_and_seeds",
        "candidate_minus_baseline": summary,
        "seed_deltas": seed_deltas,
        "cluster_deltas": cluster_deltas,
    }


def _positive_stable(comparison: Mapping[str, Any], *, minimum: float = 0.0) -> bool:
    summary = comparison["candidate_minus_baseline"]
    if summary["mean"] is None or float(summary["mean"]) < minimum:
        return False
    seed_values = [value for value in comparison["seed_deltas"].values() if value is not None]
    return bool(float(summary["bootstrap_95_low"]) > 0 or (seed_values and all(value > 0 for value in seed_values)))


def _negative_stable(comparison: Mapping[str, Any]) -> bool:
    summary = comparison["candidate_minus_baseline"]
    if summary["mean"] is None or float(summary["mean"]) >= 0:
        return False
    seed_values = [value for value in comparison["seed_deltas"].values() if value is not None]
    return bool(float(summary["bootstrap_95_high"]) < 0 or (seed_values and all(value < 0 for value in seed_values)))


def apply_decision_gate(
    aggregate: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    oracle_vs_fixed3 = comparisons["oracle_vs_fixed_k3"]
    oracle_vs_random = comparisons["oracle_vs_random_matched_commit"]
    primary_fixed3 = any(
        _positive_stable(oracle_vs_fixed3[metric], minimum=0.10)
        for metric in ("slip_recovery_success", "overall_task_success")
    )
    random_advantage = any(
        (
            oracle_vs_random[metric]["candidate_minus_baseline"]["mean"] is not None
            and float(oracle_vs_random[metric]["candidate_minus_baseline"]["mean"]) >= 0.05
        )
        or _positive_stable(oracle_vs_random[metric])
        for metric in CORE_METRICS
    )
    near_neighbor_advantage = {}
    for baseline in ("gripper_commit", "self_consistency_commit"):
        comparison = comparisons[f"oracle_vs_{baseline}"]
        near_neighbor_advantage[baseline] = any(_positive_stable(comparison[metric]) for metric in CORE_METRICS)
    attached_delta = float(oracle_vs_fixed3["attached_task_success"]["candidate_minus_baseline"]["mean"])
    attached_preserved = attached_delta >= -0.05

    oracle = aggregate["oracle_branch_safe_commit"]
    fixed1 = aggregate["fixed_k1"]
    fixed1_success_comparable = (
        float(oracle["overall_task_success"]["mean"]) >= float(fixed1["overall_task_success"]["mean"]) - 0.05
    )
    fixed1_calls = float(fixed1["policy_forward_calls"]["mean"])
    oracle_calls = float(oracle["policy_forward_calls"]["mean"])
    call_reduction = (fixed1_calls - oracle_calls) / fixed1_calls
    fixed1_efficiency = fixed1_success_comparable and call_reduction >= 0.20

    behavior_reduced = any(
        _negative_stable(oracle_vs_fixed3[metric])
        for metric in ("failure_continuation_rate", "premature_commitment_rate")
    )
    gate = {
        "oracle_vs_fixed_k3_primary_effect": primary_fixed3,
        "oracle_vs_random_unique_effect": random_advantage,
        "oracle_vs_gripper_stable_effect": near_neighbor_advantage["gripper_commit"],
        "oracle_vs_self_consistency_stable_effect": near_neighbor_advantage["self_consistency_commit"],
        "attached_degradation_pp": 100.0 * attached_delta,
        "attached_preserved_within_5pp": attached_preserved,
        "fixed_k1_overall_success_comparable_within_5pp": fixed1_success_comparable,
        "policy_forward_call_reduction_vs_fixed_k1": call_reduction,
        "fixed_k1_efficiency_gate": fixed1_efficiency,
        "behavior_error_reduced": behavior_reduced,
    }
    passed = (
        primary_fixed3
        and random_advantage
        and all(near_neighbor_advantage.values())
        and attached_preserved
        and fixed1_efficiency
        and behavior_reduced
    )
    return ("GO_COUNTERFACTUAL_COMMITMENT" if passed else "STOP_FRESH_FAMILY"), gate


def _fmt_rate(summary: Mapping[str, Any]) -> str:
    if summary.get("mean") is None:
        return "n/a"
    mean = 100 * float(summary["mean"])
    low = 100 * float(summary["bootstrap_95_low"])
    high = 100 * float(summary["bootstrap_95_high"])
    return f"{mean:.1f}% [{low:.1f}, {high:.1f}]"


def _fmt_pp(comparison: Mapping[str, Any]) -> str:
    summary = comparison["candidate_minus_baseline"]
    if summary.get("mean") is None:
        return "n/a"
    mean = 100 * float(summary["mean"])
    low = 100 * float(summary["bootstrap_95_low"])
    high = 100 * float(summary["bootstrap_95_high"])
    return f"{mean:+.1f} [{low:+.1f}, {high:+.1f}] pp"


def write_pareto(path: Path, aggregate: Mapping[str, Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for method in METHODS:
        calls = float(aggregate[method]["policy_forward_calls"]["mean"])
        overall = float(aggregate[method]["overall_task_success"]["mean"])
        recovery = float(aggregate[method]["slip_recovery_success"]["mean"])
        axes[0].scatter(calls, overall, s=45)
        axes[0].annotate(method, (calls, overall), fontsize=7, xytext=(3, 3), textcoords="offset points")
        axes[1].scatter(calls, recovery, s=45)
        axes[1].annotate(method, (calls, recovery), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].set(xlabel="Policy forward calls / episode", ylabel="Overall task success", title="Success-efficiency")
    axes[1].set(xlabel="Policy forward calls / episode", ylabel="Slip recovery success", title="Recovery-efficiency")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.03, 1.03)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final paired statistics and decision gate for Oracle Plan-Commit")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2"),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/libero-oracle-commit-final-v1"),
    )
    parser.add_argument(
        "--episode-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/libero-full-episode-v2-128"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/fresh_vla_oracle_commit_final_gate.md"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.runs_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    test_groups = sorted(
        (group for group in manifest["groups"] if group["split"] == "test"),
        key=lambda row: row["pair_id"],
    )
    eligibility = {
        str(group["pair_id"]): bool(group["feedback_reveal_time"] <= group["action_divergence_time"])
        for group in test_groups
    }
    subgroup_ids = {
        "all": sorted(eligibility),
        "primary_feedback_le_divergence": sorted(pair_id for pair_id, value in eligibility.items() if value),
        "reverse_feedback_gt_divergence": sorted(pair_id for pair_id, value in eligibility.items() if not value),
    }
    source_state_clusters: dict[str, list[str]] = defaultdict(list)
    for group in test_groups:
        pair_id = str(group["pair_id"])
        if eligibility[pair_id]:
            source_state_clusters[str(group["source_initial_state_index"])].append(pair_id)
    source_state_clusters = {source: sorted(pair_ids) for source, pair_ids in sorted(source_state_clusters.items())}
    evaluation_git_sha = frozen_git_sha(args.runs_root, args.seeds)
    latency = measured_latency_by_seed(args.runs_root, args.baseline_root, args.seeds)
    all_groups: dict[str, dict[int, dict[str, dict[str, float | None]]]] = {method: {} for method in METHODS}
    for method in METHODS:
        for seed in args.seeds:
            isolated, end_to_end, reach = load_method_rows(
                method,
                seed,
                baseline_root=args.baseline_root,
                new_root=args.runs_root,
                fixed_latency=latency[seed],
            )
            all_groups[method][seed] = group_metrics(isolated, end_to_end, reach, eligibility)
            if all_groups[method][seed].keys() != eligibility.keys():
                raise ValueError(f"test group mismatch for {method} seed {seed}")

    seed_summaries = {}
    aggregate = {}
    comparisons = {}
    for subgroup_index, (subgroup, group_ids) in enumerate(subgroup_ids.items()):
        seed_summaries[subgroup] = {}
        aggregate[subgroup] = {}
        for method_index, method in enumerate(METHODS):
            seed_summaries[subgroup][method] = {
                str(seed): summarize_groups(all_groups[method][seed], group_ids) for seed in args.seeds
            }
            aggregate[subgroup][method] = {
                metric: bootstrap_groups(
                    all_groups[method],
                    group_ids,
                    metric,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=7100 + subgroup_index * 1000 + method_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(METRICS)
            }
        comparisons[subgroup] = {}
        for baseline_index, baseline in enumerate(
            ("fixed_k1", "fixed_k3", "random_matched_commit", "gripper_commit", "self_consistency_commit")
        ):
            comparisons[subgroup][f"oracle_vs_{baseline}"] = {
                metric: paired_delta(
                    all_groups[baseline],
                    all_groups["oracle_branch_safe_commit"],
                    group_ids,
                    metric,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=9100 + subgroup_index * 1000 + baseline_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(METRICS)
            }

    primary = "primary_feedback_le_divergence"
    group_decision, gate = apply_decision_gate(aggregate[primary], comparisons[primary])
    cluster_aggregate = {}
    for method_index, method in enumerate(METHODS):
        cluster_aggregate[method] = {
            metric: bootstrap_clusters(
                all_groups[method],
                source_state_clusters,
                metric,
                bootstrap_samples=args.bootstrap_samples,
                seed=17100 + method_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
    cluster_comparisons = {}
    for baseline_index, baseline in enumerate(
        ("fixed_k1", "fixed_k3", "random_matched_commit", "gripper_commit", "self_consistency_commit")
    ):
        cluster_comparisons[f"oracle_vs_{baseline}"] = {
            metric: paired_delta_clusters(
                all_groups[baseline],
                all_groups["oracle_branch_safe_commit"],
                source_state_clusters,
                metric,
                bootstrap_samples=args.bootstrap_samples,
                seed=19100 + baseline_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
    cluster_decision, cluster_gate = apply_decision_gate(cluster_aggregate, cluster_comparisons)
    decision = (
        "GO_COUNTERFACTUAL_COMMITMENT"
        if group_decision == cluster_decision == "GO_COUNTERFACTUAL_COMMITMENT"
        else "STOP_FRESH_FAMILY"
    )
    payload = {
        "methods": list(METHODS),
        "seeds": list(args.seeds),
        "statistical_unit": "snapshot group; seeds averaged within group before paired bootstrap",
        "source_state_sensitivity_unit": (
            "source_initial_state_index; snapshot groups and seeds averaged within source before paired bootstrap"
        ),
        "evaluation_git_sha": evaluation_git_sha,
        "subgroup_ids": subgroup_ids,
        "source_state_clusters": source_state_clusters,
        "fixed_k_provenance": fixed_baseline_provenance(args.baseline_root, args.seeds),
        "fixed_k_wall_time": {
            "source": "same-seed median per-call latency measured in new single-sample commit runs",
            "seconds_per_call": {str(seed): value for seed, value in latency.items()},
        },
        "seed_summaries": seed_summaries,
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "decision_gate": gate,
        "source_state_sensitivity": {
            "aggregate": cluster_aggregate,
            "paired_comparisons": cluster_comparisons,
            "decision_gate": cluster_gate,
            "decision": cluster_decision,
        },
        "snapshot_group_decision": group_decision,
        "decision": decision,
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pareto_path = output_dir / "success_efficiency_pareto.png"
    write_pareto(pareto_path, aggregate[primary])

    decision_payload = {
        "decision": decision,
        "decision_gate": gate,
        "source_state_decision_gate": cluster_gate,
        "snapshot_group_decision": group_decision,
        "source_state_sensitivity_decision": cluster_decision,
        "results": str(results_path),
        "pareto_plot": str(pareto_path),
        "report": str(args.report.resolve()),
        "primary_group_count": len(subgroup_ids[primary]),
        "source_state_count": len(source_state_clusters),
        "reverse_group_count": len(subgroup_ids["reverse_feedback_gt_divergence"]),
    }
    decision_path = args.runs_root / "final_decision.json"
    decision_path.write_text(json.dumps(decision_payload, indent=2, sort_keys=True) + "\n")

    absolute = aggregate[primary]
    paired = comparisons[primary]
    lines = [
        "# FRESH-VLA Oracle Plan-Commit 最终上界裁决",
        "",
        "## 实验完整性",
        "",
        f"- 冻结 Full-H checkpoints：seeds={list(args.seeds)}，未重训、未修改权重；",
        (
            f"- 配对 test snapshot groups：{len(subgroup_ids[primary])}；对应独立 source initial states："
            f"{len(source_state_clusters)}；"
        ),
        "- 该 test split 已参与上一阶段最终裁决，本轮是 locked/post-hoc 上界实验，不是新盲测集；",
        (
            "- `feedback_reveal_time > action_divergence_time` 反向组："
            f"{len(subgroup_ids['reverse_feedback_gt_divergence'])}；"
        ),
        "- fixed K=1/2/3 复用既有结果；新增方法完成 isolated、end-to-end 和 deterministic reach；",
        "- 统计单位为 snapshot group，先在 group 内跨 seed 平均，再做 paired bootstrap；",
        "- 另以 source initial state 为聚类单位做保守敏感性分析，最终 GO 必须同时通过两套门槛；",
        (
            "- teacher 绝对时钟在偏离轨迹上无效，因此实际 Oracle 是 privileged runtime grasp/lift event "
            "interrupt 上界；事件 outcome 从未进入 Pi0.5 输入或动作选择。"
        ),
        "",
        "## 主结果",
        "",
        "括号为 snapshot-group bootstrap 95% CI。",
        "",
        "| 方法 | Overall | Attached | Slip recovery | Isolated recovery | Forward calls | Inference s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        lines.append(
            f"| `{method}` | {_fmt_rate(absolute[method]['overall_task_success'])} | "
            f"{_fmt_rate(absolute[method]['attached_task_success'])} | "
            f"{_fmt_rate(absolute[method]['slip_recovery_success'])} | "
            f"{_fmt_rate(absolute[method]['isolated_recovery_success'])} | "
            f"{float(absolute[method]['policy_forward_calls']['mean']):.1f} | "
            f"{float(absolute[method]['inference_wall_seconds']['mean']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Oracle 配对差异",
            "",
            "| 对照 | Overall | Slip recovery | Isolated recovery | Failure continuation | Premature commitment |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for baseline in ("fixed_k3", "random_matched_commit", "gripper_commit", "self_consistency_commit", "fixed_k1"):
        comparison = paired[f"oracle_vs_{baseline}"]
        lines.append(
            f"| `{baseline}` | {_fmt_pp(comparison['overall_task_success'])} | "
            f"{_fmt_pp(comparison['slip_recovery_success'])} | "
            f"{_fmt_pp(comparison['isolated_recovery_success'])} | "
            f"{_fmt_pp(comparison['failure_continuation_rate'])} | "
            f"{_fmt_pp(comparison['premature_commitment_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## 边界可识别性限制",
            "",
            (
                "本数据的 feedback reveal、action divergence 和物理事件标签完全重合，因此两个 Oracle "
                "调度在本轮没有可识别差异。该事实限制机制归因，但不影响判断‘Oracle 执行承诺上界"
                "是否优于固定或近邻控制’。"
            ),
            (
                "本轮只能裁决 privileged runtime-event-aligned Plan-Commit 上界，不能独立比较 branch-safe "
                "与 feedback-reveal 两种边界标签；若输出 STOP，其证据含义是连该更强事件上界也未通过。"
            ),
            "",
            "## 预注册门槛",
            "",
        ]
    )
    for name, value in gate.items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "## Source-state 聚类敏感性门槛", ""])
    for name, value in cluster_gate.items():
        lines.append(f"- `{name}`: `{value}`")
    lines.append(f"- `source_state_sensitivity_decision`: `{cluster_decision}`")
    lines.extend(
        [
            "",
            (
                "fixed-K 历史结果缺少逐调用计时，因此其 wall-clock 由相同 seed 新运行的单样本方法"
                "每次调用中位数估算；真实 invocation/forward-call count 未估算。最终效率门槛优先"
                "使用真实 forward-call count。"
            ),
            (
                "历史 fixed-K JSON 记录了精确 seed、split、protocol 和 rows，但远程 policy socket payload "
                "当时没有嵌入 checkpoint hash；本轮保存其 JSON SHA256，并以原 runner 路径约定、当前冻结"
                "权重实测 SHA256 和单组逐协议 parity 共同绑定。该限制不会被表述成密码学级历史证明。"
            ),
            "",
            f"机器可读结果：`{results_path}`",
            "",
            f"Pareto 图：`{pareto_path}`",
            "",
            decision,
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps(decision_payload, sort_keys=True))


if __name__ == "__main__":
    main()
