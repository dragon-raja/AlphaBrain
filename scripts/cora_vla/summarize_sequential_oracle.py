from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_libero_closed_loop import _atomic_write_json
from evaluate_sequential_oracle import METHODS


SEEDS = (41, 42, 43)
OUTCOMES = ("attached", "slipped")
ORACLES = ("oracle_short_physical", "oracle_policy_continuation")


def paired_bootstrap(values: Sequence[float], *, seed: int = 20260717) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        raise ValueError("bootstrap requires at least one snapshot group")
    generator = np.random.default_rng(seed)
    samples = array[generator.integers(0, len(array), size=(20000, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "group_count": int(len(array)),
    }


def load_rows(root: Path) -> dict[tuple[int, str, str], list[dict[str, Any]]]:
    merged: dict[tuple[int, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in sorted(root.glob("seed*.json")):
        payload = json.loads(path.read_text())
        if payload.get("experiment") != "cora_sequential_oracle":
            continue
        if payload.get("status") != "complete" or payload.get("run_kind") != "formal":
            continue
        key = (int(payload["seed"]), str(payload["outcome"]), str(payload["method"]))
        for row in payload["rows"]:
            pair_id = str(row["pair_id"])
            if pair_id in merged[key]:
                raise ValueError(f"duplicate formal row for {key} {pair_id}")
            merged[key][pair_id] = row
    expected = {(seed, outcome, method) for seed in SEEDS for outcome in OUTCOMES for method in METHODS}
    if set(merged) != expected:
        missing = sorted(expected - set(merged))
        raise ValueError(f"formal matrix incomplete; missing={missing}")
    result = {}
    for key, by_pair in merged.items():
        if len(by_pair) != 13:
            raise ValueError(f"{key} has {len(by_pair)} groups instead of 13")
        result[key] = [by_pair[pair_id] for pair_id in sorted(by_pair)]
    return result


def metric_value(row: Mapping[str, Any], metric: str) -> float:
    return float(row[metric])


def group_metric(
    rows: Mapping[tuple[int, str, str], Sequence[Mapping[str, Any]]],
    method: str,
    outcome: str,
    metric: str,
) -> dict[str, float]:
    by_pair: dict[str, list[float]] = defaultdict(list)
    for seed in SEEDS:
        for row in rows[(seed, outcome, method)]:
            by_pair[str(row["pair_id"])].append(metric_value(row, metric))
    return {pair_id: float(np.mean(values)) for pair_id, values in by_pair.items()}


def overall_group_metric(
    rows: Mapping[tuple[int, str, str], Sequence[Mapping[str, Any]]],
    method: str,
    metric: str,
) -> dict[str, float]:
    attached = group_metric(rows, method, "attached", metric)
    slipped = group_metric(rows, method, "slipped", metric)
    return {pair_id: (attached[pair_id] + slipped[pair_id]) / 2 for pair_id in attached}


def paired_difference(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    if set(left) != set(right):
        raise ValueError("paired comparison group sets differ")
    return paired_bootstrap([left[pair_id] - right[pair_id] for pair_id in sorted(left)])


def seed_success(
    rows: Mapping[tuple[int, str, str], Sequence[Mapping[str, Any]]],
    method: str,
    outcome: str,
) -> dict[str, float]:
    return {
        str(seed): float(np.mean([row["success"] for row in rows[(seed, outcome, method)]]))
        for seed in SEEDS
    }


def decision_diagnostics(method_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = []
    pool = []
    conditional = {str(j): {"eligible": 0, "success": 0} for j in (1, 2, 4, 8, 16)}
    for row in method_rows:
        flags = [bool(decision["selected_immediate_correct_preexecution"]) for decision in row["decisions"]]
        selected.extend(flags)
        pool.extend(float(decision["pool_immediate_correct_rate"]) for decision in row["decisions"])
        for j in (1, 2, 4, 8, 16):
            if len(flags) >= j and all(flags[:j]):
                conditional[str(j)]["eligible"] += 1
                conditional[str(j)]["success"] += int(bool(row["success"]))
    for row in conditional.values():
        row["success_rate"] = None if row["eligible"] == 0 else row["success"] / row["eligible"]
    return {
        "selected_immediate_correct_rate": float(np.mean(selected)),
        "mean_pool_immediate_correct_rate": float(np.mean(pool)),
        "episode_success_given_first_j_correct": conditional,
    }


def aggregate(
    rows: Mapping[tuple[int, str, str], Sequence[Mapping[str, Any]]],
    onpolicy: Mapping[str, Any],
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for method in METHODS:
        attached = group_metric(rows, method, "attached", "success")
        slipped = group_metric(rows, method, "slipped", "success")
        overall = overall_group_metric(rows, method, "success")
        all_rows = [row for seed in SEEDS for outcome in OUTCOMES for row in rows[(seed, outcome, method)]]
        methods[method] = {
            "attached_success": paired_bootstrap(list(attached.values())),
            "slip_recovery_success": paired_bootstrap(list(slipped.values())),
            "overall_success": paired_bootstrap(list(overall.values())),
            "by_seed_attached_success": seed_success(rows, method, "attached"),
            "by_seed_slip_success": seed_success(rows, method, "slipped"),
            "stable_regrasp": float(np.mean([row["stable_regrasp"] for row in all_rows if row["outcome"] == "slipped"])),
            "failure_continuation": float(np.mean([row["failure_continuation"] for row in all_rows if row["outcome"] == "slipped"])),
            "premature_commitment": float(np.mean([row["premature_commitment"] for row in all_rows if row["outcome"] == "slipped"])),
            "progress_auc": float(np.mean([row["progress_auc"] for row in all_rows])),
            "completion_actions": float(np.mean([row["actions"] for row in all_rows])),
            "wall_seconds": float(np.mean([row["wall_seconds"] for row in all_rows])),
            "decision_diagnostics": decision_diagnostics(all_rows),
        }
        if method != "single_sample":
            comparisons[method] = {
                "vs_single": {
                    "attached_success": paired_difference(attached, group_metric(rows, "single_sample", "attached", "success")),
                    "slip_recovery_success": paired_difference(slipped, group_metric(rows, "single_sample", "slipped", "success")),
                    "overall_success": paired_difference(overall, overall_group_metric(rows, "single_sample", "success")),
                },
                "vs_random": {
                    "slip_recovery_success": paired_difference(slipped, group_metric(rows, "random_pick_N", "slipped", "success")),
                    "overall_success": paired_difference(overall, overall_group_metric(rows, "random_pick_N", "success")),
                },
            }

    teacher_direction = comparisons["oracle_teacher_distance"]["vs_single"]["slip_recovery_success"]["mean"] > 0
    qualifying = []
    gate_audit = {}
    for oracle in ORACLES:
        vs_single = comparisons[oracle]["vs_single"]
        vs_random = comparisons[oracle]["vs_random"]
        seed_diffs = [
            methods[oracle]["by_seed_slip_success"][str(seed)]
            - methods["single_sample"]["by_seed_slip_success"][str(seed)]
            for seed in SEEDS
        ]
        random_significant = (
            vs_random["slip_recovery_success"]["ci95_low"] > 0
            or vs_random["overall_success"]["ci95_low"] > 0
        )
        direction_supported = all(diff > 0 for diff in seed_diffs) or vs_single["slip_recovery_success"]["ci95_low"] > 0
        checks = {
            "slip_gain_at_least_15pp": vs_single["slip_recovery_success"]["mean"] >= 0.15,
            "overall_gain_at_least_5pp": vs_single["overall_success"]["mean"] >= 0.05,
            "attached_degradation_at_most_5pp": vs_single["attached_success"]["mean"] >= -0.05,
            "significantly_better_than_random": random_significant,
            "onpolicy_recall16_at_least_60pct": float(onpolicy["overall"]["recall@16"]) >= 0.60,
            "three_seed_direction_or_ci": direction_supported,
            "teacher_distance_same_direction": teacher_direction,
        }
        gate_audit[oracle] = {"checks": checks, "seed_slip_differences": seed_diffs}
        if all(checks.values()):
            qualifying.append(oracle)
    decision = "GO_CORA_ENERGY_ROUTING" if qualifying else "STOP_CORA_ROUTING"
    return {
        "experiment": "cora_sequential_oracle",
        "snapshot_group_count": 13,
        "seeds": list(SEEDS),
        "candidate_count": 16,
        "execution_horizon": 2,
        "confirmation_groups_accessed": False,
        "onpolicy_support_decision": onpolicy["decision"],
        "onpolicy_recall@16": onpolicy["overall"]["recall@16"],
        "methods": methods,
        "comparisons": comparisons,
        "gate_audit": gate_audit,
        "qualifying_oracles": qualifying,
        "decision": decision,
    }


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_report(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# CORA-VLA Sequential Oracle 最终 Gate",
        "",
        "本报告只使用冻结的 13 个 validation snapshot groups；confirmation groups 未访问。所有方法固定 N=16（single 为同一候选流的 candidate 0）、K=2、最长 320 actions，基础 Full-H 参数冻结。",
        "",
        "## 闭环主结果",
        "",
        "| 方法 | Attached成功 | Slip恢复成功 | Overall成功 | Failure continuation | Premature commitment | 平均actions | 平均wall time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = result["methods"][method]
        lines.append(
            f"| `{method}` | {pct(row['attached_success']['mean'])} | {pct(row['slip_recovery_success']['mean'])} | "
            f"{pct(row['overall_success']['mean'])} | {pct(row['failure_continuation'])} | "
            f"{pct(row['premature_commitment'])} | {row['completion_actions']:.1f} | {row['wall_seconds']:.1f}s |"
        )
    lines.extend(["", "## 相对 Full-H single", ""])
    for method in METHODS[1:]:
        comparison = result["comparisons"][method]["vs_single"]
        lines.append(
            f"- `{method}`：slip {pct(comparison['slip_recovery_success']['mean'])} "
            f"(95% CI [{pct(comparison['slip_recovery_success']['ci95_low'])}, {pct(comparison['slip_recovery_success']['ci95_high'])}])；"
            f"overall {pct(comparison['overall_success']['mean'])}；attached {pct(comparison['attached_success']['mean'])}。"
        )
    lines.extend(["", "## 裁决审计", ""])
    lines.append(f"On-policy correct-mode recall@16={pct(result['onpolicy_recall@16'])}，候选支持必要条件已满足。")
    for oracle, audit in result["gate_audit"].items():
        failed = [name for name, passed in audit["checks"].items() if not passed]
        lines.append(f"- `{oracle}`：" + ("全部通过" if not failed else "未通过 " + "、".join(failed)))
    lines.extend(
        [
            "",
            "统计单位为 snapshot group；三个 seed 先在组内聚合，再进行 paired group bootstrap。候选、帧与 replan 没有被当成独立样本。",
            "",
            "## 最终结论",
            "",
            result["decision"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CORA sequential routing Gate")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--onpolicy-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.root)
    result = aggregate(rows, json.loads(args.onpolicy_decision.read_text()))
    _atomic_write_json(args.output, result)
    write_report(args.report, result)
    print(result["decision"])


if __name__ == "__main__":
    main()
