from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def paired_group_bootstrap(
    values: Mapping[str, Sequence[float]], *, samples: int = 10000, seed: int = 20260717
) -> dict[str, float]:
    groups = sorted(values)
    matrix = np.asarray([values[group] for group in groups], dtype=np.float64)
    group_means = matrix.mean(axis=1)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(samples, len(groups)))
    bootstrap = group_means[indices].mean(axis=1)
    return {
        "mean": float(group_means.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "group_count": len(groups),
    }


def aggregate(payloads: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(payloads) != 3:
        raise ValueError("formal Gate 1 requires exactly three checkpoint seeds")
    if {int(payload["checkpoint_seed"]) for payload in payloads} != {41, 42, 43}:
        raise ValueError("formal Gate 1 requires checkpoint seeds 41, 42, and 43")
    if any(payload["status"] != "complete" for payload in payloads):
        raise ValueError("all Gate 1 payloads must be complete")
    group_sets = [
        {(str(row["pair_id"]), str(row["outcome"])) for row in payload["rows"]}
        for payload in payloads
    ]
    if any(groups != group_sets[0] for groups in group_sets[1:]):
        raise ValueError("checkpoint seeds do not cover identical group/outcome rows")

    by_seed = {}
    curves = {
        outcome: {
            field: {str(size): [] for size in (1, 4, 8, 16, 32)}
            for field in ("joint_recall", "action_recall", "effect_recall", "physical_recall")
        }
        for outcome in ("attached", "slipped")
    }
    agreements = {
        outcome: {"action_physical_agreement": [], "joint_physical_agreement": []}
        for outcome in ("attached", "slipped")
    }
    group_metrics: dict[str, dict[str, list[float]]] = {
        name: {}
        for name in (
            "attached_joint16",
            "slipped_joint16",
            "slipped_joint32",
            "slipped_physical1",
            "slipped_physical16",
            "slipped_physical_gain16",
        )
    }
    leakage_passed = True
    for payload in payloads:
        checkpoint_seed = int(payload["checkpoint_seed"])
        rows = payload["rows"]
        attached = [row for row in rows if row["outcome"] == "attached"]
        slipped = [row for row in rows if row["outcome"] == "slipped"]
        seed_summary = {
            "attached_joint_recall@16": float(np.mean([row["joint_recall"]["16"] for row in attached])),
            "slipped_joint_recall@16": float(np.mean([row["joint_recall"]["16"] for row in slipped])),
            "slipped_joint_recall@32": float(np.mean([row["joint_recall"]["32"] for row in slipped])),
            "slipped_physical_success@1": float(np.mean([row["physical_recall"]["1"] for row in slipped])),
            "slipped_physical_success@16": float(np.mean([row["physical_recall"]["16"] for row in slipped])),
        }
        seed_summary["slipped_physical_gain@16_vs_1"] = (
            seed_summary["slipped_physical_success@16"] - seed_summary["slipped_physical_success@1"]
        )
        by_seed[str(checkpoint_seed)] = seed_summary
        leakage_passed = leakage_passed and bool(payload["pre_feedback_leakage_passed"])
        for row in rows:
            pair_id = str(row["pair_id"])
            outcome = str(row["outcome"])
            for field in curves[outcome]:
                for size in curves[outcome][field]:
                    curves[outcome][field][size].append(float(row[field][size]))
            for field in agreements[outcome]:
                agreements[outcome][field].append(float(row[field]))
            if outcome == "attached":
                group_metrics["attached_joint16"].setdefault(pair_id, []).append(
                    float(row["joint_recall"]["16"])
                )
            else:
                physical1 = float(row["physical_recall"]["1"])
                physical16 = float(row["physical_recall"]["16"])
                group_metrics["slipped_joint16"].setdefault(pair_id, []).append(
                    float(row["joint_recall"]["16"])
                )
                group_metrics["slipped_joint32"].setdefault(pair_id, []).append(
                    float(row["joint_recall"]["32"])
                )
                group_metrics["slipped_physical1"].setdefault(pair_id, []).append(physical1)
                group_metrics["slipped_physical16"].setdefault(pair_id, []).append(physical16)
                group_metrics["slipped_physical_gain16"].setdefault(pair_id, []).append(
                    physical16 - physical1
                )

    estimates = {name: paired_group_bootstrap(values) for name, values in group_metrics.items()}
    mean_curves = {
        outcome: {
            field: {size: float(np.mean(values)) for size, values in by_size.items()}
            for field, by_size in by_field.items()
        }
        for outcome, by_field in curves.items()
    }
    mean_agreements = {
        outcome: {field: float(np.mean(values)) for field, values in by_field.items()}
        for outcome, by_field in agreements.items()
    }
    thresholds = {
        "attached_joint_recall@16": estimates["attached_joint16"]["mean"] >= 0.80,
        "slipped_joint_recall@16": estimates["slipped_joint16"]["mean"] >= 0.50,
        "slipped_physical_gain@16_vs_1": estimates["slipped_physical_gain16"]["mean"] >= 0.15,
        "pre_feedback_no_leakage": leakage_passed,
    }
    if not leakage_passed:
        decision = "COUNTERFACTUAL_DATA_LEAKAGE"
    elif estimates["slipped_joint32"]["mean"] < 0.40:
        decision = "BASE_POLICY_LACKS_RECOVERY_SUPPORT"
    elif all(thresholds.values()):
        decision = "PASS_CORA_GATE1"
    else:
        decision = "STOP_CORA_CANDIDATE_SUPPORT"
    return {
        "experiment": "cora_gate1_candidate_support",
        "decision": decision,
        "checkpoint_seeds": [41, 42, 43],
        "by_seed": by_seed,
        "mean_recall_curves": mean_curves,
        "mean_label_agreements": mean_agreements,
        "group_level_estimates": estimates,
        "thresholds_passed": thresholds,
        "pre_feedback_leakage_passed": leakage_passed,
    }


def percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def markdown(result: Mapping[str, object]) -> str:
    estimates = result["group_level_estimates"]
    lines = [
        "# CORA-VLA Gate 1：基础策略候选支持度",
        "",
        "本报告严格使用既有 validation snapshot groups；新 confirmation groups 保持封存，未用于本 Gate。基础 Full-H Pi0.5 全部冻结，本阶段未训练任何新模型。",
        "",
        "## 主要结果",
        "",
        "| 指标 | 跨 seed 组级均值 | 95% bootstrap CI |",
        "|---|---:|---:|",
    ]
    labels = {
        "attached_joint16": "Attached 联合 correct-mode recall@16",
        "slipped_joint16": "Slipped 联合 correct-mode recall@16",
        "slipped_joint32": "Slipped 联合 correct-mode recall@32",
        "slipped_physical1": "Slipped physical success@1",
        "slipped_physical16": "Slipped oracle physical success@16",
        "slipped_physical_gain16": "Slipped oracle @16 相对 @1 增益",
    }
    for key, label in labels.items():
        estimate = estimates[key]
        lines.append(
            f"| {label} | {percent(estimate['mean'])} | [{percent(estimate['ci95_low'])}, {percent(estimate['ci95_high'])}] |"
        )
    lines.extend(["", "## 各 checkpoint seed", ""])
    for seed, values in result["by_seed"].items():
        lines.append(
            f"- seed {seed}：attached recall@16={percent(values['attached_joint_recall@16'])}，"
            f"slipped recall@16={percent(values['slipped_joint_recall@16'])}，"
            f"slipped physical @1/@16={percent(values['slipped_physical_success@1'])}/"
            f"{percent(values['slipped_physical_success@16'])}。"
        )
    lines.extend(
        [
            "",
            "## 完整 recall@N",
            "",
            "| 分支 | N | Action | EEF effect | 联合主标签 | 短时物理标签 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    curves = result["mean_recall_curves"]
    for outcome in ("attached", "slipped"):
        for size in ("1", "4", "8", "16", "32"):
            lines.append(
                f"| {outcome} | {size} | {percent(curves[outcome]['action_recall'][size])} | "
                f"{percent(curves[outcome]['effect_recall'][size])} | "
                f"{percent(curves[outcome]['joint_recall'][size])} | "
                f"{percent(curves[outcome]['physical_recall'][size])} |"
            )
    agreements = result["mean_label_agreements"]
    lines.extend(
        [
            "",
            "标签一致率："
            f"attached action/physical={percent(agreements['attached']['action_physical_agreement'])}、"
            f"joint/physical={percent(agreements['attached']['joint_physical_agreement'])}；"
            f"slipped action/physical={percent(agreements['slipped']['action_physical_agreement'])}、"
            f"joint/physical={percent(agreements['slipped']['joint_physical_agreement'])}。",
        ]
    )
    lines.extend(
        [
            "",
            "## 解释与边界",
            "",
            "联合 correct-mode 标签要求 action 距离和 K=2 EEF-effect 距离同时支持正确 continuation。物理上界还要求分支特定即时行为正确，并由同一 teacher 完成后续任务；因此它不是单纯的 teacher 可救回率。",
            "",
            f"反馈前泄漏审计：{'通过' if result['pre_feedback_leakage_passed'] else '失败'}。统计单位为 snapshot group，CI 未把候选或帧当作独立样本。",
            "",
            "## Gate 1 裁决",
            "",
            f"**{result['decision']}**",
            "",
        ]
    )
    if result["decision"] == "PASS_CORA_GATE1":
        lines.append("基础策略候选分布具有足够支持度，可以进入 Counterfactual Outcome Energy 的 Gate 2；这尚不证明 reranker 有效，更不构成闭环 Go。")
    else:
        lines.append("按预注册停止规则，本路线不进入能量模型训练。")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize formal CORA Gate 1 payloads")
    parser.add_argument("--inputs", nargs=3, type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = [json.loads(path.read_text()) for path in args.inputs]
    result = aggregate(payloads)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown(result))
    print(json.dumps({"decision": result["decision"], "output": str(args.output_json)}, sort_keys=True))


if __name__ == "__main__":
    main()
