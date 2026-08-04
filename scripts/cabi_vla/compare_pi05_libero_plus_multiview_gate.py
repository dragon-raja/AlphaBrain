from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def bootstrap_mean(values: Sequence[float], *, samples: int = 20_000, seed: int = 20260804) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a non-empty finite vector")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(array.mean()),
        "ci95": [float(low), float(high)],
        "independent_group_count": int(len(array)),
        "bootstrap_resamples": int(samples),
    }


def read_gap_group_scores(output_dir: Path) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    paths = sorted(output_dir.glob("episodes-shard-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no episode shards in {output_dir}")
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row["pair_key"]).startswith("gap::"):
                continue
            condition = str(row["condition"])
            if condition not in {"canonical", "official_camera"}:
                continue
            group = f"{row['suite']}::{row['base_task']}"
            grouped[group][condition].append(float(bool(row["success"])))
    result = {}
    for group, conditions in sorted(grouped.items()):
        if set(conditions) != {"canonical", "official_camera"}:
            raise ValueError(f"incomplete gap conditions for {group}: {sorted(conditions)}")
        result[group] = {
            condition: float(np.mean(values))
            for condition, values in conditions.items()
        }
    if not result:
        raise ValueError(f"no gap rows in {output_dir}")
    return result


def compare_run(
    scores: Mapping[str, Mapping[str, float]],
    baseline: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    if set(scores) != set(baseline):
        missing = sorted(set(baseline) - set(scores))
        extra = sorted(set(scores) - set(baseline))
        raise ValueError(f"run group mismatch: missing={missing} extra={extra}")
    groups = sorted(scores)
    canonical = np.asarray([scores[group]["canonical"] for group in groups])
    camera = np.asarray([scores[group]["official_camera"] for group in groups])
    baseline_canonical = np.asarray([baseline[group]["canonical"] for group in groups])
    baseline_camera = np.asarray([baseline[group]["official_camera"] for group in groups])
    return {
        "independent_group_count": len(groups),
        "canonical_success": bootstrap_mean(canonical),
        "official_camera_success": bootstrap_mean(camera),
        "view_generalization_gap": bootstrap_mean(canonical - camera),
        "paired_minus_official_baseline": {
            "canonical_success": bootstrap_mean(canonical - baseline_canonical),
            "official_camera_success": bootstrap_mean(camera - baseline_camera),
            "gap_reduction": bootstrap_mean(
                (baseline_canonical - baseline_camera) - (canonical - camera)
            ),
        },
    }


def build_report(
    baseline_name: str,
    baseline_scores: Mapping[str, Mapping[str, float]],
    runs: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    baseline = compare_run(baseline_scores, baseline_scores)
    comparisons = {
        name: compare_run(scores, baseline_scores)
        for name, scores in runs.items()
    }
    best_name = max(
        comparisons,
        key=lambda name: (
            comparisons[name]["official_camera_success"]["mean"],
            comparisons[name]["canonical_success"]["mean"],
            name,
        ),
    )
    best = comparisons[best_name]
    camera_gain = best["paired_minus_official_baseline"]["official_camera_success"]
    canonical_delta = best["paired_minus_official_baseline"]["canonical_success"]
    valid = best["canonical_success"]["mean"] >= 0.70
    material_gain = camera_gain["mean"] >= 0.05 and canonical_delta["mean"] >= -0.05
    significant_gain = camera_gain["ci95"][0] > 0.0 and canonical_delta["mean"] >= -0.05
    action_data_effect = None
    visual_data_effect = None
    if {"action_b025", "action_b100"} <= set(comparisons):
        action_data_effect = (
            comparisons["action_b100"]["official_camera_success"]["mean"]
            - comparisons["action_b025"]["official_camera_success"]["mean"]
        )
    if {"visual_b025", "visual_b100"} <= set(comparisons):
        visual_data_effect = (
            comparisons["visual_b100"]["official_camera_success"]["mean"]
            - comparisons["visual_b025"]["official_camera_success"]["mean"]
        )
    visual_adapter_effect = None
    if {"action_b100", "visual_b100"} <= set(comparisons):
        visual_adapter_effect = (
            comparisons["visual_b100"]["official_camera_success"]["mean"]
            - comparisons["action_b100"]["official_camera_success"]["mean"]
        )
    return {
        "schema_version": 1,
        "study": "pi05_libero_plus_strong_multiview_baseline_gate",
        "baseline_name": baseline_name,
        "baseline": baseline,
        "runs": comparisons,
        "best_run": best_name,
        "effects": {
            "action_100_minus_25_camera_success": action_data_effect,
            "visual_100_minus_25_camera_success": visual_data_effect,
            "visual_lora_minus_action_only_at_100_camera_success": visual_adapter_effect,
        },
        "gates": {
            "BASELINE_VALID": valid,
            "MATERIAL_MULTIVIEW_GAIN": material_gain,
            "PAIRED_SIGNIFICANT_MULTIVIEW_GAIN": significant_gain,
        },
    }


def write_chinese_report(report: Mapping[str, Any], output: Path) -> None:
    names = [report["baseline_name"], *report["runs"]]
    values = {report["baseline_name"]: report["baseline"], **report["runs"]}
    lines = [
        "# Pi0.5 × LIBERO-Plus 强多视角基线门控",
        "",
        "> 独立统计单位为 40 个 `suite × 基础任务`；每个任务内先平均两个初始状态，再做成对 bootstrap。",
        "",
        "| 模型 | Canonical | 官方相机扰动 | 视角缺口 | 相对官方 Pi0.5 的扰动成功率变化 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        row = values[name]
        delta = row["paired_minus_official_baseline"]["official_camera_success"]
        lines.append(
            f"| `{name}` | {row['canonical_success']['mean']:.1%} | "
            f"{row['official_camera_success']['mean']:.1%} | "
            f"{row['view_generalization_gap']['mean']:.1%} | "
            f"{delta['mean']:+.1%} [{delta['ci95'][0]:+.1%}, {delta['ci95'][1]:+.1%}] |"
        )
    best = report["runs"][report["best_run"]]
    best_delta = best["paired_minus_official_baseline"]["official_camera_success"]
    lines.extend(
        [
            "",
            "## 自动判定",
            "",
            f"- 最佳训练版本：`{report['best_run']}`。",
            f"- 最佳版本的相机扰动成功率：{best['official_camera_success']['mean']:.1%}。",
            f"- 相对官方 Pi0.5：{best_delta['mean']:+.1%}，95% CI "
            f"[{best_delta['ci95'][0]:+.1%}, {best_delta['ci95'][1]:+.1%}]。",
            f"- Canonical 有效性：{'通过' if report['gates']['BASELINE_VALID'] else '失败'}。",
            f"- 至少 5 个百分点且 canonical 退化不超过 5 点："
            f"{'通过' if report['gates']['MATERIAL_MULTIVIEW_GAIN'] else '未通过'}。",
            f"- 成对 95% CI 排除 0："
            f"{'通过' if report['gates']['PAIRED_SIGNIFICANT_MULTIVIEW_GAIN'] else '未通过'}。",
            "",
            "## 因果拆分",
            "",
        ]
    )
    labels = {
        "action_100_minus_25_camera_success": "Action-only 使用 100% 而非 25% 数据",
        "visual_100_minus_25_camera_success": "视觉 LoRA 使用 100% 而非 25% 数据",
        "visual_lora_minus_action_only_at_100_camera_success": "100% 数据下视觉 LoRA 相对 Action-only",
    }
    for key, label in labels.items():
        value = report["effects"].get(key)
        if value is not None:
            lines.append(f"- {label}：{value:+.1%}。")
    lines.extend(
        [
            "",
            "此门控只判断多视角训练和视觉适配能否缩小官方相机扰动缺口；"
            "候选视角、主动感知和方法创新结论必须在最佳版本的后续评测中单独给出。",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def render_figure(report: Mapping[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    names = [report["baseline_name"], *report["runs"]]
    values = {report["baseline_name"]: report["baseline"], **report["runs"]}
    x = np.arange(len(names))
    width = 0.36
    canonical = [values[name]["canonical_success"]["mean"] for name in names]
    camera = [values[name]["official_camera_success"]["mean"] for name in names]
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(x - width / 2, canonical, width, label="Canonical", color="#3976af")
    axis.bar(x + width / 2, camera, width, label="Official camera perturbation", color="#d05b44")
    axis.set_xticks(x, names, rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Full-task success rate")
    axis.set_title("Pi0.5 strong multiview baseline gate")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=OUTPUT_DIR")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=OUTPUT_DIR")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare paired LIBERO-Plus multiview gate runs")
    parser.add_argument("--baseline", type=parse_named_path, required=True)
    parser.add_argument("--run", type=parse_named_path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_name, baseline_path = args.baseline
    runs = dict(args.run)
    if len(runs) != len(args.run):
        raise ValueError("duplicate run names")
    report = build_report(
        baseline_name,
        read_gap_group_scores(baseline_path),
        {name: read_gap_group_scores(path) for name, path in runs.items()},
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_chinese_report(report, args.output_report)
    render_figure(report, args.output_figure)
    print(json.dumps({"best_run": report["best_run"], "gates": report["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
