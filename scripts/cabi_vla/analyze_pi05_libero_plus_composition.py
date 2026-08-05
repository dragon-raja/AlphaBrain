from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pi05_libero_plus_views import bootstrap_mean


CONDITIONS = (
    "canonical",
    "camera_only",
    "background_only",
    "camera_background",
)


def read_composition_group_scores(output_dir: Path) -> dict[str, dict[str, float]]:
    paths = sorted(output_dir.glob("episodes-shard-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no episode shards in {output_dir}")
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    seen: set[str] = set()
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row["pair_key"]).startswith("composition::"):
                continue
            episode_id = str(row["episode_id"])
            if episode_id in seen:
                raise ValueError(f"duplicate episode id: {episode_id}")
            seen.add(episode_id)
            condition = str(row["condition"])
            pair_key = str(row["pair_key"])
            if condition in pairs[pair_key]:
                raise ValueError(f"duplicate condition {condition} for {pair_key}")
            pairs[pair_key][condition] = row
    if not pairs:
        raise ValueError(f"no composition rows in {output_dir}")

    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pair_key, rows in sorted(pairs.items()):
        if set(rows) != set(CONDITIONS):
            raise ValueError(
                f"incomplete composition pair {pair_key}: {sorted(rows)}"
            )
        state_hashes = {
            str(row.get("initial_metrics", {}).get("physics_state_sha256", ""))
            for row in rows.values()
        }
        if "" in state_hashes or len(state_hashes) != 1:
            raise ValueError(
                f"physical initial state mismatch for {pair_key}: {sorted(state_hashes)}"
            )
        representative = rows["canonical"]
        group = f"{representative['suite']}::{representative['base_task']}"
        for condition in CONDITIONS:
            grouped[group][condition].append(float(bool(rows[condition]["success"])))

    scores = {
        group: {
            condition: float(np.mean(values[condition]))
            for condition in CONDITIONS
        }
        for group, values in sorted(grouped.items())
    }
    if any(set(values) != set(CONDITIONS) for values in scores.values()):
        raise ValueError("one or more independent groups are incomplete")
    return scores


def summarize_run(scores: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    groups = sorted(scores)
    if not groups:
        raise ValueError("composition scores are empty")
    values = {
        condition: np.asarray(
            [float(scores[group][condition]) for group in groups],
            dtype=np.float64,
        )
        for condition in CONDITIONS
    }
    effects = {
        "camera_gap_canonical_background": values["canonical"] - values["camera_only"],
        "background_gap_canonical_camera": values["canonical"] - values["background_only"],
        "combined_gap": values["canonical"] - values["camera_background"],
        "camera_gap_unseen_background": (
            values["background_only"] - values["camera_background"]
        ),
    }
    effects["negative_composition_interaction"] = (
        effects["camera_gap_unseen_background"]
        - effects["camera_gap_canonical_background"]
    )
    condition_summary = {
        condition: bootstrap_mean(values[condition], samples=20_000)
        for condition in CONDITIONS
    }
    canonical_mean = condition_summary["canonical"]["mean"]
    return {
        "independent_group_count": len(groups),
        "conditions": condition_summary,
        "effects": {
            name: bootstrap_mean(value, samples=20_000)
            for name, value in effects.items()
        },
        "camera_background_retention": (
            condition_summary["camera_background"]["mean"] / canonical_mean
            if canonical_mean > 0
            else None
        ),
    }


def compare_runs(
    candidate: Mapping[str, Mapping[str, float]],
    reference: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    if set(candidate) != set(reference):
        missing = sorted(set(reference) - set(candidate))
        extra = sorted(set(candidate) - set(reference))
        raise ValueError(f"run group mismatch: missing={missing} extra={extra}")
    groups = sorted(reference)
    result = {}
    for condition in CONDITIONS:
        differences = [
            float(candidate[group][condition]) - float(reference[group][condition])
            for group in groups
        ]
        result[f"{condition}_success"] = bootstrap_mean(
            differences,
            samples=20_000,
        )
    candidate_combined_gap = [
        float(candidate[group]["canonical"])
        - float(candidate[group]["camera_background"])
        for group in groups
    ]
    reference_combined_gap = [
        float(reference[group]["canonical"])
        - float(reference[group]["camera_background"])
        for group in groups
    ]
    result["combined_gap_reduction"] = bootstrap_mean(
        np.asarray(reference_combined_gap) - np.asarray(candidate_combined_gap),
        samples=20_000,
    )
    return result


def _decision(candidate: Mapping[str, Any]) -> tuple[str, dict[str, bool]]:
    conditions = candidate["conditions"]
    effects = candidate["effects"]
    valid = conditions["canonical"]["mean"] >= 0.70
    combined = effects["combined_gap"]
    unseen_camera = effects["camera_gap_unseen_background"]
    interaction = effects["negative_composition_interaction"]
    residual = valid and (
        (combined["mean"] >= 0.10 and combined["ci95"][0] > 0.0)
        or (unseen_camera["mean"] >= 0.10 and unseen_camera["ci95"][0] > 0.0)
        or (interaction["mean"] >= 0.10 and interaction["ci95"][0] > 0.0)
    )
    sufficient = valid and (
        combined["mean"] <= 0.05
        and combined["ci95"][1] <= 0.10
        and unseen_camera["mean"] <= 0.05
        and unseen_camera["ci95"][1] <= 0.10
        and candidate["camera_background_retention"] is not None
        and candidate["camera_background_retention"] >= 0.95
    )
    gates = {
        "BASELINE_VALID": valid,
        "RESIDUAL_COMPOSITION_GAP": residual,
        "MULTIVIEW_SUFFICIENT_WITHIN_TESTED_COMPOSITION": sufficient,
    }
    if not valid:
        return "BASELINE_INVALID", gates
    if residual:
        return "RESIDUAL_CAMERA_SCENE_COMPOSITION_GAP_CONFIRMED", gates
    if sufficient:
        return "MULTIVIEW_DATA_SUFFICIENT_ON_TESTED_PLUS_COMPOSITION", gates
    return "CAMERA_SCENE_COMPOSITION_RESULT_INCONCLUSIVE", gates


def build_report(
    scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    reference_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    if reference_name not in scores or candidate_name not in scores:
        raise ValueError("reference and candidate names must be present")
    group_sets = {name: set(values) for name, values in scores.items()}
    expected = group_sets[reference_name]
    for name, groups in group_sets.items():
        if groups != expected:
            raise ValueError(f"independent group mismatch for {name}")
    runs = {name: summarize_run(values) for name, values in scores.items()}
    comparison = compare_runs(scores[candidate_name], scores[reference_name])
    decision, gates = _decision(runs[candidate_name])
    return {
        "schema_version": 1,
        "study": "pi05_libero_plus_camera_background_composition",
        "independent_statistical_unit": "suite_x_base_task",
        "reference_name": reference_name,
        "candidate_name": candidate_name,
        "runs": runs,
        "candidate_minus_reference": comparison,
        "gates": gates,
        "decision": decision,
        "scope": {
            "scene_shift": "paired unseen table and background appearance textures",
            "not_tested": [
                "novel kitchen geometry",
                "novel object layout",
                "moving camera during an episode",
                "real-robot domain shift",
            ],
        },
    }


def _percent_interval(metric: Mapping[str, Any]) -> str:
    return (
        f"{metric['mean']:.1%} "
        f"[{metric['ci95'][0]:.1%}, {metric['ci95'][1]:.1%}]"
    )


def write_chinese_report(report: Mapping[str, Any], output: Path) -> None:
    names = list(report["runs"])
    labels = {
        "canonical": "原始相机 + 原始背景",
        "camera_only": "扰动相机 + 原始背景",
        "background_only": "原始相机 + 新桌面/背景",
        "camera_background": "扰动相机 + 新桌面/背景",
    }
    lines = [
        "# Pi0.5 × LIBERO-Plus 相机—背景组合泛化",
        "",
        "> 严格四条件配对：同一任务、同一语言、同一物体布局、同一初始状态；"
        "只改变第三方相机和桌面/背景外观。独立统计单位为 40 个基础任务，"
        "每个任务内先平均两个初始状态，再做 20,000 次成对 bootstrap。",
        "",
        "## 四条件闭环成功率",
        "",
        "| 模型 | 原始相机+原始背景 | 扰动相机+原始背景 | 原始相机+新背景 | 扰动相机+新背景 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        conditions = report["runs"][name]["conditions"]
        lines.append(
            f"| `{name}` | "
            + " | ".join(f"{conditions[key]['mean']:.1%}" for key in CONDITIONS)
            + " |"
        )
    lines.extend(["", "## 关键效应", ""])
    effect_labels = {
        "camera_gap_canonical_background": "原始背景上的相机缺口",
        "background_gap_canonical_camera": "原始相机下的背景缺口",
        "combined_gap": "相机与背景同时变化的总缺口",
        "camera_gap_unseen_background": "新背景上的相机缺口",
        "negative_composition_interaction": "额外组合惩罚",
    }
    for name in names:
        lines.append(f"### `{name}`")
        lines.append("")
        for key, label in effect_labels.items():
            lines.append(
                f"- {label}：{_percent_interval(report['runs'][name]['effects'][key])}。"
            )
        lines.append(
            f"- 组合条件保留率：{report['runs'][name]['camera_background_retention']:.1%}。"
        )
        lines.append("")
    comparison = report["candidate_minus_reference"]
    lines.extend(
        [
            "## 强多视角模型相对官方模型",
            "",
            f"- 组合条件成功率变化：{_percent_interval(comparison['camera_background_success'])}。",
            f"- 组合缺口缩小：{_percent_interval(comparison['combined_gap_reduction'])}。",
            "",
            "## 自动判定",
            "",
            f"**`{report['decision']}`**",
            "",
            f"- 基线有效：{'是' if report['gates']['BASELINE_VALID'] else '否'}。",
            "- 存在至少 10 个百分点且置信区间排除 0 的残余组合缺口："
            f"{'是' if report['gates']['RESIDUAL_COMPOSITION_GAP'] else '否'}。",
            "- 在本测试范围内可认为多视角数据充分："
            f"{'是' if report['gates']['MULTIVIEW_SUFFICIENT_WITHIN_TESTED_COMPOSITION'] else '否'}。",
            "",
            "## 边界",
            "",
            "这里的“新场景”仅指未见过的桌面与背景纹理，不等同于新厨房几何、"
            "新物体布局、执行中移动相机或真实机器人域迁移。若本门控通过，只能说明"
            "当前多视角训练解决了这组 LIBERO-Plus 外观组合，不代表视角泛化被普遍解决。",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def render_figure(report: Mapping[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    names = list(report["runs"])
    colors = ["#3976af", "#d05b44", "#4b9b69", "#8a5aa8"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(names))
    width = 0.18
    for index, condition in enumerate(CONDITIONS):
        axes[0].bar(
            x + (index - 1.5) * width,
            [report["runs"][name]["conditions"][condition]["mean"] for name in names],
            width,
            label=condition.replace("_", " "),
            color=colors[index],
        )
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Full-task success rate")
    axes[0].set_title("Paired 2 x 2 conditions")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    effect_names = [
        "camera_gap_canonical_background",
        "background_gap_canonical_camera",
        "combined_gap",
        "negative_composition_interaction",
    ]
    labels = ["camera", "background", "combined", "interaction"]
    x_effect = np.arange(len(effect_names))
    effect_width = 0.8 / len(names)
    for run_index, name in enumerate(names):
        means = [report["runs"][name]["effects"][key]["mean"] for key in effect_names]
        low = [
            mean - report["runs"][name]["effects"][key]["ci95"][0]
            for mean, key in zip(means, effect_names, strict=True)
        ]
        high = [
            report["runs"][name]["effects"][key]["ci95"][1] - mean
            for mean, key in zip(means, effect_names, strict=True)
        ]
        axes[1].bar(
            x_effect + (run_index - (len(names) - 1) / 2) * effect_width,
            means,
            effect_width,
            yerr=np.asarray([low, high]),
            capsize=3,
            label=name,
        )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x_effect, labels)
    axes[1].set_ylabel("Success-rate gap")
    axes[1].set_title("Group-level paired effects (95% CI)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
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
    parser = argparse.ArgumentParser(
        description="Analyze paired LIBERO-Plus camera by background composition runs"
    )
    parser.add_argument("--run", type=parse_named_path, action="append", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = dict(args.run)
    if len(paths) != len(args.run):
        raise ValueError("duplicate run names")
    report = build_report(
        {name: read_composition_group_scores(path) for name, path in paths.items()},
        reference_name=args.reference,
        candidate_name=args.candidate,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_chinese_report(report, args.output_report)
    render_figure(report, args.output_figure)
    print(json.dumps({"decision": report["decision"], "gates": report["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
