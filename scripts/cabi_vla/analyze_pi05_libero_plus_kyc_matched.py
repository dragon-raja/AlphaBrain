from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from analyze_pi05_libero_plus_composition import (
    CONDITIONS,
    read_composition_group_metadata,
    read_composition_group_scores,
    summarize_run,
)
from analyze_pi05_libero_plus_views import bootstrap_mean


BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260805
INTERPRETATION_BOUNDARIES = {
    "paired_joint_ood_stress_test": {
        "classification": "paired_joint_ood_stress_test",
        "strict_seen_factor_composition": False,
        "factor_category_separated_training": False,
        "reason": (
            "The camera/background factors are jointly out of the camera-only "
            "training distribution; they are not a held pairing of two "
            "individually seen factor categories."
        ),
    },
    "factor_separated_category_composition": {
        "classification": "factor_separated_category_composition",
        "strict_seen_factor_composition": False,
        "factor_category_separated_training": True,
        "joint_factor_training_episode_count": 0,
        "reason": (
            "Camera-only and background-only factor categories are both present "
            "in training, while their joint condition is held out. Public Goal "
            "RLDS metadata does not preserve exact per-episode camera and texture "
            "identities, so exact held-pair coverage cannot be claimed."
        ),
    },
}


def _mean_seed_scores(
    runs: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, float]]:
    if not runs:
        raise ValueError("at least one seed is required")
    group_sets = {seed: set(scores) for seed, scores in runs.items()}
    expected = next(iter(group_sets.values()))
    for seed, groups in group_sets.items():
        if groups != expected:
            raise ValueError(f"group mismatch for seed {seed}")
    return {
        group: {
            condition: float(
                np.mean(
                    [runs[seed][group][condition] for seed in sorted(runs)]
                )
            )
            for condition in CONDITIONS
        }
        for group in sorted(expected)
    }


def _paired_method_effects(
    control: Mapping[str, Mapping[str, float]],
    kyc: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    if set(control) != set(kyc):
        raise ValueError("Control and KYC group sets differ")
    groups = sorted(control)
    effects: dict[str, Any] = {}
    for condition in CONDITIONS:
        effects[f"{condition}_success"] = bootstrap_mean(
            [kyc[group][condition] - control[group][condition] for group in groups],
            samples=20_000,
        )

    control_camera_gap = np.asarray(
        [
            control[group]["canonical"] - control[group]["camera_only"]
            for group in groups
        ]
    )
    kyc_camera_gap = np.asarray(
        [
            kyc[group]["canonical"] - kyc[group]["camera_only"]
            for group in groups
        ]
    )
    control_joint_gap = np.asarray(
        [
            control[group]["canonical"] - control[group]["camera_background"]
            for group in groups
        ]
    )
    kyc_joint_gap = np.asarray(
        [
            kyc[group]["canonical"] - kyc[group]["camera_background"]
            for group in groups
        ]
    )
    effects["camera_gap_reduction"] = bootstrap_mean(
        control_camera_gap - kyc_camera_gap,
        samples=20_000,
    )
    effects["joint_gap_reduction"] = bootstrap_mean(
        control_joint_gap - kyc_joint_gap,
        samples=20_000,
    )
    return effects


def _crossed_bootstrap(
    values_by_seed: Mapping[int, Mapping[str, float]],
    *,
    seed: int,
) -> dict[str, Any]:
    seeds = sorted(values_by_seed)
    if not seeds:
        raise ValueError("crossed bootstrap requires at least one seed")
    groups = sorted(values_by_seed[seeds[0]])
    if not groups:
        raise ValueError("crossed bootstrap requires at least one group")
    expected = set(groups)
    for training_seed in seeds:
        if set(values_by_seed[training_seed]) != expected:
            raise ValueError("crossed bootstrap requires identical groups per seed")
    values = np.asarray(
        [
            [values_by_seed[training_seed][group] for group in groups]
            for training_seed in seeds
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    sampled_seeds = generator.integers(
        0,
        len(seeds),
        size=(BOOTSTRAP_SAMPLES, len(seeds)),
    )
    sampled_groups = generator.integers(
        0,
        len(groups),
        size=(BOOTSTRAP_SAMPLES, len(groups)),
    )
    distribution = values[
        sampled_seeds[:, :, None],
        sampled_groups[:, None, :],
    ].mean(axis=(1, 2))
    low, high = np.quantile(distribution, [0.025, 0.975]).tolist()
    return {
        "mean": float(values.mean()),
        "ci95": [float(low), float(high)],
        "training_seed_count": len(seeds),
        "independent_group_count": len(groups),
        "bootstrap_resamples": BOOTSTRAP_SAMPLES,
        "bootstrap_scheme": "crossed_training_seed_and_base_task",
    }


def _crossed_method_effects(
    control_runs: Mapping[int, Mapping[str, Mapping[str, float]]],
    kyc_runs: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    if set(control_runs) != set(kyc_runs):
        raise ValueError("Control and KYC seed sets differ")
    seeds = sorted(control_runs)
    expected = set(control_runs[seeds[0]])
    for seed in seeds:
        if set(control_runs[seed]) != expected or set(kyc_runs[seed]) != expected:
            raise ValueError(f"Control and KYC group sets differ for seed {seed}")

    values: dict[str, dict[int, dict[str, float]]] = {
        f"{condition}_success": {
            seed: {
                group: (
                    kyc_runs[seed][group][condition]
                    - control_runs[seed][group][condition]
                )
                for group in sorted(expected)
            }
            for seed in seeds
        }
        for condition in CONDITIONS
    }
    values["camera_gap_reduction"] = {
        seed: {
            group: (
                control_runs[seed][group]["canonical"]
                - control_runs[seed][group]["camera_only"]
                - kyc_runs[seed][group]["canonical"]
                + kyc_runs[seed][group]["camera_only"]
            )
            for group in sorted(expected)
        }
        for seed in seeds
    }
    values["joint_gap_reduction"] = {
        seed: {
            group: (
                control_runs[seed][group]["canonical"]
                - control_runs[seed][group]["camera_background"]
                - kyc_runs[seed][group]["canonical"]
                + kyc_runs[seed][group]["camera_background"]
            )
            for group in sorted(expected)
        }
        for seed in seeds
    }
    return {
        name: _crossed_bootstrap(
            per_seed,
            seed=BOOTSTRAP_SEED + index,
        )
        for index, (name, per_seed) in enumerate(values.items())
    }


def _decision(effects: Mapping[str, Mapping[str, Any]]) -> str:
    camera = effects["camera_only_success"]
    joint = effects["camera_background_success"]
    canonical = effects["canonical_success"]
    target = (camera, joint)
    if any(metric["mean"] >= 0.05 and metric["ci95"][0] > 0 for metric in target):
        if canonical["mean"] >= -0.05:
            return "KYC_INCREMENTAL_VALUE_CONFIRMED"
    if any(metric["mean"] <= -0.05 and metric["ci95"][1] < 0 for metric in target):
        return "KYC_DEGRADES_VIEW_GENERALIZATION"
    if all(metric["ci95"][1] < 0.05 for metric in target):
        return "KYC_NO_MEANINGFUL_INCREMENTAL_VALUE"
    return "KYC_INCREMENTAL_VALUE_INCONCLUSIVE"


def _baseline_valid(control: Mapping[str, Any]) -> bool:
    return float(control["conditions"]["canonical"]["mean"]) >= 0.20


def build_report(
    control_runs: Mapping[int, Mapping[str, Mapping[str, float]]],
    kyc_runs: Mapping[int, Mapping[str, Mapping[str, float]]],
    *,
    interpretation: str = "paired_joint_ood_stress_test",
) -> dict[str, Any]:
    if set(control_runs) != set(kyc_runs):
        raise ValueError("Control and KYC seed sets differ")
    if interpretation not in INTERPRETATION_BOUNDARIES:
        raise ValueError(f"unsupported interpretation boundary: {interpretation}")
    control_mean = _mean_seed_scores(control_runs)
    kyc_mean = _mean_seed_scores(kyc_runs)
    effects = _crossed_method_effects(control_runs, kyc_runs)
    control_summary = summarize_run(control_mean)
    kyc_summary = summarize_run(kyc_mean)
    baseline_valid = _baseline_valid(control_summary)
    per_seed = {
        str(seed): {
            "control": summarize_run(control_runs[seed]),
            "kyc": summarize_run(kyc_runs[seed]),
            "kyc_minus_control": _paired_method_effects(
                control_runs[seed], kyc_runs[seed]
            ),
        }
        for seed in sorted(control_runs)
    }
    return {
        "schema_version": 1,
        "study": "pi05_libero_plus_matched_kyc",
        "independent_statistical_unit": "suite_x_base_task",
        "uncertainty": "crossed_training_seed_and_base_task_bootstrap",
        "seeds": sorted(control_runs),
        "cross_seed": {
            "control": control_summary,
            "kyc": kyc_summary,
            "kyc_minus_control": effects,
        },
        "per_seed": per_seed,
        "gates": {
            "BASELINE_VALID": baseline_valid,
            "baseline_threshold": 0.20,
        },
        "decision": (
            _decision(effects)
            if baseline_valid
            else "BASELINE_INVALID_OR_DATA_INSUFFICIENT"
        ),
        "interpretation_boundary": INTERPRETATION_BOUNDARIES[interpretation],
    }


def _pct(metric: Mapping[str, Any]) -> str:
    return (
        f"{metric['mean']:.1%} "
        f"[{metric['ci95'][0]:.1%}, {metric['ci95'][1]:.1%}]"
    )


def write_chinese_report(report: Mapping[str, Any], output: Path) -> None:
    cross = report["cross_seed"]
    labels = {
        "canonical": "原始相机 + 原始背景",
        "camera_only": "扰动相机 + 原始背景",
        "background_only": "原始相机 + 新背景",
        "camera_background": "扰动相机 + 新背景",
    }
    lines = [
        "# Pi0.5 上 KYC 的完全匹配增量验证",
        "",
        "> Control 与 KYC 使用相同多视角数据、视觉 LoRA、训练步数、优化器、"
        "随机种子和闭环协议。唯一差异是射线图使用规范相机位姿（Control）还是"
        "当前真实相机位姿（KYC）。独立统计单位为基础任务，先在种子内聚合初态，"
        "再对训练种子和基础任务做 20,000 次 crossed paired bootstrap。",
        "",
        "## 跨种子闭环成功率",
        "",
        "| 条件 | Control | KYC | KYC - Control |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        lines.append(
            f"| {labels[condition]} | "
            f"{cross['control']['conditions'][condition]['mean']:.1%} | "
            f"{cross['kyc']['conditions'][condition]['mean']:.1%} | "
            f"{_pct(cross['kyc_minus_control'][condition + '_success'])} |"
        )
    boundary = report["interpretation_boundary"]
    if boundary["classification"] == "factor_separated_category_composition":
        interpretation_lines = [
            "本实验回答的是：训练分别包含相机单因素与背景单因素轨迹、但不包含",
            "二者联合轨迹时，Pi0.5 是否仍出现组合缺口，以及 KYC 是否减轻该缺口。",
            "公开 Goal RLDS 没有逐 episode 的具体相机和纹理 ID，因此这是因子类别",
            "分离组合实验，不能升级为精确位姿×纹理留配对结论。",
        ]
    else:
        interpretation_lines = [
            "本实验回答的是：在 Pi0.5、保留腕部相机、并已做多视角训练时，显式",
            "真实相机几何是否还能提供额外闭环收益。它是配对的相机与背景联合域外",
            "压力测试，不是严格的“两个因素分别见过、唯独组合未见过”实验；后者需",
            "由单独的因子配对数据划分回答。",
        ]
    lines.extend(
        [
            "",
            "## 关键差值",
            "",
            f"- 原始背景上的相机缺口缩小：{_pct(cross['kyc_minus_control']['camera_gap_reduction'])}。",
            f"- 相机与背景同时变化的总缺口缩小：{_pct(cross['kyc_minus_control']['joint_gap_reduction'])}。",
            "",
            "## 每个随机种子",
            "",
            "| 种子 | Control 组合成功率 | KYC 组合成功率 | 差值 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for seed, values in report["per_seed"].items():
        lines.append(
            f"| {seed} | "
            f"{values['control']['conditions']['camera_background']['mean']:.1%} | "
            f"{values['kyc']['conditions']['camera_background']['mean']:.1%} | "
            f"{values['kyc_minus_control']['camera_background_success']['mean']:+.1%} |"
        )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            f"**`{report['decision']}`**",
            "",
            *interpretation_lines,
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def render_figure(report: Mapping[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    cross = report["cross_seed"]
    labels = ["Canonical", "Camera", "Background", "Camera+BG"]
    x = np.arange(len(CONDITIONS))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for offset, method, color in (
        (-width / 2, "control", "#4477AA"),
        (width / 2, "kyc", "#CC6677"),
    ):
        axes[0].bar(
            x + offset,
            [cross[method]["conditions"][key]["mean"] for key in CONDITIONS],
            width,
            label=method.upper(),
            color=color,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Full-task success rate")
    axes[0].set_title("Matched closed-loop conditions")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    keys = [
        "camera_only_success",
        "camera_background_success",
        "camera_gap_reduction",
        "joint_gap_reduction",
    ]
    effect_labels = ["Camera success", "Joint success", "Camera gap", "Joint gap"]
    means = [cross["kyc_minus_control"][key]["mean"] for key in keys]
    low = [
        mean - cross["kyc_minus_control"][key]["ci95"][0]
        for mean, key in zip(means, keys, strict=True)
    ]
    high = [
        cross["kyc_minus_control"][key]["ci95"][1] - mean
        for mean, key in zip(means, keys, strict=True)
    ]
    axes[1].bar(
        np.arange(len(keys)),
        means,
        yerr=np.asarray([low, high]),
        capsize=4,
        color=["#66A61E", "#1B9E77", "#7570B3", "#E6AB02"],
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(keys)), effect_labels, rotation=15)
    axes[1].set_ylabel("KYC minus Control")
    axes[1].set_title("Paired task-level effects (95% CI)")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _named_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SEED=OUTPUT_DIR")
    seed, path = value.split("=", 1)
    try:
        parsed_seed = int(seed)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    if not path:
        raise argparse.ArgumentTypeError("output directory is empty")
    return parsed_seed, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze matched Pi0.5 KYC runs")
    parser.add_argument("--control", type=_named_path, action="append", required=True)
    parser.add_argument("--kyc", type=_named_path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument(
        "--interpretation",
        choices=sorted(INTERPRETATION_BOUNDARIES),
        default="paired_joint_ood_stress_test",
    )
    return parser.parse_args()


def _load_runs(values: Sequence[tuple[int, Path]]) -> dict[int, dict[str, dict[str, float]]]:
    paths = dict(values)
    if len(paths) != len(values):
        raise ValueError("duplicate seed")
    return {seed: read_composition_group_scores(path) for seed, path in paths.items()}


def main() -> None:
    args = parse_args()
    control = _load_runs(args.control)
    kyc = _load_runs(args.kyc)

    all_paths = [path for _, path in args.control + args.kyc]
    metadata = [read_composition_group_metadata(path) for path in all_paths]
    if any(current != metadata[0] for current in metadata[1:]):
        raise ValueError("composition protocol metadata differs across runs")

    report = build_report(control, kyc, interpretation=args.interpretation)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_chinese_report(report, args.output_report)
    render_figure(report, args.output_figure)
    print(json.dumps({"decision": report["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
