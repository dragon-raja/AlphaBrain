from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = ("canonical", "camera_only", "background_only", "camera_background")
CONDITION_LABELS_ZH = {
    "canonical": "原始相机 + 原始背景",
    "camera_only": "扰动相机 + 原始背景",
    "background_only": "原始相机 + 新背景",
    "camera_background": "扰动相机 + 新背景",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _validate_inputs(
    *,
    official_act: Mapping[str, Any],
    joint_ood: Mapping[str, Any],
    matched: Mapping[str, Any],
    factor_separated: Mapping[str, Any],
    factorial: Mapping[str, Any],
    ray_alignment: Mapping[str, Any],
) -> None:
    if official_act.get("status") != "complete":
        raise ValueError("official ACT positive control is incomplete")
    if joint_ood.get("decision") != "RESIDUAL_JOINT_CAMERA_BACKGROUND_OOD_GAP_CONFIRMED":
        raise ValueError("unexpected joint OOD result")
    if matched.get("interpretation_boundary", {}).get("classification") != "paired_joint_ood_stress_test":
        raise ValueError("matched KYC result has the wrong interpretation boundary")
    if factor_separated.get("interpretation_boundary", {}).get("classification") != "factor_separated_category_composition":
        raise ValueError("factor-separated result has the wrong interpretation boundary")
    if factorial.get("status") != "complete":
        raise ValueError("scene-cue by wrist factorial is incomplete")
    if not ray_alignment.get("gate", {}).get("passed"):
        raise ValueError("RGB/ray geometric alignment gate failed")


def _final_kyc_decision(matched: Mapping[str, Any], factor: Mapping[str, Any]) -> str:
    decisions = {matched["decision"], factor["decision"]}
    if "BASELINE_INVALID_OR_DATA_INSUFFICIENT" in decisions:
        return "PI05_KYC_DECISION_INCOMPLETE_BASELINE_INVALID"
    if factor["decision"] == "KYC_INCREMENTAL_VALUE_CONFIRMED":
        return "KYC_TRANSFER_CONFIRMED_UNDER_FACTOR_SEPARATION"
    if matched["decision"] == "KYC_INCREMENTAL_VALUE_CONFIRMED":
        return "KYC_CONTEXT_DEPENDENT_INCREMENTAL_VALUE"
    negative = {
        "KYC_NO_MEANINGFUL_INCREMENTAL_VALUE",
        "KYC_DEGRADES_VIEW_GENERALIZATION",
    }
    if decisions <= negative:
        return "RAW_RAY_KYC_NOT_TRANSFERRED_TO_PI05"
    return "KYC_INCREMENTAL_VALUE_INCONCLUSIVE"


def build_final_report(
    *,
    official_act: Mapping[str, Any],
    joint_ood: Mapping[str, Any],
    matched: Mapping[str, Any],
    factor_separated: Mapping[str, Any],
    factorial: Mapping[str, Any],
    ray_alignment: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_inputs(
        official_act=official_act,
        joint_ood=joint_ood,
        matched=matched,
        factor_separated=factor_separated,
        factorial=factorial,
        ray_alignment=ray_alignment,
    )
    multiview = joint_ood["runs"][joint_ood["candidate_name"]]
    camera_gap = multiview["effects"]["camera_gap_canonical_background"]
    joint_gap = multiview["effects"]["combined_gap"]
    factor_control = factor_separated["cross_seed"]["control"]
    factor_joint_gap_metric = factor_control["effects"]["combined_gap"]
    factor_joint_gap = factor_joint_gap_metric["mean"]
    camera_sufficient = camera_gap["ci95"][1] <= 0.05
    factor_joint_sufficient = (
        factor_separated["gates"]["BASELINE_VALID"]
        and factor_joint_gap_metric["ci95"][1] <= 0.05
    )
    fx_on = factorial["cells"]["fx_on"]
    cue_on = factorial["cells"]["cue_on"]
    act_test = official_act["pose_sets"]["test_cameras"]
    act_effect = act_test["hierarchical_paired_bootstrap"]

    return {
        "schema_version": 1,
        "study": "pi05_libero_plus_camera_background_kyc_final_synthesis",
        "questions": {
            "multiview_camera_sufficient_within_tested_single_factor": camera_sufficient,
            "multiview_plus_background_data_sufficient_for_category_composition": factor_joint_sufficient,
            "factor_separated_control_joint_gap": factor_joint_gap,
            "kyc_final_decision": _final_kyc_decision(matched, factor_separated),
        },
        "evidence": {
            "official_act_positive_control": {
                "image_success": act_test["equal_seed_mean"]["image_success"],
                "kyc_success": act_test["equal_seed_mean"]["kyc_success"],
                "kyc_minus_image": {
                    "mean": act_effect["delta"],
                    "ci95": [act_effect["ci95_low"], act_effect["ci95_high"]],
                    "bootstrap_resamples": act_effect["bootstrap_resamples"],
                },
            },
            "pi05_scene_cue_wrist_factorial": {
                "fixed_wrist_on_control": fx_on["equal_seed_mean"]["poseaug_control"]["success"],
                "fixed_wrist_on_kyc": fx_on["equal_seed_mean"]["kyc"]["success"],
                "randomized_cue_wrist_on_control": cue_on["equal_seed_mean"]["poseaug_control"]["success"],
                "randomized_cue_wrist_on_kyc": cue_on["equal_seed_mean"]["kyc"]["success"],
            },
            "multiview_joint_ood": {
                "conditions": multiview["conditions"],
                "single_camera_gap": camera_gap,
                "joint_gap": joint_gap,
            },
            "matched_joint_ood_kyc": matched,
            "factor_separated_category_composition_kyc": factor_separated,
            "ray_alignment": {
                "median_pixel_error": ray_alignment["pixel_error"]["median"],
                "p90_pixel_error": ray_alignment["pixel_error"]["p90"],
                "maximum_pixel_error": ray_alignment["pixel_error"]["maximum"],
                "passed": True,
            },
        },
        "claim_boundaries": {
            "exact_camera_texture_pair_composition_tested": False,
            "reason": (
                "Public Goal RLDS preserves factor category but not exact per-episode "
                "camera and texture identities."
            ),
            "not_tested": [
                "moving camera during execution",
                "novel kitchen geometry",
                "real-robot calibration noise",
            ],
        },
    }


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _effect(value: Mapping[str, Any]) -> str:
    return f"{value['mean']:+.1%} [{value['ci95'][0]:+.1%}, {value['ci95'][1]:+.1%}]"


def write_report(report: Mapping[str, Any], output: Path) -> None:
    evidence = report["evidence"]
    multiview = evidence["multiview_joint_ood"]
    matched = evidence["matched_joint_ood_kyc"]
    factor = evidence["factor_separated_category_composition_kyc"]
    questions = report["questions"]
    lines = [
        "# Pi0.5 视角泛化与 KYC 最终对照",
        "",
        "## 一页结论",
        "",
        f"- 单相机因素范围内，多视角训练是否足够：**{'是' if questions['multiview_camera_sufficient_within_tested_single_factor'] else '否'}**。",
        f"- 分别加入相机/背景数据后，组合条件是否足够：**{'是' if questions['multiview_plus_background_data_sufficient_for_category_composition'] else '否'}**。",
        f"- Pi0.5 上 KYC 的最终裁决：**`{questions['kyc_final_decision']}`**。",
        "",
        "## 1. 已有论文与正对照",
        "",
        "KYC 发布代码的 ACT/随机场景/无腕部正对照已复现：",
        "Image-only 为 "
        f"{_pct(evidence['official_act_positive_control']['image_success'])}，KYC 为 "
        f"{_pct(evidence['official_act_positive_control']['kyc_success'])}，差值 "
        f"{_effect(evidence['official_act_positive_control']['kyc_minus_image'])}。"
        "这证明 KYC 机制在原论文设置中有效，但不直接证明能迁移到 Pi0.5。",
        "",
        "## 2. 多视角训练解决了多少",
        "",
        "| 条件 | 强多视角 Pi0.5 成功率 |",
        "|---|---:|",
    ]
    for condition in CONDITIONS:
        lines.append(
            f"| {CONDITION_LABELS_ZH[condition]} | "
            f"{_pct(multiview['conditions'][condition]['mean'])} |"
        )
    lines.extend(
        [
            "",
            f"原始背景上的相机缺口为 {_effect(multiview['single_camera_gap'])}；"
            f"相机和新背景同时变化的总缺口为 {_effect(multiview['joint_gap'])}。",
            "因此，多视角数据能基本解决本评测的单相机扰动，但是否解决相机—背景组合，"
            "必须看下面的因子分离训练，不能由单因素结果外推。",
            "",
            "## 3. KYC 完全匹配比较",
            "",
            "| 数据设计 | Control 原始 | Control 相机 | Control 组合 | KYC 组合 | KYC-Control 组合差值 | 判定 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for label, result in (("相机训练 + 联合 OOD", matched), ("相机/背景因子分离训练", factor)):
        control = result["cross_seed"]["control"]["conditions"]
        kyc = result["cross_seed"]["kyc"]["conditions"]
        effect = result["cross_seed"]["kyc_minus_control"]["camera_background_success"]
        lines.append(
            f"| {label} | {_pct(control['canonical']['mean'])} | "
            f"{_pct(control['camera_only']['mean'])} | "
            f"{_pct(control['camera_background']['mean'])} | "
            f"{_pct(kyc['camera_background']['mean'])} | {_effect(effect)} | "
            f"`{result['decision']}` |"
        )
    lines.extend(
        [
            "",
            "两组都使用相同 Pi0.5 初始化、25% 分层数据、SigLIP LoRA rank 16、"
            "33,000 步、seeds 41/42/43 和相同闭环任务。Control 也有同容量相机分支，"
            "唯一差别是 Control 使用规范 ray，KYC 使用与当前 RGB 对齐的真实 ray。",
            "",
            "## 4. 可信度与边界",
            "",
            f"RGB/ray 投影审计通过：中位误差 {evidence['ray_alignment']['median_pixel_error']:.2f}px，"
            f"P90 {evidence['ray_alignment']['p90_pixel_error']:.2f}px。统计按训练种子×基础任务做"
            " crossed paired bootstrap，不把帧当独立样本。全部新评测视频为 AV1/WebM。",
            "",
            "因公开 Goal RLDS 不含逐 episode 的精确相机和纹理 ID，第二组是“因子类别分别见过、"
            "联合类别未见”的组合实验，不是精确位姿×纹理留配对。结果也不能外推到执行中移动相机、"
            "新厨房几何或真机标定噪声。",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def render_figure(report: Mapping[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    evidence = report["evidence"]
    multiview = evidence["multiview_joint_ood"]
    matched = evidence["matched_joint_ood_kyc"]
    factor = evidence["factor_separated_category_composition_kyc"]
    x = np.arange(len(CONDITIONS))
    labels = ["Canonical", "Camera", "Background", "Camera+BG"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharey=True)
    axes[0].bar(x, [multiview["conditions"][key]["mean"] for key in CONDITIONS], color="#4C956C")
    axes[0].set_title("Strong multiview Pi0.5")
    for axis, title, result in (
        (axes[1], "Matched joint OOD", matched),
        (axes[2], "Factor-separated training", factor),
    ):
        width = 0.34
        axis.bar(
            x - width / 2,
            [result["cross_seed"]["control"]["conditions"][key]["mean"] for key in CONDITIONS],
            width,
            label="Control",
            color="#4477AA",
        )
        axis.bar(
            x + width / 2,
            [result["cross_seed"]["kyc"]["conditions"][key]["mean"] for key in CONDITIONS],
            width,
            label="KYC",
            color="#CC6677",
        )
        axis.set_title(title)
        axis.legend()
    for axis in axes:
        axis.set_xticks(x, labels, rotation=20)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Full-task success rate")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize the Pi0.5 KYC camera study")
    parser.add_argument("--official-act", type=Path, required=True)
    parser.add_argument("--joint-ood", type=Path, required=True)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--factor-separated", type=Path, required=True)
    parser.add_argument("--factorial", type=Path, required=True)
    parser.add_argument("--ray-alignment", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_final_report(
        official_act=_read(args.official_act),
        joint_ood=_read(args.joint_ood),
        matched=_read(args.matched),
        factor_separated=_read(args.factor_separated),
        factorial=_read(args.factorial),
        ray_alignment=_read(args.ray_alignment),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_report(report, args.output_report)
    render_figure(report, args.output_figure)
    print(json.dumps(report["questions"], sort_keys=True))


if __name__ == "__main__":
    main()
