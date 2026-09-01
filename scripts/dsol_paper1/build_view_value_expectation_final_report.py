#!/usr/bin/env python3
"""Build the final Chinese report for the noise-marginalized view study."""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_view_revalidation_brief_pdf as base  # noqa: E402

METHOD_LABELS = {
    "canonical": "规范视角",
    "deterministic_random_candidate": "确定性随机",
    "calibration_global_fixed_pose": "校准集固定视角",
    "visibility_increment_selector": "可见性增量",
    "entity_visibility_hmean_selector": "实体可见性调和均值",
    "accel_ensemble_selector": "Accel 八噪声",
}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(pattern: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        rows.extend(json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


def text(
    fig: plt.Figure,
    x: float,
    y: float,
    value: str,
    size: float = 11,
    color: str | None = None,
    weight: str = "normal",
    **kwargs: Any,
) -> Any:
    return fig.text(
        x,
        y,
        value,
        fontsize=size,
        color=color or base.INK,
        va="top",
        fontweight=weight,
        linespacing=1.45,
        **kwargs,
    )


def table(
    fig: plt.Figure,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    rect: Sequence[float],
    widths: Sequence[float] | None = None,
    fontsize: float = 9.5,
) -> Any:
    ax = fig.add_axes(rect)
    ax.axis("off")
    result = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=widths,
        bbox=[0, 0, 1, 1],
        cellLoc="left",
        colLoc="left",
    )
    result.auto_set_font_size(False)
    result.set_fontsize(fontsize)
    for (row, _column), cell in result.get_celld().items():
        cell.set_edgecolor(base.LINE)
        cell.set_linewidth(0.6)
        cell.PAD = 0.07
        if row == 0:
            cell.set_facecolor(base.INK)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white" if row % 2 else "#EDF2F5")
    return ax


def clean(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=base.LINE, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def note(fig: plt.Figure, value: str, color: str = base.INK) -> None:
    text(fig, 0.055, 0.115, value, 10, color, "bold")


def save_page(pdf: PdfPages, fig: plt.Figure, page: int, preview_dir: Path) -> None:
    base.save_page(pdf, fig, page, preview_dir)


def candidate_group(candidate_id: str) -> str:
    if candidate_id == "canonical":
        return "canonical"
    if candidate_id.startswith("broad_train_"):
        return "train-support"
    if candidate_id.startswith("broad_heldout_"):
        return "held-out"
    return "other"


def final_decision(calibration: Mapping[str, Any], heldout: Mapping[str, Any]) -> dict[str, Any]:
    calibration_status = str(calibration["status"])
    selector = dict(heldout["selector_population_gate"])
    selector_status = str(selector["status"])
    if selector_status == "INCONCLUSIVE_NO_DIRECTIONAL_CLAIM":
        combined = "INCONCLUSIVE_NO_DIRECTIONAL_CLAIM"
    elif calibration_status == "STABLE_VIEW_HEADROOM_CONFIRMED":
        combined = (
            "VIEW_HEADROOM_AND_SELECTOR_GAIN_CONFIRMED"
            if selector_status == "SELECTOR_GAIN_CONFIRMED"
            else "VIEW_HEADROOM_CONFIRMED_SELECTOR_NOT_CONFIRMED"
        )
    else:
        combined = "VIEW_HEADROOM_NOT_CONFIRMED"
    return {
        "schema": "dsol_view_value_expectation_final_decision_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": combined,
        "calibration_headroom_gate": calibration_status,
        "heldout_selector_gate": selector_status,
        "selector_method": selector["best_rule_frozen_on_calibration"],
        "cross_checkpoint_mean_gain_pp": selector["cross_checkpoint_mean_gain_pp"],
        "cross_checkpoint_mean_harm_probability": selector["cross_checkpoint_mean_harm_probability"],
        "direction_consistent_positive": selector["direction_consistent_positive"],
        "noise_repeats_per_condition": heldout["noise_repeats_per_condition"],
        "claim_scope": (
            "fixed-state E0 counterfactual views in the frozen 97-view LIBERO bank; "
            "not official LIBERO-Plus full-benchmark performance and not physical active camera acquisition"
        ),
    }


def page_overview(
    pdf: PdfPages,
    preview: Path,
    population: Mapping[str, Any],
    calibration: Mapping[str, Any],
    heldout: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    fig, canvas = base.new_page(
        "视角价值期望：正式随机性复验",
        "AlphaBrain Pi0.5 · LIBERO exact-state E0 · source-disjoint calibration/test · 完整闭环",
        1,
    )
    calibration_states = population["population"]["calibration"]["states"]
    heldout_states = population["population"]["heldout_test"]["states"]
    base.metric(
        fig,
        canvas,
        0.06,
        0.59,
        0.19,
        0.17,
        str(len(calibration_states)),
        "校准物理状态",
        "8 任务，每任务 2 状态",
        base.BLUE,
    )
    base.metric(
        fig,
        canvas,
        0.28,
        0.59,
        0.19,
        0.17,
        str(len(heldout_states)),
        "独立测试状态",
        "来源轨迹与校准集不重叠",
        base.TEAL,
    )
    base.metric(
        fig,
        canvas,
        0.50,
        0.59,
        0.19,
        0.17,
        "97",
        "每状态候选视角",
        "规范 1 + 训练域 64 + 留出 32",
        base.GOLD,
    )
    base.metric(
        fig,
        canvas,
        0.72,
        0.59,
        0.22,
        0.17,
        str(heldout["noise_repeats_per_condition"]),
        "每条件完整噪声轨迹",
        "每次重规划显式共享 10×7 高斯噪声",
        base.RED,
    )
    table(
        fig,
        ["正式门", "结果", "解释边界"],
        [
            [
                "候选空间 Headroom",
                calibration["status"],
                "仅判断部分状态是否稳定存在更好视角",
            ],
            [
                "Held-out 选择器",
                heldout["selector_population_gate"]["status"],
                "规则完全由校准集冻结，测试结果不参与选视角",
            ],
            [
                "综合状态",
                decision["status"],
                "不外推为物理主动相机或官方 Plus 全榜单结论",
            ],
        ],
        [0.055, 0.28, 0.89, 0.235],
        [0.22, 0.37, 0.41],
        9.5,
    )
    note(
        fig,
        "核心估计：固定状态与 checkpoint 后，对完整 episode 的 replanning flow noise 求期望，再比较候选视角与规范视角。",
    )
    save_page(pdf, fig, 1, preview)


def page_design(pdf: PdfPages, preview: Path, root: Path, population: Mapping[str, Any]) -> None:
    fig, _canvas = base.new_page(
        "实验设计：先筛候选，再用独立噪声确认",
        "所有视角恢复同一 MuJoCo 物理状态；noise repeat 只估计策略随机性，不扩充任务样本量。",
        2,
    )
    table(
        fig,
        ["阶段", "状态", "候选/状态", "新增噪声", "用途"],
        [
            ["A", "校准 16", "97", "4", "粗筛，不作效果主张"],
            ["B", "校准 16", "24", "8", "几何分层收缩"],
            ["C", "校准 16", "6", "16", "冻结状态级候选"],
            ["D", "校准 16", "候选 + 规范", "64", "确认期望 Headroom"],
            ["E", "测试 48", "冻结规则", "32", "三 checkpoint 主检验"],
            ["F", "测试 48", "冻结规则", "+32", "仅在精度不足时自动开启"],
        ],
        [0.055, 0.46, 0.49, 0.31],
        [0.10, 0.18, 0.17, 0.16, 0.39],
        9.4,
    )
    diagnostic = root / "visibility-scan/analysis/visibility_diagnostics.png"
    if diagnostic.exists():
        base.add_image(fig, diagnostic, [0.59, 0.43, 0.35, 0.34])
    text(fig, 0.055, 0.39, "冻结的六种 held-out 规则", 13, weight="bold")
    text(
        fig,
        0.06,
        0.34,
        "规范视角｜确定性随机｜校准集全局固定｜可见性增量｜实体可见性调和均值｜Accel 八噪声",
        10.5,
    )
    counts = Counter(row["task_id"] for row in population["population"]["heldout_test"]["states"])
    text(
        fig,
        0.06,
        0.26,
        "测试任务分层：" + "；".join(f"{task}={count}" for task, count in sorted(counts.items())),
        8.8,
        base.MUTED,
    )
    note(fig, "这是一套中间状态 continuation 诊断：由真实轨迹状态接管并闭环到成功或超时，不是单步动作准确率。")
    save_page(pdf, fig, 2, preview)


def page_calibration(pdf: PdfPages, preview: Path, calibration: Mapping[str, Any]) -> None:
    fig, _canvas = base.new_page(
        "校准确认：候选池是否稳定存在行为收益空间",
        "候选在 C 阶段冻结；D 阶段使用全新的 64 条完整噪声序列，禁止看结果后换候选。",
        3,
    )
    states = calibration["states"]
    x = np.asarray([row["canonical_success"] * 100 for row in states])
    y = np.asarray([row["candidate_success"] * 100 for row in states])
    colors = [base.TEAL if row["strong_state"] else base.GRAY for row in states]
    ax = fig.add_axes([0.07, 0.39, 0.43, 0.38])
    ax.scatter(x, y, c=colors, s=55, edgecolors="white", linewidth=0.6)
    ax.plot([0, 100], [0, 100], color=base.INK, linewidth=1, linestyle="--")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("规范视角期望成功率 %")
    ax.set_ylabel("冻结候选期望成功率 %")
    ax.set_title("每个点是一个独立来源状态", loc="left", fontweight="bold")
    clean(ax)
    base.metric(
        fig,
        fig.axes[0],
        0.56,
        0.59,
        0.17,
        0.17,
        f"{calibration['strong_state_count']}/{calibration['state_count']}",
        "强候选状态",
        "候选≥80%，且增益≥20pp",
        base.TEAL,
    )
    base.metric(
        fig,
        fig.axes[0],
        0.76,
        0.59,
        0.17,
        0.17,
        f"{calibration['source_equal_success_gain_pp']:+.1f}pp",
        "来源等权平均增益",
        "配对 source bootstrap 见原始 JSON",
        base.BLUE,
    )
    rows = []
    by_task = defaultdict(list)
    for row in states:
        by_task[row["task_id"]].append(row)
    for task, task_rows in sorted(by_task.items()):
        rows.append(
            [
                task,
                str(len(task_rows)),
                str(sum(row["strong_state"] for row in task_rows)),
                f"{np.mean([row['success_gain'] for row in task_rows]) * 100:+.1f}pp",
            ]
        )
    table(
        fig,
        ["任务", "状态", "强状态", "平均增益"],
        rows,
        [0.55, 0.20, 0.39, 0.31],
        [0.50, 0.15, 0.17, 0.18],
        7.8,
    )
    note(fig, f"门槛结论：{calibration['status']}。绿色点才计入稳定 Headroom，不以一次成功的 Best-of-97 充当证据。")
    save_page(pdf, fig, 3, preview)


def page_heldout(pdf: PdfPages, preview: Path, heldout: Mapping[str, Any]) -> None:
    fig, _canvas = base.new_page(
        "Held-out 主检验：六种规则能否兑现候选空间",
        "Seed 41 完整比较；每个状态和规则使用相同显式噪声序列，误差线为任务分层 source bootstrap 95% CI。",
        4,
    )
    methods = heldout["checkpoint_seeds"]["41"]
    order = [method for method in METHOD_LABELS if method in methods]
    rates = [methods[method]["success_rate"] * 100 for method in order]
    gains = [methods[method]["success_gain_pp"] for method in order]
    gain_ci = [methods[method]["success_gain_task_stratified_bootstrap_95_pp"] for method in order]
    ax = fig.add_axes([0.07, 0.46, 0.42, 0.31])
    bars = ax.bar(
        np.arange(len(order)),
        rates,
        color=[
            base.BLUE,
            *[base.TEAL, base.GOLD, base.RED, base.GRAY, "#7B6EA8"][: len(order) - 1],
        ],
    )
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xticks(np.arange(len(order)), [METHOD_LABELS[item] for item in order], rotation=24, ha="right")
    ax.set_ylabel("完整闭环成功率 %")
    ax.set_ylim(0, max([*rates, 10]) + 15)
    clean(ax)
    ax = fig.add_axes([0.57, 0.46, 0.36, 0.31])
    y = np.arange(len(order))
    lower = [gain - ci[0] for gain, ci in zip(gains, gain_ci, strict=True)]
    upper = [ci[1] - gain for gain, ci in zip(gains, gain_ci, strict=True)]
    ax.errorbar(gains, y, xerr=[lower, upper], fmt="o", color=base.RED, capsize=3)
    ax.axvline(0, color=base.INK, linewidth=1)
    ax.axvline(5, color=base.GOLD, linewidth=1, linestyle="--")
    ax.set_yticks(y, [METHOD_LABELS[item] for item in order])
    ax.set_xlabel("相对规范视角增益（百分点）")
    ax.set_title("配对效应与 95% CI", loc="left", fontweight="bold")
    clean(ax)
    rows = []
    for method in order:
        values = methods[method]
        counts = values["paired_episode_counts"]
        rows.append(
            [
                METHOD_LABELS[method],
                f"{values['success_gain_pp']:+.1f}pp",
                f"{values['rescue_probability'] * 100:.1f}%",
                f"{values['harm_probability'] * 100:.1f}%",
                f"{counts['rescue']}/{counts['harm']}",
            ]
        )
    table(
        fig,
        ["规则", "成功增益", "Rescue", "Harm", "救回/伤害次数"],
        rows,
        [0.055, 0.17, 0.89, 0.22],
        [0.34, 0.18, 0.16, 0.16, 0.16],
        8.6,
    )
    note(fig, "Rescue/Harm 是同一状态、同一噪声下的成对行为变化；不能把噪声重复当作更多独立任务扩大显著性。")
    save_page(pdf, fig, 4, preview)


def page_cross_seed(pdf: PdfPages, preview: Path, heldout: Mapping[str, Any]) -> None:
    fig, _canvas = base.new_page(
        "三 checkpoint 与噪声收敛：结果是否可复现",
        "Seed 42/43 只复核校准集冻结的最佳非规范规则，避免在测试集继续挑方法。",
        5,
    )
    gate = heldout["selector_population_gate"]
    seeds = gate["seed_results"]
    labels = [str(row["checkpoint_seed"]) for row in seeds]
    gains = [row["success_gain_pp"] for row in seeds]
    cis = [row["success_gain_ci_95_pp"] for row in seeds]
    ax = fig.add_axes([0.07, 0.46, 0.39, 0.31])
    y = np.arange(len(seeds))
    ax.errorbar(
        gains,
        y,
        xerr=[
            [gain - ci[0] for gain, ci in zip(gains, cis, strict=True)],
            [ci[1] - gain for gain, ci in zip(gains, cis, strict=True)],
        ],
        fmt="o",
        color=base.RED,
        capsize=4,
    )
    ax.axvline(0, color=base.INK, linewidth=1)
    ax.axvline(5, color=base.GOLD, linestyle="--", linewidth=1)
    ax.set_yticks(y, [f"训练 seed {label}" for label in labels])
    ax.set_xlabel("冻结规则相对规范视角增益（百分点）")
    ax.set_title("训练 seed 复现", loc="left", fontweight="bold")
    clean(ax)
    ax = fig.add_axes([0.55, 0.46, 0.38, 0.31])
    method = gate["best_rule_frozen_on_calibration"]
    for seed, color in zip(("41", "42", "43"), (base.BLUE, base.TEAL, base.RED), strict=True):
        curve = heldout["noise_convergence"][seed]
        xs = sorted(int(value) for value in curve)
        ys = [curve[str(value)][method] * 100 for value in xs]
        ax.plot(xs, ys, marker="o", color=color, label=f"seed {seed}")
    ax.set_xlabel("每条件累计噪声轨迹数")
    ax.set_ylabel("冻结规则成功率 %")
    ax.set_title("4/8/16/32/64 前缀收敛", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    clean(ax)
    table(
        fig,
        ["检查项", "结果", "正式含义"],
        [
            ["跨 seed 平均增益", f"{gate['cross_checkpoint_mean_gain_pp']:+.2f}pp", "实用门槛为 +5pp"],
            ["方向一致", str(gate["direction_consistent_positive"]), "三个训练 seed 均需为正"],
            ["各 seed CI 排除 0", str(gate["each_checkpoint_ci_excludes_zero"]), "source-level 配对区间"],
            ["Harm", f"{gate['cross_checkpoint_mean_harm_probability'] * 100:.2f}%", "不得超过 5%"],
            ["最终精度", str(gate["final_precision_halfwidth_at_most_5pp"]), "F 后仍不足则只能报不确定"],
        ],
        [0.055, 0.17, 0.89, 0.22],
        [0.26, 0.22, 0.52],
        9,
    )
    note(fig, f"正式选择器门：{gate['status']}；冻结规则：{METHOD_LABELS.get(method, method)}。")
    save_page(pdf, fig, 5, preview)


def page_accel_decision(
    pdf: PdfPages,
    preview: Path,
    accel_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    calibration: Mapping[str, Any],
    heldout: Mapping[str, Any],
) -> None:
    fig, _canvas = base.new_page(
        "Accel 关系与最终裁决",
        "Accel 使用八个共享 flow-noise 对 97 个视角求 mean accel_3；它是一个冻结基线，不获得测试结果反馈。",
        6,
    )
    groups = Counter(candidate_group(row["selected_candidate_id"]) for row in accel_rows)
    labels = ["canonical", "train-support", "held-out"]
    values = [groups[label] for label in labels]
    ax = fig.add_axes([0.07, 0.48, 0.36, 0.29])
    bars = ax.bar(
        ["规范", "训练支持", "留出视角"],
        values,
        color=[base.BLUE, base.TEAL, base.GOLD],
    )
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("被 Accel 选中的状态数")
    ax.set_title("64 个固定状态的 Top-1 类型", loc="left", fontweight="bold")
    clean(ax)
    margins = np.asarray([row["top2_margin"] for row in accel_rows])
    text(fig, 0.50, 0.75, "Accel 八噪声排名审计", 15, weight="bold")
    text(
        fig,
        0.50,
        0.68,
        f"候选：97 / 状态\n状态：{len(accel_rows)}\nTop-2 均值间隔中位数：{np.median(margins):.5f}\n"
        "作用：比较低曲率偏好与真实闭环价值是否一致。",
        11.5,
    )
    gate = heldout["selector_population_gate"]
    table(
        fig,
        ["问题", "正式结果", "可写结论"],
        [
            ["候选池存在稳定 Headroom？", calibration["status"], "D 银行独立确认"],
            ["冻结规则能跨来源选出收益？", gate["status"], "以三训练 seed 与配对 CI 裁决"],
            ["是否可把 Accel 当 view value？", "见 held-out Accel 规则", "只能由闭环增益决定，不能由排名本身决定"],
            ["综合状态", decision["status"], "限定于固定状态 E0 反事实视角"],
        ],
        [0.055, 0.19, 0.89, 0.23],
        [0.30, 0.35, 0.35],
        8.6,
    )
    note(
        fig,
        "论文作用：把“候选中偶尔有成功视角”升级为噪声边缘化的期望检验，并区分候选空间、可部署选择规则与主动相机获取。",
    )
    save_page(pdf, fig, 6, preview)


def write_markdown(
    path: Path,
    decision: Mapping[str, Any],
    calibration: Mapping[str, Any],
    heldout: Mapping[str, Any],
) -> None:
    gate = heldout["selector_population_gate"]
    lines = [
        "# 视角价值期望正式复验",
        "",
        "## 一句话结论",
        "",
        f"**{decision['status']}**",
        "",
        "本结论仅覆盖固定物理状态下的 97 个反事实外部视角，"
        "不等同于官方 LIBERO-Plus 全量成绩，也不证明物理主动相机获取有效。",
        "",
        "## 正式门",
        "",
        "| 检验 | 结果 |",
        "|---|---|",
        f"| 校准候选 Headroom | `{calibration['status']}` |",
        f"| Held-out 选择器 | `{gate['status']}` |",
        f"| 冻结规则 | `{gate['best_rule_frozen_on_calibration']}` |",
        f"| 三 checkpoint 平均增益 | `{gate['cross_checkpoint_mean_gain_pp']:+.2f}pp` |",
        f"| 平均 Harm | `{gate['cross_checkpoint_mean_harm_probability'] * 100:.2f}%` |",
        f"| 每条件噪声轨迹 | `{heldout['noise_repeats_per_condition']}` |",
        "",
        "## 三训练 Seed",
        "",
        "| Seed | 成功率 | 相对规范增益 | 95% CI | Harm |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in gate["seed_results"]:
        low, high = row["success_gain_ci_95_pp"]
        lines.append(
            f"| {row['checkpoint_seed']} | {row['success_rate'] * 100:.2f}% | "
            f"{row['success_gain_pp']:+.2f}pp | [{low:+.2f}, {high:+.2f}] | "
            f"{row['harm_probability'] * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 统计边界",
            "",
            "- 独立单位是 source demonstration group；noise repeat 是状态内嵌套样本。",
            "- 所有视角在同一 state/repeat/replan 下共享逐元素相同的显式 flow noise。",
            "- 测试规则只由校准集冻结，不使用 held-out 闭环结果、未来状态或人工路由。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_metrics_csv(path: Path, heldout: Mapping[str, Any]) -> None:
    rows = []
    for seed, methods in heldout["checkpoint_seeds"].items():
        for method, values in methods.items():
            rows.append(
                {
                    "checkpoint_seed": seed,
                    "selector_method": method,
                    "success_rate": values["success_rate"],
                    "success_gain_pp": values["success_gain_pp"],
                    "gain_ci_low_pp": values["success_gain_task_stratified_bootstrap_95_pp"][0],
                    "gain_ci_high_pp": values["success_gain_task_stratified_bootstrap_95_pp"][1],
                    "harm_probability": values["harm_probability"],
                    "rescue_probability": values["rescue_probability"],
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    calibration = load_json(root / "calibration/analysis/analysis.json")
    final_analysis = root / "heldout/analysis-final/primary-analysis.json"
    heldout_path = final_analysis if final_analysis.exists() else root / "heldout/analysis-primary/primary-analysis.json"
    heldout = load_json(heldout_path)
    population = load_json(root / "population/population.json")
    accel_rows = load_jsonl(str(root / "accel-ensemble/rank-shard-*.jsonl"))
    if len(accel_rows) != 64:
        raise ValueError(f"expected 64 Accel states, found {len(accel_rows)}")
    decision = final_decision(calibration, heldout)
    output.mkdir(parents=True, exist_ok=True)
    preview = output / "previews"
    atomic_json(output / "final_decision.json", decision)
    write_markdown(output / "view_value_expectation_final_zh.md", decision, calibration, heldout)
    write_metrics_csv(output / "heldout_metrics.csv", heldout)
    base.configure_font()
    pdf_path = output / "view_value_expectation_final_zh.pdf"
    with PdfPages(pdf_path) as pdf:
        page_overview(pdf, preview, population, calibration, heldout, decision)
        page_design(pdf, preview, root, population)
        page_calibration(pdf, preview, calibration)
        page_heldout(pdf, preview, heldout)
        page_cross_seed(pdf, preview, heldout)
        page_accel_decision(pdf, preview, accel_rows, decision, calibration, heldout)
    print(
        json.dumps(
            {
                "status": "PASS",
                "decision": decision["status"],
                "pdf": str(pdf_path),
                "analysis": str(heldout_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
