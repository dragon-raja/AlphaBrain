#!/usr/bin/env python3
"""Build the stage-one Chinese brief as one continuous evidence chain."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

import build_accel_expanded_brief_pdf as expanded
import build_view_revalidation_brief_pdf as base


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "docs/dsol_paper1/view_revalidation_stage1_integrated_v2_20260825_zh.pdf"
)

MODELS = expanded.MODELS
MODEL_LABELS = expanded.MODEL_LABELS
MODEL_SHORT = {
    "broad64_practical": "实用非配对",
    "broad64_state_matched": "状态匹配",
    "broad64_paired_fm": "配对 FM",
    "broad64_paired_consistency": "配对+一致性",
}
MODEL_COLORS = (base.BLUE, base.TEAL, base.GOLD, base.RED)


def _save(pdf, fig, preview_dir: Path | None, page_number: int) -> None:
    base.save_page(pdf, fig, page_number, preview_dir)


def _section_title(fig, x: float, y: float, title: str, color: str) -> None:
    fig.text(x, y, title, fontsize=10.5, fontweight="bold", color=color, va="top")


def _status_badge(fig, x: float, y: float, text: str, color: str) -> None:
    fig.text(
        x,
        y,
        text,
        fontsize=8.5,
        fontweight="bold",
        color=base.WHITE,
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.35,rounding_size=0.25",
            "facecolor": color,
            "edgecolor": color,
        },
    )


def page_accel_chain(pdf, preview_dir, page_number: int = 10) -> None:
    fig, canvas = base.new_page(
        "Accel 视角选择假设：统一证据链",
        "同一假设依次接受行为、测量、支持域、任务信息与跨任务一致性检验；任何一环失败都不能直接部署为选择器。",
        page_number,
    )
    base.box(canvas, 0.055, 0.755, 0.89, 0.07, face="#EDF3F7", edge="#C8D7E2")
    fig.text(0.075, 0.79, "研究问题", fontsize=9.5, fontweight="bold", color=base.BLUE, va="center")
    fig.text(
        0.18,
        0.79,
        "在同一物理状态的多个相机候选中，选择最低 Accel 的观测，是否能稳定找到更有任务价值的视角？",
        fontsize=10.5,
        color=base.INK,
        va="center",
    )

    gates = [
        (
            "A",
            "行为价值",
            "候选中是否存在\n更高闭环成功率？\nAccel 是否兑现？",
            "SR / Oracle@4",
            base.BLUE,
        ),
        (
            "B",
            "测量可靠性",
            "改变 flow noise 后\n视角排序是否\n仍可重复？",
            "Spearman / Top-1",
            base.TEAL,
        ),
        (
            "C",
            "支持域归因",
            "低分来自规范记忆、\n训练位姿，还是\n同范围留出位姿？",
            "命中率 / 富集倍数",
            base.GOLD,
        ),
        (
            "D",
            "任务信息对齐",
            "能否偏好高可见性，\n同时拒绝 Blind、\nLook-away 与黑屏？",
            "平均排名 / 信息差",
            base.RED,
        ),
        (
            "E",
            "关系可泛化性",
            "上述关系是否跨\n任务与操作阶段\n保持同一方向？",
            "task / stage 分解",
            base.GRAY,
        ),
    ]
    xs = np.linspace(0.055, 0.775, 5)
    for index, (letter, title, detail, metric, color) in enumerate(gates):
        x = float(xs[index])
        base.box(canvas, x, 0.39, 0.17, 0.29, face=base.WHITE, edge=base.LINE)
        canvas.add_patch(Rectangle((x, 0.62), 0.17, 0.06, transform=canvas.transAxes, color=color))
        fig.text(x + 0.018, 0.65, f"Gate {letter}", fontsize=9, fontweight="bold", color=base.WHITE, va="center")
        fig.text(x + 0.018, 0.585, title, fontsize=11, fontweight="bold", color=base.INK, va="top")
        fig.text(x + 0.018, 0.535, detail, fontsize=9.0, color=base.INK, va="top", linespacing=1.45)
        fig.text(x + 0.018, 0.415, metric, fontsize=8.0, color=color, fontweight="bold", va="bottom")
        if index < 4:
            base.process_arrow(canvas, x + 0.172, 0.535, float(xs[index + 1]) - 0.008)

    base.box(canvas, 0.055, 0.235, 0.89, 0.09, face="#FFF7E8", edge="#E6D2A8")
    fig.text(0.075, 0.28, "裁决规则", fontsize=9.5, fontweight="bold", color=base.GOLD, va="center")
    fig.text(
        0.17,
        0.28,
        "直接视角选择至少要求：候选有行为空间、分数跨噪声稳定、偏好与任务信息同向、并在任务/阶段间可复现。",
        fontsize=10.0,
        color=base.INK,
        va="center",
    )
    base.box(canvas, 0.055, 0.095, 0.89, 0.09, face="#FFF2F3", edge="#E6C5CA")
    fig.text(0.075, 0.14, "当前结论", fontsize=9.5, fontweight="bold", color=base.RED, va="center")
    fig.text(
        0.17,
        0.14,
        "后续六页按 Gate 顺序给证据。它们不是两套实验，而是同一选择假设的行为层、测量层和归因层。",
        fontsize=10.0,
        color=base.INK,
        va="center",
    )
    _save(pdf, fig, preview_dir, page_number)


def page_gate_a_behavior(pdf, preview_dir, page_number: int = 11) -> None:
    fig, canvas = base.new_page(
        "Gate A：候选视角有行为收益空间，但 Accel 未兑现",
        "21 个冻结中间状态；四个角色视角都运行完整闭环。该页回答行为问题，所有百分比都是闭环成功率。",
        page_number,
    )
    base.box(canvas, 0.055, 0.75, 0.89, 0.075, face="#EDF3F7", edge="#C8D7E2")
    fig.text(
        0.075,
        0.787,
        "指标：SR_accel = 最低 Accel 视角的闭环成功率｜SR_can = 规范视角成功率｜Oracle@4 = 四个候选中至少一个成功的比例｜选择增益 = SR_accel − SR_can",
        fontsize=9.0,
        color=base.INK,
        va="center",
    )

    selected = np.asarray([57.1, 57.1, 47.6, 57.1])
    canonical = np.asarray([57.1, 52.4, 52.4, 57.1])
    oracle = np.asarray([85.7, 76.2, 71.4, 76.2])
    x = np.arange(len(MODELS))
    ax = fig.add_axes([0.07, 0.35, 0.56, 0.35])
    width = 0.24
    ax.bar(x - width, canonical, width, color=base.BLUE, label="规范视角 SRc")
    ax.bar(x, selected, width, color=base.RED, label="Accel 所选 SR_accel")
    ax.bar(x + width, oracle, width, color=base.TEAL, label="Oracle@4 上限")
    ax.set_ylim(0, 100)
    ax.set_ylabel("完整闭环成功率（%）")
    ax.set_xticks(x, [MODEL_SHORT[key] for key in MODELS], rotation=8)
    ax.grid(axis="y", color=base.LINE)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    for offset, values in ((-width, canonical), (0, selected), (width, oracle)):
        for index, value in enumerate(values):
            ax.text(index + offset, value + 2, f"{value:.1f}", ha="center", fontsize=7.7, color=base.INK)

    base.box(canvas, 0.665, 0.35, 0.28, 0.35)
    _section_title(fig, 0.69, 0.665, "如何读这张图", base.BLUE)
    fig.text(
        0.69,
        0.62,
        "1. 绿色高于蓝色：候选池中确实有更好视角。\n\n"
        "2. 红色若高于蓝色：Accel 把候选空间转化为收益。\n\n"
        "3. 实测选择增益为 0、+4.8、−4.8、0pp；均远小于绿色留下的 19.0–28.6pp 空间。",
        fontsize=9.2,
        color=base.INK,
        va="top",
        linespacing=1.45,
    )

    base.box(canvas, 0.055, 0.15, 0.89, 0.12, face="#FFF7E8", edge="#E6D2A8")
    _status_badge(fig, 0.12, 0.21, "候选空间：PASS", base.TEAL)
    _status_badge(fig, 0.295, 0.21, "选择转化：FAIL", base.RED)
    fig.text(
        0.405,
        0.225,
        "按 6 条 source demonstration 等权后的预注册差值仅 −2.5pp 至 +3.3pp，\n"
        "因此状态级 +4.8pp 不能解释为稳定行为增益。",
        fontsize=9.4,
        color=base.INK,
        va="center",
        linespacing=1.4,
    )
    fig.text(
        0.055,
        0.095,
        "本页结论：问题不是“候选池没有好视角”，而是最低 Accel 没有把已存在的行为上限可靠选出来。",
        fontsize=10.2,
        fontweight="bold",
        color=base.RED,
    )
    _save(pdf, fig, preview_dir, page_number)


def page_gate_b_reliability(pdf, preview_dir, summary: dict, page_number: int = 12) -> None:
    fig, canvas = base.new_page(
        "Gate B：单次 Accel 排名不可重复，只能做多噪声诊断",
        "96 个冻结 test 状态 × 97 个正常候选 × 6 个共享 flow-noise seed；每个噪声下重新得到完整视角排序。",
        page_number,
    )
    base.box(canvas, 0.055, 0.74, 0.89, 0.085, face="#EDF3F7", edge="#C8D7E2")
    fig.text(
        0.075,
        0.798,
        "指标定义",
        fontsize=9.2,
        fontweight="bold",
        color=base.BLUE,
        va="center",
    )
    fig.text(
        0.15,
        0.798,
        "排名 Spearman：两次 97 视角完整排序的相关性（1=完全一致，0=无关）｜Top-1 全同：同一精确位姿在 6 次中都获胜｜间隔：ensemble 第一、二名分差",
        fontsize=8.6,
        color=base.INK,
        va="center",
    )

    models = summary["models"]
    x = np.arange(len(MODELS))
    rho = np.asarray([models[key]["mean_pairwise_rank_spearman"] for key in MODELS])
    exact = 100 * np.asarray([models[key]["all_seed_top1_agreement_rate"] for key in MODELS])
    category = 100 * np.asarray([models[key]["all_seed_category_agreement_rate"] for key in MODELS])
    margin = 100 * np.asarray([models[key]["mean_ensemble_top1_relative_margin"] for key in MODELS])

    ax = fig.add_axes([0.07, 0.37, 0.39, 0.29])
    bars = ax.bar(x, rho, color=MODEL_COLORS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("平均 Spearman ρ")
    ax.set_xticks(x, [MODEL_SHORT[key] for key in MODELS], rotation=10)
    ax.grid(axis="y", color=base.LINE)
    ax.set_title("完整排序跨噪声一致性", loc="left", fontsize=10.5, fontweight="bold")
    for bar, value in zip(bars, rho):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=8)

    ax = fig.add_axes([0.54, 0.37, 0.39, 0.29])
    width = 0.35
    ax.bar(x - width / 2, exact, width, color=base.RED, label="精确 Top-1 六次全同")
    ax.bar(x + width / 2, category, width, color=base.BLUE, label="支持区域六次全同")
    ax.set_ylim(0, 15)
    ax.set_ylabel("状态比例（%）")
    ax.set_xticks(x, [MODEL_SHORT[key] for key in MODELS], rotation=10)
    ax.grid(axis="y", color=base.LINE)
    ax.set_title("赢家是否可重复", loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.6)

    headers = ["模型", "ρ", "精确 Top-1 全同", "区域全同", "ensemble 前二间隔"]
    rows = [
        [MODEL_SHORT[key], f"{rho[i]:.3f}", f"{exact[i]:.1f}%", f"{category[i]:.1f}%", f"{margin[i]:.2f}%"]
        for i, key in enumerate(MODELS)
    ]
    ax = fig.add_axes([0.07, 0.18, 0.86, 0.13])
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.35)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(base.LINE)
        cell.set_facecolor(base.INK if row == 0 else (base.WHITE if row % 2 else "#EEF2F5"))
        if row == 0:
            cell.get_text().set_color(base.WHITE)
            cell.get_text().set_fontweight("bold")

    fig.text(
        0.055,
        0.105,
        "本页结论：排序相关性仅 0.287–0.430，精确赢家六次全同仅 2.1–4.2%；单次 argmin 主要受 flow noise 影响。后续 Accel 必须做多噪声 ensemble，且仍只可作为诊断分数。",
        fontsize=9.7,
        fontweight="bold",
        color=base.RED,
    )
    _save(pdf, fig, preview_dir, page_number)


def _category_enrichment(summary: dict, model: str, category: str) -> float:
    base_rate = {
        "canonical": 1 / 97,
        "broad64_training_support": 64 / 97,
        "broad32_heldout": 32 / 97,
    }[category]
    selected = summary["models"][model]["ensemble_selected_category_rates"][category]
    return selected / base_rate


def page_gate_c_support(pdf, preview_dir, summary: dict, page_number: int = 13) -> None:
    fig, canvas = base.new_page(
        "Gate C：低-Accel 域可越过训练位姿，但规范视角仍是强吸引点",
        "97 个正常候选由 1 个规范位姿、64 个训练精确位姿和 32 个同范围留出精确位姿组成；测试物理状态全部独立留出。",
        page_number,
    )
    base.box(canvas, 0.055, 0.735, 0.89, 0.09, face="#EDF3F7", edge="#C8D7E2")
    fig.text(
        0.075,
        0.795,
        "指标：Top-1 占比 = 96 个状态中 ensemble 最低 Accel 落入该组的比例；富集倍数 Eg = Top-1 占比 ÷ 该组候选数占比。Eg=1 表示按候选数量随机命中。",
        fontsize=8.8,
        color=base.INK,
        va="center",
    )
    fig.text(
        0.075,
        0.755,
        "注意：“训练支持”只表示该相机位姿进入过全局训练 catalog；“留出”表示精确位姿未训练但仍在同一宽范围内，不是极端 OOD。",
        fontsize=8.2,
        color=base.MUTED,
        va="center",
    )

    categories = ("canonical", "broad64_training_support", "broad32_heldout")
    labels = ("规范 1/97", "训练支持 64/97", "同范围留出 32/97")
    colors = (base.BLUE, base.TEAL, base.GOLD)
    x = np.arange(len(MODELS))
    ax = fig.add_axes([0.07, 0.37, 0.51, 0.29])
    bottom = np.zeros(len(MODELS))
    for category, label, color in zip(categories, labels, colors):
        values = 100 * np.asarray(
            [summary["models"][model]["ensemble_selected_category_rates"][category] for model in MODELS]
        )
        bars = ax.bar(x, values, bottom=bottom, color=color, label=label)
        for bar, value, start in zip(bars, values, bottom):
            if value >= 8:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    start + value / 2,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=base.WHITE if color != base.GOLD else base.INK,
                )
        bottom += values
    ax.set_ylim(0, 100)
    ax.set_ylabel("ensemble Top-1 所属组（%）")
    ax.set_xticks(x, [MODEL_SHORT[key] for key in MODELS], rotation=8)
    ax.legend(frameon=False, fontsize=7.4, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_title("原始组命中率", loc="left", fontsize=10.5, fontweight="bold")

    enrich = np.asarray(
        [[_category_enrichment(summary, model, category) for category in categories] for model in MODELS]
    )
    ax = fig.add_axes([0.64, 0.37, 0.29, 0.29])
    image = ax.imshow(enrich, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(np.arange(3), ["规范", "训练支持", "留出"])
    ax.set_yticks(np.arange(4), [MODEL_SHORT[key] for key in MODELS])
    ax.set_title("按候选数校正后的富集 Eg", loc="left", fontsize=10.5, fontweight="bold")
    for row in range(4):
        for column in range(3):
            text_value = f"{enrich[row, column]:.2f}×" if column else f"{enrich[row, column]:.1f}×"
            ax.text(column, row, text_value, ha="center", va="center", fontsize=8, color=base.INK)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    colorbar.set_label("Eg（1=按数量随机）", fontsize=8)

    base.box(canvas, 0.055, 0.16, 0.89, 0.13, face=base.WHITE)
    _section_title(fig, 0.08, 0.26, "读图与裁决", base.TEAL)
    fig.text(
        0.08,
        0.222,
        "• 规范视角虽然只有 1 个候选，却获得 21.9–33.3% 的 Top-1，富集 21–32×：存在强 canonical attractor。\n"
        "• 实用非配对与状态匹配的留出富集分别为 1.23×、1.07×：低-Accel 域能跨越训练精确位姿。\n"
        "• 配对 FM 与一致性留出富集降至 0.82×、0.76×：当前配对目标没有扩大该域。",
        fontsize=9.0,
        color=base.INK,
        va="top",
        linespacing=1.35,
    )
    fig.text(
        0.055,
        0.095,
        "本页结论：Broad practical 显示同范围视角兼容性泛化，但“能兼容留出位姿”仍不等于“能识别更有任务信息的位姿”。",
        fontsize=9.8,
        fontweight="bold",
        color=base.GOLD,
    )
    _save(pdf, fig, preview_dir, page_number)


def page_gate_d_semantics(pdf, preview_dir, summary: dict, page_number: int = 14) -> None:
    fig, canvas = base.new_page(
        "Gate D：Accel 能拒绝明显坏观测，但没有对齐任务信息价值",
        "在同一冻结状态内加入最高可见性、等位移控制、Blind、Look-away 和相机黑屏；比较它们在完整候选中的 ensemble 排名。",
        page_number,
    )
    base.box(canvas, 0.055, 0.75, 0.89, 0.075, face="#EDF3F7", edge="#C8D7E2")
    fig.text(
        0.075,
        0.787,
        "指标：平均排名越低越受 Accel 偏好（约 104 个候选）；信息差 Δrank = rank(最高可见性) − rank(等位移控制)，负值才表示任务信息视角更受偏好。",
        fontsize=9.0,
        color=base.INK,
        va="center",
    )
    roles = (
        "canonical",
        "strong_info",
        "matched_control",
        "blind",
        "look_away",
        "external_blackout",
        "wrist_blackout",
        "all_camera_blackout",
    )
    role_labels = ("规范", "最高可见性", "等位移控制", "Blind", "Look-away", "外部黑屏", "腕部黑屏", "全部黑屏")
    values = np.asarray(
        [[summary["models"][model]["mean_role_ranks"][role] for role in roles] for model in MODELS]
    )
    ax = fig.add_axes([0.13, 0.35, 0.78, 0.33])
    image = ax.imshow(values, cmap="RdYlGn_r", vmin=1, vmax=104, aspect="auto")
    ax.set_xticks(np.arange(len(roles)), role_labels, rotation=24, ha="right")
    ax.set_yticks(np.arange(len(MODELS)), [MODEL_SHORT[key] for key in MODELS])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{values[row, column]:.1f}",
                ha="center",
                va="center",
                fontsize=7.8,
                color=base.WHITE if values[row, column] > 75 else base.INK,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.022, pad=0.02)
    colorbar.set_label("平均排名（低=偏好）")

    gaps = values[:, 1] - values[:, 2]
    base.box(canvas, 0.055, 0.15, 0.42, 0.13, face="#EAF3F0", edge="#BFD8D0")
    _section_title(fig, 0.08, 0.25, "通过：坏观测拒绝", base.TEAL)
    fig.text(
        0.08,
        0.215,
        "Look-away 平均第 86–92；全部黑屏约第 101/104。\nAccel 对明显无效输入具有稳定的末位排序能力。",
        fontsize=9.2,
        color=base.INK,
        va="top",
        linespacing=1.35,
    )
    base.box(canvas, 0.525, 0.15, 0.42, 0.13, face="#FFF2F3", edge="#E6C5CA")
    _section_title(fig, 0.55, 0.25, "未通过：任务信息对齐", base.RED)
    fig.text(
        0.55,
        0.215,
        "四模型 Δrank = " + ", ".join(f"{gap:+.2f}" for gap in gaps) + "。\n"
        "仅一致性模型接近零，其他模型反而更偏好控制视角。",
        fontsize=9.2,
        color=base.INK,
        va="top",
        linespacing=1.35,
    )
    fig.text(
        0.055,
        0.095,
        "本页结论：Accel 测到的是“生成轨迹是否顺滑/输入是否明显失效”，不是“该观测是否增加了当前任务证据”。两者不能互换。",
        fontsize=9.7,
        fontweight="bold",
        color=base.RED,
    )
    _save(pdf, fig, preview_dir, page_number)


def _stage_from_pair_key(pair_key: str) -> str:
    return expanded.stage_from_pair_key(pair_key)


def _gap_matrix(rows: list[dict[str, str]], key_name: str) -> tuple[list[str], np.ndarray]:
    labels = sorted({row[key_name] if key_name == "task_id" else _stage_from_pair_key(row["pair_key"]) for row in rows})
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = row[key_name] if key_name == "task_id" else _stage_from_pair_key(row["pair_key"])
        grouped[(row["model"], key)].append(
            float(row["mean_rank_strong_info"]) - float(row["mean_rank_matched_control"])
        )
    matrix = np.asarray(
        [[float(np.mean(grouped[(model, label)])) for label in labels] for model in MODELS]
    )
    return labels, matrix


def page_gate_e_generality(pdf, preview_dir, rows: list[dict[str, str]], page_number: int = 15) -> None:
    fig, canvas = base.new_page(
        "Gate E：信息偏好在任务与阶段间翻转，不能形成统一规则",
        "把同一个 Δrank 指标按 8 个任务和轨迹四阶段拆开；每个 test 状态仍使用 6-noise ensemble。",
        page_number,
    )
    base.box(canvas, 0.055, 0.75, 0.89, 0.075, face="#EDF3F7", edge="#C8D7E2")
    fig.text(
        0.075,
        0.787,
        "读法：Δrank < 0 表示最高可见性视角优于等位移控制；Δrank > 0 表示控制视角反而更受 Accel 偏好。稳定方法应在多数任务/阶段保持同一负方向。",
        fontsize=9.0,
        color=base.INK,
        va="center",
    )
    tasks, task_gaps = _gap_matrix(rows, "task_id")
    task_labels = [expanded.TASK_LABELS.get(task, task) for task in tasks]
    limit = max(10, float(np.max(np.abs(task_gaps))))
    ax = fig.add_axes([0.08, 0.36, 0.57, 0.34])
    image = ax.imshow(task_gaps, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(tasks)), task_labels, rotation=35, ha="right", fontsize=7.1)
    ax.set_yticks(np.arange(len(MODELS)), [MODEL_SHORT[key] for key in MODELS], fontsize=8)
    ax.set_title("按任务分解的 Δrank", loc="left", fontsize=10.5, fontweight="bold")
    for row_index in range(task_gaps.shape[0]):
        for column in range(task_gaps.shape[1]):
            ax.text(column, row_index, f"{task_gaps[row_index, column]:+.0f}", ha="center", va="center", fontsize=6.8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.02)
    colorbar.set_label("负=信息视角更优")

    stages, stage_gaps = _gap_matrix(rows, "stage")
    ax = fig.add_axes([0.72, 0.36, 0.21, 0.34])
    for index, (model, color) in enumerate(zip(MODELS, MODEL_COLORS)):
        ax.plot(np.arange(len(stages)), stage_gaps[index], marker="o", color=color, linewidth=1.7, label=MODEL_SHORT[model])
    ax.axhline(0, color=base.INK, linewidth=1)
    ax.set_xticks(np.arange(len(stages)), [value.replace("stage-", "阶段") for value in stages], rotation=20)
    ax.set_title("按轨迹阶段分解", loc="left", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", color=base.LINE)
    ax.legend(frameon=False, fontsize=6.8)

    base.box(canvas, 0.055, 0.15, 0.89, 0.13, face=base.WHITE)
    _section_title(fig, 0.08, 0.25, "为什么这一步必要", base.GOLD)
    fig.text(
        0.08,
        0.215,
        "总体平均可能把“某些任务偏好信息视角、另一些任务偏好控制视角”抵消成接近零。任务热图和阶段曲线都出现正负翻转，\n"
        "说明当前关系依赖任务内容与操作进程；不能据全局均值写成通用视角价值规律。",
        fontsize=9.3,
        color=base.INK,
        va="top",
        linespacing=1.4,
    )
    fig.text(
        0.055,
        0.095,
        "本页结论：跨任务/阶段一致性未通过。下一轮闭环验证必须按任务均衡抽样，不能只挑 Accel 信号最强的状态。",
        fontsize=9.7,
        fontweight="bold",
        color=base.RED,
    )
    _save(pdf, fig, preview_dir, page_number)


def page_accel_decision(pdf, preview_dir, summary: dict, page_number: int = 16) -> None:
    fig, canvas = base.new_page(
        "Accel 方法裁决：保留为兼容性诊断，停止直接视角选择",
        "把前五个 Gate 合并；“识别坏观测”与“选择高价值观测”在证据上被明确分开。",
        page_number,
    )
    headers = ["Gate", "检验问题", "关键证据", "裁决"]
    rows = [
        ["A1", "候选有无行为空间", "Oracle@4 比 canonical 高 19.0–28.6pp", "PASS"],
        ["A2", "Accel 是否兑现空间", "选择增益 −4.8 至 +4.8pp；宏平均不稳定", "FAIL"],
        ["B", "跨 flow noise 可重复", "ρ=0.287–0.430；精确全同 2.1–4.2%", "FAIL"],
        ["C", "兼容域能否到留出位姿", "practical/state 留出富集 1.23×/1.07×", "PARTIAL"],
        ["D1", "能否拒绝坏观测", "Look-away 与全黑稳定接近末位", "PASS"],
        ["D2", "能否偏好任务信息", "信息−控制排名差不稳定且多为正", "FAIL"],
        ["E", "能否跨任务/阶段复现", "任务与阶段均发生方向翻转", "FAIL"],
    ]
    ax = fig.add_axes([0.055, 0.40, 0.89, 0.39])
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left", colLoc="left", colWidths=[0.07, 0.25, 0.52, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.48)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(base.LINE)
        if row == 0:
            cell.set_facecolor(base.INK)
            cell.get_text().set_color(base.WHITE)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(base.WHITE if row % 2 else "#EEF2F5")
            if column == 3:
                value = rows[row - 1][3]
                color = base.TEAL if value == "PASS" else (base.GOLD if value == "PARTIAL" else base.RED)
                cell.get_text().set_color(color)
                cell.get_text().set_fontweight("bold")

    relation = summary["cross_model_ensemble_relations"][
        "broad64_state_matched__vs__broad64_paired_fm"
    ]["mean_ensemble_rank_spearman"]
    consistency = summary["models"]["broad64_paired_consistency"]
    diagnoses = [
        (
            "Broad practical",
            "Camera Full 最强；留出富集 1.23×。\n宽覆盖扩大了被动兼容域。",
            base.TEAL,
        ),
        (
            "Pairing + 普通 FM",
            f"与状态匹配的排序相关性 {relation:.3f}。\npair identity 未形成独立结构。",
            base.GOLD,
        ),
        (
            "当前 consistency",
            f"离散度 {100 * consistency['mean_accel_relative_spread']:.2f}%，前二间隔 {100 * consistency['mean_ensemble_top1_relative_margin']:.2f}%。\n主要是压平，而非信息对齐。",
            base.RED,
        ),
    ]
    for index, (title, detail, color) in enumerate(diagnoses):
        x = 0.055 + 0.30 * index
        base.box(canvas, x, 0.18, 0.285, 0.14, face=base.WHITE)
        _section_title(fig, x + 0.02, 0.29, title, color)
        fig.text(x + 0.02, 0.252, detail, fontsize=8.3, color=base.INK, va="top", linespacing=1.35)

    base.box(canvas, 0.055, 0.075, 0.89, 0.06, face="#FFF2F3", edge="#E6C5CA")
    _status_badge(fig, 0.15, 0.105, "DIRECT SELECTOR：HOLD", base.RED)
    fig.text(
        0.29,
        0.105,
        "允许用途：6-noise ensemble 兼容性图谱与坏观测过滤。禁止用途：把单次最低 Accel 直接解释为主动视角价值。",
        fontsize=9.3,
        color=base.INK,
        va="center",
    )
    _save(pdf, fig, preview_dir, page_number)


def page_takeaway_v2(pdf, preview_dir, page_number: int = 18) -> None:
    fig, canvas = base.new_page(
        "第一阶段综合结论：训练覆盖有效，信息利用仍未解决",
        "被动相机鲁棒性、信息视角闭环信号与 Accel 诊断已经放回同一研究逻辑中。",
        page_number,
    )
    sections = [
        (
            "已经成立",
            base.TEAL,
            [
                "Broad64 practical：Camera Full 75.9→82.2%",
                "候选视角存在 19.0–28.6pp 行为上限",
                "Accel 能拒绝 Look-away 与黑屏",
            ],
        ),
        (
            "尚未成立",
            base.GOLD,
            [
                "多视角训练会自动利用新增任务证据",
                "same-state pairing 在普通 FM 下产生独特收益",
                "最低 Accel 等价于最高闭环视角价值",
            ],
        ),
        (
            "当前停止",
            base.RED,
            [
                "Accel 单次 argmin 主动选择",
                "当前 paired-consistency recipe",
                "KYC raw-ray 双相机扩展",
            ],
        ),
    ]
    for index, (title, color, bullets) in enumerate(sections):
        x = 0.055 + 0.31 * index
        base.box(canvas, x, 0.34, 0.27, 0.42)
        canvas.add_patch(Rectangle((x, 0.69), 0.27, 0.07, transform=canvas.transAxes, color=color))
        fig.text(x + 0.025, 0.725, title, fontsize=12, fontweight="bold", color=base.WHITE, va="center")
        y = 0.63
        for bullet in bullets:
            fig.text(x + 0.028, y, "•", fontsize=12, color=color, va="top")
            fig.text(x + 0.052, y, bullet, fontsize=9.6, color=base.INK, va="top", wrap=True)
            y -= 0.105

    base.box(canvas, 0.055, 0.19, 0.89, 0.085, face="#EAF3F0", edge="#BFD8D0")
    fig.text(0.075, 0.232, "下一步", fontsize=9.5, fontweight="bold", color=base.TEAL, va="center")
    fig.text(
        0.16,
        0.232,
        "用任务均衡的完整闭环 shortlist 联合检验“可见性增量 × 兼容性”，把视角是否可执行与是否有任务价值建模为两个轴。",
        fontsize=10.2,
        color=base.INK,
        va="center",
    )
    base.box(canvas, 0.055, 0.075, 0.89, 0.07, face="#FFF7E8", edge="#E6D2A8")
    fig.text(
        0.075,
        0.11,
        "一句话：宽视角数据解决了一部分“看得惯”，但当前模型与指标仍没有解决“知道哪一眼更值得看”。",
        fontsize=11,
        fontweight="bold",
        color=base.INK,
        va="center",
    )
    _save(pdf, fig, preview_dir, page_number)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()

    base.configure_font()
    summary = expanded.load_json(expanded.DEFAULT_ANALYSIS / "summary.json")
    rows = expanded.load_csv(expanded.DEFAULT_ANALYSIS / "state_stability.csv")
    if summary.get("status") != "PASS":
        raise ValueError("expanded Accel analysis is not PASS")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        base.page_cover(pdf, args.preview_dir)
        base.page_questions(pdf, args.preview_dir)
        base.page_data(pdf, args.preview_dir)
        base.page_training(pdf, args.preview_dir)
        base.page_passive(pdf, args.preview_dir)
        base.page_formal_benchmarks(pdf, args.preview_dir)
        base.page_m0(pdf, args.preview_dir)
        base.page_m1(pdf, args.preview_dir)
        base.page_m1_task_breakdown(pdf, args.preview_dir)
        page_accel_chain(pdf, args.preview_dir, 10)
        page_gate_a_behavior(pdf, args.preview_dir, 11)
        page_gate_b_reliability(pdf, args.preview_dir, summary, 12)
        page_gate_c_support(pdf, args.preview_dir, summary, 13)
        page_gate_d_semantics(pdf, args.preview_dir, summary, 14)
        page_gate_e_generality(pdf, args.preview_dir, rows, 15)
        page_accel_decision(pdf, args.preview_dir, summary, 16)
        base.page_kyc_cvc_boundary(pdf, args.preview_dir, 17)
        page_takeaway_v2(pdf, args.preview_dir, 18)


if __name__ == "__main__":
    main()
