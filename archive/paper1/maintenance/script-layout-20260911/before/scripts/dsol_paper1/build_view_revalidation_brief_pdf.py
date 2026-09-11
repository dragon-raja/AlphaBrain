#!/usr/bin/env python3
"""Build a compact Chinese PDF brief for the view revalidation program."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

import build_accel_expanded_brief_pdf as expanded


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "docs/dsol_paper1/figures"
PASSIVE_FIGURE = FIGURE_DIR / "view_revalidation_data_passive_interim.png"
MECHANISM_FIGURE = FIGURE_DIR / "view_revalidation_m0_m1_accel_interim.png"
ACCEL_AUDIT_FIGURE = FIGURE_DIR / "accel_prefix_selector_audit.png"
M0_MONTAGE = Path(
    "/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m0-v1/"
    "operational-three-task-scan-v2/manual_audit_renders_v1/states/"
    "1a785a92de084d0b/visibility_extremes.png"
)
M1_VIEW_ROLE_DIR = FIGURE_DIR / "m1_view_roles"
M1_VIEW_ROLE_IMAGES = {
    "canonical": M1_VIEW_ROLE_DIR / "canonical.png",
    "strong_info": M1_VIEW_ROLE_DIR / "strong_info.png",
    "matched_control": M1_VIEW_ROLE_DIR / "matched_control.png",
    "blind": M1_VIEW_ROLE_DIR / "blind.png",
}
FORMAL_ROOT = Path(
    "/share/longjunyu/alphabrain/experiments/dsol-view-revalidation-m-b-v1/"
)
CAMERA_FORMAL_METRICS = FORMAL_ROOT / "camera_full/multiseed_metrics.json"
ORIGINAL_FORMAL_METRICS = FORMAL_ROOT / "original_full/multiseed_metrics.json"
M1_CROSS_MODEL_METRICS = Path(
    "/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m1-v2/"
    "cross-model-analysis/metrics.json"
)
GATE_A97_ROOT = Path(
    "/share/longjunyu/alphabrain/experiments/dsol-accel-gate-a97-v1"
)
GATE_A97_MODELS = (
    "broad64-practical",
    "broad64-state-matched",
    "broad64-paired-fm",
    "broad64-paired-consistency",
)
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

BG = "#F7F8FA"
INK = "#17212B"
MUTED = "#63707D"
LINE = "#D9DFE5"
BLUE = "#3D7EA6"
TEAL = "#3F8C7A"
GOLD = "#C79432"
RED = "#BE5968"
GRAY = "#8A9299"
WHITE = "#FFFFFF"


def configure_font() -> None:
    font_manager.fontManager.addfont(str(FONT_PATH))
    family = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
        }
    )


def new_page(title: str, subtitle: str, page_number: int):
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=BG)
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_axis_off()
    canvas.add_patch(Rectangle((0, 0.965), 1, 0.035, transform=canvas.transAxes, color=INK))
    fig.text(0.055, 0.905, title, fontsize=24, fontweight="bold", color=INK, va="top")
    fig.text(0.057, 0.855, subtitle, fontsize=10.5, color=MUTED, va="top")
    fig.text(0.055, 0.025, "VLA View Generalization × Active-Ready Perception", fontsize=7.5, color=MUTED)
    fig.text(0.95, 0.025, f"{page_number:02d}", fontsize=8, color=MUTED, ha="right")
    return fig, canvas


def box(canvas, x, y, w, h, face=WHITE, edge=LINE, radius=0.008):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        transform=canvas.transAxes,
        linewidth=0.8,
        edgecolor=edge,
        facecolor=face,
    )
    canvas.add_patch(patch)
    return patch


def metric(fig, canvas, x, y, w, h, value, label, note, color):
    box(canvas, x, y, w, h)
    canvas.add_patch(Rectangle((x, y), 0.008, h, transform=canvas.transAxes, color=color))
    fig.text(x + 0.025, y + h - 0.045, value, fontsize=24, fontweight="bold", color=color, va="top")
    fig.text(x + 0.025, y + h - 0.105, label, fontsize=10.5, fontweight="bold", color=INK, va="top")
    fig.text(x + 0.025, y + 0.025, note, fontsize=8.5, color=MUTED, va="bottom")


def section_label(fig, x, y, text, color):
    fig.text(
        x,
        y,
        text,
        fontsize=8,
        fontweight="bold",
        color=WHITE,
        bbox={"boxstyle": "round,pad=0.35,rounding_size=0.3", "facecolor": color, "edgecolor": color},
        va="center",
    )


def process_step(fig, canvas, x, y, w, h, number, title, detail, color):
    box(canvas, x, y, w, h, face=WHITE, edge=LINE)
    fig.text(x + 0.018, y + h - 0.032, number, fontsize=8.5, fontweight="bold", color=color, va="top")
    fig.text(x + 0.018, y + h - 0.065, title, fontsize=10.2, fontweight="bold", color=INK, va="top")
    fig.text(x + 0.018, y + h - 0.092, detail, fontsize=7.4, color=MUTED, va="top", linespacing=1.25)


def process_arrow(canvas, x1, y, x2):
    canvas.add_patch(
        FancyArrowPatch(
            (x1, y),
            (x2, y),
            transform=canvas.transAxes,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=GRAY,
        )
    )


def add_image(fig, image, rect, crop=None):
    source = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.copy()
    if crop is not None:
        width, height = source.size
        left, top, right, bottom = crop
        source = source.crop((int(left * width), int(top * height), int(right * width), int(bottom * height)))
    ax = fig.add_axes(rect)
    ax.imshow(source)
    ax.set_axis_off()
    return ax


def save_page(pdf, fig, page_number, preview_dir):
    pdf.savefig(fig, facecolor=BG, bbox_inches=None)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"page-{page_number:02d}.png", dpi=120, facecolor=BG)
    plt.close(fig)


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def page_cover(pdf, preview_dir):
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=BG)
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_axis_off()
    canvas.add_patch(Rectangle((0, 0), 0.025, 1, transform=canvas.transAxes, color=TEAL))
    fig.text(0.075, 0.86, "VLA 视角泛化与 Active-Ready 感知", fontsize=27, fontweight="bold", color=INK)
    fig.text(0.075, 0.79, "宽视角训练、信息视角与闭环利用重验证", fontsize=17, color=MUTED)
    section_label(fig, 0.078, 0.71, "第一阶段最终报告 · M-A / M-B 完成", TEAL)

    metric(fig, canvas, 0.075, 0.43, 0.25, 0.20, "75.9 → 82.2%", "Camera Full 成功率", "Broad64 三 seed 均值", TEAL)
    metric(fig, canvas, 0.365, 0.43, 0.25, 0.20, "+5.15pp", "按 base task 配对增益", "95% CI [+0.99, +9.62]", BLUE)
    metric(fig, canvas, 0.655, 0.43, 0.25, 0.20, "−4.75pp", "当前一致性方案", "相对 practical，CI [−7.85, −1.73]", RED)

    box(canvas, 0.075, 0.16, 0.83, 0.17, face="#EAF3F0", edge="#BFD8D0")
    fig.text(0.105, 0.285, "当前最稳妥结论", fontsize=10, fontweight="bold", color=TEAL, va="top")
    fig.text(
        0.105,
        0.235,
        "宽且真实的相机位姿覆盖稳定提高 Pi0.5 的被动相机鲁棒性；当前 paired-consistency 方案\n"
        "没有额外价值并明显损害基础能力。M1 显示信息利用方向性信号，Accel 仍不是可靠选择器。",
        fontsize=13,
        color=INK,
        va="top",
        linespacing=1.55,
    )
    fig.text(0.075, 0.045, "2026-08-24 · AlphaBrain / Pi0.5 / LIBERO-Plus", fontsize=8.5, color=MUTED)
    save_page(pdf, fig, 1, preview_dir)


def page_questions(pdf, preview_dir):
    fig, canvas = new_page("研究问题与证据链", "先验证 coverage，再检验 pairing，最后回答新增可见信息能否转化为闭环收益。", 2)
    items = [
        ("01", "Coverage", "宽、多样相机训练数据能否改善被动视角泛化？", "CONFIRMED · 3 SEEDS", TEAL),
        ("02", "Pairing", "当前 paired-consistency 训练方案是否优于 practical coverage？", "CURRENT RECIPE REJECTED", RED),
        ("03", "Information use", "新视角增加任务实体可见像素后，策略能否改善完整闭环？", "DEVELOPMENT SUPPORT", BLUE),
    ]
    y_values = [0.63, 0.41, 0.19]
    for y, (number, name, question, status, color) in zip(y_values, items):
        box(canvas, 0.065, y, 0.87, 0.155)
        fig.text(0.09, y + 0.115, number, fontsize=17, fontweight="bold", color=color, va="top")
        fig.text(0.16, y + 0.12, name, fontsize=14, fontweight="bold", color=INK, va="top")
        fig.text(0.16, y + 0.066, question, fontsize=11, color=MUTED, va="top")
        fig.text(0.79, y + 0.084, status, fontsize=9, fontweight="bold", color=color, ha="center")
    fig.text(0.065, 0.10, "证据顺序", fontsize=9, fontweight="bold", color=MUTED)
    fig.text(
        0.16,
        0.10,
        "Broad64 数据 → 训练对照 → Exact-state → Camera Full / Original Full → M0 → M1 → Accel",
        fontsize=10.5,
        color=INK,
    )
    save_page(pdf, fig, 2, preview_dir)


def page_data(pdf, preview_dir):
    fig, canvas = new_page("数据构造：从窄扰动升级为宽视角支持", "同一 simulator state 恢复，只改变相机位姿；episode 级划分避免状态和图像泄漏。", 3)
    metric(fig, canvas, 0.06, 0.61, 0.19, 0.18, "38,193", "same-state 记录", "400 源 episodes", BLUE)
    metric(fig, canvas, 0.27, 0.61, 0.19, 0.18, "8", "训练任务", "四个 LIBERO suite", TEAL)
    metric(fig, canvas, 0.48, 0.61, 0.19, 0.18, "31,440", "训练记录", "Val 2,701 / Test 4,052", GOLD)
    metric(fig, canvas, 0.69, 0.61, 0.19, 0.18, "2.7GB", "数据规模", "状态、图像、动作与哈希", RED)

    headers = ["视角集", "数量", "水平环绕角", "俯仰角", "距离比例", "角色"]
    rows = [
        ["Legacy narrow", "8", "±12°", "±7°", "0.94–1.06", "旧实验锚点"],
        ["Broad training", "64", "约 ±60°", "约 ±25°", "0.90–1.25", "正式训练"],
        ["Broad held-out", "32", "约 ±58°", "约 ±24°", "0.91–1.24", "同范围留出"],
        ["Wide extrapolation", "24", "约 -84°–79°", "约 -37°–39°", "0.75–1.50", "范围外泛化"],
        ["Extreme / blackout", "8+", "最高 ±180°", "-60°–45°", "0.50–2.00", "仅评测"],
    ]
    ax = fig.add_axes([0.06, 0.20, 0.87, 0.36])
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.65)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(LINE)
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(WHITE if row % 2 else "#EEF2F5")
    fig.text(0.06, 0.125, "关键修正", fontsize=9, fontweight="bold", color=TEAL)
    fig.text(
        0.145,
        0.125,
        "Canonical-unique 与 exact-repeat 分离；practical unpaired、state-matched、paired FM、paired consistency 分离；极端无信息视角不进入动作训练。",
        fontsize=9.5,
        color=INK,
    )
    save_page(pdf, fig, 3, preview_dir)


def page_training(pdf, preview_dir):
    fig, canvas = new_page("训练对照：覆盖、重复、配对和一致性分别控制", "模型相同、更新步数相同、global batch 相同；只改变数据组织和一致性目标。", 4)
    headers = ["方法", "独立状态", "同状态跨视角", "显式一致性", "报告角色"]
    rows = [
        ["Canonical-unique", "高", "否", "否", "训练控制"],
        ["Canonical-repeat", "低，精确重复", "否", "否", "重复效应控制"],
        ["Image-Aug unique", "高", "否", "否", "增强基线"],
        ["Broad practical", "高", "否", "否", "主模型候选"],
        ["Broad state-matched", "与 paired 相同", "否", "否", "数据组织消融"],
        ["Broad paired FM", "与 paired 相同", "是", "否", "数据组织消融"],
        ["Broad paired consistency", "与 paired 相同", "是", "是", "算法消融"],
    ]
    ax = fig.add_axes([0.055, 0.28, 0.89, 0.48])
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.7)
    table.scale(1, 1.65)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(LINE)
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(WHITE if row % 2 else "#EEF2F5")
            if row in (4, 7):
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_color(TEAL if row == 4 else BLUE)

    box(canvas, 0.06, 0.12, 0.87, 0.12, face="#EDF3F7", edge="#C8D7E2")
    fig.text(0.085, 0.205, "正式设置", fontsize=9, fontweight="bold", color=BLUE, va="top")
    fig.text(
        0.18,
        0.205,
        "Pi0.5 + PaliGemma 3B；冻结语言主干；视觉 rank-16 LoRA + action expert；2,000 updates；global batch 32；BF16；seeds 41/42/43。",
        fontsize=9.5,
        color=INK,
        va="top",
    )
    fig.text(
        0.18,
        0.155,
        "报告规则：先用 Camera Full 与 Original Full 选择有效主模型；未过基础能力门的方法只作消融，不进入主动视角主结论。",
        fontsize=9.5,
        color=MUTED,
        va="top",
    )
    save_page(pdf, fig, 4, preview_dir)


def page_passive(pdf, preview_dir):
    fig, canvas = new_page(
        "结果 1：宽视角覆盖显著改善被动相机泛化",
        "中图是 LIBERO 来源状态的自建诊断；右图才是 LIBERO-Plus 官方 Camera track，Pooled 为 1,599 条总分。",
        5,
    )
    add_image(fig, PASSIVE_FIGURE, [0.035, 0.20, 0.93, 0.59])
    box(canvas, 0.06, 0.08, 0.87, 0.085, face="#EAF3F0", edge="#BFD8D0")
    fig.text(0.085, 0.13, "结论", fontsize=9.5, fontweight="bold", color=TEAL, va="center")
    fig.text(
        0.15,
        0.13,
        "Exact-state 中 Broad practical 远高于 Canonical 与 Image-Aug；正式三 seed Camera Full 结果见下一页。",
        fontsize=10,
        color=INK,
        va="center",
    )
    save_page(pdf, fig, 5, preview_dir)


def page_formal_benchmarks(pdf, preview_dir, *, clarify_estimands=False):
    camera = load_json(CAMERA_FORMAL_METRICS)
    original = load_json(ORIGINAL_FORMAL_METRICS)
    fig, canvas = new_page(
        "结果 2：三 seed 正式基准完成",
        "Camera Full 测视角泛化；Original Full 测基础能力保持。差值按 40 个 base task 配对统计。",
        6,
    )

    labels = ["Official\nfrozen", "Broad64\npractical", "Paired +\nconsistency"]
    colors = [GRAY, TEAL, RED]
    camera_values = [
        100 * camera["baseline_success_rate"],
        100 * camera["methods"]["broad64_practical"]["cross_seed_mean_success_rate"],
        100 * camera["methods"]["broad64_paired_consistency"]["cross_seed_mean_success_rate"],
    ]
    original_values = [
        100 * original["baseline_success_rate"],
        100 * original["methods"]["broad64_practical"]["cross_seed_mean_success_rate"],
        100 * original["methods"]["broad64_paired_consistency"]["cross_seed_mean_success_rate"],
    ]

    ax = fig.add_axes([0.06, 0.41, 0.40, 0.35])
    x = range(3)
    bars = ax.bar(x, camera_values, color=colors, width=0.62)
    ax.set_title("A. LIBERO-Plus Camera Full", loc="left", fontweight="bold")
    ax.set_ylabel("闭环成功率 (%)")
    ax.set_xticks(list(x), labels, fontsize=8.5)
    ax.set_ylim(0, 103)
    style = {"color": INK, "fontsize": 9, "ha": "center", "va": "bottom"}
    for bar, value in zip(bars, camera_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.3, f"{value:.1f}%", **style)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)

    ax = fig.add_axes([0.54, 0.41, 0.40, 0.35])
    bars = ax.bar(x, original_values, color=colors, width=0.62)
    ax.set_title("B. Original LIBERO Full", loc="left", fontweight="bold")
    ax.set_ylabel("闭环成功率 (%)")
    ax.set_xticks(list(x), labels, fontsize=8.5)
    ax.set_ylim(0, 103)
    for bar, value in zip(bars, original_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.3, f"{value:.1f}%", **style)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)

    if clarify_estimands:
        fig.text(
            0.06, 0.365,
            "柱图：episode 汇总后对训练 seed 平均；下方差值：40 个任务等权配对，因此不等于柱高直接相减。",
            fontsize=9.3, color=MUTED,
        )

    metric(fig, canvas, 0.06, 0.16, 0.26, 0.18, "+5.15pp", "Practical：Camera 增益", "95% CI [+0.99, +9.62]", TEAL)
    metric(fig, canvas, 0.37, 0.16, 0.26, 0.18, "−2.53pp", "Practical：Original 退化", "95% CI [−4.05, −1.15]", GOLD)
    metric(fig, canvas, 0.68, 0.16, 0.26, 0.18, "−9.25pp", "Consistency vs practical", "Original，CI [−13.32, −5.80]", RED)
    fig.text(
        0.06,
        0.095,
        "裁决：Broad64 practical 通过预注册的相机增益门和 5pp retention 门；当前 paired-consistency 方案跨三个 seed 均更差。",
        fontsize=10.2,
        color=INK,
    )
    save_page(pdf, fig, 6, preview_dir)


def page_m0(pdf, preview_dir):
    fig, canvas = new_page(
        "结果 3：M0 先构造可控的观测差异",
        "M0 不评价策略优劣；它只回答：同一物理状态下，哪些视角增加任务实体可见像素，哪些只是普通换位姿？",
        7,
    )

    steps = [
        ("01", "冻结状态", "同一 MuJoCo state\n机器人与物体不变", BLUE),
        ("02", "渲染候选", "每状态 88 个视角\n含宽视角与极端负对照", TEAL),
        ("03", "实例分割", "统计任务实体\n在各相机的可见像素", GOLD),
        ("04", "计算 ΔI", "相对 canonical\n得到可见性增减", RED),
        ("05", "冻结四类", "Canonical / Info\nControl / Blind", BLUE),
    ]
    xs = [0.055, 0.245, 0.435, 0.625, 0.815]
    for x, step in zip(xs, steps):
        process_step(fig, canvas, x, 0.675, 0.135, 0.12, *step)
    for left, right in zip(xs[:-1], xs[1:]):
        process_arrow(canvas, left + 0.14, 0.735, right - 0.005)

    roles = [
        ("canonical", "规范视角", "ΔI = 0", BLUE),
        ("strong_info", "信息视角", "ΔI = +1.40pp", TEAL),
        ("matched_control", "位姿匹配对照", "ΔI = −0.02pp", GOLD),
        ("blind", "盲视角", "ΔI = −1.50pp", RED),
    ]
    role_xs = [0.055, 0.285, 0.515, 0.745]
    for x, (key, title, delta, color) in zip(role_xs, roles):
        box(canvas, x, 0.39, 0.20, 0.225)
        add_image(fig, M1_VIEW_ROLE_IMAGES[key], [x + 0.008, 0.465, 0.184, 0.132])
        canvas.add_patch(Rectangle((x, 0.39), 0.006, 0.225, transform=canvas.transAxes, color=color))
        fig.text(x + 0.018, 0.442, title, fontsize=9.5, fontweight="bold", color=INK, va="top")
        fig.text(x + 0.018, 0.411, delta, fontsize=8.2, color=color, va="top")

    fig.text(
        0.055,
        0.365,
        "真实同状态示例：每格左半为外部相机，右半为固定腕部相机。Info 与 Control 的相机移动幅度相近，但可见性增量不同。",
        fontsize=8.2,
        color=MUTED,
    )

    box(canvas, 0.055, 0.12, 0.56, 0.205, face="#EDF3F7", edge="#C8D7E2")
    fig.text(0.078, 0.295, "可见性定义", fontsize=10.2, fontweight="bold", color=BLUE, va="top")
    fig.text(
        0.078,
        0.252,
        r"$I_{task}(v)=\mathrm{mean}_{camera,entity}\;\frac{\mathrm{visible\ pixels}}{224\times224}$",
        fontsize=11.2,
        color=INK,
        va="top",
    )
    fig.text(0.078, 0.205, r"$\Delta I(v)=I_{task}(v)-I_{task}(canonical)$", fontsize=11.2, color=INK, va="top")
    fig.text(
        0.078,
        0.155,
        "Matched-control 还必须与 Strong-info 具有相近的相机平移和旋转，\n"
        "因此后续可以把“普通相机移动”与“新增任务可见信息”分开。",
        fontsize=8.5,
        color=MUTED,
        va="top",
        linespacing=1.35,
    )

    box(canvas, 0.645, 0.12, 0.30, 0.205)
    fig.text(0.67, 0.295, "筛选结果", fontsize=10.2, fontweight="bold", color=INK, va="top")
    fig.text(0.67, 0.252, "180 states / 15,840 candidates", fontsize=9.5, color=BLUE, va="top")
    fig.text(0.67, 0.215, "Crossed-orbit 最大 ΔI：+20.85pp", fontsize=8.8, color=INK, va="top")
    fig.text(0.67, 0.181, "Look-away / blackout 正增量：0%", fontsize=8.8, color=INK, va="top")
    fig.text(0.67, 0.145, "21 states 通过人工视觉审计进入 M1", fontsize=8.8, color=TEAL, va="top")
    save_page(pdf, fig, 7, preview_dir)


def page_m1(pdf, preview_dir):
    fig, canvas = new_page(
        "结果 4：M1 检验新增可见信息能否改善闭环 continuation",
        "从筛选后的中间状态恢复并执行到官方任务成功或超时；这不是从任务初始状态开始的标准 benchmark 成功率。",
        8,
    )

    steps = [
        ("01", "恢复同一状态", "21 个 frame states\n归属于 6 条独立演示", BLUE),
        ("02", "替换外部视角", "Canonical / Info /\nControl / Blind", TEAL),
        ("03", "双相机策略输入", "外部相机变化\n腕部相机正常动态更新", GOLD),
        ("04", "完整闭环", "Pi0.5 持续重规划\n直到成功或超时", RED),
    ]
    xs = [0.06, 0.285, 0.51, 0.735]
    for x, step in zip(xs, steps):
        process_step(fig, canvas, x, 0.67, 0.17, 0.125, *step)
    for left, right in zip(xs[:-1], xs[1:]):
        process_arrow(canvas, left + 0.175, 0.732, right - 0.005)

    payload = json.loads(M1_CROSS_MODEL_METRICS.read_text(encoding="utf-8"))
    model_order = ["official", "broad64-practical"]
    model_labels = ["Official（冻结参考）", "Broad64 practical（主模型）"]
    condition_order = [
        "canonical_both",
        "strong_info_both",
        "matched_control_both",
        "blind_both",
    ]
    condition_labels = ["规范", "信息", "位姿对照", "Blind"]
    condition_colors = [BLUE, TEAL, GOLD, RED]
    rates = {
        (row["model"], row["condition"]): 100 * float(row["state_success_rate"])
        for row in payload["condition_success"]
    }
    comparisons = {
        row["model"]: row
        for row in payload["within_model_comparisons"]
        if row["comparison"] == "information_specificity_both"
    }

    ax = fig.add_axes([0.06, 0.31, 0.53, 0.31])
    x = list(range(len(model_order)))
    width = 0.19
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for offset, condition, label, color in zip(
        offsets, condition_order, condition_labels, condition_colors
    ):
        values = [rates[(model, condition)] for model in model_order]
        ax.bar([value + offset for value in x], values, width=width, color=color, label=label)
    ax.set_title("主分析：两个有效参考 × 四种观测条件", loc="left", fontsize=10.2, fontweight="bold")
    ax.set_ylabel("闭环 continuation 成功率 (%)", fontsize=8.2)
    ax.set_xticks(x, model_labels, fontsize=8.0)
    ax.set_ylim(0, 82)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, loc="upper center", fontsize=7.3, frameon=False, bbox_to_anchor=(0.5, 1.17))

    box(canvas, 0.62, 0.31, 0.325, 0.31, face="#EAF3F0", edge="#BFD8D0")
    fig.text(0.642, 0.59, "主统计：信息是否优于等幅换视角", fontsize=10.2, fontweight="bold", color=TEAL, va="top")
    fig.text(0.642, 0.55, r"信息特异性 $=SR_{info}-SR_{control}$", fontsize=8.8, color=INK, va="top")
    y = 0.495
    for model, label in zip(model_order, model_labels):
        row = comparisons[model]
        value = float(row["difference_pp"])
        interval = f"[{float(row['ci_low_pp']):+.1f},{float(row['ci_high_pp']):+.1f}]"
        color = TEAL if row["ci_low_pp"] > 0 else (RED if row["ci_high_pp"] < 0 else INK)
        fig.text(0.642, y, label, fontsize=8.0, color=INK, va="top")
        fig.text(0.91, y, f"{value:+.1f}pp {interval}", fontsize=8.0, color=color, fontweight="bold", ha="right", va="top")
        y -= 0.055
    fig.text(
        0.642,
        0.385,
        "仅主模型的 CI 排除 0：新增可见信息\n相对等幅位姿对照有正向信号。",
        fontsize=8.3,
        color=INK,
        va="top",
        linespacing=1.35,
    )

    box(canvas, 0.06, 0.09, 0.885, 0.15, face="#FFF7E8", edge="#E6D2A8")
    fig.text(0.08, 0.21, "如何解释", fontsize=9.2, fontweight="bold", color=GOLD, va="top")
    fig.text(
        0.08,
        0.172,
        "• Broad64 practical 已由前页的正式 benchmark 门选为主模型，不是根据本页结果事后挑选。\n"
        "• 其信息视角成功率 71.4%，规范视角 57.1%，等幅位姿对照 52.4%；关键因果量 Info−Control 为 +13.3pp。\n"
        "• 其余训练组织已完成相同评测，但只用于附录稳健性审计；本页仍是中间状态 continuation，不是标准初始状态 benchmark。",
        fontsize=8.5,
        color=INK,
        va="top",
        linespacing=1.35,
    )
    save_page(pdf, fig, 8, preview_dir)


def page_m1_task_breakdown(pdf, preview_dir):
    fig, canvas = new_page(
        "结果 4B：分任务拆解，平均增益并不均匀",
        "以下仍是 selected-state closed-loop continuation；括号内为每个任务的中间状态数，不是独立任务 episode 数。",
        9,
    )

    task_labels = ["酒瓶放酒架\n(n=2)", "黑碗放抽屉\n(n=10)", "杯子放微波炉\n(n=9)", "三任务宏平均"]
    series = {
        "Canonical": ([0.0, 50.0, 77.8, 42.6], BLUE),
        "Strong-info": ([0.0, 90.0, 66.7, 52.2], TEAL),
        "Matched-control": ([0.0, 70.0, 44.4, 38.1], GOLD),
        "Blind": ([0.0, 0.0, 55.6, 18.5], RED),
    }
    ax = fig.add_axes([0.055, 0.37, 0.89, 0.40])
    x = list(range(len(task_labels)))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for offset, (name, (values, color)) in zip(offsets, series.items()):
        bars = ax.bar([value + offset for value in x], values, width=width, label=name, color=color)
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.7,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    color=INK,
                )
    ax.set_ylabel("闭环 continuation 成功率 (%)", fontsize=8.5)
    ax.set_xticks(x, task_labels, fontsize=8.3)
    ax.set_ylim(0, 103)
    ax.legend(ncol=4, loc="upper left", fontsize=7.8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)

    headers = ["任务", "Info − Canonical", "Info − Control", "解释"]
    rows = [
        ["酒瓶放酒架", "+0.0pp", "+0.0pp", "四条件全失败，不能判断视角价值"],
        ["黑碗放抽屉", "+40.0pp", "+20.0pp", "当前总增益的主要来源"],
        ["杯子放微波炉", "−11.1pp", "+22.2pp", "低于规范，但高于等幅位姿对照"],
        ["任务宏平均", "+9.6pp", "+14.1pp", "仅 3 个任务，不能外推为普遍结论"],
    ]
    table_ax = fig.add_axes([0.055, 0.17, 0.89, 0.16])
    table_ax.set_axis_off()
    table = table_ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.45)
    widths = [0.20, 0.17, 0.17, 0.46]
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(LINE)
        cell.set_linewidth(0.7)
        cell.set_width(widths[col])
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(WHITE if row % 2 else "#EEF2F5")
            if col in (1, 2):
                cell.get_text().set_fontweight("bold")

    box(canvas, 0.055, 0.075, 0.89, 0.065, face="#FFF7E8", edge="#E6D2A8")
    fig.text(0.075, 0.108, "审慎结论", fontsize=8.8, fontweight="bold", color=GOLD, va="center")
    fig.text(
        0.16,
        0.108,
        "现有正信号不是跨任务一致提升；它足以支持扩大验证，但不足以声称信息视角普遍提高完整任务成功率。",
        fontsize=8.8,
        color=INK,
        va="center",
    )
    save_page(pdf, fig, 9, preview_dir)


def page_m1_gate_a97_crosswalk(pdf, preview_dir, page_number=9):
    """Put the role-conditioned M1 and operational-bank Gate A97 results side by side."""
    fig, canvas = new_page(
        "范围对照：A97 不承担“信息视角有效”的因果验证",
        "使用同一个 seed41 checkpoint，但两套起始状态交集为 0；本页用于解释研究范围，不能把上下表的原始成功率直接比较。",
        page_number,
    )
    m1 = load_json(M1_CROSS_MODEL_METRICS)
    m1_rates = {
        (row["model"], row["condition"]): 100 * float(row["state_success_rate"])
        for row in m1["condition_success"]
    }
    m1_specificity = {
        row["model"]: row
        for row in m1["within_model_comparisons"]
        if row["comparison"] == "information_specificity_both"
    }
    m1_models = [
        ("official", "Official"),
        ("broad64-practical", "Broad practical"),
        ("broad64-state-matched", "State-matched"),
        ("broad64-paired-fm", "Paired FM"),
        ("broad64-paired-consistency", "Paired+consistency"),
    ]
    m1_rows = []
    for model, label in m1_models:
        specificity = m1_specificity[model]
        marker = " *" if specificity["ci_low_pp"] > 0 else ""
        m1_rows.append(
            [
                label,
                f"{m1_rates[(model, 'canonical_both')]:.1f}%",
                f"{m1_rates[(model, 'strong_info_both')]:.1f}%",
                f"{m1_rates[(model, 'matched_control_both')]:.1f}%",
                f"{m1_rates[(model, 'blind_both')]:.1f}%",
                f"{specificity['difference_pp']:+.1f}pp{marker}",
            ]
        )

    fig.text(0.055, 0.80, "A. M1 角色化视角：21 个筛选状态，3 个任务，6 个独立演示组", fontsize=10.5, fontweight="bold", color=TEAL)
    ax = fig.add_axes([0.055, 0.555, 0.89, 0.225])
    ax.set_axis_off()
    table = ax.table(
        cellText=m1_rows,
        colLabels=["模型", "规范", "Strong-info", "Matched-control", "Blind", "Info−Control"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1, 1.38)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(LINE)
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#EAF3F0" if row == 2 else (WHITE if row % 2 else "#EEF2F5"))
            if row == 2:
                cell.get_text().set_fontweight("bold")

    a97 = load_gate_a97_shortlist()
    gate_models = [
        ("broad64-practical", "Broad practical"),
        ("broad64-state-matched", "State-matched"),
        ("broad64-paired-fm", "Paired FM"),
        ("broad64-paired-consistency", "Paired+consistency"),
    ]
    gate_rows = []
    for model, label in gate_models:
        row = a97[model]
        success = row["condition_success"]
        gate_rows.append(
            [
                label,
                f"{100 * success['canonical']['state_success_rate']:.1f}%",
                f"{100 * success['visibility_top1']['state_success_rate']:.1f}%",
                f"{100 * success['accel_ensemble']['state_success_rate']:.1f}%",
                f"{100 * success['accel_top10_visibility']['state_success_rate']:.1f}%",
                f"{100 * success['random_operational']['state_success_rate']:.1f}%",
                f"{100 * row['oracle_at_shortlist_state_rate']:.1f}%",
            ]
        )

    fig.text(0.055, 0.515, "B. Gate A97 自动选择：96 个跨任务/阶段均衡状态（8 个任务）", fontsize=10.5, fontweight="bold", color=BLUE)
    ax = fig.add_axes([0.055, 0.305, 0.89, 0.19])
    ax.set_axis_off()
    table = ax.table(
        cellText=gate_rows,
        colLabels=["模型", "规范", "可见性最高", "Accel 六噪声", "Accel Top10+可见性", "随机", "Oracle@6"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.7)
    table.scale(1, 1.40)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(LINE)
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#EAF3F0" if row == 1 else (WHITE if row % 2 else "#EEF2F5"))
            if row == 1:
                cell.get_text().set_fontweight("bold")

    box(canvas, 0.055, 0.075, 0.89, 0.17, face="#FFF7E8", edge="#E6D2A8")
    fig.text(0.078, 0.215, "如何连接两张表", fontsize=9.4, fontweight="bold", color=GOLD, va="top")
    fig.text(
        0.078,
        0.18,
        "1. 没有额外训练：两表使用同一 Broad64 practical checkpoint；M1 与 A97 的状态交集为 0。\n"
        "2. M1：21 个信息阈值/人工审计状态，3 任务且分布为 2/10/9，平均轨迹阶段 25.9%；Strong-info 不使用 Accel。\n"
        "3. A97：8 任务 × 12 状态，平均轨迹阶段 48.8%；“可见性最高”不使用 Accel，只有 Top10+可见性是混合规则。\n"
        "4. M1 正信号集中于抽屉任务（50%→90%）；微波炉下降（77.8%→66.7%），酒架均为 0%，所以 A 没有建立普遍增益。\n"
        "5. A97 否定的是“在普通状态中直接最大化可见像素能普遍增益”；Paired FM 的规范视角也只是六种规则中最好，并非全部 97 个候选最优。",
        fontsize=7.8,
        color=INK,
        va="top",
        linespacing=1.28,
    )
    save_page(pdf, fig, page_number, preview_dir)


def page_accel_principle(pdf, preview_dir):
    fig, canvas = new_page(
        "Accel：原理、实现与复现边界",
        "核心曲率公式实现正确；本地跨视角排序是原文之外的扩展，不是原论文完整数值复现。",
        10,
    )
    steps = [
        ("1", "固定状态", "只改变相机观测", BLUE),
        ("2", "共享噪声", "候选使用同一 x0", TEAL),
        ("3", "记录轨迹", "10 个 velocity v_t", GOLD),
        ("4", "计算 Accel", "越低表示轨迹越直", RED),
        ("5", "本地扩展", "选择 argmin_v Accel", GRAY),
    ]
    x_values = [0.055, 0.235, 0.415, 0.595, 0.775]
    for index, (number, title, detail, color) in enumerate(steps):
        process_step(fig, canvas, x_values[index], 0.69, 0.15, 0.10, number, title, detail, color)
        if index < len(steps) - 1:
            process_arrow(canvas, x_values[index] + 0.153, 0.74, x_values[index + 1] - 0.008)

    box(canvas, 0.055, 0.53, 0.89, 0.10, face="#EDF3F7", edge="#C8D7E2")
    fig.text(0.08, 0.595, "离散定义", fontsize=10, fontweight="bold", color=BLUE, va="center")
    fig.text(
        0.205,
        0.605,
        r"$\mathrm{accel}_p = p\,\sum_{t=1}^{p-1}\|v_t-v_{t-1}\|_2\,/\,\sum_{t=0}^{p-1}\|v_t\|_2$",
        fontsize=15,
        color=INK,
        va="center",
    )
    fig.text(0.205, 0.542, "低分 = 该次动作生成更确定；不等于该视角更有任务价值。", fontsize=9, color=MUTED)

    audits = [
        (
            0.055,
            "一致项",
            TEAL,
            ["10-step Euler velocity trace", "原文 accel_p 公式", "单次生成，无重采样", "保存 p=2...10；候选共享 x0"],
        ),
        (
            0.365,
            "未复现项",
            RED,
            ["K=32 后验重采样相关性", "CUSUM + conformal 检测", "论文 TPR / detection lead", "原论文完整 benchmark 数值"],
        ),
        (
            0.675,
            "任务边界",
            GOLD,
            ["原文：动作不确定性/失败检测", "本地：固定状态跨视角选择", "argmin Accel 是新增假设", "负结果不反驳原文 detector"],
        ),
    ]
    for x, title, color, bullets in audits:
        box(canvas, x, 0.20, 0.27, 0.26)
        canvas.add_patch(Rectangle((x, 0.405), 0.27, 0.055, transform=canvas.transAxes, color=color))
        fig.text(x + 0.02, 0.432, title, fontsize=11, fontweight="bold", color=WHITE, va="center")
        y = 0.365
        for bullet in bullets:
            fig.text(x + 0.022, y, "•", fontsize=10, color=color, va="top")
            fig.text(x + 0.044, y, bullet, fontsize=8.8, color=INK, va="top")
            y -= 0.047

    box(canvas, 0.055, 0.09, 0.89, 0.065, face="#FFF7E8", edge="#E6D2A8")
    fig.text(0.075, 0.122, "审计结论", fontsize=9, fontweight="bold", color=GOLD, va="center")
    fig.text(
        0.16,
        0.122,
        "可称为“公式级实现 + 跨视角扩展实验”；不能称为“原论文完整复现”。",
        fontsize=10.2,
        color=INK,
        va="center",
    )
    save_page(pdf, fig, 10, preview_dir)


def page_accel_reliability(pdf, preview_dir, expanded_summary, page_number=12):
    """Explain the fixed-state ranking audit before showing behavior results."""
    fig, canvas = new_page(
        "闭环前先审计：Accel 的 97 视角排序是否可重复",
        "主模型的同一冻结状态、同一 97 视角，改变 6 次 flow 初始噪声；这里只测排名可靠性，不执行机器人动作。",
        page_number,
    )
    row = expanded_summary["models"]["broad64_practical"]
    spearman = row["mean_pairwise_rank_spearman"]
    top1 = row["all_seed_top1_agreement_rate"]
    margin = row["mean_ensemble_top1_relative_margin"]

    metric(
        fig,
        canvas,
        0.055,
        0.62,
        0.25,
        0.17,
        f"{spearman:.2f}",
        "排名 Spearman",
        "1=两次完整排序相同；0=无稳定次序",
        GOLD,
    )
    metric(
        fig,
        canvas,
        0.375,
        0.62,
        0.25,
        0.17,
        f"{100 * top1:.1f}%",
        "六次 Top-1 完全相同",
        "96 个状态中仅 2 个选中同一精确位姿",
        RED,
    )
    metric(
        fig,
        canvas,
        0.695,
        0.62,
        0.25,
        0.17,
        f"{100 * margin:.1f}%",
        "Ensemble 前二相对间隔",
        "第一名与第二名非常接近，选择易翻转",
        BLUE,
    )

    definitions = [
        (
            "排名 Spearman 怎么算",
            "任选两次噪声：\n"
            "对 97 个视角各自排序，\n"
            "再计算两份名次表相关性。\n\n"
            "0.42 = 中等偏弱一致性。",
            GOLD,
        ),
        (
            "Top-1 完全相同是什么意思",
            "同一状态重复 6 次：\n"
            "仅 2.1% 的状态六次都选中\n"
            "完全相同的精确相机位姿。\n\n"
            "单噪声 argmin 不可靠。",
            RED,
        ),
        (
            "前二间隔为什么重要",
            "先对 6 次 Accel 取均值，\n"
            "再比较第一、第二名分数。\n\n"
            "平均仅差 3.2%，说明\n"
            "大量候选近似并列。",
            BLUE,
        ),
    ]
    xs = [0.055, 0.365, 0.675]
    for x, (title, detail, color) in zip(xs, definitions):
        box(canvas, x, 0.30, 0.27, 0.245)
        canvas.add_patch(Rectangle((x, 0.49), 0.27, 0.055, transform=canvas.transAxes, color=color))
        fig.text(x + 0.018, 0.517, title, fontsize=9.6, fontweight="bold", color=WHITE, va="center")
        fig.text(x + 0.02, 0.455, detail, fontsize=8.5, color=INK, va="top", linespacing=1.35)

    box(canvas, 0.055, 0.09, 0.89, 0.14, face="#FFF7E8", edge="#E6D2A8")
    fig.text(0.08, 0.195, "这一页能下什么结论", fontsize=9.6, fontweight="bold", color=GOLD, va="top")
    fig.text(
        0.08,
        0.158,
        "Accel 对初始 flow noise 较敏感，因此正式闭环必须同时报告单噪声与六噪声均值选择。"
        "但排名不稳定不等于行为一定无效，下一页仍需真实执行所选视角。",
        fontsize=9.2,
        color=INK,
        va="top",
        linespacing=1.4,
    )
    save_page(pdf, fig, page_number, preview_dir)


def page_accel_results(pdf, preview_dir):
    fig, canvas = new_page(
        "Legacy Gate A：四个角色视角中的闭环选择",
        "这是早期 21 状态 × 4 角色视角结果；它提出问题，但不再承担 97 候选池的最终裁决。",
        11,
    )
    headers = ["模型", "选 Canonical", "所选成功", "Canonical", "任一候选", "差值"]
    rows = [
        ["Broad practical", "18/21", "57.1%", "57.1%", "85.7%", "0.0pp"],
        ["Broad state-matched", "16/21", "57.1%", "52.4%", "76.2%", "+4.8pp"],
        ["Broad paired FM", "15/21", "47.6%", "52.4%", "71.4%", "−4.8pp"],
        ["Broad paired consistency", "18/21", "57.1%", "57.1%", "76.2%", "0.0pp"],
    ]
    ax = fig.add_axes([0.06, 0.61, 0.88, 0.20])
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.42)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(LINE)
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(WHITE if row % 2 else "#EEF2F5")
    fig.text(
        0.06,
        0.575,
        "表中为 21 个状态的描述性比例；预注册主统计按 6 条 source demonstrations 等权，差值范围 −2.5pp 至 +3.3pp。",
        fontsize=8.2,
        color=MUTED,
    )

    add_image(fig, ACCEL_AUDIT_FIGURE, [0.04, 0.085, 0.63, 0.46])
    box(canvas, 0.69, 0.115, 0.255, 0.41)
    fig.text(0.715, 0.49, "如何读结果", fontsize=11, fontweight="bold", color=INK, va="top")
    observations = [
        (TEAL, "候选池有空间", "Broad practical 的任一候选成功率\n比 Canonical 高 28.6pp。"),
        (RED, "Accel 没转化空间", "所选成功率仍为 57.1%，\n没有得到行为收益。"),
        (GOLD, "后验扫参不成立", "个别前缀的正峰不跨模型稳定，\n且会随坐标尺度改变。"),
    ]
    y = 0.43
    for color, title, detail in observations:
        fig.text(0.715, y, title, fontsize=9.5, fontweight="bold", color=color, va="top")
        fig.text(0.715, y - 0.038, detail, fontsize=8.6, color=INK, va="top", linespacing=1.35)
        y -= 0.115
    box(canvas, 0.69, 0.065, 0.255, 0.04, face="#FFF2F3", edge="#E6C5CA")
    fig.text(0.817, 0.085, "argmin Accel selector：HOLD", fontsize=8.8, fontweight="bold", color=RED, ha="center", va="center")
    save_page(pdf, fig, 11, preview_dir)


def page_accel_expanded_guide(pdf, preview_dir, page_number=12):
    fig, canvas = new_page(
        "从 Legacy Gate A 到 97 候选正式 Gate A",
        "先用扩大排名审计选择器稳定性，再把同一 97 候选选择过程送入正式完整闭环。",
        page_number,
    )
    box(canvas, 0.055, 0.56, 0.42, 0.24, face="#EEF4F8", edge="#C6D7E1")
    fig.text(0.08, 0.755, "旧表：21 状态 × 4 个角色视角", fontsize=12, fontweight="bold", color=BLUE)
    fig.text(
        0.08,
        0.705,
        "每个角色都实际运行完整闭环\n"
        "→ Accel 在 4 个角色中选一个\n"
        "→ 查询该角色的闭环成功/失败\n\n"
        "回答：argmin Accel 能否选到行为更好的视角？",
        fontsize=10.0,
        color=INK,
        va="top",
        linespacing=1.45,
    )
    box(canvas, 0.525, 0.56, 0.42, 0.24, face="#EAF3F0", edge="#BFD8D0")
    fig.text(0.55, 0.755, "新表：96 状态 × 97 个视角 × 6 噪声", fontsize=12, fontweight="bold", color=TEAL)
    fig.text(
        0.55,
        0.705,
        "每个视角只运行固定状态动作流推理\n"
        "→ 对 6 个 flow-noise 的 Accel 取均值\n"
        "→ 统计最小值落入哪个相机支持区域\n\n"
        "回答：排名是否稳定、兼容域在哪里、能否排斥坏观测？",
        fontsize=10.0,
        color=INK,
        va="top",
        linespacing=1.45,
    )

    fig.text(0.055, 0.505, "以 Broad practical 为例", fontsize=11, fontweight="bold", color=INK)
    headers = ["区域", "候选数", "被选状态", "原始占比", "均匀候选基准", "单位候选富集"]
    rows = [
        ["规范视角", "1", "21 / 96", "21.9%", "1 / 97 = 1.0%", "21.2×"],
        ["训练支持", "64", "36 / 96", "37.5%", "64 / 97 = 66.0%", "0.57×"],
        ["同范围留出", "32", "39 / 96", "40.6%", "32 / 97 = 33.0%", "1.23×"],
    ]
    ax = fig.add_axes([0.06, 0.305, 0.88, 0.17])
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1, 1.48)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(LINE)
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(WHITE if row % 2 else "#EEF2F5")

    box(canvas, 0.055, 0.09, 0.89, 0.15, face="#FFF7E8", edge="#E6D2A8")
    fig.text(0.08, 0.205, "三个不能混淆的点", fontsize=10, fontweight="bold", color=GOLD)
    fig.text(
        0.08,
        0.17,
        "1. 新表的百分比不是成功率，而是 96 个状态中 Top-1 所属区域的比例。\n"
        "2. “训练支持”指该精确相机位姿进入过全局训练 catalog；不表示同一 test 状态见过该位姿。\n"
        "3. “留出”是同一宽范围内的精确位姿留出；Blind / Look-away / 黑相机才是极端诊断。",
        fontsize=9.4,
        color=INK,
        va="top",
        linespacing=1.42,
    )
    save_page(pdf, fig, page_number, preview_dir)


def load_gate_a97_shortlist() -> dict[str, dict]:
    payload = {}
    for model in GATE_A97_MODELS:
        path = GATE_A97_ROOT / "shortlist" / model / "analysis.json"
        row = load_json(path)
        if row.get("status") != "PASS" or row.get("state_count") != 96:
            raise ValueError(f"invalid Gate A97 shortlist analysis: {path}")
        payload[model] = row
    return payload


def page_accel_gate_a97(pdf, preview_dir, page_number=19):
    fig, canvas = new_page(
        "97 视角闭环选择：Accel 没有改善主模型成功率",
        "Original LIBERO 自定义诊断集：8 任务 × 3 条 test demo × 4 个轨迹阶段；不是全量 LIBERO 或 LIBERO-Plus Camera Full。",
        page_number,
    )
    payload = load_gate_a97_shortlist()
    primary = payload["broad64-practical"]
    conditions = [
        ("canonical", "规范", BLUE),
        ("accel_single_noise", "Accel 单噪声", GRAY),
        ("accel_ensemble", "Accel 六噪声", RED),
        ("visibility_top1", "可见性最高", GOLD),
        ("accel_top10_visibility", "Accel Top10+可见性", TEAL),
        ("random_operational", "随机视角", "#A9AFB5"),
    ]

    ax = fig.add_axes([0.055, 0.36, 0.59, 0.42])
    labels = [label for _, label, _ in conditions] + ["Oracle@6"]
    values = [
        100 * primary["condition_success"][condition]["state_success_rate"]
        for condition, _, _ in conditions
    ] + [100 * primary["oracle_at_shortlist_state_rate"]]
    colors = [color for _, _, color in conditions] + [INK]
    bars = ax.bar(range(len(labels)), values, color=colors, width=0.68)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}", fontsize=7.5, ha="center")
    ax.set_ylabel("完整闭环成功率 (%)", fontsize=8.4)
    ax.set_xticks(range(len(labels)), labels, fontsize=7.0, rotation=12)
    ax.set_ylim(72, 96)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)

    box(canvas, 0.68, 0.36, 0.265, 0.42, face="#EEF4F8", edge="#C6D7E1")
    fig.text(0.705, 0.735, "先把三个百分比说清楚", fontsize=10.2, fontweight="bold", color=BLUE, va="top")
    fig.text(0.705, 0.68, "84.4%  规范视角", fontsize=11.0, fontweight="bold", color=BLUE, va="top")
    fig.text(0.705, 0.645, "81 / 96 个状态完成任务", fontsize=8.2, color=INK, va="top")
    fig.text(0.705, 0.585, "83.3%  六噪声 Accel", fontsize=11.0, fontweight="bold", color=RED, va="top")
    fig.text(0.705, 0.55, "80 / 96；相对规范 −1.0pp", fontsize=8.2, color=INK, va="top")
    fig.text(0.705, 0.49, "91.7%  Oracle@6", fontsize=11.0, fontweight="bold", color=INK, va="top")
    fig.text(
        0.705,
        0.455,
        "六种已执行条件中，只要任一成功\n就记为成功；这是事后上限，不是算法。",
        fontsize=8.2,
        color=INK,
        va="top",
        linespacing=1.35,
    )
    comparison = primary["paired_source_bootstrap"]["accel_ensemble_vs_canonical"]
    fig.text(
        0.705,
        0.385,
        f"配对 95% CI：[{comparison['ci_low_pp']:+.1f}, {comparison['ci_high_pp']:+.1f}]pp",
        fontsize=8.0,
        color=MUTED,
        va="top",
    )

    box(canvas, 0.055, 0.095, 0.89, 0.18, face="#FFF7E8", edge="#E6D2A8")
    fig.text(0.08, 0.24, "结论", fontsize=9.8, fontweight="bold", color=GOLD, va="top")
    fig.text(
        0.08,
        0.205,
        "• 主模型上，Accel 单噪声与规范相同；六噪声平均反而低 1.0pp，95% CI 跨 0，不能证明提升。\n"
        "• 可见性最高也低 2.1pp；Top10+可见性高 2.1pp，但 CI 同样跨 0。现有规则没有稳定识别行为最优视角。\n"
        "• 其他三种训练组织的六噪声差值为 +2.1、−4.2、−6.2pp，只作为稳健性审计；其中 consistency 已在基准门被判为基础能力受损。",
        fontsize=8.7,
        color=INK,
        va="top",
        linespacing=1.4,
    )
    save_page(pdf, fig, page_number, preview_dir)


def selected_oracle_rates(protocol: dict, result_root: Path) -> dict[str, float]:
    rows = {}
    for path in sorted(result_root.glob("episodes-shard-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows[(row["pair_key"], row["selected_candidate_id"])] = bool(row["success"])
    conditions = list(protocol["selected_states"][0]["selected_candidates"])
    return {
        condition: 100
        * sum(
            rows[(state["pair_key"], state["selected_candidates"][condition])]
            for state in protocol["selected_states"]
        )
        / len(protocol["selected_states"])
        for condition in conditions
    }


def page_accel_failure_oracle(pdf, preview_dir, page_number=20):
    fig, canvas = new_page(
        "只看规范视角失败的 15 个状态：换视角能救回几个",
        "原 96 状态中规范视角成功 81 个、失败 15 个；本页只对这 15 个失败状态逐一执行全部 97 个候选视角。",
        page_number,
    )
    result_root = GATE_A97_ROOT / "oracle97-canonical-failures" / "broad64-practical"
    analysis = load_json(result_root / "analysis.json")
    protocol = load_json(GATE_A97_ROOT / "protocols/broad64-practical-canonical-failure-oracle.json")
    if analysis.get("status") != "PASS" or analysis.get("state_count") != 15:
        raise ValueError("canonical-failure Oracle@97 analysis is incomplete")
    rates = selected_oracle_rates(protocol, result_root)
    oracle_rate = 100 * analysis["oracle_at_97_success_rate"]
    mean_fraction = 100 * analysis["mean_successful_candidate_fraction"]

    metric(fig, canvas, 0.055, 0.61, 0.25, 0.18, "15 / 96", "条件母集", "只分析规范视角失败状态", BLUE)
    metric(fig, canvas, 0.375, 0.61, 0.25, 0.18, "12 / 15", "候选池可救回", "Oracle@97 = 80.0%，事后上限", TEAL)
    metric(fig, canvas, 0.695, 0.61, 0.25, 0.18, "5 / 15", "六噪声 Accel 救回", "33.3%，与随机视角相同", RED)

    condition_order = [
        ("canonical", "规范重放", BLUE),
        ("accel_single_noise", "Accel 单噪声", GRAY),
        ("accel_ensemble", "Accel 六噪声", RED),
        ("visibility_top1", "可见性最高", GOLD),
        ("accel_top10_visibility", "Top10+可见性", TEAL),
        ("random_operational", "随机", "#A9AFB5"),
    ]
    ax = fig.add_axes([0.06, 0.30, 0.39, 0.27])
    labels = [label for _, label, _ in condition_order] + ["Oracle@97"]
    values = [round(rates[key] * 15 / 100) for key, _, _ in condition_order] + [round(oracle_rate * 15 / 100)]
    colors = [color for _, _, color in condition_order] + [INK]
    bars = ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.7)
    ax.set_xlim(0, 15.8)
    ax.set_xticks([0, 3, 6, 9, 12, 15])
    ax.set_xlabel("15 个条件失败状态中，当前规则执行成功的数量", fontsize=7.8)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value + 0.25, bar.get_y() + bar.get_height() / 2, f"{value}/15", va="center", fontsize=7.2)

    success_counts = [row["successful_candidate_count"] for row in analysis["state_rows"]]
    bin_labels = ["0 个\n不可救", "1–24 个\n少量可救", "25–49 个\n部分可救", "50–97 个\n多数可救"]
    bin_counts = [
        sum(value == 0 for value in success_counts),
        sum(1 <= value <= 24 for value in success_counts),
        sum(25 <= value <= 49 for value in success_counts),
        sum(50 <= value <= 97 for value in success_counts),
    ]
    ax2 = fig.add_axes([0.53, 0.30, 0.415, 0.27])
    bars2 = ax2.bar(range(4), bin_counts, color=[RED, GOLD, BLUE, TEAL], width=0.65)
    ax2.set_ylim(0, 5)
    ax2.set_xlabel("每个失败状态在 97 个候选中有多少个视角能成功", fontsize=7.8)
    ax2.set_ylabel("状态数量（共 15 个）", fontsize=7.8)
    ax2.set_xticks(range(4), bin_labels, fontsize=7.2)
    ax2.set_yticks(range(0, 6))
    for bar, value in zip(bars2, bin_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.12, str(value), ha="center", fontsize=8.0)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color=LINE, linewidth=0.8)
    ax2.set_axisbelow(True)

    box(canvas, 0.055, 0.075, 0.89, 0.15, face="#EAF3F0", edge="#BFD8D0")
    fig.text(0.08, 0.19, "判定", fontsize=9.8, fontweight="bold", color=TEAL, va="top")
    fig.text(
        0.08,
        0.155,
        "定义：某规则的救援率 = 该规则在这 15 个规范失败状态中执行成功的数量 ÷ 15；它不是前页 96 状态总体成功率。\n"
        "Oracle@97 表示事后查看同一状态的 97 次完整闭环，只要其中至少一个成功就计为可救；12/15 证明候选池存在真实视角空间。\n"
        "Accel 六噪声、可见性最高和随机视角均为 5/15；Accel 没有识别出这份空间。规范重放为 1/15，反映闭环 GPU/仿真重放并非完全确定。",
        fontsize=8.8,
        color=INK,
        va="top",
        linespacing=1.4,
    )
    save_page(pdf, fig, page_number, preview_dir)


def page_kyc_cvc_boundary(pdf, preview_dir, page_number=19):
    fig, canvas = new_page(
        "方法边界：KYC 与跨视角一致性",
        "两者算法不同，但在标准 external + wrist Pi0.5 中暴露出相同的稳定视觉捷径问题。",
        page_number,
    )
    headers = ["方法 / 协议", "基线", "方法", "差值", "本地裁决"]
    rows = [
        ["KYC · matched Pi0.5", "44.49%", "43.33%", "−1.16pp", "无显式几何增量"],
        ["KYC · 双相机", "20.71%", "15.71%", "−5.00pp", "停止 raw-ray 扩展"],
        ["CVC recipe · Camera Full", "82.16%", "77.78%", "−4.75pp", "当前方案失败"],
        ["CVC recipe · Original Full", "93.92%", "84.67%", "−9.25pp", "基础能力受损"],
    ]
    ax = fig.add_axes([0.06, 0.48, 0.88, 0.28])
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.72)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(LINE)
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(WHITE if row % 2 else "#EEF2F5")

    box(canvas, 0.06, 0.25, 0.42, 0.16, face="#FFF2F3", edge="#E6C5CA")
    fig.text(0.085, 0.37, "停止", fontsize=10.5, fontweight="bold", color=RED, va="top")
    fig.text(
        0.085,
        0.325,
        "KYC raw-ray 与无条件跨视角一致性不再作为\n标准双相机 Pi0.5 的研究主线。",
        fontsize=10.5,
        color=INK,
        va="top",
        linespacing=1.5,
    )
    box(canvas, 0.52, 0.25, 0.42, 0.16, face="#EAF3F0", edge="#BFD8D0")
    fig.text(0.545, 0.37, "保留", fontsize=10.5, fontweight="bold", color=TEAL, va="top")
    fig.text(
        0.545,
        0.325,
        "把二者作为单外部相机受控协议的强基线，\n研究重点转向冗余、缺失与互补视角的区分。",
        fontsize=10.5,
        color=INK,
        va="top",
        linespacing=1.5,
    )
    fig.text(
        0.06,
        0.13,
        "注意：正式 CVC 比较同时改变了独立状态数量和数据组织，因此否定的是当前组合 recipe；\n"
        "它足以支持停止投入，但不能被写成对所有 paired consistency 的普遍反例。",
        fontsize=10,
        color=MUTED,
        linespacing=1.45,
    )
    save_page(pdf, fig, page_number, preview_dir)


def page_takeaway(pdf, preview_dir, page_number=22):
    fig, canvas = new_page("第一阶段结论：已经回答什么，还缺什么", "M-A / M-B 与 97 候选 Gate A 已完成；长期研究计划中的后置验证没有被冒充为已完成。", page_number)
    sections = [
        ("第一阶段完成", TEAL, ["Broad64 数据、训练和七臂诊断", "三 seed Camera / Original Full", "M0、M1 与 97 候选闭环 Gate A"]),
        ("后置未完成", GOLD, ["LIBERO-Plus Full 非相机副作用", "更大任务分布的 Blind–Reveal 确认", "RoboCasa 跨 benchmark 与真机"]),
        ("当前停止", RED, ["KYC raw-ray 双相机扩展", "当前 paired-consistency recipe", "Accel 直接作为主动视角选择器"]),
    ]
    x_values = [0.055, 0.365, 0.675]
    for x, (title, color, bullets) in zip(x_values, sections):
        box(canvas, x, 0.31, 0.27, 0.46)
        canvas.add_patch(Rectangle((x, 0.70), 0.27, 0.07, transform=canvas.transAxes, color=color))
        fig.text(x + 0.025, 0.735, title, fontsize=12, fontweight="bold", color=WHITE, va="center")
        y = 0.64
        for bullet in bullets:
            fig.text(x + 0.03, y, "•", fontsize=13, color=color, va="top")
            fig.text(x + 0.055, y, bullet, fontsize=10.2, color=INK, va="top", wrap=True)
            y -= 0.11
    box(canvas, 0.055, 0.13, 0.89, 0.105, face="#EAF3F0", edge="#BFD8D0")
    fig.text(0.08, 0.19, "一句话", fontsize=10, fontweight="bold", color=TEAL, va="center")
    fig.text(
        0.16,
        0.19,
        "第一阶段已足以否定当前 Accel/KYC/CVC 配方的稳定普适增益；下一阶段应围绕可救援信息关系设计，而不是继续追局部调参。",
        fontsize=11,
        color=INK,
        va="center",
    )
    save_page(pdf, fig, page_number, preview_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/dsol_paper1/view_revalidation_stage1_final_brief_20260824_zh.pdf",
    )
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()

    configure_font()
    expanded_catalog = expanded.load_json(expanded.DEFAULT_CATALOG)
    expanded_protocol = expanded.load_json(expanded.DEFAULT_PROTOCOL)
    expanded_summary = expanded.load_json(expanded.DEFAULT_ANALYSIS / "summary.json")
    if expanded_summary.get("status") != "PASS":
        raise ValueError("expanded Accel analysis is not PASS")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        page_cover(pdf, args.preview_dir)
        page_questions(pdf, args.preview_dir)
        page_data(pdf, args.preview_dir)
        page_training(pdf, args.preview_dir)
        page_passive(pdf, args.preview_dir)
        page_formal_benchmarks(pdf, args.preview_dir)
        page_m0(pdf, args.preview_dir)
        page_m1(pdf, args.preview_dir)
        page_m1_task_breakdown(pdf, args.preview_dir)
        page_accel_principle(pdf, args.preview_dir)
        expanded.page_definition(
            pdf, args.preview_dir, expanded_catalog, expanded_protocol, 11
        )
        page_accel_reliability(pdf, args.preview_dir, expanded_summary, 12)
        page_accel_gate_a97(pdf, args.preview_dir, 13)
        page_accel_failure_oracle(pdf, args.preview_dir, 14)
        page_kyc_cvc_boundary(pdf, args.preview_dir, 15)
        page_takeaway(pdf, args.preview_dir, 16)

if __name__ == "__main__":
    main()
