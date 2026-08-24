#!/usr/bin/env python3
"""Build a compact Chinese PDF brief for the view revalidation program."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "docs/dsol_paper1/figures"
PASSIVE_FIGURE = FIGURE_DIR / "view_revalidation_data_passive_interim.png"
MECHANISM_FIGURE = FIGURE_DIR / "view_revalidation_m0_m1_accel_interim.png"
M0_MONTAGE = Path(
    "/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m0-v1/"
    "operational-three-task-scan-v2/manual_audit_renders_v1/states/"
    "1a785a92de084d0b/visibility_extremes.png"
)
FORMAL_ROOT = Path(
    "/share/longjunyu/alphabrain/experiments/dsol-view-revalidation-m-b-v1/"
)
CAMERA_FORMAL_METRICS = FORMAL_ROOT / "camera_full/multiseed_metrics.json"
ORIGINAL_FORMAL_METRICS = FORMAL_ROOT / "original_full/multiseed_metrics.json"
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
    headers = ["方法", "独立状态", "同状态跨视角", "显式一致性", "主要问题"]
    rows = [
        ["Canonical-unique", "高", "否", "否", "固定视角 continuation"],
        ["Canonical-repeat", "低，精确重复", "否", "否", "有效 batch 与重复效应"],
        ["Image-Aug unique", "高", "否", "否", "图像增强能否替代相机变化"],
        ["Broad practical", "高", "否", "否", "普通宽覆盖实用上限"],
        ["Broad state-matched", "与 paired 相同", "否", "否", "严格状态和曝光预算控制"],
        ["Broad paired FM", "与 paired 相同", "是", "否", "pairing 数据本身"],
        ["Broad paired consistency", "与 paired 相同", "是", "是", "显式 action-flow 一致性"],
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
        "M-B 已完成：Broad practical 与 Broad paired consistency 均完成 3 seeds、Camera Full 与 Original Full。",
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


def page_formal_benchmarks(pdf, preview_dir):
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
    fig, canvas = new_page("结果 3：信息视角是稀疏上尾，不等于随机换视角", "第一轮保持旧定义：任务实体在多相机中的等权可见像素比例。", 7)
    add_image(fig, M0_MONTAGE, [0.055, 0.49, 0.89, 0.28])
    add_image(fig, MECHANISM_FIGURE, [0.055, 0.14, 0.42, 0.28], crop=(0.0, 0.08, 0.335, 1.0))
    box(canvas, 0.51, 0.14, 0.43, 0.28)
    fig.text(0.54, 0.375, "M0 扫描", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.54, 0.335, "180 states · 15,840 candidates", fontsize=10, color=BLUE)
    fig.text(0.54, 0.285, "• Crossed-orbit 最大 ΔI：+0.2085", fontsize=10, color=INK)
    fig.text(0.54, 0.245, "• Look-away 正增量比例：0%", fontsize=10, color=INK)
    fig.text(0.54, 0.205, "• Sensor blackout 正增量比例：0%", fontsize=10, color=INK)
    fig.text(0.54, 0.16, "21 个 test states 通过 AI-assisted 视觉审计后进入 M1。", fontsize=9, color=MUTED)
    save_page(pdf, fig, 7, preview_dir)


def page_m1(pdf, preview_dir):
    fig, canvas = new_page("结果 4：宽覆盖模型出现闭环信息特异性", "同一冻结物理状态，分别执行 Canonical、Strong-info、Matched-control 和 Blind 到成功或超时。", 8)
    add_image(fig, MECHANISM_FIGURE, [0.045, 0.18, 0.57, 0.59], crop=(0.335, 0.08, 0.625, 1.0))
    metric(fig, canvas, 0.65, 0.57, 0.28, 0.17, "+13.3pp", "Broad practical 信息特异性", "cluster 95% CI [+3.3, +26.7]", TEAL)
    metric(fig, canvas, 0.65, 0.36, 0.28, 0.17, "+34.2pp", "Strong-info 相对 Blind", "cluster 95% CI [+4.2, +67.5]", BLUE)
    box(canvas, 0.65, 0.15, 0.28, 0.16, face="#FFF7E8", edge="#E6D2A8")
    fig.text(0.675, 0.275, "限制", fontsize=10, fontweight="bold", color=GOLD, va="top")
    fig.text(
        0.675,
        0.235,
        "仅 6 个独立 source demonstrations；\nexternal-only 在五个模型中均为 0%；\n当前结果仍是 development evidence。",
        fontsize=9.2,
        color=INK,
        va="top",
        linespacing=1.45,
    )
    save_page(pdf, fig, 8, preview_dir)


def page_accel_and_next(pdf, preview_dir):
    fig, canvas = new_page("Accel 裁决与第一阶段状态", "Accel 目前更像训练熟悉度指标，而不是完整的 view-value selector。", 9)
    add_image(fig, MECHANISM_FIGURE, [0.045, 0.34, 0.50, 0.42], crop=(0.625, 0.08, 1.0, 1.0))
    box(canvas, 0.585, 0.48, 0.35, 0.28)
    fig.text(0.615, 0.705, "Accel 观察", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.615, 0.655, "• 每 21 个状态有 15–18 个选择 canonical", fontsize=10, color=INK)
    fig.text(0.615, 0.61, "• 相对 canonical 成功率：-2.5pp 到 +3.3pp", fontsize=10, color=INK)
    fig.text(0.615, 0.565, "• Broad practical 任一候选可成功：85.7%", fontsize=10, color=INK)
    fig.text(0.615, 0.52, "• Accel 未能捕获上述 oracle headroom", fontsize=10, color=RED)

    box(canvas, 0.055, 0.13, 0.88, 0.16, face="#EDF3F7", edge="#C8D7E2")
    fig.text(0.08, 0.25, "M-B 正式确认", fontsize=10, fontweight="bold", color=TEAL, va="top")
    fig.text(
        0.205,
        0.25,
        "状态 COMPLETE：6 个 Camera Full 模型 + 6 个 Original Full 模型；seeds 41/42/43。",
        fontsize=10.5,
        color=INK,
        va="top",
    )
    fig.text(
        0.205,
        0.195,
        "正式裁决：coverage 通过；当前 consistency 方案失败；Accel dynamic shortlist 继续 HOLD。",
        fontsize=9.5,
        color=MUTED,
        va="top",
    )
    save_page(pdf, fig, 9, preview_dir)


def page_kyc_cvc_boundary(pdf, preview_dir):
    fig, canvas = new_page(
        "方法边界：KYC 与跨视角一致性",
        "两者算法不同，但在标准 external + wrist Pi0.5 中暴露出相同的稳定视觉捷径问题。",
        10,
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
    save_page(pdf, fig, 10, preview_dir)


def page_takeaway(pdf, preview_dir):
    fig, canvas = new_page("第一阶段结论：已经回答什么，还缺什么", "M-A / M-B 已完成；长期研究计划中的后置验证没有被冒充为已完成。", 11)
    sections = [
        ("第一阶段完成", TEAL, ["Broad64 数据、训练和七臂诊断", "三 seed Camera / Original Full", "M0、M1 与 Accel fixed-state"]),
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
        "第一阶段已经完成并足以裁决当前方法；下一阶段若继续，应扩大信息互补验证，而不是继续追 KYC/CVC 调参。",
        fontsize=11,
        color=INK,
        va="center",
    )
    save_page(pdf, fig, 11, preview_dir)


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
        page_accel_and_next(pdf, args.preview_dir)
        page_kyc_cvc_boundary(pdf, args.preview_dir)
        page_takeaway(pdf, args.preview_dir)

if __name__ == "__main__":
    main()
