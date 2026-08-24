#!/usr/bin/env python3
"""Build a compact Chinese brief for the expanded Accel view diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "configs/dsol_paper1/libero_view_catalog_v2_m1.json"
DEFAULT_PROTOCOL = Path(
    "/share/longjunyu/alphabrain/experiments/"
    "dsol-accel-expanded-diagnostic-v1/protocol/"
    "accel_expanded_8task_test96_protocol.json"
)
DEFAULT_ANALYSIS = Path(
    "/share/longjunyu/alphabrain/experiments/"
    "dsol-accel-expanded-diagnostic-v1/analysis"
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

MODELS = (
    "broad64_practical",
    "broad64_state_matched",
    "broad64_paired_fm",
    "broad64_paired_consistency",
)
MODEL_LABELS = {
    "broad64_practical": "实用非配对",
    "broad64_state_matched": "状态匹配重复",
    "broad64_paired_fm": "同状态配对 FM",
    "broad64_paired_consistency": "配对 + 一致性",
}
TASK_LABELS = {
    "goal_cream_cheese_bowl": "奶油奶酪入碗",
    "goal_top_drawer_bowl": "碗入上层抽屉",
    "goal_wine_rack": "酒瓶入架",
    "libero10_book_caddy": "书入收纳架",
    "libero10_bowl_bottom_drawer": "碗入下层抽屉",
    "libero10_mug_microwave": "杯入微波炉",
    "object_cream_cheese_basket": "奶油奶酪入篮",
    "spatial_drawer_bowl_plate": "抽屉碗盘空间任务",
}
CATEGORIES = (
    "canonical",
    "broad64_training_support",
    "broad32_heldout",
)
CATEGORY_LABELS = {
    "canonical": "规范视角",
    "broad64_training_support": "64 个训练支持视角",
    "broad32_heldout": "32 个同范围留出视角",
}
CATEGORY_COLORS = {
    "canonical": BLUE,
    "broad64_training_support": TEAL,
    "broad32_heldout": GOLD,
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def page(title: str, subtitle: str, number: int):
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=BG)
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_axis_off()
    canvas.add_patch(Rectangle((0, 0.965), 1, 0.035, color=INK))
    fig.text(0.055, 0.91, title, fontsize=23, fontweight="bold", color=INK, va="top")
    fig.text(0.057, 0.855, subtitle, fontsize=10.5, color=MUTED, va="top")
    fig.text(0.055, 0.025, "Expanded Accel View Diagnostic · 8 tasks / 96 states", fontsize=7.5, color=MUTED)
    fig.text(0.95, 0.025, f"{number:02d}", fontsize=8, color=MUTED, ha="right")
    return fig, canvas


def box(canvas, x, y, w, h, face=WHITE, edge=LINE):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        transform=canvas.transAxes,
        linewidth=0.8,
        edgecolor=edge,
        facecolor=face,
    )
    canvas.add_patch(patch)


def save(pdf: PdfPages, fig, preview_dir: Path | None, number: int) -> None:
    pdf.savefig(fig, facecolor=BG)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"page-{number:02d}.png", dpi=140, facecolor=BG)
    plt.close(fig)


def pose_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([row["azimuth_deg"] for row in rows], dtype=float),
        np.asarray([row["elevation_deg"] for row in rows], dtype=float),
        np.asarray([row["radius_scale"] for row in rows], dtype=float),
    )


def page_definition(pdf, preview_dir, catalog: dict, protocol: dict) -> None:
    fig, canvas = page(
        "训练支持视角与留出视角：严格定义",
        "两者共享相同的宽位姿范围；差别是具体相机位姿是否进入过训练。Blind / Look-away 属于另一组极端诊断。",
        1,
    )
    ax = fig.add_axes([0.07, 0.19, 0.49, 0.57], facecolor=WHITE)
    train_az, train_el, train_r = pose_arrays(catalog["broad_training_64"])
    held_az, held_el, held_r = pose_arrays(catalog["broad_heldout_32"])
    ax.scatter(
        train_az,
        train_el,
        s=45 + 90 * (train_r - train_r.min()) / max(np.ptp(train_r), 1e-6),
        color=TEAL,
        alpha=0.78,
        label="训练支持：64 个精确位姿",
    )
    ax.scatter(
        held_az,
        held_el,
        s=55 + 90 * (held_r - held_r.min()) / max(np.ptp(held_r), 1e-6),
        facecolor=WHITE,
        edgecolor=GOLD,
        linewidth=1.4,
        label="同范围留出：32 个未训练位姿",
    )
    ax.scatter([0], [0], marker="*", s=210, color=BLUE, edgecolor=WHITE, linewidth=0.8, label="规范视角")
    ax.axhline(0, color=LINE, linewidth=0.8)
    ax.axvline(0, color=LINE, linewidth=0.8)
    ax.set_xlim(-65, 65)
    ax.set_ylim(-28, 28)
    ax.set_xlabel("水平绕台角（方位角，度）")
    ax.set_ylabel("上下俯仰偏移（度）")
    ax.set_title("97 个 operational 候选的位姿分布", loc="left", fontweight="bold", color=INK)
    ax.grid(color=LINE, alpha=0.7)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    box(canvas, 0.60, 0.57, 0.34, 0.20)
    fig.text(0.625, 0.735, "训练支持视角", fontsize=12, fontweight="bold", color=TEAL)
    fig.text(
        0.625,
        0.69,
        "64 个具体相机位姿已经用于 Broad64 微调。\n若模型偏向这里，可能反映训练熟悉度。",
        fontsize=9.6,
        color=INK,
        linespacing=1.45,
        va="top",
    )
    box(canvas, 0.60, 0.34, 0.34, 0.18)
    fig.text(0.625, 0.485, "同范围留出视角", fontsize=12, fontweight="bold", color=GOLD)
    fig.text(
        0.625,
        0.44,
        "32 个具体位姿未参与训练，但仍位于同一采样范围。\n衡量宽视角域内的插值泛化，不代表极端 OOD。",
        fontsize=9.6,
        color=INK,
        linespacing=1.45,
        va="top",
    )
    box(canvas, 0.60, 0.11, 0.34, 0.18, face="#FCEFF1", edge="#E8C3CA")
    fig.text(0.625, 0.255, "极端诊断视角", fontsize=12, fontweight="bold", color=RED)
    fig.text(
        0.625,
        0.21,
        "Blind、Look-away 与三类黑相机不进入 97 视角 Top-1\n支持统计；它们专门测试坏观测识别与视觉依赖。",
        fontsize=9.6,
        color=INK,
        linespacing=1.45,
        va="top",
    )
    fig.text(
        0.07,
        0.105,
        f"实测规模：{len(protocol['task_counts'])} 个任务 × 12 个 test 状态 = {protocol['selected_state_count']} 个冻结物理状态；"
        "每个状态渲染 97 个 operational 候选，并增加信息、控制和极端诊断视角。",
        fontsize=9.2,
        color=MUTED,
    )
    save(pdf, fig, preview_dir, 1)


def page_stability(pdf, preview_dir, summary: dict) -> None:
    fig, canvas = page(
        "扩大实测（一）：Accel 选择是否稳定",
        "每个模型使用 6 个共享 flow-noise seed；一次采样的最低 Accel 视角只有在跨噪声稳定后才可解释。",
        2,
    )
    x = np.arange(len(MODELS))
    labels = [MODEL_LABELS[key] for key in MODELS]
    models = summary["models"]

    ax = fig.add_axes([0.07, 0.46, 0.40, 0.31])
    values = [models[key]["mean_pairwise_rank_spearman"] for key in MODELS]
    bars = ax.bar(x, values, color=[BLUE, TEAL, GOLD, RED])
    ax.set_ylim(0, 1)
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("跨噪声排名 Spearman")
    ax.set_title("全部 97 视角排名的跨噪声一致性", loc="left", fontweight="bold")
    ax.grid(axis="y", color=LINE)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}", ha="center", fontsize=9)

    ax = fig.add_axes([0.55, 0.46, 0.38, 0.31])
    exact = [100 * models[key]["all_seed_top1_agreement_rate"] for key in MODELS]
    category = [100 * models[key]["all_seed_category_agreement_rate"] for key in MODELS]
    width = 0.36
    ax.bar(x - width / 2, exact, width, color=RED, label="精确 Top-1 六次全同")
    ax.bar(x + width / 2, category, width, color=BLUE, label="所属区域六次全同")
    ax.set_ylim(0, 100)
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("状态占比（%）")
    ax.set_title("选择结果的全 seed 一致率", loc="left", fontweight="bold")
    ax.grid(axis="y", color=LINE)
    ax.legend(frameon=False, fontsize=8)

    box(canvas, 0.07, 0.13, 0.86, 0.23, face="#EEF4F8", edge="#C6D7E1")
    fig.text(0.10, 0.315, "解释", fontsize=10.5, fontweight="bold", color=BLUE)
    paired_relation = summary["cross_model_ensemble_relations"][
        "broad64_state_matched__vs__broad64_paired_fm"
    ]["mean_ensemble_rank_spearman"]
    consistency = models["broad64_paired_consistency"]
    fig.text(
        0.10,
        0.27,
        "• 精确 Top-1 对 flow noise 高度敏感；单次生成得到的 10/21 等计数不能作为稳定模型属性。\n"
        f"• 状态匹配重复与 Paired-FM 的 ensemble 排名相关性为 {paired_relation:.3f}：普通 FM 下 pairing 未形成独立结构。\n"
        f"• 一致性模型的跨视角相对离散度仅 {100 * consistency['mean_accel_relative_spread']:.2f}%、Top-1 间隔 "
        f"{100 * consistency['mean_ensemble_top1_relative_margin']:.2f}%：主要效果是压平。\n"
        "• 后续只采用 6-noise 平均排名，并把 Accel 定位为兼容性诊断，而非已验证的闭环视角价值。",
        fontsize=9.5,
        color=INK,
        linespacing=1.42,
        va="top",
    )
    save(pdf, fig, preview_dir, 2)


def page_support(pdf, preview_dir, summary: dict) -> None:
    fig, canvas = page(
        "扩大实测（二）：模型最低 Accel 视角落在哪里",
        "对每个冻结状态，先对 6 个 flow-noise seed 的 Accel 取均值，再在 97 个视角中选择 Top-1。",
        3,
    )
    models = summary["models"]
    labels = [MODEL_LABELS[key] for key in MODELS]
    x = np.arange(len(MODELS))
    fig.text(
        0.08,
        0.795,
        "候选池组成：规范 1/97（1.0%）｜训练支持 64/97（66.0%）｜留出 32/97（33.0%）",
        fontsize=8.8,
        color=MUTED,
    )
    ax = fig.add_axes([0.08, 0.37, 0.84, 0.39])
    bottom = np.zeros(len(MODELS))
    for category in CATEGORIES:
        values = np.asarray(
            [100 * models[key]["ensemble_selected_category_rates"][category] for key in MODELS]
        )
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            color=CATEGORY_COLORS[category],
            label=CATEGORY_LABELS[category],
        )
        for bar, value, base in zip(bars, values, bottom):
            if value >= 7:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=WHITE if category != "broad32_heldout" else INK,
                )
        bottom += values
    ax.set_ylim(0, 100)
    ax.set_xticks(x, labels)
    ax.set_ylabel("96 个状态中的占比（%）")
    ax.set_title("6-noise ensemble Top-1 的支持区域", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.13))

    box(canvas, 0.08, 0.11, 0.84, 0.18, face=WHITE)
    fig.text(0.105, 0.255, "应当怎样读", fontsize=10.5, fontweight="bold", color=TEAL)
    fig.text(
        0.105,
        0.215,
        "留出视角被选中，说明模型的低-Accel 区域没有被 64 个训练位姿完全锁死；"
        "但它只证明内部 flow 更平滑，不能单独证明该视角闭环更好。\n"
        "训练支持占比也不能直接解释为过拟合，因为训练和留出候选数量不同（64 对 32）；"
        "正式比较需同时看单位候选命中率与闭环结果。",
        fontsize=10.0,
        color=INK,
        linespacing=1.5,
        va="top",
    )
    save(pdf, fig, preview_dir, 3)


def page_roles(pdf, preview_dir, summary: dict) -> None:
    fig, canvas = page(
        "扩大实测（三）：信息视角与坏观测诊断",
        "平均排名越低，Accel 越偏好该视角；完整候选约 104 个，黑相机若接近末位表示坏观测诊断有效。",
        4,
    )
    role_order = (
        "canonical",
        "strong_info",
        "matched_control",
        "blind",
        "look_away",
        "external_blackout",
        "wrist_blackout",
        "all_camera_blackout",
    )
    role_labels = (
        "规范",
        "最高可见性",
        "等位移控制",
        "Blind",
        "Look-away",
        "外部黑屏",
        "腕部黑屏",
        "全部黑屏",
    )
    values = np.asarray(
        [
            [summary["models"][model]["mean_role_ranks"][role] for role in role_order]
            for model in MODELS
        ]
    )
    ax = fig.add_axes([0.16, 0.25, 0.74, 0.51])
    image = ax.imshow(values, cmap="RdYlGn_r", vmin=1, vmax=104, aspect="auto")
    ax.set_xticks(np.arange(len(role_order)), role_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(MODELS)), [MODEL_LABELS[key] for key in MODELS])
    ax.set_title("各诊断视角在完整候选中的平均 Accel 排名", loc="left", fontweight="bold")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = WHITE if values[row, column] > 72 else INK
            ax.text(column, row, f"{values[row, column]:.1f}", ha="center", va="center", fontsize=8, color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("平均名次（越低越偏好）")

    box(canvas, 0.08, 0.09, 0.84, 0.105, face=WHITE)
    fig.text(
        0.105,
        0.165,
        "结论：黑相机和 Look-away 稳定落在末端，Accel 能识别明显无效观测；最高可见性视角通常只处于中游，\n"
        "尚未显示“可见实体更多 ⇒ flow 更平滑”。因此 Accel 可用于过滤坏视角，但不能替代任务信息价值。",
        fontsize=9.8,
        color=INK,
        linespacing=1.45,
        va="top",
    )
    save(pdf, fig, preview_dir, 4)


def task_category_rates(rows: list[dict[str, str]], category: str) -> np.ndarray:
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[(row["model"], row["task_id"])][row["ensemble_category"]] += 1
    tasks = list(TASK_LABELS)
    return np.asarray(
        [
            [
                100 * grouped[(model, task)][category] / sum(grouped[(model, task)].values())
                for task in tasks
            ]
            for model in MODELS
        ]
    )


def page_tasks(pdf, preview_dir, rows: list[dict[str, str]]) -> None:
    fig, canvas = page(
        "扩大实测（四）：8 个任务上的支持区域分布",
        "每格为该任务 12 个冻结 test 状态中，6-noise ensemble Top-1 落入对应区域的比例。",
        5,
    )
    tasks = list(TASK_LABELS)
    task_labels = [TASK_LABELS[task] for task in tasks]
    canonical = task_category_rates(rows, "canonical")
    heldout = task_category_rates(rows, "broad32_heldout")
    for index, (values, title, cmap) in enumerate(
        ((canonical, "规范视角 Top-1 占比", "Blues"), (heldout, "留出视角 Top-1 占比", "YlOrBr"))
    ):
        ax = fig.add_axes([0.08 + 0.46 * index, 0.25, 0.39, 0.51])
        image = ax.imshow(values, vmin=0, vmax=100, cmap=cmap, aspect="auto")
        ax.set_xticks(np.arange(len(tasks)), task_labels, rotation=40, ha="right", fontsize=7.3)
        ax.set_yticks(np.arange(len(MODELS)), [MODEL_LABELS[key] for key in MODELS], fontsize=8)
        ax.set_title(title, loc="left", fontweight="bold")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                color = WHITE if values[row, column] >= 55 else INK
                ax.text(column, row, f"{values[row, column]:.0f}", ha="center", va="center", fontsize=7, color=color)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.02)
        colorbar.set_label("%")
    fig.text(
        0.08,
        0.12,
        "用途：检查总体平均是否被单一任务支配。若不同任务的偏好方向相反，就不能把全局占比写成统一的模型规律；"
        "后续闭环 shortlist 应按任务均衡抽样，而不是只挑 Accel 信号最强的状态。",
        fontsize=10.0,
        color=INK,
        linespacing=1.5,
    )
    save(pdf, fig, preview_dir, 5)


def stage_from_pair_key(pair_key: str) -> str:
    match = re.search(r"stage-(\d+)", pair_key)
    if match is None:
        raise ValueError(f"pair key has no stage: {pair_key}")
    return f"stage-{int(match.group(1)):02d}"


def stage_category_rates(
    rows: list[dict[str, str]], category: str
) -> tuple[list[str], np.ndarray]:
    stages = sorted({stage_from_pair_key(row["pair_key"]) for row in rows})
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[(row["model"], stage_from_pair_key(row["pair_key"]))][
            row["ensemble_category"]
        ] += 1
    values = np.asarray(
        [
            [
                100
                * grouped[(model, stage)][category]
                / sum(grouped[(model, stage)].values())
                for stage in stages
            ]
            for model in MODELS
        ]
    )
    return stages, values


def stage_rank_gap(rows: list[dict[str, str]], stages: list[str]) -> np.ndarray:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], stage_from_pair_key(row["pair_key"]))].append(
            float(row["mean_rank_strong_info"])
            - float(row["mean_rank_matched_control"])
        )
    return np.asarray(
        [
            [float(np.mean(grouped[(model, stage)])) for stage in stages]
            for model in MODELS
        ]
    )


def page_stages(pdf, preview_dir, rows: list[dict[str, str]]) -> None:
    fig, canvas = page(
        "扩大实测（五）：选择偏好是否随操作阶段变化",
        "每个任务从轨迹中均衡抽取 stage-00..03；阶段仅表示轨迹进程，不在不同任务间强行赋予同一子目标语义。",
        6,
    )
    stages, canonical = stage_category_rates(rows, "canonical")
    _, heldout = stage_category_rates(rows, "broad32_heldout")
    labels = [value.replace("stage-", "阶段 ") for value in stages]
    for index, (values, title, cmap) in enumerate(
        (
            (canonical, "规范视角 Top-1 占比", "Blues"),
            (heldout, "留出视角 Top-1 占比", "YlOrBr"),
        )
    ):
        ax = fig.add_axes([0.07 + 0.31 * index, 0.39, 0.27, 0.35])
        image = ax.imshow(values, vmin=0, vmax=100, cmap=cmap, aspect="auto")
        ax.set_xticks(np.arange(len(stages)), labels, rotation=20, ha="right", fontsize=8)
        ax.set_yticks(
            np.arange(len(MODELS)),
            [MODEL_LABELS[key] for key in MODELS] if index == 0 else [],
            fontsize=8,
        )
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                color = WHITE if values[row, column] >= 55 else INK
                ax.text(
                    column,
                    row,
                    f"{values[row, column]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=color,
                )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
        colorbar.set_label("%", fontsize=8)

    gaps = stage_rank_gap(rows, stages)
    ax = fig.add_axes([0.69, 0.39, 0.25, 0.35])
    colors = (BLUE, TEAL, GOLD, RED)
    for row, (model, color) in enumerate(zip(MODELS, colors)):
        ax.plot(
            np.arange(len(stages)),
            gaps[row],
            marker="o",
            linewidth=1.8,
            color=color,
            label=MODEL_LABELS[model],
        )
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(np.arange(len(stages)), labels, rotation=20, ha="right", fontsize=8)
    ax.set_title("可见性是否带来 Accel 偏好", loc="left", fontweight="bold", fontsize=10)
    ax.grid(axis="y", color=LINE)
    ax.legend(frameon=False, fontsize=6.8, loc="best")

    box(canvas, 0.07, 0.11, 0.87, 0.18, face=WHITE)
    fig.text(0.095, 0.255, "判读规则", fontsize=10.5, fontweight="bold", color=TEAL)
    fig.text(
        0.095,
        0.215,
        "右图小于 0 才表示最高可见性视角比等位移控制更受 Accel 偏好；若不同阶段围绕 0 波动，说明没有稳定的阶段特异信息对齐。\n"
        "该分析只检查内部兼容性随阶段的变化；实际任务价值仍以同状态完整闭环成功率为准。",
        fontsize=9.8,
        color=INK,
        linespacing=1.5,
        va="top",
    )
    save(pdf, fig, preview_dir, 6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    configure_font()
    catalog = load_json(args.catalog)
    protocol = load_json(args.protocol)
    summary = load_json(args.analysis_dir / "summary.json")
    if summary.get("status") != "PASS":
        raise ValueError("expanded Accel analysis is not PASS")
    rows = load_csv(args.analysis_dir / "state_stability.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        page_definition(pdf, args.preview_dir, catalog, protocol)
        page_stability(pdf, args.preview_dir, summary)
        page_support(pdf, args.preview_dir, summary)
        page_roles(pdf, args.preview_dir, summary)
        page_tasks(pdf, args.preview_dir, rows)
        page_stages(pdf, args.preview_dir, rows)
    print(args.output)


if __name__ == "__main__":
    main()
