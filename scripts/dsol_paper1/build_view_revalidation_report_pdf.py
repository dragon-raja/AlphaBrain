#!/usr/bin/env python3
"""Build the current Chinese report from frozen results and descriptive audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

import build_view_revalidation_brief_pdf as base


ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-all8-v1")
EXT = DATA / "independent-source-extension"
AUDIT = EXT / "common-failure-audit-20260831/common_failure_audit.json"
METHODS = [
    "canonical",
    "validation_global_fixed",
    "visibility_gain_gated",
    "visibility_hmean_entity",
    "visibility_mean",
    "visibility_min_entity",
]
LABELS = ["规范视角", "验证集固定视角", "可见性增量门控", "实体调和平均", "实体算术平均", "最小实体可见性"]
TASKS = [
    "goal_cream_cheese_bowl",
    "goal_top_drawer_bowl",
    "goal_wine_rack",
    "libero10_book_caddy",
    "libero10_bowl_bottom_drawer",
    "libero10_mug_microwave",
    "object_cream_cheese_basket",
    "spatial_drawer_bowl_plate",
]
TASK_LABELS = [
    "奶酪放碗",
    "碗放上层抽屉",
    "酒瓶放酒架",
    "书放收纳盒后格",
    "碗放底层抽屉",
    "杯子放微波炉",
    "奶酪放篮子",
    "抽屉取碗放盘",
]
INPUTS = set()


def load(path):
    path = Path(path)
    INPUTS.add(path)
    return json.loads(path.read_text(encoding="utf-8"))


def text(fig, x, y, value, size=12, color=None, weight="normal", **kwargs):
    return fig.text(
        x, y, value, fontsize=size, color=color or base.INK, va="top", fontweight=weight, linespacing=1.5, **kwargs
    )


def heading(fig, value, x=0.055, y=0.78):
    text(fig, x, y, value, 14, weight="bold")


def note(fig, value, color=None):
    text(fig, 0.055, 0.13, value, 11, color or base.INK, weight="bold")


def table(fig, headers, rows, rect, widths=None, fontsize=11):
    ax = fig.add_axes(rect)
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=headers, colWidths=widths, bbox=[0, 0, 1, 1], cellLoc="left", colLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor(base.LINE)
        cell.set_linewidth(0.6)
        cell.PAD = 0.08
        if r == 0:
            cell.set_facecolor(base.INK)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white" if r % 2 else "#EDF2F5")
    return ax


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=base.LINE)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)


def finish(pdf, fig, previews, number):
    base.save_page(pdf, fig, number, previews)


def rate(analysis, method):
    return analysis["selector_summary"][method]["mean_repeat_success_rate"] * 100


def page_cover(pdf, previews, combined):
    fig, _ = base.new_page(
        "视角覆盖、可见性与闭环价值", "整合报告 · 2026-08-31 · 保留正式基准，更新构造任务与独立来源复验", 1
    )
    heading(fig, "两个问题，分别作答")
    text(fig, 0.065, 0.68, "宽视角训练是否有效？", 19, base.BLUE, "bold")
    text(fig, 0.065, 0.61, "Camera Full：75.9% → 82.2%\n三个训练 seed 的正式结果保持不变。", 15)
    text(fig, 0.535, 0.68, "可见性最高是否最值得看？", 19, base.RED, "bold")
    text(
        fig,
        0.535,
        0.61,
        f"42 来源复验：74.05% → {rate(combined, 'visibility_gain_gated'):.2f}%\n当前等权像素规则没有稳定收益。",
        15,
    )
    table(
        fig,
        ["新增证据", "规模", "结论边界"],
        [
            ["独立来源复验", "8 任务 / 42 源演示 / 84 状态", "同一 Broad64 practical checkpoint"],
            ["随机性复验", "5 个 flow-noise / 2,520 条闭环", "不是 5 个训练 seed"],
            ["共性问题审计", "区域映射 / 实体增量 / 阶段 / 支持域", "描述性归因，不冒充因果证明"],
        ],
        [0.055, 0.265, 0.89, 0.215],
        [0.24, 0.40, 0.36],
    )
    note(fig, "核心更新：分数缺少任务区域语义；构造已有可见差异，但行为证据与独立泛化仍需补强。")
    finish(pdf, fig, previews, 1)


def page_map(pdf, previews):
    fig, _ = base.new_page(
        "先区分评测母集，再比较成功率",
        "本报告中所有 SR 均来自闭环；初始状态、任务构成与候选规则不同，不能跨母集直接相减。",
        2,
    )
    table(
        fig,
        ["实验", "起点与规模", "回答的问题"],
        [
            ["正式 Camera / Original Full", "官方任务初态；完整评测协议", "视角鲁棒性与基础能力保持"],
            ["早期 A：角色化四视角", "3 任务 / 21 中间状态 / 6 源演示", "Info 相对等幅 Control 的局部信号"],
            ["扩展 A：构造与发现", "8 任务 / 32 validation 状态 / 97 视角", "候选行为上限及噪声重复性"],
            ["扩展 A：冻结规则测试", "8 任务 / 84 test 状态 / 42 源演示", "选择规则能否跨来源兑现收益"],
            ["独立 B：Accel 97 视角诊断", "8 任务 / 96 状态 / 通用候选池", "Accel 排序与闭环选择能力"],
        ],
        [0.055, 0.34, 0.89, 0.43],
        [0.29, 0.40, 0.31],
        11,
    )
    text(
        fig,
        0.065,
        0.265,
        "中间状态 continuation：恢复专家轨迹中的真实状态，随后由策略持续重规划，直到任务成功或超时。",
        12,
    )
    note(fig, "新增 84 状态属于自建 LIBERO 构造诊断；使用 LIBERO-Plus runtime，不等于官方 Plus 全量分数。")
    finish(pdf, fig, previews, 2)


def page_legacy_a(pdf, previews):
    fig, _ = base.new_page(
        "早期 A 的正信号：主要来自抽屉任务", "保留原任务分解作为发现依据；本页不与新增 84 状态的成功率直接比较。", 7
    )
    m1 = load(base.M1_CROSS_MODEL_METRICS)
    rows = [r for r in m1["condition_success"] if r["model"] == "broad64-practical"]
    values = {r["condition"]: r["state_success_rate"] * 100 for r in rows}
    labels = ["规范", "高可见性", "等幅位姿对照", "Blind"]
    data = [values[k] for k in ["canonical_both", "strong_info_both", "matched_control_both", "blind_both"]]
    ax = fig.add_axes([0.08, 0.40, 0.40, 0.34])
    bars = ax.bar(labels, data, color=[base.BLUE, base.TEAL, base.GOLD, base.RED])
    ax.set_ylim(0, 100)
    ax.set_ylabel("21 状态闭环成功率 %")
    ax.bar_label(bars, fmt="%.1f", padding=4)
    clean(ax)
    text(fig, 0.55, 0.74, "Broad64 practical 主模型", 16, weight="bold")
    text(
        fig,
        0.55,
        0.66,
        "3 个任务 / 6 条源演示\nInfo − Control：+13.3pp\n源演示 bootstrap：+3.3 至 +26.7pp\n\n这是扩大验证的起点，不是最终泛化结论。",
        13,
    )
    table(
        fig,
        ["任务", "状态数", "Info − 规范", "Info − Control"],
        [
            ["酒瓶放酒架", "2", "0.0pp", "0.0pp"],
            ["黑碗放抽屉", "10", "+40.0pp", "+20.0pp"],
            ["杯子放微波炉", "9", "-11.1pp", "+22.2pp"],
        ],
        [0.055, 0.19, 0.89, 0.16],
        [0.34, 0.16, 0.25, 0.25],
        11,
    )
    note(fig, "柱图按状态等权；右侧差值按源演示等权，因此数值不同。任务贡献不均，早期正信号仍需来源复验。")
    finish(pdf, fig, previews, 7)


def page_design(pdf, previews):
    fig, _ = base.new_page(
        "扩展 A：固定策略，扩大来源并冻结选择规则",
        "本轮没有重新训练模型；扩大的是评测状态与候选扫描。所有视角共享同一物理初态和配对噪声。",
        8,
    )
    table(
        fig,
        ["步骤", "数据与处理", "防止的混淆"],
        [
            ["构造", "8 任务；静态无碰撞遮挡；保留动态腕部相机", "不改变动作目标与任务成功判据"],
            ["候选", "规范 1 + 训练目录 64 + 同范围留出 32", "每个状态均扫描完整 97 视角"],
            ["选择", "验证集固定规则；测试不查 policy outcome", "避免用成功轨迹反向挑测试视角"],
            ["闭环", "固定所选外部位姿；K=5；执行至成功或超时", "不是只看单个 action chunk"],
            ["统计", "每源演示 2 状态；每规则 5 flow-noise", "按源演示 bootstrap，另报任务等权"],
        ],
        [0.055, 0.40, 0.89, 0.37],
        [0.12, 0.51, 0.37],
        11,
    )
    text(fig, 0.065, 0.345, r"$I(v)=\frac{1}{|C||E|}\sum_{c\in C,e\in E}\frac{P_{e,c}(v)}{224\times224}$", 23, base.BLUE)
    text(fig, 0.54, 0.34, "P：该实体在图像上的可见像素数\nC：外部与腕部相机；E：任务实体列表", 12)
    text(fig, 0.065, 0.23, "门控规则：先选 I 最大的视角；若相对规范增量小于 0.5pp，则保留规范视角。", 12)
    note(fig, "等权指实体和相机权重相同；分母仍是整张图面积，并非实体自身面积，也不衡量决策消歧。")
    finish(pdf, fig, previews, 8)


def page_headroom(pdf, previews, repeat):
    fig, _ = base.new_page(
        "候选中存在经验上限，但可见性没有兑现",
        "32 个 validation 状态、16 条源演示；97 视角单次发现后，冻结 8 个代表候选并使用 3 个新噪声复验。",
        9,
    )
    vals = [
        100 * repeat[k]
        for k in [
            "canonical_mean_repeat_success_rate",
            "visibility_mean_repeat_success_rate",
            "best_shortlist_oracle_mean_repeat_success_rate",
        ]
    ]
    ax = fig.add_axes([0.09, 0.38, 0.41, 0.38])
    bars = ax.bar(["规范", "可见性最高", "事后 Best-of-8"], vals, color=[base.BLUE, base.RED, base.TEAL])
    ax.set_ylim(0, 100)
    ax.set_ylabel("三噪声平均闭环成功率 %")
    ax.bar_label(bars, fmt="%.2f", padding=4)
    clean(ax)
    text(fig, 0.56, 0.75, "两个不同的结论", 17, weight="bold")
    text(
        fig,
        0.56,
        0.67,
        "候选空间：事后上限高于规范 +12.50pp。\n当前可见性：反而低于规范 4.17pp。\n\n5/32 个状态有“规范多数失败、\n候选多数成功”的经验标签。",
        13,
    )
    table(
        fig,
        ["比较", "差值", "源演示 bootstrap 95% CI"],
        [
            ["可见性 − 规范", "-4.17pp", "[-10.42, +2.08]"],
            ["事后 Best-of-8 − 规范", "+12.50pp", "[+6.25, +18.75]"],
        ],
        [0.055, 0.21, 0.89, 0.14],
        [0.43, 0.19, 0.38],
        12,
    )
    note(fig, "Best-of-8 使用同批复验结果选最优，仍有选择乐观偏差；该区间不是新噪声下可部署收益的保证。")
    finish(pdf, fig, previews, 9)


def page_test(pdf, previews, cohorts):
    fig, _ = base.new_page(
        "独立来源复验：初始弱正信号未复现",
        "同一 checkpoint、同一 97 视角目录、同一冻结规则；新增 18 源演示与原 24 源演示互不重叠。",
        10,
    )
    ax = fig.add_axes([0.07, 0.48, 0.42, 0.29])
    x = np.arange(3)
    for offset, key, label, color in [
        (-0.18, "canonical", "规范", base.BLUE),
        (0.18, "visibility_gain_gated", "可见性门控", base.RED),
    ]:
        vals = [rate(a, key) for a in cohorts]
        bars = ax.bar(x + offset, vals, 0.34, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=10)
    ax.set_xticks(x, ["初始 24 来源", "新增 18 来源", "合并 42 来源"])
    ax.set_ylim(0, 102)
    ax.set_ylabel("五噪声闭环成功率 %")
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=10)
    clean(ax)
    text(fig, 0.55, 0.75, "新增来源：-11.67pp", 20, base.RED, "bold")
    text(fig, 0.55, 0.67, "95% CI：[-18.89, -4.44]\n五个 flow-noise 差值全部为负。\n任务等权差值仍为 -12.14pp。", 13)
    text(fig, 0.55, 0.52, "合并：-4.29pp，CI [-8.57, 0.00]\n没有稳定收益；不能表述为普遍显著负效应。", 12)
    rows = []
    for m, label in zip(METHODS, LABELS):
        s = cohorts[2]["selector_summary"][m]
        rows.append(
            [
                label,
                f"{rate(cohorts[0], m):.2f}%",
                f"{rate(cohorts[1], m):.2f}%",
                f"{rate(cohorts[2], m):.2f}%",
                f"{s['difference_from_canonical_pp']:+.2f}pp",
            ]
        )
    table(
        fig,
        ["选择规则", "初始来源", "新增来源", "合并", "合并差值"],
        rows,
        [0.055, 0.18, 0.89, 0.245],
        [0.30, 0.175, 0.175, 0.175, 0.175],
        10.5,
    )
    note(fig, "调和平均与最小实体规则也未恢复稳定收益；不能仅靠改一个平均方式解决当前问题。")
    finish(pdf, fig, previews, 10)


def page_task_noise(pdf, previews, cohorts, audit):
    fig, _ = base.new_page(
        "任务与噪声分解：变化不只是总体均值",
        "下表只比较同一规则相对同母集规范视角的百分点差；任务级小样本用于诊断，不单独宣称显著性。",
        11,
    )
    rows = []
    for task, label in zip(TASKS, TASK_LABELS):
        vals = []
        for a in cohorts:
            t = a["task_summary"].get(task)
            vals.append(
                "无新增来源"
                if t is None
                else f"{100 * (t['visibility_gain_gated']['mean_repeat_success_rate'] - t['canonical']['mean_repeat_success_rate']):+.1f}pp"
            )
        rows.append([label, *vals])
    table(
        fig,
        ["任务", "初始 24 来源", "新增 18 来源", "合并"],
        rows,
        [0.055, 0.31, 0.51, 0.45],
        [0.40, 0.20, 0.20, 0.20],
        10.5,
    )
    ax = fig.add_axes([0.65, 0.43, 0.29, 0.30])
    noise_items = sorted(audit["per_noise_difference_pp"].items())
    vals = [value for _, value in noise_items]
    bars = ax.bar([str(seed)[-2:] for seed, _ in noise_items], vals, color=base.RED)
    ax.axhline(0, color=base.INK, lw=0.8)
    ax.set_ylim(-13, 2)
    ax.set_xlabel("flow-noise 编号后两位")
    ax.set_ylabel("合并选择增益 pp")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
    clean(ax)
    text(fig, 0.62, 0.30, "新增来源使书、底层抽屉、微波炉的\n正向点估计反转；不能只保留早期正例。", 12)
    note(fig, "共性：收益依赖具体轨迹状态；两个奶酪任务接近满分，适合伤害控制，不适合证明信息收益。")
    finish(pdf, fig, previews, 11)


def montage_tile(scan_path, candidate):
    scan = load(scan_path)
    ranked = sorted(
        [r for r in scan["records"] if r.get("status") != "INVALID"], key=lambda r: float(r["delta_visibility"])
    )
    chosen = ranked[:6] + ranked[-6:]
    canonical = next(r for r in ranked if r["pose_id"] == "canonical")
    chosen = list({r["pose_id"]: r for r in [canonical] + [r for r in chosen if r["pose_id"] != "canonical"]}.values())
    index = next(i for i, r in enumerate(chosen) if r["pose_id"] == candidate)
    path = Path(scan_path).with_name("visibility_extremes.png")
    INPUTS.add(path)
    im = Image.open(path)
    width = 448
    if im.width != min(5, len(chosen)) * width:
        raise ValueError("unexpected montage layout")
    x = (index % 5) * width
    y = (index // 5) * 264
    return im.crop((x, y, x + width, y + 224))


def page_proxy(pdf, previews, audit):
    fig, _ = base.new_page(
        "找到的具体共性：任务区域被整物体像素替代", "原始同状态渲染与实例分割统计；以下为新增来源中的上层抽屉实例。", 12
    )
    example = next(e for e in audit["examples"] if "demo_10::stage-02" in e["pair_key"])
    for x, candidate, label in [
        (0.06, "canonical", "规范视角"),
        (0.53, example["selected_candidate_id"], "可见性最高视角"),
    ]:
        ax = fig.add_axes([x, 0.465, 0.41, 0.29])
        ax.imshow(montage_tile(example["scan_path"], candidate))
        ax.axis("off")
        text(fig, x, 0.79, label, 15, weight="bold")
        text(fig, x, 0.448, "每格左：外部相机；右：腕部相机", 10, base.MUTED)
    table(
        fig,
        ["统计对象", "实体级可见面积增量", "含义"],
        [
            ["被操作的碗", "0.00pp", "碗没有新增可见像素"],
            ["top_region → 整个柜体", "+3.08pp", "实际统计不是抽屉内部 ROI"],
            ["两实体再等权平均", "+1.54pp", "因此被门控选中；5 次成功由 5/5 降为 3/5"],
        ],
        [0.055, 0.22, 0.89, 0.18],
        [0.29, 0.25, 0.46],
        11,
    )
    note(fig, "55 次换视角中有 10 次目标像素不增；其中 8 次由整柜体增益主导。证明指标代理有缺口，尚不证明全部失败机制。")
    finish(pdf, fig, previews, 12)


def page_common(pdf, previews, audit):
    fig, _ = base.new_page(
        "不仅是像素定义：阶段、支持域与通道仍耦合",
        "以下均为事后分组描述；各组任务与状态不相同，不能把组间差值解释为因果效应。",
        13,
    )
    rows = []
    for label, field, key in [
        ("轨迹较早阶段", "stage_bin", "early_le_0.35"),
        ("轨迹较后阶段", "stage_bin", "later_gt_0.35"),
        ("选中训练目录位姿", "catalog_group", "broad_training_64"),
        ("选中同范围留出位姿", "catalog_group", "broad_heldout_32"),
    ]:
        r = audit["breakdown"][field][key]
        rows.append(
            [label, str(r["states"]), f"{r['difference_pp']:+.2f}pp", f"{r['stable_rescue']} / {r['stable_harm']}"]
        )
    table(
        fig,
        ["分组", "状态数", "可见性门控 − 规范", "多数 Rescue / Harm"],
        rows,
        [0.055, 0.48, 0.89, 0.28],
        [0.36, 0.15, 0.24, 0.25],
        12,
    )
    text(fig, 0.065, 0.42, "训练目录位姿仍有失败", 15, base.BLUE, "bold")
    text(
        fig,
        0.065,
        0.36,
        "目录内视角也有 4 个 Harm；因此不能把所有失败\n归因于 exact-pose OOD。目录参数不等于全部视觉输入同分布。",
        11,
    )
    text(fig, 0.545, 0.42, "腕部冗余尚未隔离", 15, base.BLUE, "bold")
    text(
        fig,
        0.545,
        0.36,
        "冻结状态下 wrist 可见性增量严格为 0；\n但执行时 wrist 正常更新，是否提供捷径尚未做本轮消融。",
        11,
    )
    text(fig, 0.065, 0.23, "早 / 后阶段按轨迹比例 0.35 划分，尚不是抓取、搬运、放置等语义阶段标注。", 11, base.MUTED)
    note(fig, "已排除明显配对错误：420 个状态-噪声组物理哈希一致；145 次同视角重复的成功判定完全一致。")
    finish(pdf, fig, previews, 13)


def page_accel(pdf, previews):
    fig, _ = base.new_page(
        "Accel 与可见性：两种代理，尚不能替代行为价值",
        "此页保留独立 B 诊断：8 任务 / 96 中间状态；未以可见性收益筛选，不能直接反驳 A 的角色化检验。",
        14,
    )
    a = load(base.GATE_A97_ROOT / "shortlist/broad64-practical/analysis.json")
    if a.get("status") != "PASS" or a.get("state_count") != 96:
        raise ValueError("invalid frozen Accel shortlist analysis")
    cond = a["condition_success"]
    pairs = [
        ("canonical", "规范"),
        ("accel_single_noise", "Accel 单噪声"),
        ("accel_ensemble", "Accel 六噪声"),
        ("visibility_top1", "可见性最高"),
        ("accel_top10_visibility", "Accel Top10+可见性"),
        ("random_operational", "随机候选"),
    ]
    rows = [[label, f"{100 * cond[key]['state_success_rate']:.1f}%"] for key, label in pairs]
    rows.append(["六种已执行条件的事后并集", f"{100 * a['oracle_at_shortlist_state_rate']:.1f}%"])
    table(fig, ["选择规则", "闭环成功率"], rows, [0.055, 0.27, 0.43, 0.49], [0.72, 0.28], 11)
    text(fig, 0.54, 0.75, "Accel 的实际含义", 16, weight="bold")
    text(
        fig,
        0.54,
        0.68,
        "测量 flow velocity 的相对变化。\n本地已核对公式，不是论文全协议数值复现。\n低值表示该次生成轨迹更直，\n并不直接测量新增任务证据。",
        12,
    )
    text(fig, 0.54, 0.49, "排序与行为应分别判断", 16, weight="bold")
    text(
        fig,
        0.54,
        0.42,
        "97 候选重复 6 次噪声：\n平均 Spearman 约 0.41；Top-1 全同约 2.1%。\n但排序变化不必然改变行为，\n最终仍以左表真实闭环为准。",
        12,
    )
    note(fig, "共性在于代理与目标错位：像素面积不等于任务证据；flow 直线性也不等于闭环收益。两者均待机制验证。")
    finish(pdf, fig, previews, 14)


def page_next(pdf, previews):
    fig, _ = base.new_page(
        "样本充分性：规则已有负证据，机制尚待验证",
        "下一轮优先提高任务区域标注、匹配对照和验证划分的质量，再扩大样本覆盖。",
        15,
    )
    table(
        fig,
        ["问题", "当前判断", "下一项可证伪验证"],
        [
            ["当前像素选择规则可靠？", "没有跨来源稳定收益", "保留为冻结弱基线，不在现有 test 调参"],
            ["正确任务区域更清楚？", "部分 region 回退为整物体", "标注抽屉内部 / 接触区；核对 ROI 分割"],
            ["收益来自任务证据？", "当前 6 规则测试缺等幅 Control", "在开发集构造 Reveal 与位移匹配对照"],
            ["视角 OOD 或腕部捷径？", "现有分组不能因果区分", "相同状态做训练支持匹配与 wrist-on/off"],
            ["最优候选能否被学习？", "只见事后行为上限", "开发集定视角；新噪声与新来源验证"],
        ],
        [0.055, 0.32, 0.89, 0.44],
        [0.25, 0.32, 0.43],
        11,
    )
    text(fig, 0.065, 0.25, "新增训练与规则设计仅使用开发来源；当前已查看的 42 条 test 来源不再承担后续确认性测试。", 12)
    note(fig, "建议顺序：修 ROI 与人工审计 → 冻结候选并独立复验 → 再检验关系/阶段条件化的最小选择器。")
    finish(pdf, fig, previews, 15)


def page_scope(pdf, previews, audit):
    fig, _ = base.new_page(
        "统计与复现边界", "本次更新包含离线审计与报告生成；未新增训练、未重新采样测试结果、未修改原可见性公式。", 16
    )
    table(
        fig,
        ["术语", "本报告中的严格定义"],
        [
            ["5 个 seed", "5 个推理 flow-noise；新增选择器评测只使用 1 个训练 checkpoint"],
            ["95% CI", "按源演示成组重采样；对既定任务集与噪声集合条件化的区间"],
            [
                "多数成功 / Rescue / Harm",
                "多数成功：5 次至少 3 次；Rescue：规范多数失败、选择后多数成功\nHarm：规范多数成功、选择后多数失败；均为描述性标签",
            ],
            ["Oracle / Best-of-8", "事后查结果得到的经验上限；bootstrap 不消除取最大值的乐观偏差"],
            ["人工质量门", "全量人工审查回执 0/84；本次仅复核有限示例，不能视为全量人工通过"],
        ],
        [0.055, 0.38, 0.89, 0.38],
        [0.27, 0.73],
        11,
    )
    text(fig, 0.065, 0.345, "多规则与事后分组属于探索性诊断；所示区间未作多重比较校正。", 10.5, base.MUTED)
    text(fig, 0.065, 0.30, "结果入口", 14, weight="bold")
    text(
        fig,
        0.065,
        0.255,
        "constructed-all8-v1 / independent-source-extension / combined-analysis / analysis.json\nconstructed-all8-v1 / independent-source-extension / common-failure-audit-20260831\nscripts/dsol_paper1/build_view_revalidation_report_pdf.py",
        10.5,
        base.MUTED,
    )
    note(fig, "最终判断：已有具体的指标缺口与复现失败证据；尚无充分依据认定主动视角无用或失败原因已经唯一定位。")
    finish(pdf, fig, previews, 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "docs/dsol_paper1/view_revalidation_report_zh.pdf")
    parser.add_argument("--preview-dir", type=Path, required=True)
    args = parser.parse_args()
    base.configure_font()
    # The installed CJK collection uses CFF outlines, not TrueType outlines.
    # Type 3 embeds those glyphs correctly in PDF readers; Type 42 does not.
    plt.rcParams.update({"axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "pdf.fonttype": 3})
    cohorts = [
        load(DATA / "dense-test-selector-eval/analysis/analysis.json"),
        load(EXT / "dense-test-selector-eval/analysis/analysis.json"),
        load(EXT / "combined-analysis/analysis.json"),
    ]
    if [(a["states"], a["source_groups"], a["episodes"]) for a in cohorts] != [
        (48, 24, 1440),
        (36, 18, 1080),
        (84, 42, 2520),
    ]:
        raise ValueError("unexpected frozen study sizes")
    if any(a["status"] != "PASS" or a["expected_repeats"] != 5 for a in cohorts):
        raise ValueError("test incomplete")
    audit = load(AUDIT)
    if audit["status"] != "PASS" or (audit["states"], audit["sources"]) != (84, 42):
        raise ValueError("common-failure audit incomplete or pairing checks failed")
    repeat = load(DATA / "dense-repeatability/analysis/analysis.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        page_cover(pdf, args.preview_dir, cohorts[2])
        page_map(pdf, args.preview_dir)
        base.page_data(pdf, args.preview_dir)
        base.page_training(pdf, args.preview_dir)
        base.page_passive(pdf, args.preview_dir)
        base.page_formal_benchmarks(pdf, args.preview_dir, clarify_estimands=True)
        page_legacy_a(pdf, args.preview_dir)
        page_design(pdf, args.preview_dir)
        page_headroom(pdf, args.preview_dir, repeat)
        page_test(pdf, args.preview_dir, cohorts)
        page_task_noise(pdf, args.preview_dir, cohorts, audit)
        page_proxy(pdf, args.preview_dir, audit)
        page_common(pdf, args.preview_dir, audit)
        page_accel(pdf, args.preview_dir)
        page_next(pdf, args.preview_dir)
        page_scope(pdf, args.preview_dir, audit)
        pdf.infodict().update(
            Title="视角覆盖、可见性与闭环价值",
            Author="AlphaBrain research workflow",
            Subject="Frozen-source replication and descriptive failure audit; 2026-08-31",
        )
    for p in [
        Path(__file__),
        Path(base.__file__),
        base.PASSIVE_FIGURE,
        base.CAMERA_FORMAL_METRICS,
        base.ORIGINAL_FORMAL_METRICS,
    ]:
        INPUTS.add(p)
    receipt = {
        "schema": "dsol_report_build_v1",
        "pages": 16,
        "output": str(args.output),
        "git_head_at_build": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUTS)},
        "pdf_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    (args.preview_dir / "build_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {"status": "COMPLETE", "pages": 16, "pdf": str(args.output), "preview_dir": str(args.preview_dir)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
