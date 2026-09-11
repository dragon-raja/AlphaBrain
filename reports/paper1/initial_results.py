#!/usr/bin/env python3
"""Figure-led 12-page current research report from audited INITIAL32 outputs."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from scipy.stats import rankdata
from PIL import Image, ImageDraw

try:
    import pymupdf as fitz
except ImportError:
    sys.path.insert(0, "/root/.cache/uv/archive-v0/XO8_y4zBEt33DOY7wfN64")
    import pymupdf as fitz
from reports.paper1.legacy_style import Brief, INK, MUTED, BLUE, TEAL, GRAY, AMBER, LINE
from AlphaBrain.research.dsol.data.artifacts import read, write, sha
from AlphaBrain.research.dsol.analysis.pipeline import load_config
from AlphaBrain.research.dsol.plotting.view_space import (
    candidate_coordinates,
    projected_field,
    success_matrix,
    spatial_field,
)

REPO = Path(__file__).resolve().parents[2]
_config = load_config(REPO / "configs/dsol_paper1/analysis/standard_initial_v1.json")
ROOT, ANALYSIS = _config.root, _config.output
DOCS = REPO / "docs/dsol_paper1"
PUBLIC = DOCS / "reports/current/vla_view_research_progress_zh.pdf"
ARCHIVE = Path("/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908/unified-research-progress-v1")
COLORS = [BLUE, TEAL]
MODEL_LABELS = ["规范视角训练", "多视角训练"]
TASK_LABELS = ["奶酪入碗", "碗入上抽屉", "酒瓶入架", "书入笔筒", "碗入下抽屉", "杯入微波炉", "奶酪入篮", "抽屉旁碗入盘"]
METHOD_LABELS = {
    "canonical": "规范视角",
    "uniform": "均匀候选",
    "min_accel": "最低平均 Accel",
    "visibility": "最大外部可见性",
    "accel10_visibility": "Accel Top-10＋可见性",
    "dev_global_fixed": "开发集全局固定",
    "dev_task_fixed": "开发集每任务固定",
    "geometry_context_ridge": "几何＋规范上下文",
    "candidate_image_ridge": "额外查询候选图",
    "initial_oracle": "每初态经验 Oracle",
}


def validate_current_texts(texts):
    if len(texts) != 12:
        raise ValueError("Expected 12 pages")
    joined = "\n".join(texts)
    for word in ["标准初态", "32", "97", "198,656", "无额外遮挡", "探索性", "开发", "初态 2、3"]:
        if word not in joined:
            raise ValueError("Missing scope: " + word)
    for bad in ["主实验尚缺", "尚未完成本版主矩阵", "62.99", "75.88"]:
        if bad in joined:
            raise ValueError("Stale result claim: " + bad)
    if "人为遮挡" not in texts[1] or "历史" not in texts[2] or "历史" not in texts[3]:
        raise ValueError("Historical conditions not distinguished")
    if "候选图" not in texts[10] or "已查看" not in texts[10]:
        raise ValueError("Selector permissions / post-hoc status missing")


class CurrentBrief(Brief):
    def __init__(self, archive):
        super().__init__()
        self.archive = archive
        self.summary = read(archive / "summary.json")
        self.release = read(ROOT / "release.json")
        self.states = read(archive / "states.json")
        with np.load(archive / "matrix.npz") as a:
            self.y = a["success"]
            self.acc = a["accel_members"]
            self.vis = a["visibility"]
            self.cids = a["candidate_ids"]
        self.q = self.y.mean(axis=-1)
        self.s = [self.summary["models"][m] for m in ["canonical", "broad"]]
        self.per_initial = read(archive / "per-initial.json")
        self.figures = []

    def new(self, title, stage, subtitle, takeaway, source, note):
        if title == "外部多视角后训练改善被动视角鲁棒性":
            title = "历史训练消融：覆盖收益明确，约束增益未确认"
            stage = "历史证据 · 不与新初态结果直接合并"
            takeaway = "支持视角覆盖的研究动机；不将未确认增益解释为配对或一致性普遍无效。"
        if title == "宽视角后训练的鲁棒性与原始任务性能权衡":
            title = "历史整体基准：Camera 鲁棒性与 Original 性能"
            stage = "历史证据 · 40 任务基准，不是本轮 8 任务矩阵"
        super().new(title, stage, subtitle, takeaway, source, note)
        self.artists[1].set_text(f"{len(self.notes):02d} / 12")
        for t in self.artists:
            if "2026-09-07" in t.get_text():
                t.set_text("VLA VIEW RESEARCH  ·  2026-09-11  ·  标准初态结果与离线选择器")
        self.artists[2].set_fontsize(25)
        # Footer takeaway is a compact conclusion, not a paragraph.
        for t in self.artists:
            if t.get_text() == takeaway:
                t.set_fontsize(17)

    def save(self, pdf):
        i = len(self.notes)
        self.fig.savefig(self.figure_dir / f"page-{i:02d}.png", dpi=140, facecolor="white")
        self.figures.append(str(self.figure_dir / f"page-{i:02d}.png"))
        super().save(pdf)

    def method(self, m, key, scope="heldout16"):
        return self.s[m]["scopes"][scope]["methods"][key]

    def overview(self, pdf):
        self.new(
            "如何利用 VLA 视角：训练覆盖、候选收益与选择",
            "研究主线 · 当前证据",
            "已完成：8 个已知任务 × 4 个标准初态 × 97 候选 × 32 套噪声 × 2 模型 = 198,656 次闭环",
            "训练扩大有效视角范围；超过规范视角的收益仍需由选择方法证明。",
            "本轮：标准初态、无额外遮挡，外部相机单次执行内固定；腕部输入随动作变化。不是动态相机策略或多路外部图像融合。",
            "先对同一初态同一视角的 32 套完整噪声轨迹求平均，再比较视角。所有主结果来自同一个新矩阵。历史遮挡恢复实验仅用于提出信息视角假设，不与当前绝对成功率相减。",
        )
        cards = [
            (0.065, "训练覆盖", "均匀候选成功率", "22.28 → 75.07%", BLUE),
            (0.38, "剩余收益", "多视角模型：规范 → Oracle", "90.23 → 96.00%", TEAL),
            (0.695, "视角利用", "最低 Accel 未超过规范", "87.70 < 90.23%", AMBER),
        ]
        for x, title, body, value, color in cards:
            self.box(x, 0.44, 0.245, 0.27)
            self.text(x + 0.015, 0.68, title, 21, color, True)
            self.text(x + 0.015, 0.60, body, 11, MUTED)
            self.text(x + 0.015, 0.54, value, 21, color, True)
        self.text(0.075, 0.365, "主问题：训练之后，何时需要选择视角，哪些信息足以指导选择？", 18, INK, True)
        self.text(0.075, 0.30, "经验 Oracle 是有限矩阵参照；指标与学习器必须与规范视角、开发集固定视角比较。", 12, MUTED)
        self.save(pdf)

    def protocol(self, pdf):
        self.new(
            "评测条件已改变：新旧百分比不能直接比较",
            "协议边界 · 必须先对齐",
            "任务名称与模型权重保持一致；初始化、场景条件和起点数量不同，不是旧结果的数值修正。",
            "当前主线采用自然场景的标准初态；人为遮挡的作用尚未做单因素对照。",
            "新协议：官方 init 0–3；10 步统一稳定；AA0；实际首帧与指标输入一致；跨机与调度重复轨迹验收通过。",
            "旧 64 起点来自演示恢复快照并注入无碰撞视觉遮挡物；新 32 起点为官方标准初始化且无额外遮挡。不能将绝对成功率变化归因于某个单独因素，也不自动恢复或扩展遮挡评测。",
        )
        rows = [
            ["项目", "旧候选实验（辅助）", "本轮主实验"],
            ["任务与模型", "8 已知任务；同一模型", "8 已知任务；权重未变"],
            ["起点", "64 个演示恢复快照", "32 个官方标准初态"],
            ["场景", "人为遮挡物", "无额外遮挡／不改背景"],
            ["任务执行", "从快照完成剩余任务", "从任务初态完整执行"],
            ["统计含义", "恢复后续跑成功率", "初态整任务成功率"],
        ]
        ax = self.fig.add_axes([0.065, 0.37, 0.87, 0.37])
        ax.axis("off")
        table = ax.table(cellText=rows, cellLoc="center", loc="center", colWidths=[0.20, 0.40, 0.40])
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 1.9)
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor(LINE)
            cell.set_facecolor("#EFF4FA" if i == 0 else "white")
        self.text(
            0.073, 0.29, "每初态先平均 32 套噪声，再取视角最大；不逐噪声挑赢家，不在任务过程中切换外部位姿。", 12, INK
        )
        self.save(pdf)

    def coverage(self, pdf):
        self.new(
            "多视角训练扩大有效范围，但规范视角仍是强基线",
            "01 · 训练覆盖",
            "新初态矩阵；两模型使用相同初态、视角与显式噪声。下图均为完整任务成功率。",
            "覆盖训练主要改善非规范视角表现；不能把覆盖收益等同于规范视角性能提升。",
            "32 初态任务等权。训练目录／留出目录指位姿目录来源，不等同于未见任务或训练来源无重叠。每模型仅一个后训练 seed。",
            "均匀视角不是同时输入 97 路相机，而是对各单视角闭环成绩等权平均。右图展示各任务的变化，避免由少数任务掩盖异质性。",
        )
        keys = ["canonical", "training_catalog_64", "heldout_catalog_32"]
        ax = self.axes([0.08, 0.34, 0.40, 0.40])
        x = np.arange(3)
        for m in range(2):
            vals = [self.s[m]["candidate_groups"][k] * 100 for k in keys]
            ax.bar(x + (m - 0.5) * 0.34, vals, 0.34, color=COLORS[m], label=MODEL_LABELS[m])
            for xx, v in zip(x + (m - 0.5) * 0.34, vals):
                ax.text(xx, v + 1.3, f"{v:.1f}", ha="center", fontsize=11)
        ax.set_xticks(x, ["规范", "训练目录 64", "留出目录 32"], fontsize=11)
        ax.set_ylim(0, 106)
        ax.set_ylabel("成功率（%）", fontsize=12)
        ax.legend(fontsize=10, frameon=False, loc="upper right")
        ax = self.axes([0.67, 0.34, 0.26, 0.40], True, False)
        gain = (self.q[1].mean(axis=1) - self.q[0].mean(axis=1)).reshape(8, 4).mean(axis=1) * 100
        ax.barh(np.arange(8), gain, color=TEAL)
        ax.set_yticks(range(8), TASK_LABELS, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("均匀候选：多视角 − 规范训练（pp）", fontsize=9)
        for i, v in enumerate(gain):
            ax.text(v + 1, i, f"{v:+.1f}", va="center", fontsize=9)
        ax.set_xlim(0, max(gain) + 12)
        e = self.summary["training_effect"]["uniform"]
        lo, hi = np.array(e["ci95"]) * 100
        self.text(
            0.08,
            0.27,
            f"均匀候选净增益：{e['mean'] * 100:+.2f} pp；分层 bootstrap 95% 区间 [{lo:+.2f}, {hi:+.2f}]。",
            12,
            INK,
        )
        self.save(pdf)

    def space(self, pdf):
        self.new(
            "候选位姿分布：成功区域与低 Accel 区域是否一致",
            "02 · 实测空间分布",
            "96 个非规范候选：横轴方位角、纵轴仰角、点大小表示半径倍率；颜色表示成功率或 Accel 偏好。",
            "候选分布可以直接复用矩阵分析；低 Accel 与高成功率并不完全重合。",
            "方位角／仰角／半径倍率是候选目录参数，不是完整 SE(3)。规范相机无该参数化，单独统计；图中按 32 初态平均。",
            "上排是噪声平均再对初态平均的完整任务成功率。下排先在每个初态内对负 Accel 排名，再取平均；颜色越高表示越偏好该候选，不是成功率。",
        )
        poses = [c["pose"] for c in self.release["protocol"]["state_blocks"][0]["candidates"][1:]]
        xyz = candidate_coordinates(self.release["protocol"]["state_blocks"][0]["candidates"])
        for row in range(2):
            for m in range(2):
                ax = self.axes([0.09 + m * 0.46, 0.535 - row * 0.27, 0.32, 0.20], True, True)
                if row == 0:
                    val = self.q[m, :, 1:].mean(axis=0) * 100
                else:
                    val = np.mean([rankdata(-v[1:]) / 96 * 100 for v in self.acc[m].mean(axis=-1)], axis=0)
                sc = projected_field(ax, xyz, val)
                if row == 1:
                    ax.set_xlabel("方位角（度）", fontsize=9, labelpad=1)
                ax.set_ylabel("仰角（度）", fontsize=9, labelpad=1)
                ax.tick_params(labelsize=8)
                ax.set_xlim(-65, 65)
                ax.set_ylim(-29, 29)
                ax.set_title(
                    MODEL_LABELS[m] + (" · 成功率" if row == 0 else " · 低 Accel 偏好百分位"), fontsize=11, pad=6
                )
                cax = self.fig.add_axes([0.435 + m * 0.46, 0.55 - row * 0.27, 0.009, 0.17])
                cb = self.fig.colorbar(sc, cax=cax)
                cb.ax.tick_params(labelsize=7)
                cb.set_ticks([0, 50, 100])
        self.save(pdf)

    def matrix(self, pdf):
        self.new(
            "完整初态—候选矩阵：训练覆盖改变失败区域",
            "02 · 全量证据，不按结果筛选",
            "每行一个标准初态，每格为 32 次完整任务的平均成功率；两个模型使用完全相同的行列顺序。",
            "全量矩阵保留任务与初态差异；不能只展示少数有利视角或成功案例。",
            "列顺序：规范视角 | 训练目录 64 | 留出目录 32。每任务四行依次为初态 0–3；黑色横线分隔任务。",
            "该图不是时间序列，也不是每个动作时刻切换相机。它对全部 32 个标准初态等权显示，没有筛选失败起点或困难任务。",
        )
        for m in range(2):
            ax = self.fig.add_axes([0.14 + m * 0.43, 0.29, 0.34, 0.45])
            im = success_matrix(ax, self.q[m])
            ax.set_title(MODEL_LABELS[m], fontsize=15, color=COLORS[m], pad=10)
            ax.set_yticks(np.arange(8) * 4 + 1.5, TASK_LABELS if m == 0 else [], fontsize=9)
            ax.set_xticks([0, 32, 80], ["规范", "训练目录", "留出目录"], fontsize=10)
            for j in range(1, 8):
                ax.axhline(j * 4 - 0.5, color="black", lw=0.4)
            for j in [0.5, 64.5]:
                ax.axvline(j, color="white", lw=0.8)
        cb = self.fig.colorbar(im, cax=self.fig.add_axes([0.932, 0.31, 0.013, 0.40]))
        cb.ax.tick_params(labelsize=9)
        cb.set_label("成功率（%）", fontsize=10)
        self.save(pdf)

    def headroom(self, pdf):
        self.new(
            "多视角训练之后，仍存在有限的经验候选收益",
            "03 · 固定摆放与逐初态上限",
            "全部 32 初态，同一候选空间、噪声与任务权重；每个初态先对噪声求平均，再选最佳视角。",
            "多视角模型的经验 Oracle 较规范视角高 5.76 pp；这不是已实现的选择器收益。",
            "所有经验最佳项均可选择规范视角。最大值来自同批结果，存在有限重复的选择偏差，不视为真实期望上限或独立复评。",
            "全局固定对所有初态使用同一个目录候选；每任务固定允许任务间不同；每初态 Oracle 允许同一任务不同初态不同。单次执行内均不换外部位姿。",
        )
        keys = ["canonical", "global_best", "task_best", "initial_oracle"]
        for m in range(2):
            ax = self.axes([0.075 + m * 0.47, 0.33, 0.39, 0.40])
            h = self.s[m]["scopes"]["all32"]["hierarchy"]
            vals = [h[k] * 100 for k in keys]
            ax.bar(range(4), vals, color=[GRAY, "#7594AE", BLUE, TEAL], width=0.68)
            ax.set_ylim(0, 106)
            ax.set_xticks(range(4), ["规范", "全局最佳", "每任务最佳", "每初态 Oracle"], fontsize=10)
            ax.set_title(MODEL_LABELS[m], fontsize=17, color=COLORS[m])
            ax.set_ylabel("成功率（%）", fontsize=12)
            for i, v in enumerate(vals):
                ax.text(i, v + 1.5, f"{v:.2f}", ha="center", fontsize=13, fontweight="bold")
            self.text(
                0.085 + m * 0.47,
                0.27,
                f"初态 Oracle − 规范：{(h['initial_oracle'] - h['canonical']) * 100:+.2f} pp",
                13,
                COLORS[m],
                True,
            )
        self.save(pdf)

    def rules(self, pdf):
        self.new(
            "指标能避开部分差视角，但尚未超过规范基线",
            "04 · 预定规则，留出初态评价",
            "仅初态 2、3：16 个评价初态；规则不使用该初态的闭环成败选视角。",
            "与随机视角相比的改善，不等于超过规范视角；两种比较必须同时报告。",
            "线段为按任务—初态配对分层 bootstrap 95% 区间。可见性使用仿真实体分割；Accel 需查询候选图，均不计实际相机获取成本。",
            "规则是事前保留的最低平均 Accel、最大外部可见性和 Accel Top-10 后可见性精排。可见性为外部相机指标，不混入腕部平均。均匀候选为矩阵期望。",
        )
        keys = ["canonical", "uniform", "min_accel", "visibility", "accel10_visibility"]
        for m in range(2):
            ax = self.axes([0.26 + m * 0.36, 0.33, 0.27, 0.40], True, False)
            for i, key in enumerate(keys):
                a = self.method(m, key)["success"]
                v = a["mean"] * 100
                lo, hi = np.array(a["ci95"]) * 100
                ax.errorbar(v, i, xerr=[[max(0, v - lo)], [max(0, hi - v)]], fmt="o", color=COLORS[m], capsize=3)
                ax.text(v, i - 0.18, f"{v:.2f}", ha="center", fontsize=10)
            ax.set_yticks(range(len(keys)), [METHOD_LABELS[k] for k in keys] if m == 0 else [], fontsize=11)
            ax.set_ylim(4.6, -0.6)
            ax.set_xlim(0, 104)
            ax.set_xlabel("成功率（%）", fontsize=11)
            ax.set_title(MODEL_LABELS[m], fontsize=15, color=COLORS[m])
        self.text(
            0.075, 0.265, "多视角模型留出结果：最低 Accel 86.52%，规范视角 88.67%；Top-10＋可见性 86.13%。", 12, INK
        )
        self.save(pdf)

    def metric_diagnostics(self, pdf):
        self.new(
            "指标诊断：排序相关性与评分噪声稳定性",
            "04 · 指标不是闭环价值的替代品",
            "左：每个初态内的候选排序相关性；右：8 个评分初噪各自的最优候选与均值最优的一致程度。",
            "本轮多视角训练后，Accel 排序相关性减弱；可见性单独解释力有限。",
            "Accel：前三步去噪速度变化的归一化分数，非机器人执行加速度；8 个评分初噪取均值。\n左图为 32 个初态内的相关系数。右图使用评分初噪，不是 32 套完整闭环噪声。",
            "Accel_3 = 3 × (||v1−v0|| + ||v2−v1||) / (||v0||+||v1||+||v2||)，v 为 Flow 去噪速度预测，范数覆盖动作序列维度。它不是机器人真实加速度。这些是机制诊断，未执行因果干预；不可据相关性确认遮挡机制、腕部捷径或内生主动感知能力。",
        )
        ax = self.axes([0.12, 0.34, 0.39, 0.39])
        values = []
        for m in range(2):
            rs = self.s[m]["rank_correlations"]
            values.extend(
                [[r[k] for r in rs if r[k] is not None] for k in ["minus_accel_spearman", "visibility_spearman"]]
            )
        bp = ax.boxplot(values, positions=[0, 1, 3, 4], widths=0.5, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], [BLUE, BLUE, TEAL, TEAL]):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
        rng = np.random.default_rng(3)
        for x, vs, color in zip([0, 1, 3, 4], values, [BLUE, BLUE, TEAL, TEAL]):
            ax.scatter(x + rng.uniform(-0.15, 0.15, len(vs)), vs, s=10, color=color, alpha=0.65)
        ax.axhline(0, color=GRAY, lw=1)
        ax.set_ylim(-1, 1)
        ax.set_ylabel("与闭环成功率的 Spearman ρ", fontsize=11)
        ax.set_xticks(
            [0, 1, 3, 4],
            ["−Accel\n规范训练", "可见性\n规范训练", "−Accel\n多视角训练", "可见性\n多视角训练"],
            fontsize=9,
        )
        ax = self.axes([0.64, 0.34, 0.29, 0.39])
        bins = np.arange(-0.0625, 1.13, 0.125)
        for m in range(2):
            v = [a["agreement_with_ensemble_winner"] for a in self.s[m]["score_noise_stability"]]
            ax.hist(v, bins=bins, alpha=0.5, color=COLORS[m], label=MODEL_LABELS[m])
        ax.set_xlabel("单噪声最优与均值最优一致比例", fontsize=10)
        ax.set_ylabel("初态数", fontsize=11)
        ax.legend(frameon=False, fontsize=10)
        med = [
            np.median(
                [v["minus_accel_spearman"] for v in s["rank_correlations"] if v["minus_accel_spearman"] is not None]
            )
            for s in self.s
        ]
        self.text(
            0.085,
            0.265,
            f"−Accel 与成功率相关系数的初态中位数：规范训练 {med[0]:.2f}，多视角训练 {med[1]:.2f}。",
            12,
            INK,
        )
        self.save(pdf)

    def learners(self, pdf):
        self.new(
            "离线选择器：区分无需候选图与候选图可查询",
            "05 · 探索性学习实验",
            "初态 0、1 开发；初态 2、3 评价。固定视觉编码器＋岭回归；超参数仅由开发集两折选择。",
            "多视角模型上，两类学习器均未超过简单的每任务固定基线。",
            "已查看本轮部分测试汇总后设计，属于探索性结果，不是完全盲测确认。仿真标签复用自完整矩阵；新增闭环 0 次，VLA 不重训。",
            "几何上下文模型用规范外部图、腕部图、任务 ID 与候选相机几何，不看候选图或分割。候选图模型额外查询各候选 RGB；不使用仿真可见性。PCA、标准化与回归均在开发折拟合，测试标签不用于训练或调参。两者仍是假定选一次固定视角，不是移动相机策略。",
        )
        keys = ["canonical", "dev_task_fixed", "geometry_context_ridge", "candidate_image_ridge"]
        ax = self.axes([0.285, 0.36, 0.36, 0.36], True, False)
        for m in range(2):
            for i, key in enumerate(keys):
                a = self.method(m, key)["gain_vs_canonical"]
                v = 100 * a["mean"]
                lo, hi = np.array(a["ci95"]) * 100
                yy = i + (m - 0.5) * 0.30
                ax.errorbar(v, yy, xerr=[[max(0, v - lo)], [max(0, hi - v)]], fmt="o", color=COLORS[m], capsize=3)
                ax.text(v, yy - 0.10, f"{v:+.2f}", fontsize=8, ha="center")
        ax.axvline(0, color=GRAY, lw=1)
        ax.set_yticks(range(4), [METHOD_LABELS[k] for k in keys], fontsize=11)
        ax.set_ylim(3.6, -0.6)
        ax.set_xlabel("相对规范视角的成功率差（pp，95% 区间）", fontsize=9)
        self.text(0.70, 0.72, "选择时的输入权限", 16, INK, True)
        self.text(0.70, 0.655, "几何＋规范上下文\n不需要候选图\n\n额外查询候选图\n需要多视角采集或渲染", 12, MUTED)
        self.text(0.70, 0.39, "蓝：规范视角训练\n绿：多视角训练", 11, MUTED)
        vals = [self.method(1, k)["success"]["mean"] * 100 for k in keys]
        self.text(
            0.075, 0.27, "多视角模型成功率：" + " / ".join(f"{v:.2f}%" for v in vals) + "（按左图顺序）。", 11, INK
        )
        self.save(pdf)

    def supporting_assets(self, output):
        """Full per-task 3D plots and exact values, without extending the slide deck."""
        directory = output / "task-space"
        directory.mkdir()
        poses = [c["pose"] for c in self.release["protocol"]["state_blocks"][0]["candidates"][1:]]
        xyz = np.array([[p["azimuth_deg"], p["elevation_deg"], p["radius_scale"]] for p in poses])
        for task in range(8):
            fig = plt.figure(figsize=(12, 8), facecolor="white")
            for row in range(2):
                for m in range(2):
                    ax = fig.add_subplot(2, 2, row * 2 + m + 1, projection="3d")
                    if row == 0:
                        value = self.q[m, task * 4 : task * 4 + 4, 1:].mean(axis=0) * 100
                    else:
                        value = np.mean(
                            [rankdata(-v[1:]) / 96 * 100 for v in self.acc[m, task * 4 : task * 4 + 4].mean(axis=-1)],
                            axis=0,
                        )
                    sc = spatial_field(ax, xyz, value)
                    ax.set_title(
                        MODEL_LABELS[m] + ("：成功率（%）" if row == 0 else "：低 Accel 偏好百分位"), fontsize=12
                    )
                    ax.set_xlabel("方位角（度）", fontsize=10)
                    ax.set_ylabel("仰角（度）", fontsize=10)
                    ax.set_zlabel("半径倍率", fontsize=9, labelpad=12)
                    ax.tick_params(labelsize=8)
                    ax.view_init(24, -60)
                    fig.colorbar(sc, ax=ax, shrink=0.60, pad=0.12)
            fig.suptitle(TASK_LABELS[task] + "｜4 个标准初态平均；96 非规范候选；无插值", fontsize=16)
            fig.subplots_adjust(top=0.91, bottom=0.08, left=0.02, right=0.98, hspace=0.18, wspace=0.04)
            fig.savefig(directory / f"task-{task + 1:02d}.png", dpi=160)
            plt.close(fig)
        with (output / "candidate-values.csv").open("w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(
                [
                    "model",
                    "task",
                    "initial_index",
                    "candidate_id",
                    "success",
                    "mean_accel",
                    "external_visibility",
                    "azimuth_deg",
                    "elevation_deg",
                    "radius_scale",
                ]
            )
            for m, model in enumerate(["canonical", "broad"]):
                for i, state in enumerate(self.states):
                    for c, cid in enumerate(self.cids):
                        pose = poses[c - 1] if c else {}
                        w.writerow(
                            [
                                model,
                                state["task_id"],
                                state["initialization"]["init_state_index"],
                                cid,
                                float(self.q[m, i, c]),
                                float(self.acc[m, i, c].mean()),
                                float(self.vis[i, c]),
                                *[pose.get(k, "") for k in ["azimuth_deg", "elevation_deg", "radius_scale"]],
                            ]
                        )
        with (output / "heldout-methods.csv").open("w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(
                [
                    "model",
                    "method",
                    "success_percent",
                    "gain_vs_canonical_pp",
                    "paired_ci95_low_pp",
                    "paired_ci95_high_pp",
                ]
            )
            for m, model in enumerate(["canonical", "broad"]):
                for key, a in self.s[m]["scopes"]["heldout16"]["methods"].items():
                    w.writerow(
                        [
                            model,
                            key,
                            a["success"]["mean"] * 100,
                            a["gain_vs_canonical"]["mean"] * 100,
                            *[x * 100 for x in a["gain_vs_canonical"]["ci95"]],
                        ]
                    )

    def synthesis(self, pdf):
        self.new(
            "研究结论与下一步：先完成视角利用的证据闭合",
            "收敛：不把探索性结果写成已完成的主动感知方法",
            "当前研究范围：已知任务、自然场景、固定外部位姿；两个模型各一个后训练 seed。",
            "已有覆盖与选择分析证据；主动获取、遮挡机制和跨任务泛化仍需独立验证。",
            "本轮不扩任务／初态，不新增遮挡或闭环评测。未来新噪声与更多初态验证须另行确认；历史 AA4 结果不与新矩阵混合统计。",
            "论文可围绕训练覆盖与视角价值的关系组织系统分析，但不能将多视角训练的已有常识本身作为全部创新。需证明评价与选择框架带来可靠新发现或可用方法。是否补遮挡，应该做同初态受控对照，不比较不同总体的历史百分比。",
        )
        blocks = [
            (
                0.06,
                "本轮支持",
                "覆盖提高候选空间整体表现\n规范视角仍是强参考\n存在逐初态经验收益\n简单指标不能等同任务价值",
            ),
            (
                0.375,
                "本轮未证明",
                "选择收益的新噪声稳定性\n跨任务／训练来源泛化\n真实相机获取净收益\n遮挡的独立因果作用",
            ),
            (
                0.69,
                "后续优先级",
                "冻结最终方法与基线\n少量新噪声确认（待授权）\n必要时补独立初态\n遮挡／主动相机另立对照",
            ),
        ]
        for x, title, text in blocks:
            self.box(x, 0.32, 0.255, 0.40)
            self.text(x + 0.015, 0.69, title, 19, BLUE, True)
            self.text(x + 0.015, 0.61, text, 12, MUTED)
        self.save(pdf)


def build():
    latest = read(ANALYSIS / "latest.json")
    source = Path(latest["archive"])
    for name, digest in latest["artifacts"].items():
        if sha(source / name) != digest:
            raise ValueError("Analysis artifact changed: " + name)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = ARCHIVE / "history" / stamp
    out.mkdir(parents=True, exist_ok=False)
    (out / "figures").mkdir()
    (out / "source").mkdir()
    for script in [
        *sorted((REPO / "AlphaBrain/research/dsol").rglob("*.py")),
        *sorted((REPO / "reports/paper1").glob("*.py")),
    ]:
        destination = out / "source" / script.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(script, destination)
    previous = {}
    for p in [PUBLIC, PUBLIC.with_suffix(".md")]:
        if p.exists():
            dest = out / ("previous-" + p.name)
            shutil.copyfile(p, dest)
            previous[str(p)] = {"sha256": sha(p), "archive": str(dest)}
    brief = CurrentBrief(source)
    brief.figure_dir = out / "figures"
    with PdfPages(out / "panels.pdf") as pdf:
        for fn in [
            brief.overview,
            brief.protocol,
            brief.page2,
            brief.page3,
            brief.coverage,
            brief.space,
            brief.matrix,
            brief.headroom,
            brief.rules,
            brief.metric_diagnostics,
            brief.learners,
            brief.synthesis,
        ]:
            fn(pdf)
    brief.supporting_assets(out)
    layout_issues = [issue for page in brief.qa for issue in page["issues"]]
    write(out / "layout.json", brief.qa)
    if layout_issues:
        raise ValueError("Out-of-page text; inspect " + str(out / "layout.json"))
    target = out / "vla_research_progress.pdf"
    with fitz.open(out / "panels.pdf") as doc:
        doc.set_toc([[1, n["title"], i + 1] for i, n in enumerate(brief.notes)])
        doc.set_metadata(
            {
                "title": "VLA 研究进展｜标准初态、训练覆盖与视角选择",
                "author": "VLA View Research",
                "subject": "2026-09-11：完整矩阵与探索性离线选择器；无新增闭环",
            }
        )
        doc.save(target, garbage=4, deflate=True)
    qa = out / "pdf-qa"
    qa.mkdir()
    contact = Image.new("RGB", (1600, 6 * 470), "#DDE5ED")
    draw = ImageDraw.Draw(contact)
    with fitz.open(target) as doc:
        texts = [p.get_text() for p in doc]
        validate_current_texts(texts)
        for i, p in enumerate(doc):
            if "\ufffd" in texts[i] or len(texts[i]) < 150:
                raise ValueError("PDF text corruption")
            png = qa / f"page-{i + 1:02d}.png"
            p.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False).save(png)
            im = Image.open(png).convert("RGB")
            im.thumbnail((780, 439))
            x = i % 2 * 800 + 10
            y = i // 2 * 470 + 25
            contact.paste(im, (x, y))
            draw.text((x, y - 18), str(i + 1), fill=INK)
    contact.save(qa / "contact.jpg", quality=93)
    notes = [
        "# VLA 研究进展｜标准初态与离线选择器（2026-09-11）",
        f"唯一 PDF：[当前报告]({PUBLIC})。本轮新增闭环 0 次，VLA 训练 0 步。",
        "## 实验范围与新旧差异",
        "两模型权重与 8 个任务不变。旧实验为 64 个演示恢复快照并添加人为遮挡；新实验为每任务 4 个官方标准初态，共 32 个，不添加遮挡或背景干预。初始化与渲染协议也不同，不能将绝对成功率变化归因于单一因素。",
        "## 选择器与统计边界",
        "初态 0、1 开发，2、3 评价。两类冻结 ResNet18 特征＋岭回归，仅开发折进行 PCA、标准化和超参数选择。任务 ID 不是对未见任务的语言泛化；候选图模型需要额外图像查询。部分测试汇总已查看，因此本轮学习器是探索性验证，不是完全盲测。",
        "区间按任务再按初态做配对分层 bootstrap，条件于实测噪声均值，仅 8 个任务。经验 Oracle 先平均噪声再取视角最大，区间不消除其最大值选择偏差。",
        "## 逐页讲解",
    ]
    for n in brief.notes:
        notes.extend([f"### 第 {n['page']} 页：{n['title']}", n["takeaway"], n["speaker_note"]])
    notes.extend(
        [
            "## 复核入口",
            f"- 统计与选择器产物：{source}",
            f"- PDF 构建、原图与逐页预览：{out}",
            f"- 各任务三维空间图：{out}/task-space",
            f"- 全部候选数值：{out}/candidate-values.csv",
            f"- 留出方法成绩：{out}/heldout-methods.csv",
            f"- 统计脚本：{REPO}/scripts/dsol_paper1/analysis/analyze_standard_initial_results_v1.py",
            f"- 报告脚本：{Path(__file__).resolve()}",
            "本次只使用已完成矩阵；没有扩充初态、任务、遮挡条件，没有开启新评测或修改冻结 release。",
        ]
    )
    md = out / "vla_view_research_progress_zh.md"
    md.write_text("\n\n".join(notes) + "\n")
    provenance = {
        "status": "PASS_STANDARD_INITIAL_RESEARCH_REPORT_12_PAGES",
        "archive": str(out),
        "public": str(PUBLIC),
        "pages": 12,
        "pdf_sha256": sha(target),
        "previous_outputs": previous,
        "analysis": str(source),
        "analysis_provenance_sha256": sha(source / "provenance.json"),
        "inputs": {
            str(Path(__file__)): sha(__file__),
            str(REPO / "scripts/dsol_paper1/reporting/build_view_research_focus_brief.py"): sha(
                REPO / "scripts/dsol_paper1/reporting/build_view_research_focus_brief.py"
            ),
            str(DOCS / "report_sources_20260907/legacy_evidence.json"): sha(
                DOCS / "report_sources_20260907/legacy_evidence.json"
            ),
        },
        "new_closed_loops": 0,
        "vla_training_steps": 0,
        "selector_status": "exploratory_offline_development_only_fit",
        "supporting_assets": {
            str(f.relative_to(out)): sha(f)
            for f in [
                *sorted((out / "task-space").glob("*.png")),
                out / "candidate-values.csv",
                out / "heldout-methods.csv",
            ]
        },
        "source_snapshots": {str(f.relative_to(out)): sha(f) for f in sorted((out / "source").rglob("*.py"))},
        "outline": [n["title"] for n in brief.notes],
        "visual_review": "rendered, awaiting inspection",
    }
    write(out / "provenance.json", provenance)
    for src, dest in [(target, PUBLIC), (md, PUBLIC.with_suffix(".md"))]:
        temp = dest.with_name(dest.name + ".building")
        shutil.copyfile(src, temp)
        os.replace(temp, dest)
    for alias in ["vla_view_research_focus_20260907_zh.pdf", "vla_view_landscape_and_metrics_20260908_zh.pdf"]:
        p = DOCS / alias
        temp = p.with_suffix(".alias-building")
        temp.symlink_to(os.path.relpath(PUBLIC, p.parent))
        os.replace(temp, p)
    write(ARCHIVE / "latest.json", provenance)
    print(json.dumps({"pdf": str(PUBLIC), "archive": str(out)}, ensure_ascii=False))
    return provenance


if __name__ == "__main__":
    import fcntl

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    with (ARCHIVE / ".report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        build()
