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
from AlphaBrain.research.dsol.analysis.view_structure import summarize_structure
from AlphaBrain.research.dsol.analysis.view_recovery import metric_reference_summary
from AlphaBrain.research.dsol.plotting.view_space import (
    candidate_coordinates,
    projected_field,
    success_matrix,
    spatial_field,
    adaptive_limits,
    absolute_space_fields,
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
            self.camera = a["camera"]
        self.q = self.y.mean(axis=-1)
        self.s = [self.summary["models"][m] for m in ["canonical", "broad"]]
        self.per_initial = read(archive / "per-initial.json")
        self.structure = summarize_structure(
            self.q, self.acc.mean(axis=-1), self.camera,
            [s["task_ordinal"] for s in self.states],
            [s["initialization"]["init_state_index"] for s in self.states], self.cids,
        )
        self.figures = []
        self.recovery = None  # Explicitly bound and hash-checked by build().
        evaluation = [i for i, s in enumerate(self.states) if s['initialization']['init_state_index'] >= 2]
        self.metric_references = metric_reference_summary(self.q, self.acc, self.vis, evaluation)

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
                t.set_text("VLA VIEW RESEARCH  ·  主矩阵 2026-09-11  ·  计划更新 2026-09-16")
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
            "如何利用 VLA 视角：训练覆盖与扰动恢复",
            "研究主线 · 当前证据",
            "已完成：8 个已知任务 × 4 个标准初态 × 97 候选 × 32 套噪声 × 2 模型 = 198,656 次闭环",
            "训练提高被动覆盖；推理前选择可恢复部分扰动损失，实际获取仍待确认。",
            "标准初态、无额外遮挡；外部相机执行内固定，腕部输入随动作变化。非任务途中动态取景。\n恢复项：留出32视角起点，RGB＋Accel 引导16查询；静态矩阵重放，尚未验证实际相机获取。",
            "先对同一初态同一视角的32套完整噪声轨迹求平均，再比较视角。覆盖和经验上限展示全部32初态；恢复和学习器只评价初态2、3的16初态，并以留出目录32个相机起点均匀平均。二者不能混为相同统计总体。所有闭环标签来自同一冻结矩阵，新分析仅训练轻量选择器和进行静态图重放。历史遮挡实验不与当前绝对成功率相减。",
        )
        cards = [
            (0.065, "训练覆盖", "全部32初态 · 均匀候选", "22.28 → 75.07%", BLUE),
            (0.38, "经验上限", "32初态 · 规范→经验Oracle", "90.23 → 96.00%", TEAL),
            (0.695, "扰动恢复", "16评价初态 · 多视角16查询",
             f"{self.recovery['models']['broad']['references']['stay']['heldout32']['success']['mean']*100:.2f} → {self.recovery['models']['broad']['methods']['local_rgb_accel']['16']['heldout32']['success']['mean']*100:.2f}%", AMBER),
        ]
        for x, title, body, value, color in cards:
            self.box(x, 0.44, 0.245, 0.27)
            self.text(x + 0.015, 0.68, title, 21, color, True)
            self.text(x + 0.015, 0.60, body, 11, MUTED)
            self.text(x + 0.015, 0.54, value, 21, color, True)
        self.text(0.075, 0.365, "主问题：相机偏离有效视角后，能否用可获得的信息恢复任务表现？", 17, INK, True)
        self.text(0.075, 0.30, "保持原位为主对照，同预算随机探查检验搜索价值；已知标定返回规范作为参考。", 12, MUTED)
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
        am = self.acc.mean(axis=-1)
        fields = absolute_space_fields(self.q, am)
        means = fields["accel"].mean(axis=1)
        reduction = 100 * (1 - means[1] / means[0])
        lower = 100 * np.mean(am[1, :, 1:] < am[0, :, 1:])
        self.new(
            "候选空间分布：成功率与初态 Accel",
            "02 · 独立色标，展开各自空间差异",
            "各图色标独立，跨图请比较色条数值；上排越亮成功率越高，下排越亮原始 Accel 越低。",
            f"平均 Accel：{means[0]:.3f} → {means[1]:.3f}（降低 {reduction:.1f}%）；{lower:.1f}% 配对位置降低。",
            "上排：32 次完整闭环的成功率；下排：同一初态 8 个评分噪声的平均 Accel，非执行全程均值。\n随后对 32 初态等权平均；96 个非规范候选，无排名变换或插值。配对位置共 3,072 个。",
            "本页每张三维图使用独立自适应色条，覆盖该图全部实测均值，不裁剪极值。颜色只表示各图内部高低，跨模型绝对值须读取各自色条，不能直接比较相同颜色。成功率仍为百分比，Accel 仍为原始数值，不作排名变换。Accel 在同一任务初态对 8 个评分噪声求均值，不是执行轨迹全程的均值；成功率来自 32 次完整闭环。每个点随后对 32 初态等权平均。三轴是任务参考点下的方位角、仰角和半径倍率，不是完整 SE(3)，同一参数跨任务并非同一世界位姿。原始 Accel 整体下降可以与候选内部排序区分能力减弱同时存在；不能据两者共同变化认定因果关系。排名图仅作为辅助诊断保留。",
        )
        xyz = candidate_coordinates(self.release["protocol"]["state_blocks"][0]["candidates"])
        for row, key in enumerate(["success", "accel"]):
            for m in range(2):
                limits = adaptive_limits(fields[key][m], minimum_span=.001 if row else 1.,
                                         bounds=(0., float("inf")) if row else (0., 100.))
                ax = self.fig.add_axes([.045 + m * .46, .535 - row * .28, .32, .235], projection="3d")
                sc = spatial_field(ax, xyz, fields[key][m], limits=limits,
                                   cmap="viridis" if row == 0 else "viridis_r")
                ax.set_xlabel("方位角 / °", fontsize=8, labelpad=-3)
                ax.set_ylabel("仰角 / °", fontsize=8, labelpad=-3)
                ax.set_zlabel("半径倍率", fontsize=8, labelpad=-1)
                ax.tick_params(labelsize=7, pad=-1)
                ax.set_box_aspect((1.6, 1, .7), zoom=1.12)
                ax.view_init(24, -60)
                ax.set_xlim(-65, 65)
                ax.set_ylim(-29, 29)
                ax.set_title(MODEL_LABELS[m] + (" · 成功率" if row == 0 else " · 原始 Accel"),
                             fontsize=11, pad=-3)
                cb = self.fig.colorbar(sc, cax=self.fig.add_axes([.393 + m * .46, .55-row*.27, .012, .17]))
                ticks = np.linspace(*limits, 4)
                cb.set_ticks(ticks)
                cb.ax.set_yticklabels([f"{v:.3f}" if row else f"{v:.1f}" for v in ticks])
                cb.ax.tick_params(labelsize=8)
                cb.set_label("原始 Accel" if row else "成功率（%）", fontsize=9)
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
        from .recovery_pages import full_query_page
        full_query_page(self, pdf)

    def metric_diagnostics(self, pdf):
        self.new(
            "低 Accel 有分区信号，但不足以确定最优视角",
            "04 · 逐初态排序，分任务检查",
            "96 非规范候选：每初态按 8 次评分的平均 Accel 排序，四组各 24 个；再统计完整任务成功率。",
            "平均空间相关较强，但不等于每个初态都能按 Accel 准确选点。",
            "左图与右图均为全部 32 初态的探索性诊断，不是新增选择器成绩；右图为每任务四初态平均。\nAccel 是 Flow 去噪速度变化分数，非机器人加速度；8 个评分初噪不同于 32 套闭环噪声。",
            "候选先在各自初态内按指标排序，再对成功率分组，避免把不同难度任务混排。四分位仅用于诊断，没有用成功标签挑阈值或重新训练规则。按初态 2、3 单独检查时，两模型的低值组到高值组趋势也仍存在，但本轮数据已查看，不能称为新的盲测。相似性只说明评分空间部分重合，不证明熟悉度或遮挡是原因。每任务差值显示多视角条件下并非所有任务都有单调信号；高成功率的饱和、多个等价好视角及评分噪声都可能影响排序。",
        )
        ax = self.axes([.105, .35, .39, .36])
        for m in range(2):
            v = np.array(self.structure["models"][m]["quartiles_all"]) * 100
            ax.plot(range(4), v, "o-", color=COLORS[m], label=MODEL_LABELS[m], lw=2)
            for i, value in enumerate(v):
                ax.text(i, value + 2.2, f"{value:.1f}", ha="center", fontsize=10, color=COLORS[m])
        ax.set_xticks(range(4), ["最低 25%", "次低", "次高", "最高 25%"], fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_ylabel("完整任务成功率（%）", fontsize=11)
        ax.set_xlabel("平均 Accel 从低到高 →", fontsize=11)
        ax.legend(frameon=False, fontsize=10, loc="center right")
        ax = self.axes([.68, .34, .24, .39], True, False)
        task_rows = self.structure["models"][1]["by_task"]
        delta = [100 * (x["quartile_success"][0] - x["quartile_success"][-1]) for x in task_rows]
        ax.barh(range(8), delta, color=[TEAL if v >= 0 else AMBER for v in delta])
        ax.set_yticks(range(8), TASK_LABELS, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color=GRAY, lw=1)
        ax.set_xlim(min(-2, min(delta)-2), max(delta)+5)
        ax.set_xlabel("低值组 − 高值组（pp）", fontsize=9)
        ax.set_title("多视角训练：分任务差异", fontsize=11)
        for i, value in enumerate(delta):
            ax.text(max(0, value)+.3, i, f"{value:+.1f}", va="center", fontsize=8)
        rho = np.median([v for v in self.structure["cross_model"]["accel_rank_rho"] if v is not None])
        overlap = np.mean(self.structure["cross_model"]["low10_overlap_fraction"]) * 100
        spatial = [v['spatial_average_accel_success_rho'] for v in self.structure['models']]
        within = [np.median([x for x in v['per_initial_accel_success_rho'] if x is not None]) for v in self.structure['models']]
        self.text(.075, .275, f"低 Accel 与高成功率的 ρ：第六页平均空间 {spatial[0]:.2f} / {spatial[1]:.2f}；逐初态中位数 {within[0]:.2f} / {within[1]:.2f}（规范 / 多视角训练）。", 10, INK)
        self.text(.075, .245, f"两模型 Accel 排序重合：ρ 中位数 {rho:.2f}；最低 10 候选平均重合 {overlap:.1f}%（随机集合期望 10.4%）。", 9, MUTED)
        self.save(pdf)

    def learners(self, pdf):
        from .recovery_pages import budget_page
        budget_page(self, pdf)

    def supporting_assets(self, output):
        """Full per-task 3D plots and exact values, without extending the slide deck."""
        directory = output / "task-space"
        directory.mkdir()
        write(output / 'metric-rule-references.json', self.metric_references)
        if self.recovery is not None:
            with (output / 'recovery-methods.csv').open('w') as f:
                writer = csv.writer(f)
                writer.writerow(['model', 'method', 'budget', 'start_scope', 'success_percent', 'gain_vs_stay_pp',
                                 'gain_ci_low_pp', 'gain_ci_high_pp', 'harmed_start_fraction', 'graph_translation_m', 'graph_rotation_deg'])
                for model, result in self.recovery['models'].items():
                    for method, budgets in result['methods'].items():
                        for budget, scopes in budgets.items():
                            for scope, row in scopes.items():
                                writer.writerow([model, method, budget, scope, row['success']['mean']*100,
                                                 row['gain_vs_stay']['mean']*100, *[v*100 for v in row['gain_vs_stay']['ci95']],
                                                 row['harmed_fraction'], row['mean_translation_m'], row['mean_rotation_deg']])
        poses = [c["pose"] for c in self.release["protocol"]["state_blocks"][0]["candidates"][1:]]
        xyz = np.array([[p["azimuth_deg"], p["elevation_deg"], p["radius_scale"]] for p in poses])
        for task in range(-1, 8):
            fig = plt.figure(figsize=(12, 8), facecolor="white")
            subset = slice(None) if task == -1 else slice(task * 4, task * 4 + 4)
            pref = np.array([np.mean([rankdata(-v[1:]) / 96 * 100
                                      for v in am[subset]], axis=0)
                             for am in self.acc.mean(axis=-1)])
            for row in range(2):
                for m in range(2):
                    ax = fig.add_subplot(2, 2, row * 2 + m + 1, projection="3d")
                    if row == 0:
                        value = self.q[m, subset, 1:].mean(axis=0) * 100
                    else:
                        value = np.mean(
                            [rankdata(-v[1:]) / 96 * 100 for v in self.acc[m, subset].mean(axis=-1)],
                            axis=0,
                        )
                    limits = adaptive_limits(value) if row == 0 else adaptive_limits(pref)
                    sc = spatial_field(ax, xyz, value, limits=limits)
                    ax.set_title(
                        MODEL_LABELS[m] + ("：成功率 %（局部色标）" if row == 0 else "：低 Accel 偏好（共享色标）"), fontsize=12
                    )
                    ax.set_xlabel("方位角（度）", fontsize=10)
                    ax.set_ylabel("仰角（度）", fontsize=10)
                    ax.set_zlabel("")
                    ax.text2D(.83, .90, "半径倍率", transform=ax.transAxes, fontsize=9)
                    ax.tick_params(labelsize=8)
                    ax.view_init(24, -60)
                    fig.colorbar(sc, ax=ax, shrink=0.60, pad=0.12)
            label = "全部 32 初态" if task == -1 else TASK_LABELS[task] + "｜4 初态"
            fig.suptitle("辅助图 · " + label + "平均；局部成功率色标与 Accel 排名，不表示绝对降低", fontsize=13)
            fig.subplots_adjust(top=0.91, bottom=0.08, left=0.02, right=0.98, hspace=0.18, wspace=0.04)
            filename = "global-relative.png" if task == -1 else f"task-{task + 1:02d}.png"
            fig.savefig(directory / filename, dpi=160)
            plt.close(fig)
        write(output / "view-structure.json", self.structure)
        # Preserve the earlier rank/noise diagnostic beside the expanded figures.
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="white")
        for m in range(2):
            values = [x['minus_accel_spearman'] for x in self.s[m]['rank_correlations']
                      if x['minus_accel_spearman'] is not None]
            axes[0].scatter(np.full(len(values), m), values, alpha=.55, s=24, color=COLORS[m])
            axes[0].plot([m-.2, m+.2], [np.median(values)]*2, color=COLORS[m], lw=3)
            stability = [x['agreement_with_ensemble_winner'] for x in self.s[m]['score_noise_stability']]
            axes[1].hist(stability, bins=np.arange(-.0625, 1.13, .125), alpha=.5, color=COLORS[m], label=MODEL_LABELS[m])
        axes[0].set_xticks([0, 1], MODEL_LABELS)
        axes[0].set_ylabel('−Accel 与成功率的 Spearman ρ')
        axes[0].set_ylim(-1, 1)
        axes[0].axhline(0, color=GRAY, lw=1)
        axes[1].set_xlabel('单评分噪声最优与 8 噪声均值最优的一致比例')
        axes[1].set_ylabel('初态数')
        axes[1].legend(frameon=False, fontsize=10)
        fig.suptitle('保留原诊断：32 初态，97 候选（含规范）；评分噪声不是闭环重复噪声', fontsize=13)
        fig.tight_layout()
        fig.savefig(output / 'metric-noise-stability.png', dpi=160)
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
        from .recovery_pages import next_page
        next_page(self, pdf)


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
    recovery_config = read(REPO / 'configs/dsol_paper1/analysis/view_recovery_v1.json')
    recovery_receipt = read(Path(recovery_config['output']) / 'latest.json')
    recovery_source = Path(recovery_receipt['archive'])
    if recovery_receipt['input_hashes']['matrix.npz'] != sha(source / 'matrix.npz'):
        raise ValueError('Recovery and main report use different matrices')
    for name, digest in recovery_receipt['artifacts'].items():
        if sha(recovery_source / name) != digest:
            raise ValueError('Recovery artifact changed: ' + name)
    brief.recovery = read(recovery_source / 'summary.json')
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
                "subject": "2026-09-14 研究更新：扰动恢复、部署权限重设与有预算离线搜索；原冻结矩阵不变，无新增闭环",
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
        "# VLA 研究进展｜标准初态与离线选择器（计划更新 2026-09-16）",
        f"唯一 PDF：[当前报告]({PUBLIC})。主矩阵不变；本轮新增科学闭环 0 次，未新增正式 VLA 训练权重。资源 smoke 不属于科学训练结果。",
        "当前两阶段推进：[条件场学习首轮结果与 RoboCasa 资产、训练及评测计划](../../protocols/conditional_field_robocasa_20260916_zh.md)。第12页更新，前11页仍是原有冻结证据，不将本轮小模型与历史候选图选择器混称同一方法。",
        "## 实验范围与新旧差异",
        "两模型权重与 8 个任务不变。旧实验为 64 个演示恢复快照并添加人为遮挡；新实验为每任务 4 个官方标准初态，共 32 个，不添加遮挡或背景干预。初始化与渲染协议也不同，不能将绝对成功率变化归因于单一因素。",
        "## 选择器与统计边界",
        "初态 0、1 开发，2、3 评价。新学习器使用查询点 RGB、腕部图和已知任务 ID，增强版加入该点 Accel 均值及标准差；不再输入规范图、规范标记或规范相对位姿。冻结 ResNet18＋开发折 PCA／岭回归，训练初态内的视角相对价值。位姿目录仅用于搜索导航。主恢复比较在留出目录 32 个起始位姿上均匀平均，全部非规范 96 起点结果另存；留出目录是 VLA 的划分，不是选择器标签留出。此前测试汇总已查看，属于探索性分析，不是完全盲测或真机认证。",
        "区间按任务再按初态做配对分层 bootstrap，条件于实测噪声均值，仅 8 个任务。经验 Oracle 先平均噪声再取视角最大，区间不消除其最大值选择偏差。",
        "## 逐页讲解",
    ]
    for n in brief.notes:
        notes.extend([f"### 第 {n['page']} 页：{n['title']}", n["takeaway"], n["speaker_note"]])
    notes.extend([
        "## 本轮新增诊断：哪些规律目前成立",
        "以下均复用同一冻结矩阵，是描述性分析，不增加选择器训练或新闭环。第六页四图分别使用独立自适应色条，以展开各自空间差异；跨图须读取色条数值，不直接比较颜色。下排为初态 8 个评分噪声的平均原始 Accel，越亮值越低，不是执行全程均值，也不做排名转换。辅助图 global-relative.png 和各任务图保留局部成功率色标与排名，仅用于看内部差异；不能用辅助图判断原始 Accel 是否整体降低。",
        "| 训练条件 | 最低 Accel 四分位 | 次低 | 次高 | 最高四分位 |\n| --- | --- | --- | --- | --- |",
    ])
    for m in range(2):
        notes.append("| " + MODEL_LABELS[m] + " | " + " | ".join(
            f"{100*v:.2f}%" for v in brief.structure['models'][m]['quartiles_all']) + " |")
    notes.extend([
        "每个初态先将 96 个非规范候选按平均 Accel 排序，四组各 24 个，再计算每组的多噪声成功率。不是把不同初态按任务难度混排，也没有将规范视角插入非规范空间图。趋势只说明区域初筛信号，不等于最小值视角最优。分任务完整数据和各任务前五个经验高成功候选见 view-structure.json；这些候选由已观察的成功标签排序，只作事后诊断。",
        "原有排序相关性与评分噪声稳定性图保留为 metric-noise-stability.png，仍使用原先含规范视角的 97 候选口径。单评分噪声最优与八噪声均值最优的一致比例，初态中位数为规范训练 50%、多视角训练 37.5%；因此一次评分的最小值位置不能视为稳定解。主 PDF 保持 12 页，新增分区与任务异质性图代替重复占页，原诊断数据未删除。",
        "### 是否主要回到了熟悉区域？",
    ])
    for m in range(2):
        rs = brief.structure['models'][m]['per_initial_closeness_success_rho']
        med = {k: float(np.median([v for v in values if v is not None])) for k, values in rs.items()}
        notes.append(f"{MODEL_LABELS[m]}：在每个初态内，越靠近规范位姿与成功率的排序相关系数中位数为：平移距离 {med['translation']:.2f}、旋转角距离 {med['rotation']:.2f}。距离使用实际相机变换，不使用图上目录角替代。这个结果与训练熟悉度解释相容，但相机距离、图像内容和可见性同时变化，不能据此认定因果机制。")
    notes.extend([
        "## 部署权限重设与有限查询重放（本轮已完成）",
        f"可复核协议与结果：{recovery_source}。输入矩阵哈希保持不变；新增闭环 0 次，仅重训轻量离线选择器并执行图搜索重放。",
        "主口径：16 个评价初态、每个初态均匀取 32 个留出起始视角。每个查询节点只给当前候选 RGB、腕部 RGB、任务 ID，以及该节点的 8 个 Accel 评分（若方法需要）。无规范图、规范标记、候选身份、仿真分割特权。缓存复用是离线仿真后端优化，每次访问仍按协议计查询，不能声称实际推理耗时已测。",
        "目录导航假设已知候选间相对位姿；六近邻图按平移/0.1m＋旋转/15°构造，连接不等于物理可达。预算 [1,4,8,16,32] 在拟合前记录，主预算为16。起点计一次查询，保留起点是合法选择；搜索中不读取未查询分数和闭环标签。随机对照采用8个固定随机重放种子，最后同样选择已访问的最好评分节点。",
        "学习器只在16个开发初态拟合；两折初态验证选正则强度，不按测试搜索曲线改规则。已知任务ID不等于未见任务语言泛化。旧含规范上下文的学习器仍归原分析历史，不再作为部署权限一致的方法主结果。",
        "以下16查询结果使用已完成的整任务闭环标签查终点；随机方法先在每个初态×起点内对重放均值，再判断改善或变差，不把起点/随机重放扩充为独立场景。",
        "| 模型 | 方法 | 成功率 | 相对保持原位 pp | 变差起点比例 | 图路径平移 m | 图路径转角 ° |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for model in ["canonical", "broad"]:
        for method in ["local_accel", "random_accel", "local_rgb", "local_rgb_accel", "random_rgb_accel"]:
            row = brief.recovery["models"][model]["methods"][str(method)]["16"]["heldout32"]
            notes.append(f"| {model} | {method} | {100*row['success']['mean']:.2f}% | {100*row['gain_vs_stay']['mean']:+.2f} | {100*row['harmed_fraction']:.2f}% | {row['mean_translation_m']:.2f} | {row['mean_rotation_deg']:.1f} |")
    notes.extend([
        "### 同预算配对比较（16查询、留出32起点）",
        "以下区间按任务—初态配对重采样，条件于既有评分与闭环噪声均值，不能消除先前查看测试汇总的探索性限制。",
    ])
    for model in ["canonical", "broad"]:
        for comparison, budgets in brief.recovery["models"][model]["paired_method_comparisons"].items():
            row = budgets["16"]["heldout32"]
            notes.append(f"{model} · {comparison}：{row['mean']*100:+.2f} pp；95%区间 [{row['ci95'][0]*100:+.2f}, {row['ci95'][1]*100:+.2f}]。")
    notes.extend([
        "路径平移、旋转累计含从最后查询点返回所选终点；米与度分别报告，不合并成收益。当前目录搜索的累计移动并不小，因此不能把有限图像查询等同于低运动成本、真实头部可执行或真机净收益。",
        "尚未做：实际相机获取后的整任务执行、新噪声独立确认、连续位姿、RoboCasa、多机器人或真机扩展。缓存评分固定为每点8噪声均值，没有验证搜索时重新抽取评分噪声的影响。",
        "下一步先冻结方法与预算，补获取—选择—执行小规模对照；同时设置原地等待、同预算随机搜索、已知标定返回规范参考。若获取改变机械臂或场景状态，当前终点矩阵不可直接复用。1°小扰动不在本候选目录内，不能据稀疏点插值成绩。RoboCasa待方法和权限明确后作为跨场景验证。",
    ])
    notes.extend(
        [
            "## 复核入口",
            f"- 统计与选择器产物：{source}",
            f"- 新部署权限选择器、查询重放与全部预算结果：{recovery_source}",
            f"- 新部署权限方法的预算与起点分布对照表：{out}/recovery-methods.csv",
            f"- PDF 构建、原图与逐页预览：{out}",
            f"- 各任务三维空间图：{out}/task-space",
            f"- 辅助全局排名与局部色标图：{out}/task-space/global-relative.png",
            f"- 全部候选数值：{out}/candidate-values.csv",
            f"- 历史规则与旧规范上下文学习器成绩：{out}/heldout-methods.csv",
            f"- 第九页单噪声、8噪声平均、可见性及组合规则：{out}/metric-rule-references.json",
            f"- 分区、任务差异、几何距离及高成功候选：{out}/view-structure.json",
            f"- 原排序与评分噪声诊断：{out}/metric-noise-stability.png",
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
        "recovery_analysis": str(recovery_source),
        "recovery_provenance_sha256": sha(recovery_source / 'provenance.json'),
        "inputs": {
            str(Path(__file__)): sha(__file__),
            str(REPO / "scripts/dsol_paper1/reporting/build_view_research_focus_brief.py"): sha(
                REPO / "scripts/dsol_paper1/reporting/build_view_research_focus_brief.py"
            ),
            str(DOCS / "reports/sources/20260907/legacy_evidence.json"): sha(
                DOCS / "reports/sources/20260907/legacy_evidence.json"
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
                out / "recovery-methods.csv",
                out / "metric-rule-references.json",
                out / "view-structure.json",
                out / "metric-noise-stability.png",
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
    registry_path = PUBLIC.parent / 'publication.json'
    registry = read(registry_path) if registry_path.exists() else {
        'scope': 'Explicit report publication; not a change to frozen evaluation results',
        'original_pdf': {'path': previous[str(PUBLIC)]['archive'], 'sha256': previous[str(PUBLIC)]['sha256']},
    }
    registry['current'] = {
        'pdf_sha256': sha(target), 'notes_sha256': sha(md), 'archive': str(out),
        'analysis': str(source), 'new_closed_loops': 0, 'vla_training_steps': 0,
        'figure_revision': '2026-09-14: independent raw colorbars; disturbed-start baseline, permission-corrected learners and finite-query offline graph replay',
        'recovery_analysis': str(recovery_source),
    }
    registry['previous_publication'] = previous
    write(registry_path, registry)
    print(json.dumps({"pdf": str(PUBLIC), "archive": str(out)}, ensure_ascii=False))
    return provenance


if __name__ == "__main__":
    import fcntl

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    with (ARCHIVE / ".report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        build()
