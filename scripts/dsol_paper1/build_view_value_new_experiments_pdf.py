#!/usr/bin/env python3
"""Build an additive report of the latest source-disjoint selector replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import build_view_revalidation_report_pdf as report


base = report.base


def page_design(pdf, previews):
    fig, _ = base.new_page(
        "本次新增实验：更换源轨迹，复验视角选择收益",
        "原报告保持不变；本补充仅说明新增来源评测及其离线分析。2026-08-31",
        1,
    )
    report.text(fig, 0.06, 0.78, "检验问题：原先的微弱正收益，是否依赖少数轨迹或一次采样？", 17, base.BLUE)
    report.table(
        fig,
        ["项目", "原批次", "本次新增", "合并分析"],
        [
            ["源演示轨迹", "24 条", "18 条，与原批次不重叠", "42 条"],
            ["物理状态", "48 个", "36 个，每条轨迹取 2 个阶段", "84 个"],
            ["任务", "8 个", "其中 7 个任务的新轨迹", "仍为原 8 个任务"],
            ["完整闭环 rollout", "1,440 条", "36 状态 × 6 规则 × 5 噪声 = 1,080", "2,520 条"],
        ],
        [0.055, 0.47, 0.89, 0.25],
        [0.18, 0.17, 0.42, 0.23],
        11,
    )
    report.text(fig, 0.065, 0.42, "每个状态实际做了什么", 15)
    report.text(
        fig,
        0.065,
        0.365,
        "恢复真实轨迹中间状态并沿用静态遮挡构造 → 渲染 97 个相机候选 → 按冻结规则选视角\n"
        "→ 从同一物理初态分别执行完整闭环 → 每种规则重复 5 个共享推理噪声。",
        12,
    )
    report.text(
        fig,
        0.065,
        0.255,
        "固定项：Broad64 practical 的同一 checkpoint；外部相机选定后固定，腕部相机正常更新；K=5。\n"
        "本次没有新增训练、任务类型或遮挡方案。新增的是独立轨迹及对应的构造状态。",
        12,
    )
    report.note(fig, "97 候选用于静态评分，不是每个测试状态跑 97 次闭环；实际比较 6 条冻结选择规则，结果不参与选视角。")
    report.finish(pdf, fig, previews, 1)


def page_results(pdf, previews, cohorts):
    fig, _ = base.new_page(
        "总体结果：原批次的弱正收益未在新增来源中复现",
        "成功率：从选定中间状态交给策略，一直执行至任务成功或超时；不等于官方 LIBERO-Plus 全量分数。",
        2,
    )
    ax = fig.add_axes([0.07, 0.49, 0.41, 0.28])
    x = np.arange(3)
    for offset, method, label, color in [
        (-0.18, "canonical", "规范视角", base.BLUE),
        (0.18, "visibility_gain_gated", "可见性门控", base.RED),
    ]:
        values = [report.rate(a, method) for a in cohorts]
        bars = ax.bar(x + offset, values, 0.34, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
    ax.set_xticks(x, ["原 24 轨迹", "新增 18 轨迹", "合并 42 轨迹"])
    ax.set_ylim(0, 106)
    ax.set_ylabel("闭环成功率 %")
    ax.legend(frameon=False, ncol=2, fontsize=10, loc="upper left")
    report.clean(ax)
    new = cohorts[1]["selector_summary"]["visibility_gain_gated"]
    report.text(fig, 0.54, 0.755, f"新增来源：{new['difference_from_canonical_pp']:+.2f}pp", 20, base.RED)
    report.text(
        fig,
        0.54,
        0.675,
        f"源轨迹配对 95% CI：[{new['difference_ci_low_pp']:.2f}, {new['difference_ci_high_pp']:.2f}]\n"
        "原批次 +1.25pp，新批次 -11.67pp。\n任务等权后仍为 -12.14pp。",
        13,
    )
    report.text(
        fig, 0.54, 0.53, "新批次规范成功率较高，反映评测初态和组成不同；\n没有额外训练。应比较同批次内的配对差值。", 11
    )
    definitions = [
        "始终保持规范位姿",
        "仅用验证结果预选一个固定视角",
        "均值最高；增量不足则保留规范",
        "用调和平均抑制实体间不均衡",
        "直接选择实体平均分最高视角",
        "优先补足分数最低的实体",
    ]
    rows = []
    for method, label, definition in zip(report.METHODS, report.LABELS, definitions):
        rows.append([label, definition, *[f"{report.rate(a, method):.2f}%" for a in cohorts]])
    report.table(
        fig,
        ["选择规则", "选视角依据", "原批次", "新增批次", "合并"],
        rows,
        [0.055, 0.19, 0.89, 0.25],
        [0.21, 0.34, 0.15, 0.15, 0.15],
        10.2,
    )
    report.note(
        fig, "可见性门控：选等权实体可见面积最高的视角；增量不足 0.5pp 时保留规范。合并差值 -4.29pp，CI [-8.57, 0.00]。"
    )
    report.finish(pdf, fig, previews, 2)


def noise_differences(protocol_path, run_root):
    protocol = report.load(protocol_path)
    expected_states = {r["pair_key"] for r in protocol["selected_states"]}
    grouped = {}
    for path in sorted(run_root.glob("episodes-shard-*.jsonl")):
        report.INPUTS.add(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["status"] != "complete":
                raise ValueError("incomplete raw evaluation")
            method = row["condition"].removeprefix("selector__")
            if method not in ("canonical", "visibility_gain_gated"):
                continue
            key = (row["evaluation_seed"], row["pair_key"])
            methods = grouped.setdefault(key, {})
            if method in methods:
                raise ValueError("duplicate evaluation")
            methods[method] = row
    seeds = sorted({seed for seed, _ in grouped})
    result = {}
    for seed in seeds:
        if {key for s, key in grouped if s == seed} != expected_states:
            raise ValueError("repeat does not cover the frozen state set")
        values = []
        for key in expected_states:
            methods = grouped[seed, key]
            if set(methods) != {"canonical", "visibility_gain_gated"}:
                raise ValueError("missing paired method")
            if len({r["policy_noise_seed"] for r in methods.values()}) != 1:
                raise ValueError("unmatched state-specific noise")
            values.append(int(methods["visibility_gain_gated"]["success"]) - int(methods["canonical"]["success"]))
        result[seed] = 100 * float(np.mean(values))
    if len(result) != 5:
        raise ValueError("expected five evaluation repeats")
    return result


def page_tasks(pdf, previews, cohorts, noises):
    fig, _ = base.new_page(
        "分任务与噪声：下降不只来自总体平均的变化",
        "差值均为同一批状态上，可见性门控成功率减去规范成功率。新增批次没有加入新的任务类型。",
        3,
    )
    rows = []
    for task, label in zip(report.TASKS, report.TASK_LABELS):
        vals = []
        for a in cohorts[:2]:
            t = a["task_summary"].get(task)
            vals.append(
                "无新增轨迹"
                if t is None
                else f"{100 * (t['visibility_gain_gated']['mean_repeat_success_rate'] - t['canonical']['mean_repeat_success_rate']):+.1f}pp"
            )
        count = sum(
            r["task_id"] == task and r["selector_method"] == "canonical" for r in cohorts[1]["state_method_rows"]
        )
        rows.append([label, str(count), *vals])
    report.table(
        fig,
        ["任务", "新增状态", "原批次差值", "新增差值"],
        rows,
        [0.055, 0.33, 0.53, 0.44],
        [0.38, 0.15, 0.23, 0.24],
        10.5,
    )
    ax = fig.add_axes([0.66, 0.45, 0.28, 0.28])
    values = list(noises.values())
    bars = ax.bar([str(seed)[-2:] for seed in noises], values, color=base.RED)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
    ax.axhline(0, color=base.INK, lw=0.8)
    ax.set_ylim(-24, 2)
    ax.set_xlabel("推理噪声编号末两位")
    ax.set_ylabel("新增批次配对增益 pp")
    report.clean(ax)
    report.text(
        fig, 0.635, 0.345, "5 个推理噪声下，差值全部为负。\n它们是同一模型的重复评测，\n不是 5 个训练 seed。", 12
    )
    report.text(
        fig,
        0.065,
        0.235,
        "书、底层抽屉与微波炉在新轨迹上由正转负；原先的优势不是稳定的任务级规律。\n"
        "两个奶酪任务接近满分，没有明显改善空间；抽屉取碗放盘没有新增轨迹，正信号尚未复验。",
        12,
    )
    report.note(fig, "任务级样本仍少；本页用于定位异质性，不据单个任务的小差值宣称显著效果。")
    report.finish(pdf, fig, previews, 3)


def page_audit(pdf, previews, audit):
    fig, _ = base.new_page(
        "补充离线分析：分数为何可能选错视角",
        "以下是已有渲染和结果的审计，不是新增训练或闭环。实例：新增来源中的上层抽屉任务。",
        4,
    )
    example = next(e for e in audit["examples"] if "demo_10::stage-02" in e["pair_key"])
    for x, candidate, label in [
        (0.055, "canonical", "规范视角"),
        (0.53, example["selected_candidate_id"], "门控选中的视角"),
    ]:
        report.text(fig, x, 0.79, label, 15)
        ax = fig.add_axes([x, 0.50, 0.41, 0.245])
        ax.imshow(report.montage_tile(example["scan_path"], candidate))
        ax.axis("off")
        report.text(fig, x, 0.475, "左：外部相机；右：腕部相机", 10, base.MUTED)
    report.table(
        fig,
        ["测量对象", "像素分数增量", "同状态闭环结果"],
        [
            ["被操作的碗", "0.00pp", "规范视角：5 次成功 5 次"],
            ["父级柜体", "+3.08pp", "选中视角：5 次成功 3 次"],
            ["两实体等权平均", "+1.54pp", "总分提高，但目标物没有更可见"],
        ],
        [0.055, 0.245, 0.89, 0.17],
        [0.30, 0.25, 0.45],
        11,
    )
    report.text(fig, 0.065, 0.205, "代码中的部分 region 名称回退为整柜体几何；当前统计不等于抽屉内部或接触区域。", 12)
    report.note(
        fig,
        "合并 55 个换视角状态中，10 个目标物像素不增，其中 8 个由父级实体增益主导；这是指标缺口，不是全部失败的因果解释。",
    )
    report.finish(pdf, fig, previews, 4)


def page_conclusions(pdf, previews):
    fig, _ = base.new_page(
        "本次可以得出的结论，以及仍未回答的问题",
        "本补充不改写原报告中已完成的宽视角训练与正式 benchmark 结果。",
        5,
    )
    report.table(
        fig,
        ["问题", "本次回答"],
        [
            ["原先那一点正收益稳健吗？", "没有跨新来源复现；同一任务也存在正负反转。"],
            ["是否只因为一次随机采样？", "新增批次 5 个噪声差值均为负，不能只归因于一次采样。"],
            ["是否已证明信息视角没有价值？", "没有。当前只检验了冻结像素规则，未验证所有更有价值的候选。"],
            ["是否已经找到全部失败原因？", "没有。发现区域像素代理问题；位姿、阶段、腕部作用仍需隔离。"],
            ["构造是否已足够全面？", "尚未增加新的任务机制或遮挡方案，也未完成全量人工质量确认。"],
        ],
        [0.055, 0.39, 0.89, 0.36],
        [0.33, 0.67],
        11.5,
    )
    report.text(fig, 0.065, 0.33, "下一步应补什么", 15, base.BLUE)
    report.text(
        fig,
        0.065,
        0.275,
        "保留当前分数作基线；在开发来源核对任务区域与遮挡，加入位移匹配的低信息对照。\n"
        "对发现的有利视角先冻结，再用新噪声和新来源复验；不能只保留成功案例。",
        12,
    )
    report.note(
        fig,
        "当前结论是“这一选择规则的收益没有稳定复现”，不是“主动视觉方向无效”。已查看的 42 条来源不再用作下一轮独立测试。",
    )
    report.finish(pdf, fig, previews, 5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=report.ROOT / "docs/dsol_paper1/view_value_new_experiments_zh.pdf"
    )
    parser.add_argument("--preview-dir", type=Path, required=True)
    args = parser.parse_args()
    base.configure_font()
    plt.rcParams.update({"pdf.fonttype": 3, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10})
    cohorts = [
        report.load(report.DATA / "dense-test-selector-eval/analysis/analysis.json"),
        report.load(report.EXT / "dense-test-selector-eval/analysis/analysis.json"),
        report.load(report.EXT / "combined-analysis/analysis.json"),
    ]
    if [(a["source_groups"], a["states"], a["episodes"]) for a in cohorts] != [
        (24, 48, 1440),
        (18, 36, 1080),
        (42, 84, 2520),
    ]:
        raise ValueError("unexpected frozen cohort sizes")
    if any(a["status"] != "PASS" or a["expected_repeats"] != 5 for a in cohorts):
        raise ValueError("incomplete result set")
    audit = report.load(report.AUDIT)
    if audit["status"] != "PASS":
        raise ValueError("audit pairing checks failed")
    noises = noise_differences(
        report.EXT / "dense-test-selector-protocol.json", report.EXT / "dense-test-selector-eval/run-five-noise"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        page_design(pdf, args.preview_dir)
        page_results(pdf, args.preview_dir, cohorts)
        page_tasks(pdf, args.preview_dir, cohorts, noises)
        page_audit(pdf, args.preview_dir, audit)
        page_conclusions(pdf, args.preview_dir)
        pdf.infodict().update(Title="新增实验：独立来源视角选择复验", Author="AlphaBrain research workflow")
    report.INPUTS.update([Path(__file__), Path(report.__file__), Path(base.__file__)])
    receipt = {
        "status": "COMPLETE",
        "pages": 5,
        "pdf": str(args.output),
        "pdf_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(report.INPUTS)},
    }
    (args.preview_dir / "build_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "input_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
