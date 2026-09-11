#!/usr/bin/env python3
"""Plot rendered preflight inputs and full-bank/confirmation behavior separately."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import build_view_revalidation_report_pdf as report
import failure_view_search as study


def preflight(root):
    protocol = study.read(root / "protocols/discovery-base.json")
    audit = study.read(report.AUDIT)
    inputs = {}
    for subroot in [study.DATA, study.DATA / "independent-source-extension"]:
        for path in sorted((subroot / "test-feature-scan").glob("shard-*.jsonl")):
            import json

            for line in path.read_text().splitlines():
                row = json.loads(line)
                inputs[row["scan_id"]] = Path(row["output_dir"]) / "scan.json"
    output = root / "preflight"
    output.mkdir(parents=True, exist_ok=True)
    states = protocol["selected_states"]
    for offset in range(0, len(states), 4):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, state in zip(axes.flat, states[offset : offset + 4]):
            ax.imshow(report.montage_tile(inputs[state["pair_key"]], "canonical"))
            ax.set_title(
                f"{state['task_id']}\n{state['pair_key'].split('::test::')[-1]} | canonical {state['screening_successes_of_5']}/5",
                fontsize=10,
            )
            ax.axis("off")
        fig.suptitle("困难状态输入核验：左外部相机，右腕部相机；不据此筛选候选", fontsize=14)
        fig.tight_layout()
        fig.savefig(output / f"canonical-contact-sheet-{offset // 4 + 1:02d}.png", dpi=130)
        plt.close(fig)
    study.atomic_json(
        output / "automated_checks.json",
        {
            "states": len(states),
            "source_audit_status": audit["status"],
            "hdf5_paths_exist": all(Path(s["hdf5"]).is_file() for s in protocol["specs"]),
            "all_static_noncolliding_occluders": all(
                s["scene_construction"]["occluder"]["contype"] == 0
                and s["scene_construction"]["occluder"]["conaffinity"] == 0
                and not s["scene_construction"]["occluder"]["has_joint"]
                for s in protocol["specs"]
            ),
            "candidate_bank_not_visibility_filtered": True,
            "formal_manual_release": False,
        },
    )


def analysis(root):
    discovery = study.read(root / "analysis/discovery.json")["states"]
    final = study.read(root / "analysis/final_analysis.json")
    states = sorted(discovery)
    candidates = sorted(
        discovery[states[0]], key=lambda c: (0 if c == "canonical" else 1 if c.startswith("broad_train") else 2, c)
    )
    matrix = np.asarray([[discovery[s][c]["mean_success"] for c in candidates] for s in states])
    fig, ax = plt.subplots(figsize=(17, 8))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(states)), [s.replace("::test::", " / ").replace("::", " / ") for s in states], fontsize=7)
    ticks = [0, *range(8, len(candidates), 8)]
    ax.set_xticks(ticks, [candidates[i] for i in ticks], rotation=45, ha="right", fontsize=8)
    ax.set_title("完整候选池：每格为 3 个发现噪声的闭环成功率；只作发现标签")
    fig.colorbar(im, ax=ax, label="闭环成功率")
    fig.tight_layout()
    fig.savefig(root / "analysis/full_bank_success_heatmap.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for ax, task in zip(axes.flat, sorted(final["tasks"])):
        task_states = [s for s in states if discovery[s]["canonical"]["task_id"] == task]
        group = [c for c in candidates if c != "canonical"]
        x = [discovery[task_states[0]][c]["azimuth_deg"] for c in group]
        y = [discovery[task_states[0]][c]["elevation_deg"] for c in group]
        gains = [np.mean([discovery[s][c]["advantage"] for s in task_states]) * 100 for c in group]
        scatter = ax.scatter(x, y, c=gains, cmap="RdBu", vmin=-100, vmax=100, s=45, edgecolors="gray", linewidths=0.3)
        ax.set_title(task, fontsize=10)
        ax.set_xlabel("相机方位角")
        ax.set_ylabel("俯仰角")
    fig.suptitle("各任务的候选收益分布：相对规范的百分点差；半径变化投影到平面", fontsize=14)
    fig.colorbar(scatter, ax=axes, label="发现阶段配对增益 pp", shrink=0.8)
    fig.savefig(root / "analysis/task_pose_advantage.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 6))
    tasks = sorted(final["tasks"])
    x = np.arange(len(tasks))
    for shift, key, label, color in [
        (-0.18, "canonical_success", "规范视角", report.base.BLUE),
        (0.18, "primary_success", "发现阶段冻结 Top-1", report.base.TEAL),
    ]:
        bars = ax.bar(x + shift, [final["tasks"][t][key] * 100 for t in tasks], 0.34, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_xticks(x, tasks, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("独立 5 噪声闭环成功率 %")
    ax.legend(frameon=False)
    ax.set_title("独立复验：不使用复验结果重新选择 Top-1；困难发现集，不是完整 benchmark")
    p = final["primary_paired_comparison"]
    fig.text(
        0.06,
        0.02,
        f"源轨迹等权增益 {p['source_equal_advantage_pp']:+.2f}pp，95% CI {p['source_bootstrap_ci95_pp']}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(root / "analysis/frozen_top1_confirmation.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["preflight", "analysis"])
    parser.add_argument("--root", type=Path, default=study.DEFAULT_ROOT)
    args = parser.parse_args()
    report.base.configure_font()
    (preflight if args.phase == "preflight" else analysis)(args.root)


if __name__ == "__main__":
    main()
