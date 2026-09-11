from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from onpolicy_support import STAGES


def bootstrap(values: Mapping[str, float], samples: int = 10000) -> dict[str, float]:
    array = np.asarray([values[key] for key in sorted(values)], dtype=np.float64)
    rng = np.random.default_rng(20260717)
    draws = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "group_count": len(array),
    }


def aggregate(payloads: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if {int(payload["seed"]) for payload in payloads} != {41, 42, 43}:
        raise ValueError("formal summary requires seeds 41, 42, 43")
    rows = [(int(payload["seed"]), row) for payload in payloads for row in payload["rows"]]
    by_seed = {}
    for payload in payloads:
        by_seed[str(payload["seed"])] = {
            f"recall@{n}": float(np.mean([row["immediate_recall"][str(n)] for row in payload["rows"]]))
            for n in (1, 4, 8, 16)
        }
        by_seed[str(payload["seed"])]["natural_success"] = float(
            np.mean([episode["success"] for episode in payload["episodes"]])
        )
    by_stage = {}
    for stage in STAGES:
        selected = [row for _, row in rows if row["stage"] == stage]
        by_stage[stage] = {
            "reachable_state_count": len(selected),
            **{
                f"recall@{n}": float(np.mean([row["immediate_recall"][str(n)] for row in selected]))
                if selected else None
                for n in (1, 4, 8, 16)
            },
            "teacher_recoverable@16": float(
                np.mean([row["teacher_recoverable_recall"]["16"] for row in selected])
            ) if selected else None,
        }
    group_values = defaultdict(list)
    for _, row in rows:
        group_values[str(row["pair_id"])].append(float(row["immediate_recall"]["16"]))
    group_means = {pair_id: float(np.mean(values)) for pair_id, values in group_values.items()}
    estimate = bootstrap(group_means)
    decision = (
        "PASS_ONPOLICY_SUPPORT"
        if estimate["mean"] >= 0.60
        else "BASE_POLICY_ONPOLICY_SUPPORT_INSUFFICIENT"
    )
    curves = {
        f"recall@{n}": float(np.mean([row["immediate_recall"][str(n)] for _, row in rows]))
        for n in (1, 4, 8, 16)
    }
    return {
        "experiment": "cora_onpolicy_candidate_support",
        "decision": decision,
        "state_count": len(rows),
        "snapshot_group_count": len(group_values),
        "overall": curves,
        "recall16_group_bootstrap": estimate,
        "by_seed": by_seed,
        "by_stage": by_stage,
        "teacher_state_slipped_recall@16": 1.0,
        "teacher_minus_onpolicy_recall@16": 1.0 - estimate["mean"],
    }


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.1f}%"


def markdown(result: Mapping[str, object]) -> str:
    e = result["recall16_group_bootstrap"]
    lines = [
        "# CORA-VLA On-policy Candidate Support",
        "",
        "状态全部来自冻结 Full-H 自己的 slipped 闭环轨迹；teacher 只用于候选执行后的可恢复性审计，不参与状态生成。Confirmation 保持封存。",
        "",
        "## 总体",
        "",
        f"共评估 {result['state_count']} 个可达 on-policy replanning states、{result['snapshot_group_count']} 个 snapshot groups。",
        "",
        "| N | Correct-mode recall |",
        "|---:|---:|",
    ]
    for n in (1, 4, 8, 16):
        lines.append(f"| {n} | {pct(result['overall'][f'recall@{n}'])} |")
    lines.extend(
        [
            "",
            f"组级 recall@16={pct(e['mean'])}，paired group bootstrap 95% CI=[{pct(e['ci95_low'])}, {pct(e['ci95_high'])}]。Teacher-state slipped recall@16=100.0%，差值={pct(result['teacher_minus_onpolicy_recall@16'])}。",
            "",
            "## 分阶段",
            "",
            "| 阶段 | 可达 seed-state 数 | recall@1 | recall@4 | recall@8 | recall@16 | teacher可恢复@16 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for stage in STAGES:
        row = result["by_stage"][stage]
        lines.append(
            f"| {stage} | {row['reachable_state_count']} | {pct(row['recall@1'])} | {pct(row['recall@4'])} | "
            f"{pct(row['recall@8'])} | {pct(row['recall@16'])} | {pct(row['teacher_recoverable@16'])} |"
        )
    lines.extend(["", "## 各 seed", ""])
    for seed, row in result["by_seed"].items():
        lines.append(
            f"- seed {seed}：recall@1/4/8/16={pct(row['recall@1'])}/{pct(row['recall@4'])}/"
            f"{pct(row['recall@8'])}/{pct(row['recall@16'])}，自然闭环成功={pct(row['natural_success'])}。"
        )
    lines.extend(["", "## Gate", "", f"**{result['decision']}**", ""])
    if result["decision"] == "PASS_ONPOLICY_SUPPORT":
        lines.append("on-policy correct-mode recall@16 达到 60% 必要条件，允许补充冻结策略 continuation 并进入一次 Sequential Oracle 上界。")
    else:
        lines.append("基础策略在自身状态分布中的候选支持不足；按预注册停止 CORA，不运行 Sequential Oracle。")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CORA on-policy support")
    parser.add_argument("--inputs", nargs=3, type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate([json.loads(path.read_text()) for path in args.inputs])
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(markdown(result))
    print(json.dumps({"decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
