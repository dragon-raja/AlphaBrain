from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


SEEDS = (41, 42, 43)


def load(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text())


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for item in path.glob(pattern) if item.is_file())


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    gate = load(args.baseline_gate)
    episodes = load(args.episode_root / "manifest.json")
    episode_quality = load(args.episode_root / "quality_report.json")
    windows_quality = load(args.windows_root / "quality_report.json")
    deviation = load(args.protocol_deviation)

    checkpoints = {}
    evaluations = {}
    corrections = {}
    for seed in SEEDS:
        source = gate["source_runs"][str(seed)]
        run_dir = Path(source["run_dir"])
        support_identity = load(
            args.recovery_root
            / f"recovery_support_base_continuation_seed{seed}_steps6902"
            / "run_identity.json"
        )
        checkpoint = Path(source["checkpoint_model_path"]).parent
        checkpoints[str(seed)] = {
            "path": str(checkpoint.resolve()),
            "model_size_bytes": int(source["checkpoint_model_size_bytes"]),
            "sha256": str(support_identity["initial_checkpoint_sha256"]),
            "optimizer_steps": int(gate["uniform_training_budget_steps"]),
        }
        evaluations[str(seed)] = {
            "isolated": load(run_dir / "closed_loop_isolated_val_gate_v2.json")["summary"],
            "end_to_end": load(run_dir / "closed_loop_end_to_end_val_gate_v2.json")["summary"],
            "deterministic_reach": load(run_dir / "deterministic_reach_val_gate_v2.json")["summary"],
        }
        corrections[str(seed)] = load(
            args.recovery_root
            / "corrections"
            / f"policy_state_repaired_seed{seed}"
            / "quality_report.json"
        )

    split_counts: dict[str, int] = {}
    for group in episodes["groups"]:
        split = str(group["split"])
        split_counts[split] = split_counts.get(split, 0) + 1
    test_groups = sorted(
        str(group["pair_id"]) for group in episodes["groups"] if group["split"] == "test"
    )
    event_times = [int(group["event_time"]) for group in episodes["groups"]]
    reveal_times = [int(group["feedback_reveal_time"]) for group in episodes["groups"]]
    divergence_times = [int(group["action_divergence_time"]) for group in episodes["groups"]]
    return {
        "schema_version": 1,
        "status": "CORA_ASSET_AUDIT_COMPLETE",
        "baseline_gate_decision": gate["decision"],
        "checkpoints": checkpoints,
        "episodes": {
            "root": str(args.episode_root.resolve()),
            "group_count": len(episodes["groups"]),
            "split_group_counts": split_counts,
            "source_initial_state_counts": episode_quality["metrics"].get("split_source_counts"),
            "attached_episode_count": len(episodes["groups"]),
            "slipped_episode_count": len(episodes["groups"]),
            "paired_video_count": count_files(args.episode_root / "paired_videos", "*.mp4"),
            "branch_video_count": count_files(args.episode_root / "videos", "*.mp4"),
            "event_time_range": [min(event_times), max(event_times)],
            "feedback_reveal_time_range": [min(reveal_times), max(reveal_times)],
            "action_divergence_time_range": [min(divergence_times), max(divergence_times)],
            "quality_passed": bool(episode_quality["passed"]),
        },
        "windows": {
            "root": str(args.windows_root.resolve()),
            **windows_quality["metrics"],
            "quality_passed": bool(windows_quality["passed"]),
        },
        "recovery_support_v2": {
            "root": str(args.recovery_root.resolve()),
            "formal_decision": load(args.recovery_root / "support_decision_val.json")["decision"],
            "correction_quality": corrections,
        },
        "validation": {
            "group_count": split_counts["val"],
            "group_ids": sorted(
                str(group["pair_id"]) for group in episodes["groups"] if group["split"] == "val"
            ),
            "fixed_k_results": evaluations,
        },
        "original_test": {
            "group_count": len(test_groups),
            "group_ids": test_groups,
            "strictly_pristine": bool(deviation["strict_original_test_groups_pristine"]),
            "affected_auxiliary_artifact_count": int(deviation["affected_artifact_count"]),
            "closed_loop_test_evaluations_run": bool(deviation["closed_loop_test_evaluations_run"]),
            "allowed_for_cora_confirmation": False,
        },
    }


def markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# CORA-VLA 资产审计",
        "",
        f"状态：`{payload['status']}`",
        "",
        "## 冻结基础策略",
        "",
        "| Seed | Optimizer steps | Checkpoint | SHA256 |",
        "| ---: | ---: | --- | --- |",
    ]
    for seed, row in payload["checkpoints"].items():
        lines.append(f"| {seed} | {row['optimizer_steps']} | `{row['path']}` | `{row['sha256']}` |")
    episodes = payload["episodes"]
    windows = payload["windows"]
    lines.extend(
        [
            "",
            "三个 checkpoint 均来自已通过 validation baseline gate 的 Full-H，CORA 不得重新训练或修改它们。",
            "",
            "## 反事实数据",
            "",
            f"- 完整 episode：`{episodes['root']}`；{episodes['group_count']} 个 snapshot groups。",
            f"- Split：`{episodes['split_group_counts']}`；按 source initial state 隔离。",
            f"- Branch：{episodes['attached_episode_count']} attached + {episodes['slipped_episode_count']} slipped/recovery。",
            f"- 视频：{episodes['paired_video_count']} 个成对视频，{episodes['branch_video_count']} 个 branch 视频。",
            f"- event/reveal/divergence 范围：{episodes['event_time_range']} / {episodes['feedback_reveal_time_range']} / {episodes['action_divergence_time_range']}。",
            f"- 滑动窗口：{windows['record_count']}；train/val/test groups 为 {windows['split_group_counts']}。",
            f"- post-feedback windows：{windows['post_feedback_record_count']}；质量门通过：`{str(windows['quality_passed']).lower()}`。",
            "",
            "## Recovery Support v2",
            "",
            f"正式结论：`{payload['recovery_support_v2']['formal_decision']}`。其 correction trajectories、policy-state failures 和逐帧图像只作为 CORA 候选/负样本资产，不再用于微调基础 VLA。",
            "",
            "| Seed | Retained groups | Windows | Full-teacher success | Frozen-policy downstream |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed, report in payload["recovery_support_v2"]["correction_quality"].items():
        metric = report["metrics"]
        lines.append(
            f"| {seed} | {metric['retained_correction_group_count']} | {metric['training_window_count']} | "
            f"{100 * metric['full_teacher_success_rate']:.1f}% | {100 * metric['frozen_policy_downstream_success_rate']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## 可用评测集合",
            "",
            f"开发只允许使用 train 与 {payload['validation']['group_count']} 个 validation groups。已有 Full-H fixed K=1/2/3、isolated、end-to-end 和 deterministic reach 结果均已发现。",
            "",
            f"原 test 的 {payload['original_test']['group_count']} 个 groups 曾被 auxiliary deterministic-reach 触碰；严格 pristine=`{str(payload['original_test']['strictly_pristine']).lower()}`。它们不得作为 CORA 最终确认集。",
            "",
            "## Confirmation 封存协议",
            "",
            "- 冻结生成 24 个新的 grasp/slip snapshot groups，source seed=`2026071701`，full-episode seed=`2026071702`。",
            "- 生成后只运行自动质量门和 snapshot fingerprint 去重，不查看 CORA 指标。",
            "- Seal 路径：`/share/longjunyu/fresh-vla/cora-vla/confirmation-v1-24/seal.json`。",
            "- 仅在 Gate 1、Gate 2、validation rerank 全部冻结且 Gate 3 正式启动时解封一次。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动发现并审计 CORA-VLA 资产")
    parser.add_argument("--baseline-gate", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--windows-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--protocol-deviation", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_audit(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown(payload))
    print(json.dumps({"status": payload["status"], "output": str(args.output_json)}, sort_keys=True))


if __name__ == "__main__":
    main()
