from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


SELECTED_PROBLEM = "候选动作物理后果驱动的恢复阶段验证"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_problem(evidence: Mapping[str, object]) -> str:
    required = {
        "stage_a_stopped": bool(evidence["stage_a_stopped"]),
        "feedback_observable": bool(evidence["feedback_observable"]),
        "local_recovery_mode_available": float(evidence["local_mode_availability"]) >= 0.90,
        "not_random_mode_selection": not bool(evidence["supports_mode_selection_bottleneck"]),
        "prompt_only_failed": not bool(evidence["supports_prompt_grounded_recovery"]),
        "regrasp_exists": float(evidence["isolated_regrasp_rate"]) >= 0.50,
        "post_regrasp_attrition": float(evidence["regrasp_to_transport_rate"]) <= 0.50,
    }
    return (
        f"SELECT_NEW_RESEARCH_PROBLEM: {SELECTED_PROBLEM}"
        if all(required.values())
        else "NO_VALID_RESEARCH_PROBLEM_YET"
    )


def parse_args() -> argparse.Namespace:
    root = Path("/share/longjunyu/fresh-vla")
    parser = argparse.ArgumentParser(description="Finalize the evidence-gated embodied research reset")
    parser.add_argument(
        "--stage-a",
        type=Path,
        default=root / "runs/libero-oracle-commit-final-v1/final_decision.json",
    )
    parser.add_argument(
        "--observability",
        type=Path,
        default=root / "research-reset/feedback_observability.json",
    )
    parser.add_argument(
        "--modes",
        type=Path,
        default=root / "research-reset/post_feedback_modes_summary.json",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=root / "research-reset/recovery_prompt_summary.json",
    )
    parser.add_argument(
        "--funnel",
        type=Path,
        default=root / "research-reset/recovery_funnel.json",
    )
    parser.add_argument(
        "--video-manifest",
        type=Path,
        default=root / "research-reset/recovery_prompt_video_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "research-reset/final_decision.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "stage_a": args.stage_a,
        "observability": args.observability,
        "modes": args.modes,
        "prompt": args.prompt,
        "funnel": args.funnel,
        "video_manifest": args.video_manifest,
    }
    payloads = {name: json.loads(path.read_text()) for name, path in paths.items()}
    mode_summary = payloads["modes"]["summary"]["0"]["slipped"]
    funnel = payloads["funnel"]["results"]["isolated"]["fixed_k3"]
    evidence = {
        "stage_a_stopped": payloads["stage_a"]["decision"] == "STOP_FRESH_FAMILY",
        "feedback_observable": bool(payloads["observability"]["supports_single_frame_observability"]),
        "pre_feedback_accuracy": payloads["observability"]["results"]["-1"]["vision_state"]["sample_accuracy"],
        "post_feedback_accuracy": payloads["observability"]["results"]["0"]["vision_state"]["sample_accuracy"],
        "shuffled_label_mean_accuracy": payloads["observability"]["shuffled_label_control"]["mean_accuracy"],
        "supports_mode_selection_bottleneck": bool(payloads["modes"]["supports_mode_selection_bottleneck"]),
        "local_mode_availability": mode_summary["any_correct_mode"]["group_bootstrap_95"]["mean"],
        "sample0_correct_mode": mode_summary["sample0_correct_mode"]["group_bootstrap_95"]["mean"],
        "supports_prompt_grounded_recovery": bool(payloads["prompt"]["supports_prompt_grounded_recovery"]),
        "original_prompt_recovery": payloads["prompt"]["absolute"]["original_task"]["recovery_success"]["mean"],
        "explicit_prompt_recovery": payloads["prompt"]["absolute"]["explicit_recovery"]["recovery_success"]["mean"],
        "isolated_regrasp_rate": funnel["absolute"]["chain_regrasp"]["snapshot_group"]["mean"],
        "isolated_transport_rate": funnel["absolute"]["chain_transport"]["snapshot_group"]["mean"],
        "isolated_recovery_rate": funnel["absolute"]["chain_success"]["snapshot_group"]["mean"],
        "regrasp_to_transport_rate": funnel["conditional"]["regrasp_to_transport"]["snapshot_group"]["mean"],
        "new_prompt_video_count": int(payloads["video_manifest"]["count"]),
        "new_prompt_videos_h264_compatible": int(payloads["video_manifest"]["already_compatible"])
        == int(payloads["video_manifest"]["count"]),
    }
    decision = choose_problem(evidence)
    if evidence["new_prompt_video_count"] != 78 or not evidence["new_prompt_videos_h264_compatible"]:
        raise ValueError("recovery-prompt video artifact gate failed")
    result = {
        "decision": decision,
        "selected_problem": SELECTED_PROBLEM if decision.startswith("SELECT_NEW_RESEARCH_PROBLEM") else None,
        "scope": "research problem selected; candidate physical-consequence oracle is the next hard gate, not an established method result",
        "evidence": evidence,
        "next_hard_gate": {
            "name": "state-clone candidate physical-consequence oracle",
            "minimum_candidate_positive_coverage": 0.70,
            "minimum_regrasp_to_transport_gain_pp": 15.0,
            "alternative_minimum_isolated_recovery_gain_pp": 10.0,
            "maximum_attached_degradation_pp": 5.0,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "output": str(args.output)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
