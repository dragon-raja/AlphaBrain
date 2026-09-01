#!/usr/bin/env python3
"""Log scalar-only final metrics to W&B, defaulting to a local offline run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--project", default="alphabrain-view-system-analysis")
    args = parser.parse_args()
    analysis: dict[str, Any] = json.loads(args.analysis.read_text())
    decision: dict[str, Any] = json.loads(args.decision.read_text())
    args.run_dir.mkdir(parents=True, exist_ok=True)

    try:
        import wandb
    except ImportError:
        receipt = {"status": "SKIPPED_WANDB_NOT_INSTALLED"}
        (args.run_dir / "wandb_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt))
        return

    mode = os.environ.get("WANDB_MODE", "offline")
    run = wandb.init(
        project=args.project,
        name="view-value-expectation-formal-v1",
        group="dsol-paper1-view-expectation",
        job_type="closed-loop-evaluation",
        dir=str(args.run_dir),
        mode=mode,
        save_code=False,
        config={
            "git_commit": git_commit(args.repo),
            "checkpoint_seeds": [41, 42, 43],
            "candidate_count": 97,
            "calibration_state_count": 16,
            "heldout_state_count": 48,
            "noise_repeats_per_condition": analysis["noise_repeats_per_condition"],
            "replan_k": 5,
            "flow_steps": 10,
            "uploads_disabled": [
                "dataset",
                "video",
                "checkpoint",
                "model_weight",
                "system_environment",
            ],
        },
        settings=wandb.Settings(
            disable_git=True,
            init_timeout=20,
            _disable_stats=True,
        ),
    )
    step = 0
    for seed, methods in sorted(analysis["checkpoint_seeds"].items()):
        for method, values in sorted(methods.items()):
            run.log(
                {
                    "checkpoint_seed": int(seed),
                    "selector_method": method,
                    "success_rate": values["success_rate"],
                    "success_gain_pp": values["success_gain_pp"],
                    "harm_probability": values["harm_probability"],
                    "rescue_probability": values["rescue_probability"],
                },
                step=step,
            )
            step += 1
    gate = analysis["selector_population_gate"]
    run.summary.update(
        {
            "final_status": decision["status"],
            "calibration_headroom_gate": decision["calibration_headroom_gate"],
            "heldout_selector_gate": decision["heldout_selector_gate"],
            "best_rule": gate["best_rule_frozen_on_calibration"],
            "cross_checkpoint_mean_gain_pp": gate["cross_checkpoint_mean_gain_pp"],
            "cross_checkpoint_mean_harm_probability": gate["cross_checkpoint_mean_harm_probability"],
        }
    )
    receipt = {
        "status": "PASS",
        "mode": mode,
        "run_id": run.id,
        "run_name": run.name,
        "run_directory": run.dir,
        "media_or_artifact_uploads": False,
    }
    run.finish()
    (args.run_dir / "wandb_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
