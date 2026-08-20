#!/usr/bin/env python3
"""Write a non-sensitive, immutable identity manifest for a DSOL training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


CRITICAL_CODE = (
    "AlphaBrain/dataloader/paligemma_datasets.py",
    "AlphaBrain/model/framework/PaliGemmaPi.py",
    "AlphaBrain/model/modules/action_model/pi0_flow_matching_head/pair_consistency.py",
    "AlphaBrain/training/train_alphabrain.py",
    "AlphaBrain/training/trainer_utils/trainer_tools.py",
    "configs/experiments/dsol_libero_broad_pairing.yaml",
    "scripts/dsol_paper1/libero_pair_records.py",
    "scripts/dsol_paper1/run_libero_pair_train.sh",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=root, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--gpu-devices", required=True)
    parser.add_argument("--main-process-port", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--scheduler-steps", type=int, required=True)
    parser.add_argument("--global-examples", type=int, required=True)
    parser.add_argument("--examples-per-item", type=int, required=True)
    parser.add_argument("--gradient-accumulation", type=int, required=True)
    parser.add_argument("--calibration", type=int, choices=(0, 1), required=True)
    parser.add_argument("--calibration-interval", type=int, required=True)
    parser.add_argument("--calibration-items", type=int, required=True)
    parser.add_argument("--skip-final-save", type=int, choices=(0, 1), required=True)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), required=True)
    parser.add_argument("--budget-decision", type=Path)
    args = parser.parse_args()

    if args.scheduler_steps < args.steps:
        parser.error("--scheduler-steps must be >= --steps")

    root = args.repo_root.resolve()
    data_manifest = args.data_root.resolve() / "manifest.json"
    model_manifest = Path(
        "/share/longjunyu/alphabrain/pretrained_models/openpi/"
        "pi05_libero_pytorch/source_manifest.json"
    )
    patch = subprocess.check_output(("git", "diff", "--binary", "--", *CRITICAL_CODE), cwd=root)
    manifest = {
        "schema": "dsol_training_run_manifest_v1",
        "arm": args.arm,
        "seed": args.seed,
        "num_gpus": args.num_gpus,
        "gpu_devices": [int(value) for value in args.gpu_devices.split(",")],
        "main_process_port": args.main_process_port,
        "steps": args.steps,
        "scheduler_total_steps": args.scheduler_steps,
        "global_model_examples_per_update": args.global_examples,
        "source_data_items_per_update": (
            args.global_examples // args.examples_per_item
        ),
        "examples_per_data_item": args.examples_per_item,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "calibration": {
            "enabled": bool(args.calibration),
            "interval": args.calibration_interval,
            "items": args.calibration_items,
        },
        "skip_final_save": bool(args.skip_final_save),
        "wandb_mode": args.wandb_mode,
        "data_root": str(args.data_root.resolve()),
        "data_manifest_sha256": sha256(data_manifest),
        "pretrained_checkpoint_manifest": str(model_manifest),
        "pretrained_checkpoint_manifest_sha256": sha256(model_manifest),
        "git_commit": git_output(root, "rev-parse", "HEAD"),
        "git_dirty": bool(git_output(root, "status", "--short")),
        "critical_tracked_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "critical_code_sha256": {
            relative: sha256(root / relative) for relative in CRITICAL_CODE
        },
        "python": sys.version.split()[0],
    }
    if args.budget_decision is not None:
        budget_decision = args.budget_decision.resolve()
        if not budget_decision.is_file():
            parser.error(f"budget decision does not exist: {budget_decision}")
        manifest["budget_decision"] = str(budget_decision)
        manifest["budget_decision_sha256"] = sha256(budget_decision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
