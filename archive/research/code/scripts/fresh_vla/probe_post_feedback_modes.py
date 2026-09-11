from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from AlphaBrain.dataloader.paligemma_datasets import FreshEpisodeWindowDataset
from AlphaBrain.model.framework.base_framework import BaseFramework


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def analyze_candidate_set(
    chunks: np.ndarray,
    correct_actions: np.ndarray,
    opposite_actions: np.ndarray,
    *,
    action_steps: int,
    mode_margin: float,
) -> dict[str, object]:
    samples = np.asarray(chunks, dtype=np.float64)[:, :action_steps]
    correct = np.asarray(correct_actions, dtype=np.float64)[:action_steps]
    opposite = np.asarray(opposite_actions, dtype=np.float64)[:action_steps]
    if samples.ndim != 3 or samples.shape[1:] != correct.shape or opposite.shape != correct.shape:
        raise ValueError("candidate and expert action shapes do not match")
    correct_rmse = np.sqrt(np.square(samples - correct[None]).mean(axis=(1, 2)))
    opposite_rmse = np.sqrt(np.square(samples - opposite[None]).mean(axis=(1, 2)))
    signed_margin = opposite_rmse - correct_rmse
    best_index = int(np.argmin(correct_rmse))
    expert_mode_distance = float(np.sqrt(np.square(correct - opposite).mean()))
    return {
        "action_steps": action_steps,
        "expert_mode_distance": expert_mode_distance,
        "sample0_correct_rmse": float(correct_rmse[0]),
        "sample0_opposite_rmse": float(opposite_rmse[0]),
        "sample0_signed_margin": float(signed_margin[0]),
        "sample0_correct_mode": bool(signed_margin[0] > mode_margin),
        "correct_mode_fraction": float(np.mean(signed_margin > mode_margin)),
        "opposite_mode_fraction": float(np.mean(signed_margin < -mode_margin)),
        "ambiguous_mode_fraction": float(np.mean(np.abs(signed_margin) <= mode_margin)),
        "any_correct_mode": bool(np.any(signed_margin > mode_margin)),
        "any_opposite_mode": bool(np.any(signed_margin < -mode_margin)),
        "covers_both_modes": bool(
            np.any(signed_margin > mode_margin) and np.any(signed_margin < -mode_margin)
        ),
        "best_correct_rmse": float(correct_rmse[best_index]),
        "best_correct_index": best_index,
        "best_correct_signed_margin": float(signed_margin[best_index]),
        "best_candidate_is_correct_mode": bool(signed_margin[best_index] > mode_margin),
        "best_discriminative_margin": float(np.max(signed_margin)),
        "mean_correct_rmse": float(np.mean(correct_rmse)),
        "candidate_action_variance": float(np.var(samples, axis=0).mean()),
        "best_of_n_relative_rmse_reduction": float(
            (correct_rmse[0] - correct_rmse[best_index]) / max(correct_rmse[0], 1e-12)
        ),
    }


def select_feedback_samples(
    dataset: FreshEpisodeWindowDataset,
    groups: list[Mapping[str, object]],
    offsets: tuple[int, ...],
) -> dict[tuple[str, int], dict[str, Mapping[str, object]]]:
    wanted = {
        (str(group["pair_id"]), int(group["feedback_reveal_time"]) + offset)
        for group in groups
        for offset in offsets
    }
    selected: dict[tuple[str, int], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for index in range(len(dataset)):
        sample = dataset[index]
        key = (str(sample["pair_id"]), int(sample["frame_index"]))
        if key in wanted:
            selected[key][str(sample["branch_id"])] = sample
    missing = [key for key in sorted(wanted) if selected[key].keys() < {"attached", "slipped"}]
    if missing:
        raise ValueError(f"missing paired feedback samples: {missing[:3]}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe post-feedback recovery modes in frozen Full-H samples")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--windows-root", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offsets", nargs="+", type=int, default=(-1, 0, 1))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--action-steps", type=int, default=3)
    parser.add_argument("--mode-margin", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 2 <= args.samples <= 16:
        raise ValueError("samples must be in [2, 16]")
    os.environ.setdefault("PRETRAINED_MODELS_DIR", "/share/longjunyu/alphabrain/pretrained_models")
    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    groups = sorted(
        (group for group in manifest["groups"] if group["split"] == "test"),
        key=lambda group: str(group["pair_id"]),
    )
    dataset = FreshEpisodeWindowDataset(
        args.windows_root,
        split="test",
        feedback_label="oracle_feedback_horizon",
        feedback_output_key="feedback_horizon",
        tasks=("grasp_slip_full_episode",),
    )
    offsets = tuple(args.offsets)
    samples = select_feedback_samples(dataset, groups, offsets)

    model = BaseFramework.from_pretrained(str(args.checkpoint))
    model = model.to(torch.bfloat16).to(args.device).eval()
    model.gripper_remap = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_horizon = int(model.action_horizon)
    if args.action_steps > model_horizon:
        raise ValueError("action_steps exceeds model horizon")

    rows = []
    with torch.inference_mode():
        for group_index, group in enumerate(groups):
            pair_id = str(group["pair_id"])
            feedback_time = int(group["feedback_reveal_time"])
            for offset in offsets:
                paired = samples[(pair_id, feedback_time + offset)]
                seeds = [stable_seed(args.seed, pair_id, offset, index) for index in range(args.samples)]
                for outcome in ("attached", "slipped"):
                    sample = paired[outcome]
                    opposite_outcome = "slipped" if outcome == "attached" else "attached"
                    chunks = []
                    for sample_seed in seeds:
                        torch.manual_seed(sample_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(sample_seed)
                        with torch.autocast(
                            "cuda",
                            dtype=torch.bfloat16,
                            enabled=args.device.startswith("cuda"),
                        ):
                            output = model.predict_action(examples=[sample])
                        chunks.append(np.asarray(output["normalized_actions"][0], dtype=np.float32))
                    row = {
                        "seed": args.seed,
                        "pair_id": pair_id,
                        "source_initial_state_index": int(group["source_initial_state_index"]),
                        "offset": offset,
                        "outcome": outcome,
                        **analyze_candidate_set(
                            np.stack(chunks),
                            np.asarray(sample["action"], dtype=np.float32),
                            np.asarray(paired[opposite_outcome]["action"], dtype=np.float32),
                            action_steps=args.action_steps,
                            mode_margin=args.mode_margin,
                        ),
                    }
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                "pair_id": pair_id,
                                "offset": offset,
                                "outcome": outcome,
                                "any_correct_mode": row["any_correct_mode"],
                                "sample0_correct_mode": row["sample0_correct_mode"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    payload = {
        "experiment": "post_feedback_mode_coverage",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_model_size_bytes": (args.checkpoint / "model.safetensors").stat().st_size,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "seed": args.seed,
        "split": "test",
        "group_count": len(groups),
        "source_state_count": len({group["source_initial_state_index"] for group in groups}),
        "samples_per_observation": args.samples,
        "action_steps": args.action_steps,
        "mode_margin": args.mode_margin,
        "offset_definition": "frame_index = feedback_reveal_time + offset",
        "policy_input_excludes_branch_outcome": True,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"complete": True, "output": str(args.output), "row_count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
