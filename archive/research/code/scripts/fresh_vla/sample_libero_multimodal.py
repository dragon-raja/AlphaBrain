from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from AlphaBrain.dataloader.paligemma_datasets import FreshEpisodeWindowDataset, FreshSnapshotDataset
from AlphaBrain.model.framework.base_framework import BaseFramework
try:
    from scripts.fresh_vla.paired_evaluation import bootstrap_summary
except ModuleNotFoundError:
    from paired_evaluation import bootstrap_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Pi0.5 suffix modes at identical pre-feedback observations")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=27183)
    parser.add_argument("--samples-per-context", type=int, default=32)
    parser.add_argument("--max-contexts", type=int, default=8)
    parser.add_argument("--mode-margin", type=float, default=0.02)
    parser.add_argument("--dataset-format", choices=("snapshot", "episode_window"), default="snapshot")
    parser.add_argument("--tasks", nargs="+")
    return parser.parse_args()


def _analyze_context(
    samples: np.ndarray,
    attached: np.ndarray,
    slipped: np.ndarray,
    horizon: int,
    *,
    mode_margin: float,
) -> dict[str, float | bool]:
    common = 0.5 * (attached + slipped)
    prefix_mse = np.square(samples[:, :horizon] - common[None, :horizon]).mean(axis=(1, 2))
    suffix = samples[:, horizon:]
    attached_distance = np.sqrt(np.square(suffix - attached[None, horizon:]).mean(axis=(1, 2)))
    slipped_distance = np.sqrt(np.square(suffix - slipped[None, horizon:]).mean(axis=(1, 2)))
    signed_margin = slipped_distance - attached_distance
    attached_fraction = float((signed_margin > mode_margin).mean())
    slipped_fraction = float((signed_margin < -mode_margin).mean())
    ambiguous_fraction = 1.0 - attached_fraction - slipped_fraction
    return {
        "common_prefix_mse": float(prefix_mse.mean()),
        "common_prefix_variance": float(samples[:, :horizon].var(axis=0).mean()),
        "suffix_variance": float(suffix.var(axis=0).mean()),
        "suffix_min_expert_distance": float(np.minimum(attached_distance, slipped_distance).mean()),
        "attached_mode_fraction": attached_fraction,
        "slipped_mode_fraction": slipped_fraction,
        "ambiguous_fraction": ambiguous_fraction,
        "covers_both_suffix_modes": attached_fraction >= 0.1 and slipped_fraction >= 0.1,
        "mode_balance": float(max(0.0, 1.0 - abs(attached_fraction - slipped_fraction))),
    }


def main() -> None:
    args = parse_args()
    if args.samples_per_context < 32:
        raise ValueError("samples_per_context must be at least 32")
    os.environ.setdefault("PRETRAINED_MODELS_DIR", "/share/longjunyu/alphabrain/pretrained_models")
    dataset_class = FreshSnapshotDataset if args.dataset_format == "snapshot" else FreshEpisodeWindowDataset
    tasks = tuple(args.tasks or (("grasp_slip",) if args.dataset_format == "snapshot" else ("grasp_slip_full_episode",)))
    dataset = dataset_class(
        args.data_root,
        split="test",
        feedback_label="oracle_feedback_horizon",
        feedback_output_key="feedback_horizon",
        tasks=tasks,
    )
    pairs = defaultdict(dict)
    for index in range(len(dataset)):
        sample = dataset[index]
        context = (sample["pair_id"], int(sample.get("frame_index", 0)))
        pairs[context][sample["branch_id"]] = sample
    contexts = []
    for context, branches in sorted(pairs.items()):
        if branches.keys() < {"attached", "slipped"}:
            continue
        horizon = int(branches["attached"]["oracle_feedback_horizon"])
        action_horizon = len(branches["attached"]["action"])
        if args.dataset_format == "episode_window" and not 0 < horizon < action_horizon:
            continue
        contexts.append(context)
    contexts = contexts[: args.max_contexts]

    model = BaseFramework.from_pretrained(str(args.checkpoint))
    model = model.to(torch.bfloat16).to(args.device).eval()
    model.gripper_remap = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_horizon = int(model.action_horizon)

    rows = []
    sampled_chunks = {}
    with torch.inference_mode():
        for context_index, context in enumerate(contexts):
            pair_id, frame_index = context
            attached_sample = pairs[context]["attached"]
            slipped_sample = pairs[context]["slipped"]
            chunks = []
            for sample_index in range(args.samples_per_context):
                sample_seed = args.seed + context_index * 10_000 + sample_index
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                    output = model.predict_action(examples=[attached_sample])
                chunks.append(np.asarray(output["normalized_actions"][0], dtype=np.float32))
            samples = np.stack(chunks)
            attached = np.asarray(attached_sample["action"], dtype=np.float32)[:model_horizon]
            slipped = np.asarray(slipped_sample["action"], dtype=np.float32)[:model_horizon]
            horizon = min(int(attached_sample["oracle_feedback_horizon"]), model_horizon - 1)
            row = {
                "pair_id": pair_id,
                "frame_index": frame_index,
                "oracle_feedback_horizon": horizon,
                **_analyze_context(
                    samples,
                    attached,
                    slipped,
                    horizon,
                    mode_margin=args.mode_margin,
                ),
            }
            rows.append(row)
            sampled_chunks[f"{pair_id}__frame{frame_index:04d}"] = samples

    scalar_metrics = (
        "common_prefix_mse",
        "common_prefix_variance",
        "suffix_variance",
        "suffix_min_expert_distance",
        "attached_mode_fraction",
        "slipped_mode_fraction",
        "ambiguous_fraction",
        "mode_balance",
    )
    summary = {
        metric: bootstrap_summary([float(row[metric]) for row in rows], seed=args.seed)
        for metric in scalar_metrics
    }
    summary["suffix_mode_coverage"] = float(np.mean([row["covers_both_suffix_modes"] for row in rows]))
    result = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "dataset_format": args.dataset_format,
        "tasks": list(tasks),
        "model_horizon": model_horizon,
        "samples_per_context": args.samples_per_context,
        "context_count": len(rows),
        "gripper_remap": False,
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), **sampled_chunks)
    print(json.dumps({"output": str(args.output), "summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
