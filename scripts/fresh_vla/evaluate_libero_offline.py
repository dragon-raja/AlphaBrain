from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from AlphaBrain.dataloader.paligemma_datasets import FreshSnapshotDataset
from AlphaBrain.model.framework.base_framework import BaseFramework
try:
    from scripts.fresh_vla.paired_evaluation import EvaluationIdentity, bootstrap_summary, per_sample_flow_metrics
except ModuleNotFoundError:
    from paired_evaluation import EvaluationIdentity, bootstrap_summary, per_sample_flow_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Pi0.5 checkpoint on paired snapshot data")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--control-output", type=Path)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--tasks", nargs="+", default=("grasp_slip",))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=9107)
    parser.add_argument("--fixed-k", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def _aggregate(rows: list[dict[str, object]], metric: str, args: argparse.Namespace) -> dict[str, float | int | None]:
    values = [float(row[metric]) for row in rows if row[metric] is not None]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "standard_error": None,
            "bootstrap_95_low": None,
            "bootstrap_95_high": None,
        }
    return bootstrap_summary(values, bootstrap_samples=args.bootstrap_samples, seed=args.seed)


def _evaluate_tasks(
    model: BaseFramework,
    args: argparse.Namespace,
    *,
    tasks: tuple[str, ...],
    output_path: Path,
) -> None:
    dataset = FreshSnapshotDataset(
        args.data_root,
        split=args.split,
        feedback_label="oracle_feedback_horizon",
        feedback_output_key="feedback_horizon",
        tasks=tasks,
    )
    sample_count = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    samples = [dataset[index] for index in range(sample_count)]
    sample_ids = tuple(str(sample["fresh_sample_id"]) for sample in samples)
    sample_content = np.stack([np.asarray(sample["action"], dtype=np.float32) for sample in samples])
    flow_times = np.asarray([args.seed + index for index in range(sample_count)], dtype=np.int64)
    identity = EvaluationIdentity(
        sample_ids=sample_ids,
        flow_times=flow_times,
        noise=np.empty((0,), dtype=np.float32),
        action_normalization={"source": "checkpoint_framework_config"},
        sample_content=sample_content,
    )
    fingerprint = identity.fingerprint()

    model_horizon = int(model.action_horizon)
    if args.fixed_k > model_horizon:
        raise ValueError(f"fixed_k={args.fixed_k} exceeds model horizon={model_horizon}")

    rows = []
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            sample = dict(sample)
            sample["feedback_horizon"] = min(int(sample["feedback_horizon"]), model_horizon)
            torch.manual_seed(args.seed + index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + index)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                output = model.forward([sample])
            per_step = np.asarray(
                [float(output[f"fresh_step_loss_{step:02d}"]) for step in range(model_horizon)],
                dtype=np.float64,
            )[None, :]
            horizon = min(int(sample["oracle_feedback_horizon"]), model_horizon)
            metrics = per_sample_flow_metrics(per_step, [horizon], fixed_k=args.fixed_k)[0]
            fixed_k_metrics = {
                f"fixed_k_{fixed_k}": float(per_step[0, :fixed_k].mean())
                for fixed_k in (1, 2, 3)
                if fixed_k <= model_horizon
            }
            rows.append(
                {
                    "sample_id": sample["fresh_sample_id"],
                    "pair_id": sample["pair_id"],
                    "branch_id": sample["branch_id"],
                    "task": sample["task"],
                    "oracle_feedback_horizon": horizon,
                    "evaluation_fingerprint": fingerprint,
                    **metrics,
                    **fixed_k_metrics,
                    "per_step": per_step[0].tolist(),
                }
            )

    summary = {
        metric: _aggregate(rows, metric, args)
        for metric in ("fixed_k", "oracle_prefix", "suffix", "full")
    }
    summary.update(
        {
            f"fixed_k_{fixed_k}": _aggregate(rows, f"fixed_k_{fixed_k}", args)
            for fixed_k in (1, 2, 3)
            if fixed_k <= model_horizon
        }
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "split": args.split,
        "tasks": list(tasks),
        "model_horizon": model_horizon,
        "fixed_k": args.fixed_k,
        "seed": args.seed,
        "evaluation_fingerprint": fingerprint,
        "summary": summary,
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_path), "summary": summary}, sort_keys=True))


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PRETRAINED_MODELS_DIR", "/share/longjunyu/alphabrain/pretrained_models")
    model = BaseFramework.from_pretrained(str(args.checkpoint))
    model = model.to(torch.bfloat16).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    output = args.output or args.checkpoint.parent / "offline_eval.json"
    _evaluate_tasks(model, args, tasks=tuple(args.tasks), output_path=output)
    if args.control_output is not None:
        _evaluate_tasks(
            model,
            args,
            tasks=("deterministic_reach",),
            output_path=args.control_output,
        )


if __name__ == "__main__":
    main()
