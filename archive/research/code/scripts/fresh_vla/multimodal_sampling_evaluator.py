from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze_samples(
    action_samples: np.ndarray,
    common_actions: np.ndarray,
    horizons: np.ndarray,
    branch_direction: np.ndarray,
    *,
    mode_margin: float = 0.2,
) -> dict[str, object]:
    samples = np.asarray(action_samples, dtype=np.float64)
    common = np.asarray(common_actions, dtype=np.float64)
    horizons = np.asarray(horizons, dtype=np.int64)
    direction = np.asarray(branch_direction, dtype=np.float64)
    if samples.ndim != 4:
        raise ValueError(f"action_samples must be [N, S, H, D], got {samples.shape}")
    if common.shape != (samples.shape[0], samples.shape[2], samples.shape[3]):
        raise ValueError(f"common_actions must be [N, H, D], got {common.shape}")
    if horizons.shape != (samples.shape[0],):
        raise ValueError(f"horizons must be [N], got {horizons.shape}")
    if direction.shape != (samples.shape[3],) or np.linalg.norm(direction) == 0:
        raise ValueError(f"branch_direction must be nonzero [D], got {direction.shape}")
    if np.any((horizons < 0) | (horizons > samples.shape[2])):
        raise ValueError("horizons are outside [0, H]")

    direction = direction / np.linalg.norm(direction)
    residual = samples - common[:, None, :, :]
    projection = np.einsum("nshd,d->nsh", residual, direction)
    steps = np.arange(samples.shape[2])[None, :]
    prefix_mask = steps < horizons[:, None]
    prefix_count = max(int(prefix_mask.sum()) * samples.shape[1], 1)
    premature = float((np.abs(projection) * prefix_mask[:, None, :]).sum() / prefix_count)
    variance = samples.var(axis=1).mean(axis=-1)
    prefix_variance = float((variance * prefix_mask).sum() / max(int(prefix_mask.sum()), 1))

    terminal = projection[:, :, -1]
    positive = (terminal > mode_margin).mean(axis=1)
    negative = (terminal < -mode_margin).mean(axis=1)
    coverage = (positive >= 0.1) & (negative >= 0.1)
    branch_motion_before_feedback = np.abs(projection) * prefix_mask[:, None, :]
    return {
        "context_count": samples.shape[0],
        "samples_per_context": samples.shape[1],
        "common_prefix_sampling_variance": prefix_variance,
        "premature_commitment": premature,
        "suffix_mode_coverage": float(coverage.mean()),
        "suffix_mode_balance": float(np.maximum(0.0, 1.0 - np.abs(positive - negative)).mean()),
        "contexts_with_branch_motion_before_feedback": float(
            (branch_motion_before_feedback.max(axis=(1, 2)) > mode_margin).mean()
        ),
        "per_context": [
            {
                "positive_fraction": float(positive[index]),
                "negative_fraction": float(negative[index]),
                "covers_both_modes": bool(coverage[index]),
            }
            for index in range(samples.shape[0])
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze 32+ sampled action chunks per counterfactual state")
    parser.add_argument("input", type=Path, help="NPZ with samples, common_actions, horizons, branch_direction")
    parser.add_argument("--mode-margin", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays = np.load(args.input)
    result = analyze_samples(
        arrays["samples"],
        arrays["common_actions"],
        arrays["horizons"],
        arrays["branch_direction"],
        mode_margin=args.mode_margin,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(args.output)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
