from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


BOOTSTRAP_SEED = 20260728


def load_successes(
    root: Path,
    *,
    arm: str,
    seed: int,
    pose_set: str,
    epoch: int,
) -> dict[int, float]:
    path = (
        root
        / f"official_act_lift_randomized_{arm}_seed{seed}"
        / f"eval_epoch_{epoch}_{pose_set}"
        / "success_by_seed.json"
    )
    payload = json.loads(path.read_text())
    result = {int(key): float(bool(value)) for key, value in payload.items()}
    if not result:
        raise ValueError(f"official KYC evaluation has no episodes: {path}")
    return result


def paired_delta(
    control: Mapping[int, float],
    kyc: Mapping[int, float],
) -> np.ndarray:
    if set(control) != set(kyc):
        raise ValueError("official image/KYC evaluations are not episode-paired")
    return np.asarray(
        [kyc[index] - control[index] for index in sorted(control)],
        dtype=np.float64,
    )


def hierarchical_bootstrap(
    deltas_by_seed: Mapping[int, np.ndarray],
    *,
    resamples: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    seeds = sorted(deltas_by_seed)
    if not seeds:
        raise ValueError("hierarchical bootstrap has no training seeds")
    rng = np.random.default_rng(seed)
    distribution = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for training_seed in sampled_seeds:
            values = deltas_by_seed[int(training_seed)]
            sampled_values = rng.choice(values, size=len(values), replace=True)
            seed_means.append(float(np.mean(sampled_values)))
        distribution[index] = float(np.mean(seed_means))
    observed = float(
        np.mean([np.mean(values) for values in deltas_by_seed.values()])
    )
    return {
        "delta": observed,
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "training_seed_count": len(seeds),
        "episodes_per_seed": int(
            min(len(values) for values in deltas_by_seed.values())
        ),
        "bootstrap_resamples": resamples,
    }


def summarize_official(
    root: Path,
    *,
    seeds: Sequence[int],
    pose_sets: Sequence[str],
    epoch: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    pose_payloads = {}
    for pose_index, pose_set in enumerate(pose_sets):
        per_seed = {}
        deltas_by_seed = {}
        for seed in seeds:
            image = load_successes(
                root,
                arm="image",
                seed=seed,
                pose_set=pose_set,
                epoch=epoch,
            )
            kyc = load_successes(
                root,
                arm="kyc",
                seed=seed,
                pose_set=pose_set,
                epoch=epoch,
            )
            deltas = paired_delta(image, kyc)
            deltas_by_seed[seed] = deltas
            per_seed[str(seed)] = {
                "episode_count": len(image),
                "image_success": float(np.mean(list(image.values()))),
                "kyc_success": float(np.mean(list(kyc.values()))),
                "kyc_minus_image": float(np.mean(deltas)),
            }
        pose_payloads[pose_set] = {
            "per_training_seed": per_seed,
            "equal_seed_mean": {
                key: float(np.mean([row[key] for row in per_seed.values()]))
                for key in (
                    "image_success",
                    "kyc_success",
                    "kyc_minus_image",
                )
            },
            "hierarchical_paired_bootstrap": hierarchical_bootstrap(
                deltas_by_seed,
                resamples=bootstrap_resamples,
                seed=BOOTSTRAP_SEED + pose_index,
            ),
        }
    return {
        "schema_version": 1,
        "status": "complete",
        "study": "official_kyc_act_lift_randomized_positive_control",
        "official_repository_commit": "e0647105",
        "official_robosuite_commit": "0df3a5f5782937fa2e8fa923910d616b73310b8d",
        "epoch": epoch,
        "training_seeds": list(seeds),
        "pose_sets": pose_payloads,
        "inference_unit": "evaluation_episode_seed",
        "aggregate_weighting": "equal_training_seed",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": bootstrap_resamples,
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the released KYC ACT Lift randomized control"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--pose-sets",
        nargs="+",
        default=["train_cameras", "test_cameras"],
    )
    parser.add_argument("--epoch", type=int, default=20_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    if parsed.output.exists():
        raise FileExistsError(f"refusing to overwrite summary: {parsed.output}")
    payload = summarize_official(
        parsed.run_root,
        seeds=parsed.seeds,
        pose_sets=parsed.pose_sets,
        epoch=parsed.epoch,
        bootstrap_resamples=parsed.bootstrap_resamples,
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                pose_set: result["equal_seed_mean"]
                for pose_set, result in payload["pose_sets"].items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
