#!/usr/bin/env python3
"""Aggregate paired fixed-condition LIBERO closed-loop results across methods."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ARMS = (
    "canonical_unique",
    "canonical_repeat",
    "image_augmentation_unique",
    "broad_unpaired_practical",
    "broad_unpaired_state_matched",
    "broad_paired_fm",
    "broad_paired_consistency",
)
CONDITIONS = (
    "canonical_both",
    "broad_heldout_both",
    "wide_extrapolation_both",
    "canonical_external_only",
    "canonical_wrist_only",
    "broad_heldout_external_only",
    "broad_heldout_wrist_only",
)
METHOD_CONTRASTS = (
    ("repeat_minus_unique", "canonical_repeat", "canonical_unique"),
    ("image_aug_minus_unique", "image_augmentation_unique", "canonical_unique"),
    ("broad_practical_minus_unique", "broad_unpaired_practical", "canonical_unique"),
    (
        "broad_state_matched_minus_unique",
        "broad_unpaired_state_matched",
        "canonical_unique",
    ),
    (
        "paired_fm_minus_state_matched",
        "broad_paired_fm",
        "broad_unpaired_state_matched",
    ),
    (
        "paired_consistency_minus_paired_fm",
        "broad_paired_consistency",
        "broad_paired_fm",
    ),
    (
        "paired_consistency_minus_state_matched",
        "broad_paired_consistency",
        "broad_unpaired_state_matched",
    ),
)


def read_rows(path: Path) -> list[dict]:
    rows = []
    for ledger in sorted(path.glob("episodes-shard-*.jsonl")):
        rows.extend(json.loads(line) for line in ledger.read_text().splitlines() if line.strip())
    if not rows:
        raise ValueError(f"no episode rows in {path}")
    return rows


def paired_bootstrap(values: np.ndarray, *, seed: int, samples: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    return np.quantile(values[draws].mean(axis=1), [0.025, 0.975]).tolist()


def index_rows(rows: list[dict]) -> dict[tuple[str, str], dict]:
    indexed = {}
    for row in rows:
        key = (str(row["pair_key"]), str(row["condition"]))
        if key in indexed:
            raise ValueError(f"duplicate episode key: {key}")
        indexed[key] = row
    return indexed


def build_summary(
    rows_by_arm: dict[str, list[dict]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict, list[dict], list[dict]]:
    if set(rows_by_arm) != set(ARMS):
        raise ValueError(f"expected arms {ARMS}, got {sorted(rows_by_arm)}")
    indexed = {arm: index_rows(rows) for arm, rows in rows_by_arm.items()}
    reference_keys = set(indexed[ARMS[0]])
    for arm in ARMS[1:]:
        if set(indexed[arm]) != reference_keys:
            raise ValueError(f"episode key mismatch for {arm}")
    physics_mismatches = []
    for key in sorted(reference_keys):
        hashes = {
            indexed[arm][key]["initial_metrics"]["physics_state_sha256"]
            for arm in ARMS
        }
        if len(hashes) != 1:
            physics_mismatches.append(key)
    if physics_mismatches:
        raise ValueError(f"cross-method physics mismatch: {physics_mismatches[:5]}")

    matrix_rows = []
    rates = defaultdict(dict)
    for arm in ARMS:
        for condition in CONDITIONS:
            selected = [
                row for (pair, current), row in indexed[arm].items() if current == condition
            ]
            successes = sum(bool(row["success"]) for row in selected)
            rate = successes / len(selected)
            rates[arm][condition] = rate
            matrix_rows.append(
                {
                    "arm": arm,
                    "condition": condition,
                    "successes": successes,
                    "episodes": len(selected),
                    "success_rate": rate,
                }
            )

    contrast_rows = []
    for contrast_index, (name, left, right) in enumerate(METHOD_CONTRASTS):
        for condition_index, condition in enumerate(CONDITIONS):
            differences = np.asarray(
                [
                    float(indexed[left][(pair_key, condition)]["success"])
                    - float(indexed[right][(pair_key, condition)]["success"])
                    for pair_key in sorted({pair for pair, _ in reference_keys})
                ],
                dtype=np.float64,
            )
            interval = paired_bootstrap(
                differences,
                seed=seed + contrast_index * 100 + condition_index,
                samples=bootstrap_samples,
            )
            contrast_rows.append(
                {
                    "contrast": name,
                    "left": left,
                    "right": right,
                    "condition": condition,
                    "difference_pp": float(differences.mean() * 100.0),
                    "ci95_low_pp": float(interval[0] * 100.0),
                    "ci95_high_pp": float(interval[1] * 100.0),
                    "paired_groups": int(len(differences)),
                }
            )

    summary = {
        "schema": "dsol_libero_fixed_condition_cross_method_summary_v1",
        "status": "PASS",
        "arms": list(ARMS),
        "conditions": list(CONDITIONS),
        "paired_group_count": len({pair for pair, _ in reference_keys}),
        "episode_count": sum(len(rows) for rows in rows_by_arm.values()),
        "cross_method_physics_mismatches": 0,
        "success_rates": rates,
        "method_contrasts": contrast_rows,
        "statistical_unit": "source HDF5 episode / pair_key",
    }
    return summary, matrix_rows, contrast_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-tag", default="quick-gate-v1")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    args = parser.parse_args()

    rows_by_arm = {}
    for arm in ARMS:
        run_id = (
            f"dsol_{arm}_{args.run_tag}_seed{args.seed}_g{args.gpus}_gb32_steps{args.steps}"
        )
        rows_by_arm[arm] = read_rows(args.root / run_id)
    summary, matrix_rows, contrast_rows = build_summary(
        rows_by_arm,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_csv(args.output_dir / "success_matrix.csv", matrix_rows)
    write_csv(args.output_dir / "method_contrasts.csv", contrast_rows)

    try:
        import matplotlib.pyplot as plt

        values = np.asarray(
            [[summary["success_rates"][arm][condition] * 100 for condition in CONDITIONS] for arm in ARMS]
        )
        figure, axis = plt.subplots(figsize=(12.5, 6.2), constrained_layout=True)
        image = axis.imshow(values, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(CONDITIONS)), CONDITIONS, rotation=30, ha="right")
        axis.set_yticks(range(len(ARMS)), ARMS)
        for row in range(len(ARMS)):
            for column in range(len(CONDITIONS)):
                color = "white" if values[row, column] < 55 else "black"
                axis.text(column, row, f"{values[row, column]:.1f}", ha="center", va="center", color=color)
        axis.set_title("LIBERO exact-state full closed-loop success (%)")
        figure.colorbar(image, ax=axis, label="Success (%)")
        figure.savefig(args.output_dir / "success_matrix.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
