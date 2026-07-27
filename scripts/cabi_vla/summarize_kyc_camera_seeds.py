from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from compare_kyc_camera_evaluations import (
    BOOTSTRAP_SEED,
    METRICS,
    TRAINING_SUPPORT,
    index_fov_rows,
    join_episode_rows,
    validate_paired_evaluations,
)


def parse_seed_spec(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise ValueError(f"seed checkpoint must use SEED=JSON syntax: {value!r}")
    seed_text, path_text = value.split("=", 1)
    seed = int(seed_text)
    if seed < 0 or not path_text:
        raise ValueError(f"invalid seed checkpoint: {value!r}")
    return seed, Path(path_text)


def is_training_support(row: Mapping[str, Any]) -> bool:
    if str(row["camera_pose"]) == "baseline":
        return True
    axis = str(row["sweep_axis"])
    support = TRAINING_SUPPORT.get(axis)
    if support is None:
        raise ValueError(f"unknown camera sweep axis: {axis!r}")
    value = float(row["sweep_value"])
    return support[0] <= value <= support[1]


def scope_matches(row: Mapping[str, Any], scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "canonical":
        return str(row["camera_pose"]) == "baseline"
    fully_supported = str(row["visibility_stratum"]) == "fully_supported"
    if scope == "fully_supported":
        return fully_supported
    if scope == "training_support":
        return fully_supported and is_training_support(row)
    if scope == "extrapolation_supported":
        return fully_supported and not is_training_support(row)
    if scope.startswith("stratum:"):
        return str(row["visibility_stratum"]) == scope.split(":", 1)[1]
    raise ValueError(f"unknown summary scope: {scope!r}")


def paired_group_bootstrap(
    differences: Mapping[int, Sequence[float]],
    *,
    resamples: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not differences or any(not values for values in differences.values()):
        raise ValueError("bootstrap differences must contain non-empty state groups")
    states = sorted(differences)
    state_means = np.asarray(
        [np.mean(differences[state]) for state in states],
        dtype=np.float64,
    )
    if len(states) == 1:
        low = high = float(state_means[0])
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(states), size=(resamples, len(states)))
        draws = state_means[indices].mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975]).tolist()
    return {
        "delta": float(np.mean(state_means)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "state_count": len(states),
    }


def summarize_seed_pairs(
    seed_rows: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    bootstrap_resamples: int,
) -> list[dict[str, Any]]:
    strata = sorted(
        {
            str(row["visibility_stratum"])
            for rows in seed_rows.values()
            for row in rows
        }
    )
    scopes = [
        "canonical",
        "training_support",
        "fully_supported",
        "extrapolation_supported",
        "all",
        *[f"stratum:{stratum}" for stratum in strata],
    ]
    summaries = []
    for scope in scopes:
        selected_by_seed = {
            seed: [row for row in rows if scope_matches(row, scope)]
            for seed, rows in seed_rows.items()
        }
        if not all(selected_by_seed.values()):
            continue
        pose_keys = {
            seed: {
                (
                    row["edge_id"],
                    int(row["canonical_state_index"]),
                    int(row["execution_horizon"]),
                    row["camera_pose"],
                )
                for row in rows
                if row["method"] == "poseaug_control"
            }
            for seed, rows in selected_by_seed.items()
        }
        if len({frozenset(keys) for keys in pose_keys.values()}) != 1:
            raise ValueError(f"scope {scope!r} is not paired across seeds")

        result: dict[str, Any] = {
            "scope": scope,
            "seed_count": len(seed_rows),
            "seeds": sorted(seed_rows),
        }
        for metric in METRICS:
            seed_metrics = []
            grouped_differences: dict[int, list[float]] = defaultdict(list)
            for seed, rows in selected_by_seed.items():
                by_method = {
                    method: {
                        (
                            row["edge_id"],
                            int(row["canonical_state_index"]),
                            int(row["execution_horizon"]),
                            row["camera_pose"],
                        ): row
                        for row in rows
                        if row["method"] == method
                    }
                    for method in ("poseaug_control", "kyc")
                }
                if set(by_method["poseaug_control"]) != set(by_method["kyc"]):
                    raise ValueError(f"scope {scope!r}, seed {seed} is not paired")
                control_values = []
                kyc_values = []
                for key in sorted(by_method["poseaug_control"]):
                    control_value = float(by_method["poseaug_control"][key][metric])
                    kyc_value = float(by_method["kyc"][key][metric])
                    control_values.append(control_value)
                    kyc_values.append(kyc_value)
                    grouped_differences[int(key[1])].append(
                        kyc_value - control_value
                    )
                seed_metrics.append(
                    {
                        "seed": seed,
                        "poseaug_control": float(np.mean(control_values)),
                        "kyc": float(np.mean(kyc_values)),
                        "delta": float(np.mean(kyc_values) - np.mean(control_values)),
                        "episode_count_per_method": len(control_values),
                    }
                )
            paired = paired_group_bootstrap(
                grouped_differences,
                resamples=bootstrap_resamples,
                seed=BOOTSTRAP_SEED,
            )
            result[metric] = {
                "poseaug_control_mean": float(
                    np.mean([row["poseaug_control"] for row in seed_metrics])
                ),
                "kyc_mean": float(np.mean([row["kyc"] for row in seed_metrics])),
                "delta": paired["delta"],
                "ci95_low": paired["ci95_low"],
                "ci95_high": paired["ci95_high"],
                "paired_state_count": paired["state_count"],
                "per_seed": seed_metrics,
            }
        summaries.append(result)
    return summaries


def load_seed_rows(
    control_paths: Mapping[int, Path],
    kyc_paths: Mapping[int, Path],
    fov_payloads: Sequence[Mapping[str, Any]],
    *,
    minimum_patch_support: int,
) -> dict[int, list[dict[str, Any]]]:
    if set(control_paths) != set(kyc_paths) or not control_paths:
        raise ValueError("control and KYC seed sets must be identical and non-empty")
    fov = index_fov_rows(fov_payloads)
    seed_rows = {}
    for seed in sorted(control_paths):
        evaluations = {
            "poseaug_control": json.loads(control_paths[seed].read_text()),
            "kyc": json.loads(kyc_paths[seed].read_text()),
        }
        indexed = validate_paired_evaluations(evaluations)
        seed_rows[seed] = join_episode_rows(
            indexed,
            fov,
            minimum_patch_support=minimum_patch_support,
        )
    return seed_rows


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize matched KYC/control camera gates across seeds"
    )
    parser.add_argument("--control", action="append", required=True, metavar="SEED=JSON")
    parser.add_argument("--kyc", action="append", required=True, metavar="SEED=JSON")
    parser.add_argument("--fov-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-patch-support", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args(args)


def _specs(values: Sequence[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        seed, path = parse_seed_spec(value)
        if seed in result:
            raise ValueError(f"duplicate seed: {seed}")
        result[seed] = path
    return result


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    if parsed.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite summary: {parsed.output_dir}")
    controls = _specs(parsed.control)
    kycs = _specs(parsed.kyc)
    fov_payloads = [json.loads(path.read_text()) for path in parsed.fov_json]
    seed_rows = load_seed_rows(
        controls,
        kycs,
        fov_payloads,
        minimum_patch_support=parsed.minimum_patch_support,
    )
    summaries = summarize_seed_pairs(
        seed_rows,
        bootstrap_resamples=parsed.bootstrap_resamples,
    )
    by_scope = {row["scope"]: row for row in summaries}
    supported = by_scope["fully_supported"]["success"]
    canonical = by_scope["canonical"]["success"]
    gate_passed = (
        (supported["delta"] >= 0.10 or supported["ci95_low"] > 0.0)
        and canonical["delta"] >= -0.05
    )
    report = {
        "schema_version": 1,
        "study": "kyc_camera_seed_summary",
        "control_evaluations": {
            str(seed): str(path) for seed, path in controls.items()
        },
        "kyc_evaluations": {
            str(seed): str(path) for seed, path in kycs.items()
        },
        "fov_json": [str(path) for path in parsed.fov_json],
        "paired_unit": "canonical_state_index",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": parsed.bootstrap_resamples,
        "minimum_patch_support": parsed.minimum_patch_support,
        "incremental_camera_metadata_gate": {
            "passed": gate_passed,
            "supported_success_delta_threshold": 0.10,
            "canonical_regression_floor": -0.05,
        },
        "summaries": summaries,
    }

    staging = (
        parsed.output_dir.parent / f".{parsed.output_dir.name}.staging-{os.getpid()}"
    )
    parsed.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        flat_rows = []
        for row in summaries:
            for metric in METRICS:
                values = row[metric]
                flat_rows.append(
                    {
                        "scope": row["scope"],
                        "metric": metric,
                        "seed_count": row["seed_count"],
                        **{
                            key: value
                            for key, value in values.items()
                            if key != "per_seed"
                        },
                    }
                )
        with (staging / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(flat_rows[0]),
            )
            writer.writeheader()
            writer.writerows(flat_rows)
        staging.rename(parsed.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output_dir": str(parsed.output_dir),
                "seed_count": len(seed_rows),
                "incremental_camera_metadata_gate": gate_passed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
