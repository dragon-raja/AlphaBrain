#!/usr/bin/env python3
"""Compute a transparent design sensitivity grid for the Paper 1 paired contrast."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any


Z_ALPHA_TWO_SIDED_005 = 1.959963984540054


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def paired_binary_power(
    *, n: int, baseline: float, effect: float, correlation: float
) -> float:
    treatment = baseline + effect
    if not (0 < baseline < 1 and 0 < treatment < 1):
        raise ValueError("baseline and treatment probabilities must be in (0, 1)")
    if not (-1 <= correlation <= 1):
        raise ValueError("correlation must be in [-1, 1]")
    variance0 = baseline * (1 - baseline)
    variance1 = treatment * (1 - treatment)
    covariance = correlation * math.sqrt(variance0 * variance1)
    variance_difference = variance0 + variance1 - 2 * covariance
    if variance_difference <= 0:
        raise ValueError("assumptions imply non-positive paired variance")
    noncentrality = effect / math.sqrt(variance_difference / n)
    return float(
        1
        - normal_cdf(Z_ALPHA_TWO_SIDED_005 - noncentrality)
        + normal_cdf(-Z_ALPHA_TWO_SIDED_005 - noncentrality)
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    prereg = json.loads(args.preregistration.read_text())
    statistics = prereg["statistics"]
    population = prereg["population"]
    groups_per_seed = (
        len(population["tasks"])
        * len(population["scene_seeds"])
        * len(population["snapshot_partitions"])
    )
    effect = float(statistics["minimum_relevant_absolute_effect"])
    target = float(statistics["target_power"])
    records = []
    for retention in (1.0, 0.9, 0.8, 0.7):
        n = int(math.floor(groups_per_seed * retention))
        for baseline in (0.2, 0.5, 0.8):
            for correlation in (0.0, 0.25, 0.5):
                power = paired_binary_power(
                    n=n,
                    baseline=baseline,
                    effect=effect,
                    correlation=correlation,
                )
                records.append(
                    {
                        "retention": retention,
                        "groups_per_policy_seed": n,
                        "baseline_rate": baseline,
                        "absolute_effect": effect,
                        "paired_correlation": correlation,
                        "approximate_power": power,
                        "meets_target": power >= target,
                    }
                )
    full_independent = next(
        item
        for item in records
        if item["retention"] == 1.0
        and item["baseline_rate"] == 0.5
        and item["paired_correlation"] == 0.0
    )
    retained_independent = next(
        item
        for item in records
        if item["retention"] == 0.8
        and item["baseline_rate"] == 0.5
        and item["paired_correlation"] == 0.0
    )
    payload = {
        "schema": "dsol_paper1_precision_power_sensitivity_v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DESIGN_SENSITIVITY_ONLY_NOT_RELEASED",
        "preregistration": str(args.preregistration.resolve()),
        "preregistration_sha256": sha256(args.preregistration),
        "independent_unit": population["independent_unit"],
        "groups_per_policy_seed_before_attrition": groups_per_seed,
        "target_power": target,
        "minimum_relevant_absolute_effect": effect,
        "approximation": (
            "two-sided normal approximation for a paired binary difference; "
            "does not model task heterogeneity, relation calibration, or policy-seed variance"
        ),
        "sensitivity_grid": records,
        "diagnosis": {
            "full_population_midrate_independent": full_independent,
            "eighty_percent_retained_midrate_independent": retained_independent,
            "release_impact": (
                "B7 remains HOLD until empirical pilot variance, relation-gate attrition, "
                "and task-stratified simulations replace this analytical sensitivity grid"
            ),
        },
    }
    atomic_json(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
