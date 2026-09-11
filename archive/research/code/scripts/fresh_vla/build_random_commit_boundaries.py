from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_boundary_map(
    groups: list[dict],
    fixed_k3_rows: list[dict],
    *,
    seed: int,
) -> dict:
    test_groups = {str(group["pair_id"]): group for group in groups if group["split"] == "test"}
    episodes: dict[str, dict[str, dict]] = {}
    for row in fixed_k3_rows:
        if int(row["execution_horizon"]) != 3:
            continue
        pair_id = str(row["pair_id"])
        episodes.setdefault(pair_id, {})[str(row["branch_outcome"])] = row
    if episodes.keys() != test_groups.keys():
        raise ValueError("fixed-K3 rows and test manifest groups do not match")

    source = {}
    target_completion_limits = {}
    for pair_id in sorted(test_groups):
        paired = episodes[pair_id]
        if paired.keys() != {"attached", "slipped"}:
            raise ValueError(f"missing paired fixed-K3 branches for {pair_id}")
        attached_event = paired["attached"].get("event_time")
        slipped_event = paired["slipped"].get("event_time")
        if attached_event != slipped_event:
            raise ValueError(f"paired branch event time mismatch for {pair_id}: {attached_event} vs {slipped_event}")
        source[pair_id] = None if attached_event is None else int(attached_event)
        target_completion_limits[pair_id] = min(
            int(paired["attached"]["completion_steps"]),
            int(paired["slipped"]["completion_steps"]),
        )

    pair_ids = sorted(source)
    values = [source[pair_id] for pair_id in pair_ids]
    original = list(values)
    if len(values) > 1:
        rng = np.random.default_rng(seed)
        for _ in range(10_000):
            candidate = [original[index] for index in rng.permutation(len(original))]
            if candidate == original:
                continue
            if all(
                candidate[index] is None or candidate[index] <= target_completion_limits[pair_id]
                for index, pair_id in enumerate(pair_ids)
            ):
                values = candidate
                break
        else:
            raise ValueError("could not build a non-identity reachable random boundary permutation")
    boundaries = {pair_id: values[index] for index, pair_id in enumerate(pair_ids)}
    return {
        "schema_version": 1,
        "seed": seed,
        "source": "paired Full-H fixed-K3 runtime event_time; null means Oracle would not shorten that group",
        "matching": (
            "constrained permutation of boundary times across snapshot groups; each boundary is reachable "
            "before both paired fixed-K3 branches terminate; branch outcome is never a lookup key"
        ),
        "source_boundaries": source,
        "target_completion_limits": target_completion_limits,
        "boundaries": boundaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build outcome-blind random commit boundaries matched to Full-H events")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-k3-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    groups = json.loads(args.manifest.read_text())["groups"]
    rows = json.loads(args.fixed_k3_results.read_text())["rows"]
    payload = build_boundary_map(groups, rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
