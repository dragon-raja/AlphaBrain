from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_libero_closed_loop import _atomic_write_json
from evaluate_physical_process_oracle import STAGES, summarize_rows
from paired_evaluation import bootstrap_summary


METRICS = (
    "success",
    "next_stage_reached",
    "transport_reached",
    "lift_reached",
    "stable_grasp_at_end",
    "drop",
    "regress",
    "progress_auc",
    "object_to_bowl_progress",
    "object_height_progress",
)


def load_and_validate(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not paths:
        raise ValueError("at least one physical-process result is required")
    payloads = [json.loads(path.read_text()) for path in paths]
    for path, payload in zip(paths, payloads, strict=True):
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete physical-process result: {path}")
        if payload.get("schema_version") != 2:
            raise ValueError(f"unsupported physical-process schema: {path}")
        if payload.get("git_dirty_at_launch"):
            raise ValueError(f"decision-bearing result came from a dirty worktree: {path}")
        if not payload.get("git_sha") or not payload.get("policy_checkpoint_sha256"):
            raise ValueError(f"result is missing source or checkpoint identity: {path}")
    identity_keys = (
        "episode_root",
        "split",
        "sample_count",
        "execution_horizon",
        "bridge_steps_by_stage",
        "post_regrasp_source",
        "state_generation_max_steps",
        "stable_grasp_steps",
        "stage_dwell_steps",
        "selection_continuations",
        "validation_continuations",
        "git_sha",
    )
    identity = {key: payloads[0].get(key) for key in identity_keys}
    for path, payload in zip(paths[1:], payloads[1:], strict=True):
        proposed = {key: payload.get(key) for key in identity_keys}
        if proposed != identity:
            raise ValueError(f"evaluation configuration mismatch: {path}")

    rows = []
    seen = set()
    for path, payload in zip(paths, payloads, strict=True):
        for row in payload["rows"]:
            key = (int(payload["seed"]), str(row["pair_id"]), str(row["stage"]))
            if key in seen:
                raise ValueError(f"duplicate seed/group/stage row {key} in {path}")
            seen.add(key)
            rows.append({**row, "seed": int(payload["seed"]), "source_file": str(path)})
    return rows, identity


def paired_metric_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    outcome_source: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    deltas_by_pair: dict[str, list[float]] = defaultdict(list)
    deltas_by_seed_source: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if not row.get("eligible", True):
            continue
        source_cluster = row.get("source_initial_state_index")
        if source_cluster is None:
            raise ValueError(f"row is missing source_initial_state_index: {row['pair_id']}")
        if outcome_source == "selection":
            metric_key = f"{metric}_rate" if metric in METRICS[:7] else metric
            sample0 = row["candidates"][0]["selection_summary"]
            oracle = row["candidates"][int(row["oracle_index"])]["selection_summary"]
            delta = float(oracle[metric_key]) - float(sample0[metric_key])
        elif outcome_source == "heldout":
            heldout = row.get("heldout_continuations", ())
            if not heldout:
                continue
            delta = float(
                np.mean(
                    [
                        float(entry["oracle"]["bridge"][metric])
                        - float(entry["sample0"]["bridge"][metric])
                        for entry in heldout
                    ]
                )
            )
        else:
            raise ValueError(f"unknown outcome source: {outcome_source}")
        deltas_by_pair[str(row["pair_id"])].append(delta)
        deltas_by_seed_source[(int(row["seed"]), str(source_cluster))].append(delta)
    if not deltas_by_seed_source:
        return {"status": "no_eligible_states"}
    seed_source_deltas = {
        key: float(np.mean(values)) for key, values in sorted(deltas_by_seed_source.items())
    }
    source_values: dict[str, list[float]] = defaultdict(list)
    per_seed: dict[str, list[float]] = defaultdict(list)
    for (run_seed, source_cluster), delta in seed_source_deltas.items():
        source_values[source_cluster].append(delta)
        per_seed[str(run_seed)].append(delta)
    source_deltas = {source: float(np.mean(values)) for source, values in sorted(source_values.items())}
    return {
        "direction": "oracle_minus_sample0",
        "outcome_source": outcome_source,
        "independent_unit": "source_initial_state_index",
        "source_cluster_level": bootstrap_summary(
            list(source_deltas.values()),
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "per_source_cluster": source_deltas,
        "per_pair_diagnostic": {
            pair_id: float(np.mean(values)) for pair_id, values in sorted(deltas_by_pair.items())
        },
        "per_seed_mean": {run_seed: float(np.mean(values)) for run_seed, values in sorted(per_seed.items())},
    }


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260715,
) -> dict[str, Any]:
    result = {
        "descriptive": summarize_rows(rows),
        "paired_oracle_vs_sample0": {},
        "paired_heldout_oracle_vs_sample0": {},
    }
    for stage in STAGES:
        selected = [row for row in rows if row["stage"] == stage]
        if not selected:
            continue
        result["paired_oracle_vs_sample0"][stage] = {
            metric: paired_metric_summary(
                selected,
                metric=metric,
                outcome_source="selection",
                bootstrap_samples=bootstrap_samples,
                seed=seed + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
        result["paired_heldout_oracle_vs_sample0"][stage] = {
            metric: paired_metric_summary(
                selected,
                metric=metric,
                outcome_source="heldout",
                bootstrap_samples=bootstrap_samples,
                seed=seed + 100 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and summarize physical-process Oracle shards")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, identity = load_and_validate(args.inputs)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "input_files": [str(path) for path in args.inputs],
        "evaluation_identity": identity,
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "group_count": len({row["pair_id"] for row in rows}),
        "source_cluster_count": len({row["source_initial_state_index"] for row in rows}),
        "row_count": len(rows),
        "summary": build_summary(
            rows,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "rows": rows,
    }
    _atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
