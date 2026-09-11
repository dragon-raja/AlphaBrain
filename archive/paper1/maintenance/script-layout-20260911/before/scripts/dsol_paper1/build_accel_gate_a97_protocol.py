#!/usr/bin/env python3
"""Build model-specific closed-loop protocols for the 97-view Accel gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SHORTLIST_CONDITIONS = (
    "canonical",
    "accel_single_noise",
    "accel_ensemble",
    "visibility_top1",
    "accel_top10_visibility",
    "random_operational",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def pose_index(catalog: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    poses: dict[str, Mapping[str, Any]] = {}
    for values in catalog.values():
        if not isinstance(values, list):
            continue
        for row in values:
            if not isinstance(row, Mapping) or "pose_id" not in row:
                continue
            pose_id = str(row["pose_id"])
            if pose_id in poses:
                raise ValueError(f"duplicate pose_id in catalog: {pose_id}")
            poses[pose_id] = row
    return poses


def indexed_state_files(root: Path, filename: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted((root / "states").glob(f"*/{filename}")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        pair_key = str(payload.get("pair_key") or payload["fixed_state_audit"]["pair_key"])
        if pair_key in result:
            raise ValueError(f"duplicate pair_key under {root}: {pair_key}")
        result[pair_key] = path
    return result


def load_render_metadata(root: Path) -> dict[str, dict[str, Mapping[str, Any]]]:
    records = indexed_state_files(root, "render_record.json")
    result = {}
    for pair_key, record_path in records.items():
        metadata_path = record_path.with_name("candidate_metadata.json")
        rows = json.loads(metadata_path.read_text(encoding="utf-8"))
        result[pair_key] = {str(row["candidate_id"]): row for row in rows}
    return result


def load_noise_run(root: Path) -> dict[str, dict[str, float]]:
    records = indexed_state_files(root, "rank_record.json")
    result = {}
    for pair_key, record_path in records.items():
        ranking = json.loads(record_path.with_name("rankings.json").read_text(encoding="utf-8"))
        rows = ranking["operational_97"]["ranking"]
        result[pair_key] = {
            str(row["candidate_id"]): float(row["accel_3"]) for row in rows
        }
    return result


def mean_scores(
    runs: Sequence[Mapping[str, Mapping[str, float]]], pair_key: str
) -> dict[str, float]:
    candidate_ids = sorted(runs[0][pair_key])
    if any(sorted(run[pair_key]) != candidate_ids for run in runs[1:]):
        raise ValueError(f"noise runs disagree on candidate bank: {pair_key}")
    return {
        candidate_id: sum(run[pair_key][candidate_id] for run in runs) / len(runs)
        for candidate_id in candidate_ids
    }


def stable_random_candidate(pair_key: str, candidates: Sequence[str], seed: int) -> str:
    digest = hashlib.sha256(f"{seed}::{pair_key}".encode()).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


def operational_metadata(
    metadata: Mapping[str, Mapping[str, Any]], scores: Mapping[str, float]
) -> dict[str, Mapping[str, Any]]:
    result = {candidate_id: metadata[candidate_id] for candidate_id in scores}
    if len(result) != 97 or "canonical" not in result:
        raise ValueError("closed-loop gate requires exactly 97 operational candidates")
    return result


def candidate_pose(
    candidate_id: str, poses: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    if candidate_id == "canonical":
        return None
    if candidate_id not in poses:
        raise ValueError(f"candidate pose absent from catalog: {candidate_id}")
    return poses[candidate_id]


def base_specs(protocol: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for spec in protocol["specs"]:
        pair_key = str(spec["pair_key"])
        if pair_key in result:
            raise ValueError(f"base protocol repeats state: {pair_key}")
        result[pair_key] = spec
    if len(result) != int(protocol["selected_state_count"]):
        raise ValueError("base protocol state count mismatch")
    return result


def selection_for_state(
    *,
    pair_key: str,
    metadata: Mapping[str, Mapping[str, Any]],
    noise_runs: Sequence[Mapping[str, Mapping[str, float]]],
    random_seed: int,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    averaged = mean_scores(noise_runs, pair_key)
    operational = operational_metadata(metadata, averaged)
    ordered = sorted(averaged, key=lambda value: (averaged[value], value))
    single_scores = noise_runs[0][pair_key]
    single_ordered = sorted(single_scores, key=lambda value: (single_scores[value], value))
    visible = sorted(
        operational,
        key=lambda value: (-float(operational[value]["visibility_score"]), value),
    )
    top10 = ordered[:10]
    hybrid = min(
        top10,
        key=lambda value: (-float(operational[value]["visibility_score"]), averaged[value], value),
    )
    candidate_ids = sorted(operational)
    selected = {
        "canonical": "canonical",
        "accel_single_noise": single_ordered[0],
        "accel_ensemble": ordered[0],
        "visibility_top1": visible[0],
        "accel_top10_visibility": hybrid,
        "random_operational": stable_random_candidate(pair_key, candidate_ids, random_seed),
    }
    annotations = {
        candidate_id: {
            "ensemble_accel_3": averaged[candidate_id],
            "ensemble_accel_rank": ordered.index(candidate_id) + 1,
            "single_noise_accel_3": single_scores[candidate_id],
            "single_noise_accel_rank": single_ordered.index(candidate_id) + 1,
            "visibility_score": float(operational[candidate_id]["visibility_score"]),
            "delta_visibility": float(operational[candidate_id]["delta_visibility"]),
            "catalog_group": str(operational[candidate_id]["catalog_group"]),
        }
        for candidate_id in operational
    }
    return selected, annotations


def make_spec(
    base: Mapping[str, Any],
    *,
    model: str,
    condition: str,
    candidate_id: str,
    annotation: Mapping[str, Any],
    poses: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    identity = f"{model}::{base['pair_key']}::{condition}::{candidate_id}"
    return {
        **dict(base),
        "condition": condition,
        "pose": candidate_pose(candidate_id, poses),
        "sensor_control": "both",
        "episode_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
        "selected_candidate_id": candidate_id,
        "selection_metadata": dict(annotation),
    }


def choose_oracle_states(
    specs: Mapping[str, Mapping[str, Any]], states_per_task: int
) -> list[str]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for spec in specs.values():
        grouped[str(spec["task_id"])].append(spec)
    selected = []
    targets = (0.125, 0.375, 0.625, 0.875)
    for task_index, task in enumerate(sorted(grouped)):
        remaining = list(grouped[task])
        for offset in range(states_per_task):
            target = targets[(task_index + offset) % len(targets)]
            choice = min(
                remaining,
                key=lambda row: (abs(float(row["stage_fraction"]) - target), str(row["pair_key"])),
            )
            selected.append(str(choice["pair_key"]))
            remaining.remove(choice)
    return selected


def build(
    *,
    base_protocol: Mapping[str, Any],
    catalog: Mapping[str, Any],
    render_metadata: Mapping[str, Mapping[str, Mapping[str, Any]]],
    noise_runs: Sequence[Mapping[str, Mapping[str, float]]],
    model: str,
    mode: str,
    random_seed: int,
    oracle_states_per_task: int,
    oracle_pair_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    if len(noise_runs) < 2:
        raise ValueError("Gate A97 ensemble requires at least two flow-noise runs")
    bases = base_specs(base_protocol)
    poses = pose_index(catalog)
    common_keys = set(bases) & set(render_metadata)
    common_keys &= set.intersection(*(set(run) for run in noise_runs))
    if common_keys != set(bases):
        raise ValueError("base protocol, render bank, and noise runs do not identify identical states")

    pair_keys = sorted(bases)
    if mode == "oracle":
        if oracle_pair_keys is None:
            pair_keys = choose_oracle_states(bases, oracle_states_per_task)
        else:
            pair_keys = list(dict.fromkeys(oracle_pair_keys))
            missing = sorted(set(pair_keys) - set(bases))
            if missing:
                raise ValueError(f"oracle pair keys absent from base protocol: {missing}")
            if not pair_keys:
                raise ValueError("oracle pair-key selection is empty")
    specs = []
    selected_states = []
    for pair_key in pair_keys:
        selected, annotations = selection_for_state(
            pair_key=pair_key,
            metadata=render_metadata[pair_key],
            noise_runs=noise_runs,
            random_seed=random_seed,
        )
        if mode == "shortlist":
            condition_candidates = [(name, selected[name]) for name in SHORTLIST_CONDITIONS]
        else:
            condition_candidates = [
                (f"candidate__{candidate_id}", candidate_id)
                for candidate_id in sorted(annotations)
            ]
        for condition, candidate_id in condition_candidates:
            specs.append(
                make_spec(
                    bases[pair_key],
                    model=model,
                    condition=condition,
                    candidate_id=candidate_id,
                    annotation=annotations[candidate_id],
                    poses=poses,
                )
            )
        selected_states.append(
            {
                "pair_key": pair_key,
                "task_id": str(bases[pair_key]["task_id"]),
                "source_episode_id": str(bases[pair_key]["episode_id_source"]),
                "source_state_index": int(bases[pair_key]["source_state_index"]),
                "stage_fraction": float(bases[pair_key]["stage_fraction"]),
                "selected_candidates": selected,
            }
        )
    expected_conditions = len(SHORTLIST_CONDITIONS) if mode == "shortlist" else 97
    return {
        "schema": "dsol_accel_gate_a97_closed_loop_protocol_v1",
        "status": "PASS",
        "analysis_role": (
            "task_balanced_97_view_selector_closed_loop"
            if mode == "shortlist"
            else (
                "targeted_97_view_exhaustive_oracle"
                if oracle_pair_keys is not None
                else "task_balanced_97_view_exhaustive_oracle_pilot"
            )
        ),
        "mode": mode,
        "model": model,
        "catalog": str(base_protocol["catalog"]),
        "statistical_unit": "source HDF5 demonstration; frame states clustered within source",
        "selected_state_count": len(pair_keys),
        "condition_count": expected_conditions,
        "episode_count": len(specs),
        "noise_run_count": len(noise_runs),
        "selection_rule": {
            "primary": "mean accel_3 over shared flow-noise runs",
            "hybrid": "highest visibility among ensemble Accel top-10",
            "random_seed": random_seed,
            "candidate_bank": "canonical + broad64 training support + broad32 held-out",
        },
        "selected_states": selected_states,
        "specs": specs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--noise-run", type=Path, action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("shortlist", "oracle"), default="shortlist")
    parser.add_argument("--oracle-states-per-task", type=int, default=1)
    parser.add_argument(
        "--oracle-pair-key-file",
        type=Path,
        help="Optional newline-delimited pair keys for a targeted exhaustive oracle run.",
    )
    parser.add_argument("--random-seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.oracle_states_per_task < 1:
        raise ValueError("oracle-states-per-task must be positive")
    oracle_pair_keys = None
    if args.oracle_pair_key_file is not None:
        oracle_pair_keys = [
            line.strip()
            for line in args.oracle_pair_key_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    base = json.loads(args.base_protocol.read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    payload = build(
        base_protocol=base,
        catalog=catalog,
        render_metadata=load_render_metadata(args.render_root),
        noise_runs=[load_noise_run(path) for path in args.noise_run],
        model=args.model,
        mode=args.mode,
        random_seed=args.random_seed,
        oracle_states_per_task=args.oracle_states_per_task,
        oracle_pair_keys=oracle_pair_keys,
    )
    payload.update(
        {
            "base_protocol": str(args.base_protocol.resolve()),
            "base_protocol_sha256": sha256(args.base_protocol),
            "catalog": str(args.catalog.resolve()),
            "catalog_sha256": sha256(args.catalog),
            "render_root": str(args.render_root.resolve()),
            "noise_runs": [str(path.resolve()) for path in args.noise_run],
        }
    )
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "mode": payload["mode"],
                "model": payload["model"],
                "selected_state_count": payload["selected_state_count"],
                "episode_count": payload["episode_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
