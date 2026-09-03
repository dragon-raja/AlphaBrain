#!/usr/bin/env python3
"""Prove exact rollout equivalence and compare measured evaluation throughput."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .audit_statewise_view_oracle_v2_run import atomic_json
    from .evaluate_dsol_libero_hdf5_views import protocol_spec_at, protocol_spec_count
    from .explicit_flow_noise import sha256_file
except ImportError:  # Direct script execution.
    from audit_statewise_view_oracle_v2_run import atomic_json
    from evaluate_dsol_libero_hdf5_views import protocol_spec_at, protocol_spec_count
    from explicit_flow_noise import sha256_file


def load_rows(patterns: Sequence[str], expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    paths = sorted({Path(value) for pattern in patterns for value in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError("no episode ledgers matched")
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            episode_id = str(row.get("episode_id"))
            if episode_id not in expected_ids:
                continue
            if episode_id in rows:
                raise ValueError(f"duplicate benchmark episode: {episode_id}")
            rows[episode_id] = row
    missing = expected_ids.difference(rows)
    if missing:
        raise ValueError(f"benchmark episode set incomplete: {len(rows)}/{len(expected_ids)}")
    return rows


def rollout_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": row.get("status"),
        "success": row.get("success"),
        "completion_steps": row.get("completion_steps"),
        "inference_calls": row.get("inference_calls"),
        "normalized_final_progress": row.get("normalized_final_progress"),
        "initial_physics_state_sha256": row.get("initial_metrics", {}).get(
            "physics_state_sha256"
        ),
        "post_wait_physics_state_sha256": row.get("initial_metrics", {}).get(
            "post_wait_physics_state_sha256_exact"
        ),
        "policy_calls": [
            {
                "replan_index": call.get("replan_index"),
                "noise_seed": call.get("noise_seed"),
                "noise_sha256": call.get("noise_sha256"),
                "action_chunk_sha256": call.get("action_chunk_sha256"),
            }
            for call in row.get("policy_calls", [])
        ],
    }


def load_latest_receipt(run_dir: Path) -> dict[str, Any]:
    receipts = sorted((run_dir / "run-manifests").glob("receipt-*.json"))
    if not receipts:
        raise FileNotFoundError(f"no completed run receipt under {run_dir}")
    receipt = json.loads(receipts[-1].read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    return {
        **receipt,
        "run_dir": str(run_dir.resolve()),
        "policy_server_count": manifest["policy_server_count"],
        "eval_worker_count": manifest["eval_worker_count"],
        "policy_cpu_threads": manifest["policy_cpu_threads"],
        "sim_cpu_threads": manifest["sim_cpu_threads"],
    }


def compare(
    *,
    protocol_path: Path,
    reference_patterns: Sequence[str],
    baseline_dir: Path,
    candidate_dir: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    count = protocol_spec_count(protocol)
    expected_ids = {
        protocol_spec_at(protocol, index)["episode_id"] for index in range(count)
    }
    datasets = {
        "pre_optimization_formal": load_rows(reference_patterns, expected_ids),
        "baseline": load_rows(
            [str(baseline_dir / "episodes-shard-*.jsonl")], expected_ids
        ),
        "candidate": load_rows(
            [str(candidate_dir / "episodes-shard-*.jsonl")], expected_ids
        ),
    }
    reference = datasets["pre_optimization_formal"]
    mismatches = []
    for label, rows in datasets.items():
        if label == "pre_optimization_formal":
            continue
        for episode_id in sorted(expected_ids):
            if rollout_signature(rows[episode_id]) != rollout_signature(reference[episode_id]):
                mismatches.append({"dataset": label, "episode_id": episode_id})
    baseline_receipt = load_latest_receipt(baseline_dir)
    candidate_receipt = load_latest_receipt(candidate_dir)
    baseline_rate = float(baseline_receipt["episodes_per_hour"])
    candidate_rate = float(candidate_receipt["episodes_per_hour"])
    return {
        "schema": "dsol_eval_throughput_benchmark_comparison_v1",
        "status": "PASS_EXACT_EQUIVALENCE" if not mismatches else "FAIL_MISMATCH",
        "benchmark_only": True,
        "excluded_from_scientific_estimands": True,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "episode_count": count,
        "compared_dataset_count": len(datasets),
        "exact_rollout_signature_match": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "baseline": baseline_receipt,
        "candidate": candidate_receipt,
        "throughput_speedup": candidate_rate / baseline_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reference-ledgers", nargs="+", required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        protocol_path=args.protocol.resolve(),
        reference_patterns=args.reference_ledgers,
        baseline_dir=args.baseline_dir.resolve(),
        candidate_dir=args.candidate_dir.resolve(),
    )
    atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "episodes": result["episode_count"],
                "speedup": result["throughput_speedup"],
            },
            sort_keys=True,
        )
    )
    if result["status"] != "PASS_EXACT_EQUIVALENCE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
