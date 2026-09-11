#!/usr/bin/env python3
"""Outcome-blind integrity audit for a compact statewise-oracle rollout wave."""

from __future__ import annotations

from pathlib import Path as _RepoPath
import sys as _sys
_REPOSITORY_ROOT = _RepoPath(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPOSITORY_ROOT))


import argparse
import glob
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.dsol_paper1.runtime.evaluate_dsol_libero_hdf5_views import protocol_spec_at, protocol_spec_count
from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank, sha256_file


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def iter_rows(patterns: Sequence[str]):
    paths = []
    for pattern in patterns:
        paths.extend(Path(value) for value in sorted(glob.glob(pattern)))
    if not paths:
        raise FileNotFoundError("no episode ledgers matched")
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    yield path, line_number, json.loads(line)


def validate_row_against_spec(row: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    for field in (
        "episode_id",
        "pair_key",
        "source_group",
        "task_id",
        "selected_candidate_id",
        "policy_repeat_id",
        "noise_bank_id",
        "environment_seed",
        "construction_spec_sha256",
        "condition",
    ):
        if row.get(field) != spec.get(field):
            raise ValueError(f"episode differs from protocol at {field}: {row.get('episode_id')}")


def audit(
    *,
    protocol_path: Path,
    noise_manifest: Path,
    run_manifest_path: Path,
    ledger_patterns: Sequence[str],
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_count = protocol_spec_count(protocol)
    expected = {
        spec["episode_id"]: spec
        for spec in (protocol_spec_at(protocol, index) for index in range(expected_count))
    }
    if len(expected) != expected_count:
        raise ValueError("protocol has duplicate expanded episode IDs")
    bank = ExplicitFlowNoiseBank(noise_manifest, verify_file=True)
    if protocol.get("noise_bank_id") != bank.manifest.get("bank_id"):
        raise ValueError("protocol and noise-bank IDs differ")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("run manifest protocol checksum mismatch")
    if run_manifest.get("noise_bank_manifest_sha256") != sha256_file(noise_manifest):
        raise ValueError("run manifest noise-bank checksum mismatch")
    if not run_manifest.get("require_explicit_noise"):
        raise ValueError("run manifest did not require explicit noise")
    seen = set()
    state_physics = defaultdict(set)
    state_environment_seed = defaultdict(set)
    state_repeat_candidates = defaultdict(set)
    noise_identity = defaultdict(set)
    call_count = 0
    for path, line_number, row in iter_rows(ledger_patterns):
        episode_id = str(row.get("episode_id"))
        if episode_id in seen:
            raise ValueError(f"duplicate episode ID: {episode_id}")
        seen.add(episode_id)
        spec = expected.get(episode_id)
        if spec is None:
            raise ValueError(f"row outside protocol at {path}:{line_number}")
        validate_row_against_spec(row, spec)
        if row.get("status") != "complete" or not row.get("explicit_flow_noise"):
            raise ValueError(f"episode is not complete explicit-noise: {episode_id}")
        calls = list(row.get("policy_calls", []))
        if len(calls) != int(row.get("inference_calls", -1)):
            raise ValueError(f"policy-call ledger length mismatch: {episode_id}")
        if [int(call["replan_index"]) for call in calls] != list(range(len(calls))):
            raise ValueError(f"noncontiguous replan ledger: {episode_id}")
        pair_key = str(row["pair_key"])
        repeat_id = int(row["policy_repeat_id"])
        state_repeat_candidates[pair_key, repeat_id].add(str(row["selected_candidate_id"]))
        initial = row["initial_metrics"]
        if initial["physics_state_sha256"] != initial["post_wait_physics_state_sha256_exact"]:
            raise ValueError(f"physics changed during camera installation: {episode_id}")
        state_physics[pair_key].add(str(initial["physics_state_sha256"]))
        state_environment_seed[pair_key].add(int(row["environment_seed"]))
        for call in calls:
            replan_index = int(call["replan_index"])
            if int(call["policy_repeat_id"]) != repeat_id:
                raise ValueError(f"policy-call repeat differs from episode: {episode_id}")
            expected_noise = bank.get(pair_key, repeat_id, replan_index)
            actual = (int(call["noise_seed"]), str(call["noise_sha256"]))
            frozen = (int(expected_noise["noise_seed"]), str(expected_noise["noise_sha256"]))
            if actual != frozen:
                raise ValueError(f"noise call differs from frozen bank: {episode_id}/{replan_index}")
            action_digest = str(call.get("action_chunk_sha256", ""))
            if len(action_digest) != 64 or any(value not in "0123456789abcdef" for value in action_digest):
                raise ValueError(f"invalid action-chunk digest: {episode_id}/{replan_index}")
            noise_identity[pair_key, repeat_id, replan_index].add(actual)
            call_count += 1
    if seen != set(expected):
        raise ValueError(f"episode set incomplete: {len(seen)}/{len(expected)}")
    if any(len(values) != 1 for values in state_physics.values()):
        raise ValueError("physics-state hashes differ across views/repeats")
    if any(len(values) != 1 for values in state_environment_seed.values()):
        raise ValueError("environment seeds differ across views/repeats")
    if any(len(values) != 1 for values in noise_identity.values()):
        raise ValueError("paired explicit noise differs across views")
    expected_candidates = {
        str(block["state"]["pair_key"]): {
            str(candidate["selected_candidate_id"]) for candidate in block["candidates"]
        }
        for block in protocol["state_blocks"]
    }
    expected_repeats = {int(value) for value in protocol["policy_repeat_ids"]}
    for pair_key, candidates in expected_candidates.items():
        for repeat_id in expected_repeats:
            if state_repeat_candidates[pair_key, repeat_id] != candidates:
                raise ValueError(f"candidate matrix incomplete: {pair_key}/{repeat_id}")
    return {
        "schema": "dsol_statewise_view_oracle_v2_run_audit",
        "status": "PASS_COMPLETE",
        "outcome_values_aggregated": False,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "noise_bank_manifest": str(noise_manifest.resolve()),
        "noise_bank_manifest_sha256": sha256_file(noise_manifest),
        "noise_bank_file_sha256": bank.manifest["noise_file_sha256"],
        "run_manifest": str(run_manifest_path.resolve()),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "episode_count": len(seen),
        "state_count": len(expected_candidates),
        "repeat_count": len(expected_repeats),
        "candidate_counts": sorted({len(values) for values in expected_candidates.values()}),
        "policy_call_count": call_count,
        "physics_hash_constant_within_state": True,
        "environment_seed_constant_within_state": True,
        "every_policy_call_matches_frozen_noise_bank": True,
        "paired_noise_identity_at_common_replan_indices": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--noise-bank-manifest", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--episode-ledgers", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        protocol_path=args.protocol.resolve(),
        noise_manifest=args.noise_bank_manifest.resolve(),
        run_manifest_path=args.run_manifest.resolve(),
        ledger_patterns=args.episode_ledgers,
    )
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "episodes": result["episode_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
