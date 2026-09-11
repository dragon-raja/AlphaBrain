#!/usr/bin/env python3
"""Fail-closed audit for partial or complete held-out expectation runs."""

from __future__ import annotations

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[3]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))


import argparse
import collections
import json
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts.dsol_paper1.explicit_flow_noise import ExplicitFlowNoiseBank, sha256_file
except ModuleNotFoundError:
    from scripts.dsol_paper1.explicit_flow_noise import ExplicitFlowNoiseBank, sha256_file


PROTOCOL_SCHEMA = "dsol_view_value_expectation_heldout_protocol_v1"
RUN_SCHEMA = "dsol_libero_hdf5_closed_loop_run_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_shards(run_dir: Path, shard_count: int) -> tuple[list[tuple[int, dict[str, Any]]], list[int], int]:
    rows: list[tuple[int, dict[str, Any]]] = []
    counts: list[int] = []
    ignored_partial_tails = 0
    for shard in range(shard_count):
        path = run_dir / f"episodes-shard-{shard:02d}.jsonl"
        if not path.exists():
            counts.append(0)
            continue
        payload = path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            payload, _separator, _tail = payload.rpartition(b"\n")
            ignored_partial_tails += 1
        shard_rows = []
        for line_number, line in enumerate(payload.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in shard {shard} line {line_number}: {error}"
                ) from error
            require(isinstance(row, dict), f"non-object row in shard {shard} line {line_number}")
            shard_rows.append(row)
            rows.append((shard, row))
        counts.append(len(shard_rows))
    return rows, counts, ignored_partial_tails


def audit(
    *,
    protocol_path: Path,
    run_dir: Path,
    run_manifest_path: Path,
    noise_manifest_path: Path,
    require_complete: bool = False,
    verify_noise_file: bool = True,
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    run_dir = run_dir.resolve()
    run_manifest_path = run_manifest_path.resolve()
    noise_manifest_path = noise_manifest_path.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    require(protocol.get("schema") == PROTOCOL_SCHEMA, "unexpected held-out protocol schema")
    require(protocol.get("status") == "PASS", "held-out protocol must have PASS status")
    specs = protocol.get("specs")
    require(isinstance(specs, list), "held-out protocol specs must be a list")
    require(len(specs) == int(protocol["episode_count"]), "protocol episode count is inconsistent")
    expected = {str(spec["episode_id"]): (index, spec) for index, spec in enumerate(specs)}
    require(len(expected) == len(specs), "protocol contains duplicate episode IDs")

    require(run_manifest.get("schema") == RUN_SCHEMA, "unexpected run-manifest schema")
    require(_resolved(run_manifest["protocol"]) == protocol_path, "run manifest points to another protocol")
    protocol_sha256 = sha256_file(protocol_path)
    require(run_manifest.get("protocol_sha256") == protocol_sha256, "run manifest protocol SHA-256 mismatch")
    require(
        _resolved(run_manifest["noise_bank_manifest"]) == noise_manifest_path,
        "run manifest points to another noise-bank manifest",
    )
    noise_manifest_sha256 = sha256_file(noise_manifest_path)
    require(
        run_manifest.get("noise_bank_manifest_sha256") == noise_manifest_sha256,
        "run manifest noise-bank manifest SHA-256 mismatch",
    )
    require(run_manifest.get("require_explicit_noise") is True, "run manifest does not require explicit noise")
    shard_count = int(run_manifest["eval_worker_count"])
    require(shard_count > 0, "run manifest has no evaluator shards")

    bank = ExplicitFlowNoiseBank(noise_manifest_path, verify_file=verify_noise_file)
    require(bank.manifest.get("bank_id") == protocol.get("bank_id"), "protocol and noise-bank IDs differ")
    expected_state_keys = {str(spec["pair_key"]) for spec in specs}
    require(
        set(bank.manifest["state_keys"]) == expected_state_keys,
        "noise-bank state population differs from protocol",
    )

    loaded, shard_counts, ignored_partial_tails = _load_shards(run_dir, shard_count)
    seen: set[str] = set()
    method_counts: collections.Counter[str] = collections.Counter()
    pair_episode_ids: dict[str, set[str]] = collections.defaultdict(set)
    pair_physics: dict[str, set[str]] = collections.defaultdict(set)
    pair_environment_seeds: dict[str, set[int]] = collections.defaultdict(set)
    noise_signatures: dict[tuple[str, int, int], tuple[int, str]] = {}
    total_policy_calls = 0

    for shard, row in loaded:
        episode_id = str(row.get("episode_id", ""))
        require(episode_id in expected, f"result episode is absent from protocol: {episode_id}")
        require(episode_id not in seen, f"duplicate result episode ID: {episode_id}")
        seen.add(episode_id)
        protocol_index, spec = expected[episode_id]
        require(protocol_index % shard_count == shard, f"result is in the wrong shard: {episode_id}")
        for field, value in spec.items():
            require(row.get(field) == value, f"result differs from protocol field {field}: {episode_id}")
        require(row.get("status") == "complete", f"result is not complete: {episode_id}")
        require(row.get("explicit_flow_noise") is True, f"result did not use explicit noise: {episode_id}")
        require(row.get("noise_bank_id") == protocol["bank_id"], f"result used another noise bank: {episode_id}")
        require(
            row.get("noise_bank_manifest_sha256") == noise_manifest_sha256,
            f"result noise-bank manifest SHA-256 mismatch: {episode_id}",
        )

        pair_key = str(row["pair_key"])
        repeat_id = int(row["policy_repeat_id"])
        method_counts[str(row["selector_method"])] += 1
        pair_episode_ids[pair_key].add(episode_id)
        pair_environment_seeds[pair_key].add(int(row["environment_seed"]))
        metrics = row.get("initial_metrics", {})
        before = str(metrics.get("physics_state_sha256", ""))
        after = str(metrics.get("post_wait_physics_state_sha256_exact", ""))
        require(before and before == after, f"physics changed during camera/wait setup: {episode_id}")
        pair_physics[pair_key].add(before)

        calls = row.get("policy_calls")
        require(isinstance(calls, list), f"policy-call ledger is missing: {episode_id}")
        require(len(calls) == int(row["inference_calls"]), f"policy-call count mismatch: {episode_id}")
        require(
            [int(call["replan_index"]) for call in calls] == list(range(len(calls))),
            f"replan indices are not contiguous: {episode_id}",
        )
        total_policy_calls += len(calls)
        for call in calls:
            replan_index = int(call["replan_index"])
            require(int(call["policy_repeat_id"]) == repeat_id, f"policy-call repeat mismatch: {episode_id}")
            expected_noise = bank.get(pair_key, repeat_id, replan_index)
            require(call.get("noise_seed") == expected_noise["noise_seed"], f"noise seed mismatch: {episode_id}")
            require(call.get("noise_sha256") == expected_noise["noise_sha256"], f"noise SHA-256 mismatch: {episode_id}")
            key = (pair_key, repeat_id, replan_index)
            signature = (int(call["noise_seed"]), str(call["noise_sha256"]))
            require(
                noise_signatures.setdefault(key, signature) == signature,
                f"paired methods forked onto different noise: {key}",
            )

    for pair_key in pair_episode_ids:
        require(len(pair_physics[pair_key]) == 1, f"physics state differs within pair: {pair_key}")
        require(
            len(pair_environment_seeds[pair_key]) == 1,
            f"environment seed differs within pair: {pair_key}",
        )

    expected_by_pair: dict[str, set[str]] = collections.defaultdict(set)
    for spec in specs:
        expected_by_pair[str(spec["pair_key"])].add(str(spec["episode_id"]))
    complete_state_matrices = sum(
        pair_episode_ids.get(pair_key, set()) == episode_ids
        for pair_key, episode_ids in expected_by_pair.items()
    )
    if require_complete:
        require(ignored_partial_tails == 0, "a shard has an incomplete trailing write")
        require(seen == set(expected), f"result episode set is incomplete: {len(seen)}/{len(expected)}")
        expected_shard_counts = [0] * shard_count
        for index in range(len(specs)):
            expected_shard_counts[index % shard_count] += 1
        require(shard_counts == expected_shard_counts, "completed shard populations differ from protocol")

    return {
        "schema": "dsol_view_value_expectation_heldout_run_audit_v1",
        "status": "PASS_COMPLETE" if require_complete else "PASS_PARTIAL_SUBSET",
        "auditor": str(Path(__file__).resolve()),
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "run_dir": str(run_dir),
        "run_manifest": str(run_manifest_path),
        "noise_bank_manifest": str(noise_manifest_path),
        "noise_bank_manifest_sha256": noise_manifest_sha256,
        "noise_file_sha256": str(bank.manifest["noise_file_sha256"]),
        "protocol_episode_count": len(specs),
        "result_episode_count": len(seen),
        "unique_episode_count": len(seen),
        "shard_count": shard_count,
        "shard_counts": shard_counts,
        "shard_min": min(shard_counts),
        "shard_max": max(shard_counts),
        "ignored_partial_tails": ignored_partial_tails,
        "method_counts": dict(sorted(method_counts.items())),
        "pairs_touched": len(pair_episode_ids),
        "complete_state_matrices": int(complete_state_matrices),
        "total_policy_calls": total_policy_calls,
        "reconstructed_policy_calls": total_policy_calls,
        "paired_noise_keys": len(noise_signatures),
        "physics_pair_violations": 0,
        "environment_seed_pair_violations": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--noise-bank-manifest", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-noise-file-sha", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    result = audit(
        protocol_path=args.protocol,
        run_dir=args.run_dir,
        run_manifest_path=args.run_manifest or args.run_dir / "run_manifest.json",
        noise_manifest_path=args.noise_bank_manifest,
        require_complete=args.require_complete,
        verify_noise_file=not args.skip_noise_file_sha,
    )
    if args.output is not None:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
