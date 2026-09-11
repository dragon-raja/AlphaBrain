#!/usr/bin/env python3
"""Fail-closed release audit for the explicit-noise view-value runner."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_rows(patterns: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def per_call_noise(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], set[str]]:
    result = defaultdict(set)
    for row in rows:
        for call in row["policy_calls"]:
            result[(int(call["policy_repeat_id"]), int(call["replan_index"]))].add(
                str(call["noise_sha256"])
            )
    return result


def action_sequence(row: Mapping[str, Any]) -> list[str]:
    return [str(call["action_chunk_sha256"]) for call in row["policy_calls"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--static-audit", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--noise-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--paired-smoke", nargs="+", required=True)
    parser.add_argument("--identical-smoke", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    static = json.loads(args.static_audit.read_text())
    population = json.loads(args.population.read_text())
    noise_receipt = json.loads(args.noise_receipt.read_text())
    require(static["status"] == "PASS_STATIC_DESIGN_RUNNER_STILL_HOLD", "static protocol audit did not pass")
    require(population["status"] == "PASS", "population audit did not pass")
    require(population["source_disjoint"] and population["state_disjoint"] and population["legacy_source_disjoint"], "population split is not source/state/legacy disjoint")

    checkpoint_text = args.checkpoint_config.read_text()
    horizons = {int(value) for value in re.findall(r"action_horizon:\s*([0-9]+)", checkpoint_text)}
    require(horizons == {10}, f"selected checkpoint action horizon is not uniquely 10: {horizons}")
    require(protocol["randomness_contract"]["policy_flow_noise"]["expected_shape_for_selected_checkpoint"] == [10, 7], "protocol/checkpoint shape mismatch")

    require(noise_receipt["status"] == "PASS" and len(noise_receipt["banks"]) == 6, "six noise banks were not materialized")
    bank_ids = set()
    root_seeds = set()
    seed_streams = set()
    bank_a_manifest = None
    for bank in noise_receipt["banks"]:
        require(bank["shape"][-2:] == [10, 7], "noise bank has the wrong action shape")
        noise_file = Path(bank["noise_file"])
        require(sha256_file(noise_file) == bank["noise_file_sha256"], "noise bank file SHA-256 mismatch")
        bank_ids.add(bank["bank_id"])
        root_seeds.add(bank["root_seed"])
        seed_streams.add(bank["seed_stream_sha256"])
        if bank["bank_id"] == "A":
            bank_a_manifest = Path(bank["manifest_path"])
    require(bank_ids == set("ABCDEF"), "noise bank identities differ from A-F")
    require(len(root_seeds) == 6 and len(seed_streams) == 6, "noise banks do not use disjoint seed streams")
    require(bank_a_manifest is not None, "bank A manifest is absent")

    paired = load_rows(args.paired_smoke)
    identical = load_rows(args.identical_smoke)
    require(len(paired) == 2 and len(identical) == 2, "smoke matrices must each contain two episodes")
    for rows in (paired, identical):
        require(all(row.get("status") == "complete" and row.get("explicit_flow_noise") for row in rows), "smoke episode was incomplete or implicit-noise")
        require(len({row["environment_seed"] for row in rows}) == 1, "environment seeds differ within smoke pair")
        require(len({row["initial_metrics"]["physics_state_sha256"] for row in rows}) == 1, "physics state differs within smoke pair")
        require(all(len(values) == 1 for values in per_call_noise(rows).values()), "paired views did not share exact explicit noise")
    require(paired[0]["selected_candidate_id"] != paired[1]["selected_candidate_id"], "paired-view smoke did not use two views")
    require(identical[0]["selected_candidate_id"] == identical[1]["selected_candidate_id"], "identical-input smoke used different views")
    require(action_sequence(identical[0]) == action_sequence(identical[1]), "same input and same explicit noise produced different actions")
    require(identical[0]["success"] == identical[1]["success"] and identical[0]["completion_steps"] == identical[1]["completion_steps"], "identical-input rollout outcome was not reproducible")

    payload = {
        "schema": "dsol_view_value_expectation_execution_release_v1",
        "status": "PASS_EXECUTION_RELEASED",
        "formal_execution_authorized": True,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "static_audit_sha256": sha256_file(args.static_audit),
        "population_sha256": sha256_file(args.population),
        "noise_receipt_sha256": sha256_file(args.noise_receipt),
        "checkpoint_config_sha256": sha256_file(args.checkpoint_config),
        "checkpoint_action_horizon": 10,
        "explicit_noise_shape": [10, 7],
        "bank_a_manifest_sha256": sha256_file(bank_a_manifest),
        "checks": {
            "six_disjoint_materialized_noise_banks": True,
            "source_and_state_disjoint_population": True,
            "legacy_source_disjoint_population": True,
            "environment_and_policy_randomness_separate": True,
            "paired_views_share_exact_per_replan_noise": True,
            "identical_input_noise_action_hashes_exact": True,
            "formal_episodes_before_release": 0,
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "formal_execution_authorized": True}, sort_keys=True))


if __name__ == "__main__":
    main()
