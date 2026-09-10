#!/usr/bin/env python3
"""Freeze an outcome-blind, matched 8-state development view landscape.

This is a new diagnostic, not a continuation of the v2 confirmation claim.
The old state keys, camera blocks, and materialized O noise are reused exactly.
No rollout outcomes, Accel scores, or test-state protocols are read.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .evaluate_dsol_libero_hdf5_views import protocol_spec_at, protocol_spec_count
    from .explicit_flow_noise import ExplicitFlowNoiseBank, sha256_file
except ImportError:
    from evaluate_dsol_libero_hdf5_views import protocol_spec_at, protocol_spec_count
    from explicit_flow_noise import ExplicitFlowNoiseBank, sha256_file


SCHEMA = "dsol_compact_view_matrix_protocol_v1"
DEFAULT_SALT = "dsol-view-landscape-v1-20260908::development::one-source-per-task::v1"
PREFIX = "view-landscape-v1-20260908::O::development"
DIAGNOSTIC_ROLE = "matched_view_landscape_v1_development_diagnostic"
SMOKE_CANDIDATES = ("canonical", "broad_train_000", "broad_heldout_000")
EXPECTED_CANDIDATES = (
    "canonical",
    *(f"broad_train_{index:03d}" for index in range(64)),
    *(f"broad_heldout_{index:03d}" for index in range(32)),
)
DEFAULT_V2_ROOT = Path(
    "/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908"
)


def encoded_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(encoded_json(value)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path: Path, value: Any) -> None:
    """Exclusive creation: even a byte-identical existing file is not overwritten."""
    with path.open("xb") as handle:
        handle.write(encoded_json(value))


def source_identity(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"frozen source checksum mismatch: {path}")
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def selection_digest(salt: str, state: Mapping[str, Any]) -> str:
    # No row position, image, task performance, view metric, or outcome is used.
    return hashlib.sha256(
        f"{salt}::{state['task_id']}::{state['source_group']}".encode("utf-8")
    ).hexdigest()


def select_development_states(
    population: Mapping[str, Any], *, salt: str = DEFAULT_SALT
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if population.get("status") != "PASS":
        raise ValueError("source v2 population must have status PASS")
    states = population["population"]["development"]["states"]
    if len(states) != 48 or len({state["pair_key"] for state in states}) != 48:
        raise ValueError("expected 48 distinct development states")
    if len({state["source_group"] for state in states}) != 48:
        raise ValueError("development source groups must be distinct")
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for state in states:
        if state.get("split") != "development" or state.get("v2_role") != "development":
            raise ValueError("non-development state in development population")
        by_task[str(state["task_id"])].append(state)
    if len(by_task) != 8 or any(len(rows) != 6 for rows in by_task.values()):
        raise ValueError("expected eight tasks with six development sources each")
    selected = []
    rankings = []
    for task_id, rows in sorted(by_task.items()):
        ranked = sorted(rows, key=lambda state: (selection_digest(salt, state), state["pair_key"]))
        selected.append(deepcopy(ranked[0]))
        rankings.append(
            {
                "task_id": task_id,
                "ranked_sources": [
                    {
                        "rank": index + 1,
                        "pair_key": state["pair_key"],
                        "source_group": state["source_group"],
                        "selection_sha256": selection_digest(salt, state),
                        "selected": index == 0,
                    }
                    for index, state in enumerate(ranked)
                ],
            }
        )
    return selected, rankings


def matching_blocks(
    protocols: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if len(protocols) != 8:
        raise ValueError("expected all eight frozen development O waves")
    reference = None
    for index, protocol in enumerate(protocols):
        if protocol.get("schema") != SCHEMA or protocol.get("status") != "PASS_FROZEN":
            raise ValueError("source compact protocol is not PASS_FROZEN")
        if protocol.get("role") != "development" or protocol.get("noise_bank_id") != "O":
            raise ValueError("only development O protocols are allowed")
        if protocol.get("policy_repeat_ids") != list(range(4 * index, 4 * index + 4)):
            raise ValueError("source O repeat waves must partition IDs 0..31")
        blocks = protocol["state_blocks"]
        if len(blocks) != 48 or len({block["state"]["pair_key"] for block in blocks}) != 48:
            raise ValueError("source wave must contain 48 distinct state blocks")
        by_key = {block["state"]["pair_key"]: block for block in blocks}
        current = []
        for state in selected:
            block = by_key[state["pair_key"]]
            if block["state"] != state:
                raise ValueError("source protocol and population state fields differ")
            if tuple(row["selected_candidate_id"] for row in block["candidates"]) != EXPECTED_CANDIDATES:
                raise ValueError("source state does not contain the ordered full 97 candidates")
            current.append(deepcopy(block))
        if reference is not None and reference != current:
            raise ValueError("selected frozen state/candidate blocks differ across O waves")
        reference = current
    assert reference is not None
    return reference


def make_protocol(
    template: Mapping[str, Any],
    blocks: Sequence[Mapping[str, Any]],
    *,
    repeats: Sequence[int],
    label: str,
    selection_path: Path,
    selection_sha256: str,
    candidates: Sequence[str] = EXPECTED_CANDIDATES,
) -> dict[str, Any]:
    candidate_set = set(candidates)
    if not candidate_set or len(candidate_set) != len(candidates):
        raise ValueError("candidate IDs must be nonempty and unique")
    if not set(repeats) <= set(range(32)) or not repeats or len(set(repeats)) != len(repeats):
        raise ValueError("repeat IDs must be a unique nonempty subset of O IDs 0..31")
    new_blocks = deepcopy(list(blocks))
    for block in new_blocks:
        block["candidates"] = [
            row for row in block["candidates"] if row["selected_candidate_id"] in candidate_set
        ]
        if len(block["candidates"]) != len(candidates):
            raise ValueError("requested candidate is absent from a state block")
    return {
        "schema": SCHEMA,
        "status": "PASS_FROZEN_DEVELOPMENT_DIAGNOSTIC",
        "phase": "matched_view_landscape_development",
        "role": "development",
        "wave_id": label,
        "diagnostic_role": DIAGNOSTIC_ROLE,
        # Same identity/expanded fields across smoke, full waves and residuals.
        "episode_identity_prefix": PREFIX,
        "noise_bank_id": "O",
        "policy_repeat_ids": list(repeats),
        "sensor_control": "both",
        "state_count": len(new_blocks),
        "candidate_count_per_state": len(candidates),
        "episode_count": len(new_blocks) * len(candidates) * len(repeats),
        "matrix_order": "state_then_candidate_then_repeat",
        "catalog": template["catalog"],
        "catalog_sha256": template["catalog_sha256"],
        "population": template["population"],
        "population_sha256": template["population_sha256"],
        "selection": str(selection_path.resolve()),
        "selection_sha256": selection_sha256,
        "claim_scope": "new outcome-blind development matched diagnostic; not v2 test confirmation",
        "camera_motion_within_rollout": False,
        "test_states_included": False,
        "rollout_outcomes_used_for_source_selection": False,
        "state_blocks": new_blocks,
    }


def expanded_specs(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for index in range(protocol_spec_count(protocol)):
        spec = protocol_spec_at(protocol, index)
        if spec["episode_id"] in result:
            raise ValueError("duplicate expanded episode ID")
        result[spec["episode_id"]] = spec
    return result


def verify_reuse_partition(
    full: Mapping[str, Any], parts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = expanded_specs(full)
    union = {}
    for part in parts:
        for episode_id, spec in expanded_specs(part).items():
            if episode_id in union:
                raise ValueError("reuse parts overlap")
            if expected.get(episode_id) != spec:
                raise ValueError("reuse part has a different expanded execution spec")
            union[episode_id] = spec
    if union != expected:
        raise ValueError("reuse parts do not cover the full wave")
    return {
        "status": "PASS_EXACT_DISJOINT_SPEC_PARTITION",
        "full_episode_count": len(expected),
        "part_episode_counts": [protocol_spec_count(part) for part in parts],
        "expanded_episode_contract_sha256": object_sha256(expected),
        "all_expanded_fields_identical": True,
        "episode_id_overlap_between_parts": False,
        "runtime_reuse_requires": [
            "each part passes the outcome-blind integrity audit independently",
            "checkpoint weights and framework/normalization configuration hashes match",
            "policy backend, code, noise-bank manifest/file and catalog hashes match",
            "replan_steps=5, wait_steps=0, flow denoising steps=10 and rollout budget match",
            "preserve every original part run manifest and ledger provenance",
            "no rerunning or selecting rows according to success outcomes",
        ],
    }


def build(v2_root: Path, output_root: Path, *, salt: str = DEFAULT_SALT) -> dict[str, Any]:
    protocol_dir = output_root / "protocols"
    if protocol_dir.exists():
        raise FileExistsError(f"refusing to overwrite protocol directory: {protocol_dir}")
    population_path = v2_root / "population" / "population-v2.json"
    noise_path = v2_root / "noise-banks" / "bank_O.manifest.json"
    population = read_json(population_path)
    selected, rankings = select_development_states(population, salt=salt)
    input_paths = [
        v2_root / "protocols" / f"dense-O-development-wave-{index:02d}.json"
        for index in range(8)
    ]
    templates = [read_json(path) for path in input_paths]
    blocks = matching_blocks(templates, selected)
    population_identity = source_identity(population_path)
    for template in templates:
        if template["population_sha256"] != population_identity["sha256"]:
            raise ValueError("source wave population checksum differs")
        if Path(template["population"]).resolve() != population_path.resolve():
            raise ValueError("source wave references a different population path")
        for key in ("catalog", "catalog_sha256", "sensor_control"):
            if template[key] != templates[0][key]:
                raise ValueError(f"source waves differ at {key}")
    if templates[0]["sensor_control"] != "both":
        raise ValueError("source sensor control must be both")
    bank = ExplicitFlowNoiseBank(noise_path, verify_file=True)
    for key, expected in {
        "bank_id": "O", "repeat_count": 32, "max_replans": 104,
        "action_horizon": 10, "action_dim": 7,
    }.items():
        if bank.manifest.get(key) != expected:
            raise ValueError(f"source noise bank differs at {key}")
    for state in selected:
        bank.get(state["pair_key"], 0, 0)
        bank.get(state["pair_key"], 31, 103)
    selection_path = protocol_dir / "selection.json"
    selection = {
        "schema": "dsol_matched_view_landscape_selection_v1",
        "status": "FROZEN_OUTCOME_BLIND_DEVELOPMENT_SUBSET",
        "salt": salt,
        "ranking_rule": "ascending sha256(salt::task_id::source_group), pair_key tie-break",
        "source_population": population_identity,
        "eligible_states": 48,
        "selected_states": 8,
        "states_per_task": 1,
        "outcomes_or_view_metrics_used_in_selection": False,
        "test_states_used": False,
        "task_rankings": rankings,
        "states": selected,
    }
    selection_hash = object_sha256(selection)
    shared = dict(selection_path=selection_path, selection_sha256=selection_hash)
    outputs = {
        f"dense-O-development-wave-{index:02d}.json": make_protocol(
            templates[0], blocks, repeats=list(range(index * 4, index * 4 + 4)),
            label=f"wave-{index:02d}", **shared,
        )
        for index in range(8)
    }
    outputs["smoke-O-development.json"] = make_protocol(
        templates[0], blocks, repeats=[0, 1], label="smoke-reusable-in-wave-00",
        candidates=SMOKE_CANDIDATES, **shared,
    )
    outputs["dense-O-development-wave-00-remainder-a.json"] = make_protocol(
        templates[0], blocks, repeats=[0, 1], label="wave-00-remainder-a",
        candidates=[value for value in EXPECTED_CANDIDATES if value not in SMOKE_CANDIDATES],
        **shared,
    )
    outputs["dense-O-development-wave-00-remainder-b.json"] = make_protocol(
        templates[0], blocks, repeats=[2, 3], label="wave-00-remainder-b", **shared,
    )
    reuse_names = [
        "smoke-O-development.json",
        "dense-O-development-wave-00-remainder-a.json",
        "dense-O-development-wave-00-remainder-b.json",
    ]
    reuse = verify_reuse_partition(
        outputs["dense-O-development-wave-00.json"], [outputs[name] for name in reuse_names]
    )
    for name, protocol in outputs.items():
        if protocol_spec_count(protocol) != protocol["episode_count"]:
            raise ValueError(f"expanded episode count mismatch: {name}")
    static_files = {}
    for state in selected:
        pairs = [("construction_spec", "construction_spec_sha256", state)]
        assets = state["static_assets"]
        pairs += [(key, f"{key}_sha256", assets) for key in (
            "visibility_scan", "render_receipt", "policy_inputs"
        )]
        for path_key, digest_key, owner in pairs:
            path = Path(owner[path_key])
            static_files[str(path.resolve())] = source_identity(path, owner[digest_key])
    scripts_dir = Path(__file__).resolve().parent
    provenance = {
        "schema": "dsol_matched_view_landscape_source_provenance_v1",
        "population": population_identity,
        "source_development_O_protocols": [source_identity(path) for path in input_paths],
        "catalog": source_identity(Path(templates[0]["catalog"]), templates[0]["catalog_sha256"]),
        "noise_manifest": source_identity(noise_path),
        "noise_file": {
            "path": str(bank.noise_path), "sha256": bank.manifest["noise_file_sha256"],
            "bytes": bank.noise_path.stat().st_size, "verified": True,
        },
        "selected_static_files": list(static_files.values()),
        "implementation": [source_identity(scripts_dir / name) for name in (
            Path(__file__).name, "evaluate_dsol_libero_hdf5_views.py", "explicit_flow_noise.py",
            "run_dsol_libero_hdf5_closed_loop_eval.sh", "audit_statewise_view_oracle_v2_run.py",
        )],
        "hdf5_policy": "paths and source state identities inherited; full HDF5 hashes not recomputed",
        "checkpoint_policy": "checkpoint not selected here; runtime must independently freeze weights and configuration",
        "outcome_ledgers_read": [],
        "required_execution": {
            "noise_bank_manifest": str(noise_path.resolve()),
            "require_explicit_noise": True, "replan_steps": 5, "wait_steps": 0,
            "flow_denoising_steps": 10, "max_replans": 104,
            "camera_fixed_within_rollout": True, "wrist_camera_deployed": True,
        },
    }
    manifest = {
        "schema": "dsol_matched_view_landscape_protocol_manifest_v1",
        "status": "PASS_READY_FOR_CHECKPOINT_AUDIT_AND_SMOKE",
        "claim_scope": "development-only matched diagnostic; no test, no confirmatory oracle claim",
        "selection": {"path": str(selection_path.resolve()), "sha256": selection_hash},
        "source_provenance": {
            "path": str((protocol_dir / "source_provenance.json").resolve()),
            "sha256": object_sha256(provenance),
        },
        "state_count": 8, "task_count": 8, "candidate_count": 97,
        "repeat_count": 32, "dense_unique_episode_budget": 24832,
        "smoke_episode_count": 48, "smoke_additional_episode_budget_if_reused": 0,
        "dense_wave_count": 8, "dense_episodes_per_wave": 3104,
        "protocols": [
            {"name": name, "path": str((protocol_dir / name).resolve()),
             "sha256": object_sha256(protocol), "episode_count": protocol["episode_count"]}
            for name, protocol in outputs.items()
        ],
        "wave_00_reuse": {**reuse, "full_protocol": "dense-O-development-wave-00.json", "parts": reuse_names},
        "execution_routes": {
            "no_reuse": "run the eight full dense waves; any separate smoke is extra work",
            "reuse_smoke": "run smoke, wave-00 remainders a/b, and full waves 01..07; never also run full wave-00",
        },
    }
    # Complete validation before claiming a new output directory. Existing output
    # trees (e.g. checkpoint-audit) are left untouched; protocols is create-only.
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_dir.mkdir(exist_ok=False)
    write_new_json(selection_path, selection)
    write_new_json(protocol_dir / "source_provenance.json", provenance)
    for name, protocol in outputs.items():
        write_new_json(protocol_dir / name, protocol)
    write_new_json(protocol_dir / "protocol_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--selection-salt", default=DEFAULT_SALT)
    args = parser.parse_args()
    result = build(args.v2_root.resolve(), args.output_root.resolve(), salt=args.selection_salt)
    print(json.dumps({
        "status": result["status"], "unique_episodes": result["dense_unique_episode_budget"],
        "states": result["state_count"], "output": str(args.output_root / "protocols"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
