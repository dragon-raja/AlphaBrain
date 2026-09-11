#!/usr/bin/env python3
"""Freeze the population, static assets, noise banks, and dense v2 protocols."""

from __future__ import annotations

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[3]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))


import argparse
import glob
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank, materialize_bank, sha256_file


COMPACT_SCHEMA = "dsol_compact_view_matrix_protocol_v1"


def stable_digest(label: str, root_seed: int) -> str:
    return hashlib.sha256(f"{root_seed}::{label}".encode()).hexdigest()


def serialized_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def freeze_json(path: Path, payload: object) -> None:
    encoded = serialized_json(payload)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to change frozen JSON: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    temporary.replace(path)


def load_scan_assets(root: Path) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for ledger in sorted((root / "visibility-scan").glob("shard-*.jsonl")):
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "PASS":
                raise ValueError(f"visibility scan failed: {row.get('scan_id')}")
            pair_key = str(row["scan_id"])
            scan_path = Path(row["output_dir"]) / "scan.json"
            scan = json.loads(scan_path.read_text(encoding="utf-8"))
            if scan.get("status") != "PASS":
                raise ValueError(f"visibility scan artifact failed: {pair_key}")
            if pair_key in assets:
                raise ValueError(f"duplicate visibility scan: {pair_key}")
            assets[pair_key] = {
                "scan": scan,
                "scan_path": str(scan_path.resolve()),
                "scan_sha256": sha256_file(scan_path),
            }
    return assets


def load_render_assets(root: Path) -> dict[str, dict[str, Any]]:
    assets = {}
    for receipt_path in sorted((root / "accel-ensemble" / "states").glob("*/render.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "PASS" or int(receipt.get("candidate_count", 0)) != 97:
            raise ValueError(f"invalid 97-view render receipt: {receipt_path}")
        artifact = Path(receipt["artifact"])
        if not artifact.is_file() or sha256_file(artifact) != receipt["artifact_sha256"]:
            raise ValueError(f"render artifact checksum mismatch: {artifact}")
        pair_key = str(receipt["pair_key"])
        if pair_key in assets:
            raise ValueError(f"duplicate render receipt: {pair_key}")
        assets[pair_key] = {
            "render_receipt": str(receipt_path.resolve()),
            "render_receipt_sha256": sha256_file(receipt_path),
            "policy_inputs": str(artifact.resolve()),
            "policy_inputs_sha256": receipt["artifact_sha256"],
            "physics_state_sha256": receipt["physics_state_sha256"],
        }
    return assets


def candidate_sort_key(candidate_id: str) -> tuple[int, str]:
    if candidate_id == "canonical":
        return (0, candidate_id)
    if candidate_id.startswith("broad_train_"):
        return (1, candidate_id)
    if candidate_id.startswith("broad_heldout_"):
        return (2, candidate_id)
    return (3, candidate_id)


def operational_candidates(scan: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = [
        row
        for row in scan["records"]
        if row.get("status", "PASS") == "PASS"
        and (
            row.get("pose_id") == "canonical"
            or str(row.get("pose_id", "")).startswith("broad_train_")
            or str(row.get("pose_id", "")).startswith("broad_heldout_")
        )
    ]
    records.sort(key=lambda row: candidate_sort_key(str(row["pose_id"])))
    if len(records) != 97 or len({row["pose_id"] for row in records}) != 97:
        raise ValueError(f"expected 97 unique operational candidates, found {len(records)}")
    return [
        {
            "selected_candidate_id": str(row["pose_id"]),
            "pose": row.get("pose"),
            "candidate_features": {
                "catalog_group": row["group"],
                "visibility_score": float(row["visibility_score"]),
                "delta_visibility": float(row["delta_visibility"]),
                "per_camera_scores": row.get("per_camera_scores"),
                "camera_displacement_from_canonical": row.get(
                    "camera_displacement_from_canonical"
                ),
            },
        }
        for row in records
    ]


def _v2_state(
    source: Mapping[str, Any],
    *,
    role: str,
    cv_fold: int | None,
    scan_asset: Mapping[str, Any],
    render_asset: Mapping[str, Any],
) -> dict[str, Any]:
    state = dict(source)
    source_pair_key = str(state["pair_key"])
    state["asset_source_pair_key"] = source_pair_key
    state["legacy_v1_split"] = state["split"]
    state["split"] = role
    state["v2_role"] = role
    state["cv_fold"] = cv_fold
    state["pair_key"] = (
        f"oracle-v2::{role}::{state['task_id']}::{state['demo_name']}::"
        f"frame-{int(state['source_state_index']):05d}"
    )
    state["static_assets"] = {
        "visibility_scan": scan_asset["scan_path"],
        "visibility_scan_sha256": scan_asset["scan_sha256"],
        **render_asset,
    }
    return state


def freeze_population(
    source_population: Path,
    *,
    scan_assets: Mapping[str, Mapping[str, Any]],
    render_assets: Mapping[str, Mapping[str, Any]],
    split_root_seed: int,
) -> dict[str, Any]:
    source_payload = json.loads(source_population.read_text(encoding="utf-8"))
    if source_payload.get("status") != "PASS":
        raise ValueError("source population did not PASS")
    original = [
        dict(row)
        for split in ("calibration", "heldout_test")
        for row in source_payload["population"][split]["states"]
    ]
    original_keys = {str(row["pair_key"]) for row in original}
    if len(original) != 64 or len(original_keys) != 64:
        raise ValueError("v2 requires exactly 64 unique source states")
    if set(scan_assets) != original_keys or set(render_assets) != original_keys:
        raise ValueError("static 97-view asset identities differ from the source population")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in original:
        by_task[str(row["task_id"])].append(row)
    if len(by_task) != 8 or any(len(rows) != 8 for rows in by_task.values()):
        raise ValueError("v2 requires eight states for each of eight tasks")
    development = []
    test = []
    split_assignments = {}
    for task_id in sorted(by_task):
        rows = by_task[task_id]
        heldout = [row for row in rows if row["split"] == "heldout_test"]
        if len(heldout) != 6:
            raise ValueError(f"{task_id} does not have six prior heldout states")
        heldout.sort(
            key=lambda row: stable_digest(
                f"test::{task_id}::{row['source_group']}", split_root_seed
            )
        )
        test_source_keys = {str(row["pair_key"]) for row in heldout[:2]}
        dev_rows = [row for row in rows if str(row["pair_key"]) not in test_source_keys]
        dev_rows.sort(
            key=lambda row: stable_digest(
                f"cv::{task_id}::{row['source_group']}", split_root_seed
            )
        )
        for fold, row in enumerate(dev_rows):
            source_key = str(row["pair_key"])
            value = _v2_state(
                row,
                role="development",
                cv_fold=fold,
                scan_asset=scan_assets[source_key],
                render_asset=render_assets[source_key],
            )
            development.append(value)
            split_assignments[source_key] = {
                "v2_pair_key": value["pair_key"],
                "role": "development",
                "cv_fold": fold,
            }
        for row in heldout[:2]:
            source_key = str(row["pair_key"])
            value = _v2_state(
                row,
                role="test",
                cv_fold=None,
                scan_asset=scan_assets[source_key],
                render_asset=render_assets[source_key],
            )
            test.append(value)
            split_assignments[source_key] = {
                "v2_pair_key": value["pair_key"],
                "role": "test",
                "cv_fold": None,
            }
    development.sort(key=lambda row: (row["task_id"], int(row["cv_fold"])))
    test.sort(key=lambda row: (row["task_id"], row["source_group"]))
    task_counts = {
        role: {
            task_id: sum(row["task_id"] == task_id for row in rows)
            for task_id in sorted(by_task)
        }
        for role, rows in (("development", development), ("test", test))
    }
    sources = {
        role: {str(row["source_group"]) for row in rows}
        for role, rows in (("development", development), ("test", test))
    }
    status = "PASS" if (
        len(development) == 48
        and len(test) == 16
        and not sources["development"].intersection(sources["test"])
        and all(count == 6 for count in task_counts["development"].values())
        and all(count == 2 for count in task_counts["test"].values())
        and all(
            {int(row["cv_fold"]) for row in development if row["task_id"] == task_id}
            == set(range(6))
            for task_id in by_task
        )
    ) else "FAIL"
    return {
        "schema": "dsol_statewise_view_oracle_population_v2",
        "status": status,
        "split_root_seed": split_root_seed,
        "selection_policy": "two_sha256_ranked_prior_heldout_sources_per_task_for_test_then_six_task_stratified_cv_folds",
        "source_population": str(source_population.resolve()),
        "source_population_sha256": sha256_file(source_population),
        "source_disjoint": not sources["development"].intersection(sources["test"]),
        "task_counts": task_counts,
        "split_assignments": split_assignments,
        "population": {
            "development": {
                "state_count": len(development),
                "source_group_count": len(sources["development"]),
                "states": development,
            },
            "test": {
                "state_count": len(test),
                "source_group_count": len(sources["test"]),
                "states": test,
            },
        },
    }


def build_dense_protocol(
    *,
    role: str,
    states: Sequence[Mapping[str, Any]],
    scan_assets: Mapping[str, Mapping[str, Any]],
    repeat_ids: Sequence[int],
    bank_id: str,
    wave_id: str,
    population_path: Path,
    catalog_path: Path,
    protocol_config: Path,
) -> dict[str, Any]:
    blocks = []
    for state in states:
        scan_asset = scan_assets[str(state["asset_source_pair_key"])]
        blocks.append(
            {
                "state": dict(state),
                "scene_construction": scan_asset["scan"]["scene_construction"],
                "candidates": operational_candidates(scan_asset["scan"]),
            }
        )
    repeats = [int(value) for value in repeat_ids]
    episode_count = len(states) * 97 * len(repeats)
    return {
        "schema": COMPACT_SCHEMA,
        "status": "PASS_FROZEN",
        "phase": "dense_discovery",
        "role": role,
        "wave_id": wave_id,
        "diagnostic_role": f"statewise_view_oracle_v2_dense_{role}",
        "episode_identity_prefix": f"oracle-v2::{bank_id}::{role}::{wave_id}",
        "noise_bank_id": bank_id,
        "policy_repeat_ids": repeats,
        "sensor_control": "both",
        "state_count": len(states),
        "candidate_count_per_state": 97,
        "episode_count": episode_count,
        "matrix_order": "state_then_candidate_then_repeat",
        "catalog": str(catalog_path.resolve()),
        "catalog_sha256": sha256_file(catalog_path),
        "population": str(population_path.resolve()),
        "population_sha256": sha256_file(population_path),
        "protocol_config": str(protocol_config.resolve()),
        "protocol_config_sha256": sha256_file(protocol_config),
        "test_outcome_embargoed_until_selector_freeze": role == "test",
        "state_blocks": blocks,
    }


def ensure_bank(
    *,
    output_dir: Path,
    bank_id: str,
    state_keys: Sequence[str],
    repeats: int,
    root_seed: int,
) -> dict[str, Any]:
    manifest_path = output_dir / f"bank_{bank_id}.manifest.json"
    if manifest_path.exists():
        bank = ExplicitFlowNoiseBank(manifest_path, verify_file=True)
        expected = {
            "bank_id": bank_id,
            "state_keys": list(state_keys),
            "repeat_count": repeats,
            "root_seed": root_seed,
            "max_replans": 104,
            "action_horizon": 10,
            "action_dim": 7,
        }
        for field, value in expected.items():
            if bank.manifest.get(field) != value:
                raise ValueError(f"existing bank {bank_id} differs at {field}")
        manifest = bank.manifest
    else:
        manifest = materialize_bank(
            output_dir=output_dir,
            bank_id=bank_id,
            state_keys=state_keys,
            repeat_count=repeats,
            max_replans=104,
            action_horizon=10,
            action_dim=7,
            root_seed=root_seed,
        )
    return {
        "bank_id": bank_id,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "noise_file": manifest["noise_file"],
        "noise_file_sha256": manifest["noise_file_sha256"],
        "shape": manifest["shape"],
    }


def build_all(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.protocol_config.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_V2_POLICY_EXECUTION":
        raise ValueError("v2 protocol config is not frozen for execution")
    legacy_root = Path(config["legacy_evidence"]["root"])
    source_population = Path(config["population"]["source_population"])
    catalog = Path(config["candidate_space"]["catalog"])
    if not catalog.is_absolute():
        catalog = args.repo_root / catalog
    scans = load_scan_assets(legacy_root)
    renders = load_render_assets(legacy_root)
    population = freeze_population(
        source_population,
        scan_assets=scans,
        render_assets=renders,
        split_root_seed=int(config["population"]["split_root_seed"]),
    )
    if population["status"] != "PASS":
        raise ValueError("v2 population failed its split audit")
    population_path = args.output_root / "population" / "population-v2.json"
    freeze_json(population_path, population)
    development = population["population"]["development"]["states"]
    test = population["population"]["test"]["states"]
    all_state_keys = sorted(row["pair_key"] for row in [*development, *test])
    smoke_state_keys = [development[0]["pair_key"]]
    bank_receipts = []
    banks = config["randomness"]["banks"]
    for bank_id in ("S", "O", "P", "Q", "R"):
        spec = banks[bank_id]
        bank_receipts.append(
            ensure_bank(
                output_dir=args.output_root / "noise-banks",
                bank_id=bank_id,
                state_keys=smoke_state_keys if bank_id == "S" else all_state_keys,
                repeats=int(spec["repeats"]),
                root_seed=int(spec["root_seed"]),
            )
        )
    protocol_paths = []
    smoke_scan = scans[str(development[0]["asset_source_pair_key"])]
    smoke_block = {
        "state": dict(development[0]),
        "scene_construction": smoke_scan["scan"]["scene_construction"],
        "candidates": operational_candidates(smoke_scan["scan"])[:2],
    }
    smoke_protocol = {
        "schema": COMPACT_SCHEMA,
        "status": "PASS_FROZEN_SMOKE_ONLY",
        "phase": "smoke",
        "role": "development",
        "wave_id": "smoke",
        "diagnostic_role": "statewise_view_oracle_v2_smoke",
        "episode_identity_prefix": "oracle-v2::S::smoke",
        "noise_bank_id": "S",
        "policy_repeat_ids": [0, 1],
        "sensor_control": "both",
        "state_count": 1,
        "candidate_count_per_state": 2,
        "episode_count": 4,
        "matrix_order": "state_then_candidate_then_repeat",
        "catalog": str(catalog.resolve()),
        "catalog_sha256": sha256_file(catalog),
        "population": str(population_path.resolve()),
        "population_sha256": sha256_file(population_path),
        "protocol_config": str(args.protocol_config.resolve()),
        "protocol_config_sha256": sha256_file(args.protocol_config),
        "state_blocks": [smoke_block],
    }
    smoke_path = args.output_root / "protocols" / "smoke-S.json"
    freeze_json(smoke_path, smoke_protocol)
    protocol_paths.append(smoke_path)
    wave_size = int(config["execution"]["dense_discovery"]["wave_size_repeats"])
    repeats = int(config["execution"]["dense_discovery"]["noise_repeats"])
    for role, states in (("development", development), ("test", test)):
        for start in range(0, repeats, wave_size):
            wave_id = f"wave-{start // wave_size:02d}"
            payload = build_dense_protocol(
                role=role,
                states=states,
                scan_assets=scans,
                repeat_ids=range(start, min(start + wave_size, repeats)),
                bank_id="O",
                wave_id=wave_id,
                population_path=population_path,
                catalog_path=catalog,
                protocol_config=args.protocol_config,
            )
            path = args.output_root / "protocols" / f"dense-O-{role}-{wave_id}.json"
            freeze_json(path, payload)
            protocol_paths.append(path)
    receipt = {
        "schema": "dsol_statewise_view_oracle_v2_preparation_receipt",
        "status": "PASS_READY_FOR_SMOKE",
        "protocol_config": str(args.protocol_config.resolve()),
        "protocol_config_sha256": sha256_file(args.protocol_config),
        "population": str(population_path.resolve()),
        "population_sha256": sha256_file(population_path),
        "development_states": len(development),
        "test_states": len(test),
        "candidate_count": 97,
        "static_visibility_scans_verified": len(scans),
        "static_policy_input_banks_verified": len(renders),
        "legacy_rollout_outcomes_pooled": False,
        "noise_banks": bank_receipts,
        "protocols": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "episode_count": json.loads(path.read_text())["episode_count"],
            }
            for path in protocol_paths
        ],
    }
    freeze_json(args.output_root / "preparation" / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=Path("configs/dsol_paper1/statewise_view_oracle_v2.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2"
        ),
    )
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    if not args.protocol_config.is_absolute():
        args.protocol_config = args.repo_root / args.protocol_config
    args.protocol_config = args.protocol_config.resolve()
    args.output_root = args.output_root.resolve()
    receipt = build_all(args)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "development_states": receipt["development_states"],
                "test_states": receipt["test_states"],
                "protocols": len(receipt["protocols"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
