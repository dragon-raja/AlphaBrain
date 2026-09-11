from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEPLOYABLE_SHAPES = {
    "agentview_image": (224, 224, 3),
    "wrist_image": (224, 224, 3),
    "robot_state": (8,),
    "vla_feature": (4096,),
    "candidates": (16, 10, 7),
    "candidate_seeds": (16, 2),
}
LABEL_SHAPES = {
    "continuation_signatures": (16, 6, 6),
    "continuation_profiles": (16, 6),
    "raw_milestones": (16, 6, 4),
    "direct_signatures": (16, 6),
    "immediate_correct": (16,),
    "failure_continuation": (16,),
    "premature_commitment": (16,),
}
FILE_FIELDS = ("deployable_file", "labels_file", "audit_file")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _partition(source_id: int, metadata: Mapping[str, Any]) -> str:
    if source_id in set(metadata["engineering_excluded_source_ids"]):
        return "engineering_excluded"
    if source_id in set(metadata["holdout_source_ids"]):
        return "holdout"
    if source_id in set(metadata["fit_source_ids"]):
        return "fit"
    return "unknown"


def _check_npz(
    path: Path,
    expected_shapes: Mapping[str, tuple[int, ...]],
    errors: list[str],
) -> None:
    try:
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(expected_shapes):
                errors.append(
                    f"{path}: keys={sorted(arrays.files)} expected={sorted(expected_shapes)}"
                )
                return
            for key, shape in expected_shapes.items():
                value = arrays[key]
                if value.shape != shape:
                    errors.append(f"{path}:{key}: shape={value.shape} expected={shape}")
                if value.dtype.hasobject:
                    errors.append(f"{path}:{key}: object dtype is forbidden")
                if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
                    errors.append(f"{path}:{key}: contains non-finite values")
    except Exception as exc:  # noqa: BLE001 - the audit must report corrupt archives.
        errors.append(f"{path}: cannot load npz ({type(exc).__name__}: {exc})")


def audit_collection(
    dataset_root: Path,
    episode_root: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    metadata = json.loads((dataset_root / "metadata.json").read_text())
    source_manifest = json.loads((episode_root / "manifest.json").read_text())
    expected_rows = {
        str(row["pair_id"]): row for row in source_manifest["groups"] if row["split"] == "train"
    }
    complete_files = sorted((dataset_root / "groups").glob("*/complete.json"))
    rows = [json.loads(path.read_text()) for path in complete_files]
    pair_counts = Counter(str(row.get("pair_id")) for row in rows)
    discovered = set(pair_counts)
    expected = set(expected_rows)
    errors: list[str] = []
    warnings: list[str] = []
    inspected_states = 0
    sealed_states = 0
    partition_groups: Counter[str] = Counter()
    partition_success: Counter[str] = Counter()

    duplicate_ids = sorted(pair_id for pair_id, count in pair_counts.items() if count != 1)
    if duplicate_ids:
        errors.append(f"duplicate pair IDs: {duplicate_ids}")
    missing = sorted(expected - discovered)
    unexpected = sorted(discovered - expected)
    if missing:
        errors.append(f"missing expected groups: {missing}")
    if unexpected:
        errors.append(f"unexpected groups: {unexpected}")
    if int(metadata["all_train_group_count"]) != len(expected):
        errors.append("metadata all_train_group_count does not match source train manifest")

    for row in sorted(rows, key=lambda item: str(item.get("pair_id"))):
        pair_id = str(row.get("pair_id"))
        if pair_id not in expected_rows:
            continue
        source_id = int(row["source_initial_state_index"])
        expected_source = int(expected_rows[pair_id]["source_initial_state_index"])
        partition = str(row["source_partition"])
        expected_partition = _partition(source_id, metadata)
        partition_groups[partition] += 1
        partition_success[partition] += int(bool(row["success"]))
        if source_id != expected_source:
            errors.append(f"{pair_id}: source={source_id} expected={expected_source}")
        if partition != expected_partition:
            errors.append(f"{pair_id}: partition={partition} expected={expected_partition}")
        states = row.get("states", [])
        state_ids = [str(state.get("state_id")) for state in states]
        if not states:
            errors.append(f"{pair_id}: no captured states")
        if len(state_ids) != len(set(state_ids)):
            errors.append(f"{pair_id}: duplicate state IDs")

        for state in states:
            state_id = str(state.get("state_id"))
            if int(state.get("candidate_count", -1)) != 16:
                errors.append(f"{pair_id}/{state_id}: candidate_count is not 16")
            if int(state.get("continuation_repeats", -1)) != 6:
                errors.append(f"{pair_id}/{state_id}: continuation_repeats is not 6")
            if int(state.get("raw_milestone_violations", -1)) != 0:
                warnings.append(f"{pair_id}/{state_id}: raw milestone violation")
            paths: dict[str, Path] = {}
            for field in FILE_FIELDS:
                relative = Path(str(state.get(field, "")))
                path = dataset_root / relative
                paths[field] = path
                if not relative.parts or not _inside(dataset_root, path):
                    errors.append(f"{pair_id}/{state_id}: unsafe {field} path")
                elif not path.is_file() or path.stat().st_size == 0:
                    errors.append(f"{pair_id}/{state_id}: missing or empty {field}")

            # Holdout labels and inputs remain sealed until every Gate 0B model is serialized.
            if partition == "holdout":
                sealed_states += 1
                continue
            if all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
                _check_npz(paths["deployable_file"], DEPLOYABLE_SHAPES, errors)
                _check_npz(paths["labels_file"], LABEL_SHAPES, errors)
                try:
                    with np.load(paths["audit_file"], allow_pickle=False) as audit:
                        if "sim_state" not in audit.files:
                            errors.append(f"{pair_id}/{state_id}: audit snapshot lacks sim_state")
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        f"{pair_id}/{state_id}: cannot load audit snapshot "
                        f"({type(exc).__name__}: {exc})"
                    )
                inspected_states += 1

    success_rates = {
        partition: partition_success[partition] / count
        for partition, count in sorted(partition_groups.items())
        if count
    }
    audit = {
        "status": "pass" if not errors else "fail",
        "dataset_root": str(dataset_root.resolve()),
        "episode_root": str(episode_root.resolve()),
        "expected_groups": len(expected),
        "completed_groups": len(rows),
        "missing_groups": missing,
        "unexpected_groups": unexpected,
        "partition_group_counts": dict(sorted(partition_groups.items())),
        "partition_teacher_success_rates": success_rates,
        "inspected_non_holdout_states": inspected_states,
        "sealed_holdout_states_not_opened": sealed_states,
        "errors": errors,
        "warnings": warnings,
    }
    if errors:
        return audit, None
    manifest = {
        **metadata,
        "status": "complete",
        "completed_groups": len(rows),
        "groups": sorted(rows, key=lambda item: str(item["pair_id"])),
        "integrity_audit": "integrity_audit.json",
        "holdout_content_status": "sealed_unopened",
    }
    return audit, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize and audit a CCV collection")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path)
    parser.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    metadata = json.loads((dataset_root / "metadata.json").read_text())
    episode_root = (args.episode_root or Path(metadata["episode_root"])).resolve()
    audit, manifest = audit_collection(dataset_root, episode_root)
    _atomic_write_json(dataset_root / "integrity_audit.json", audit)
    if args.finalize:
        if manifest is None:
            raise SystemExit("collection audit failed; manifest was not finalized")
        _atomic_write_json(dataset_root / "manifest.json", manifest)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
