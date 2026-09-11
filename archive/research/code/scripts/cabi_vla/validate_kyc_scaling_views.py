from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_view_spec(value: str) -> tuple[int, str, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise ValueError("view specs must use CATALOG_SIZE=SCENE_MODE=PATH")
    catalog_size = int(parts[0])
    scene_mode = parts[1]
    path = Path(parts[2])
    if catalog_size <= 0:
        raise ValueError("catalog size must be positive")
    if scene_mode not in {"fixed", "cue_randomized"}:
        raise ValueError("scene mode must be fixed or cue_randomized")
    return catalog_size, scene_mode, path


def validate_view(
    *,
    catalog_size: int,
    scene_mode: str,
    root: Path,
    expected_records: int,
    expected_replicas: int,
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    records_path = root / "records.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(f"incomplete scaling view: {root}")
    manifest = json.loads(manifest_path.read_text())
    camera_view = manifest.get("camera_training_view")
    if not isinstance(camera_view, Mapping):
        raise ValueError(f"{root} has no camera_training_view manifest")
    config = camera_view.get("camera_config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{root} has no camera config")

    declared_records = int(manifest["record_count"])
    replicas = int(camera_view["camera_epoch_replicas"])
    if declared_records != expected_records:
        raise ValueError(
            f"{root} record count {declared_records} != {expected_records}"
        )
    if replicas != expected_replicas:
        raise ValueError(f"{root} replicas {replicas} != {expected_replicas}")
    if int(config["camera_catalog_size"]) != catalog_size:
        raise ValueError(f"{root} catalog size does not match its view spec")
    if str(config["scene_cue_mode"]) != scene_mode:
        raise ValueError(f"{root} scene mode does not match its view spec")
    records_hash = _sha256_file(records_path)
    if records_hash != manifest["records_sha256"]:
        raise ValueError(f"{root} records hash does not match its manifest")

    key_counts: Counter[tuple[int, int]] = Counter()
    pose_by_index: dict[int, tuple[float, float, float]] = {}
    scene_modes = set()
    sample_ids = set()
    row_count = 0
    with records_path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_index = int(row["source_record_index"])
            replica = int(row["camera_epoch_replica"])
            variant = int(row["camera_variant_index"])
            if not 0 <= replica < expected_replicas:
                raise ValueError(f"{root}:{line_number} has invalid replica")
            if not 0 <= variant < catalog_size:
                raise ValueError(f"{root}:{line_number} has invalid camera index")
            if int(row["camera_catalog_size"]) != catalog_size:
                raise ValueError(f"{root}:{line_number} has wrong catalog size")
            scene_modes.add(str(row["scene_cue_mode"]))
            sample_id = str(row["sample_id"])
            if sample_id in sample_ids:
                raise ValueError(f"{root}:{line_number} has duplicate sample_id")
            sample_ids.add(sample_id)
            key_counts[(source_index, replica)] += 1
            pose = (
                float(row["camera_azimuth_deg"]),
                float(row["camera_elevation_deg"]),
                float(row["camera_radius_scale"]),
            )
            previous = pose_by_index.setdefault(variant, pose)
            if previous != pose:
                raise ValueError(
                    f"{root}:{line_number} maps camera index {variant} "
                    "to inconsistent poses"
                )
            relative_shard = Path(str(row["camera_view_file"]))
            if (
                relative_shard.is_absolute()
                or ".." in relative_shard.parts
                or not (root / relative_shard).is_file()
            ):
                raise ValueError(f"{root}:{line_number} has an invalid shard path")
            row_count += 1

    if row_count != expected_records:
        raise ValueError(f"{root} contains {row_count} records, expected {expected_records}")
    if scene_modes != {scene_mode}:
        raise ValueError(f"{root} contains scene modes {sorted(scene_modes)}")
    if any(count != 1 for count in key_counts.values()):
        raise ValueError(f"{root} has duplicate source-record replicas")
    source_count = expected_records // expected_replicas
    expected_keys = {
        (source_index, replica)
        for source_index in range(source_count)
        for replica in range(expected_replicas)
    }
    if set(key_counts) != expected_keys:
        raise ValueError(f"{root} does not cover every source-record replica")
    if len(pose_by_index) != catalog_size:
        raise ValueError(
            f"{root} observes {len(pose_by_index)} of {catalog_size} catalog entries"
        )
    replay_state_max_abs = float(camera_view["replay_state_max_abs"])
    replay_pose_max_abs = float(camera_view["replay_pose_max_abs"])
    if replay_state_max_abs > 1e-8 or replay_pose_max_abs > 1e-5:
        raise ValueError(f"{root} failed the deterministic replay gate")

    return {
        "root": str(root),
        "catalog_size": catalog_size,
        "scene_cue_mode": scene_mode,
        "record_count": row_count,
        "camera_epoch_replicas": replicas,
        "source_records_sha256": camera_view["source_records_sha256"],
        "records_sha256": records_hash,
        "observed_catalog_entries": len(pose_by_index),
        "pose_by_index": pose_by_index,
        "replay_state_max_abs": replay_state_max_abs,
        "replay_pose_max_abs": replay_pose_max_abs,
        "baseline_image_mae_max": float(camera_view["baseline_image_mae_max"]),
    }


def validate_nested_catalogs(summaries: list[Mapping[str, Any]]) -> None:
    by_scene: dict[str, list[Mapping[str, Any]]] = {}
    for summary in summaries:
        by_scene.setdefault(str(summary["scene_cue_mode"]), []).append(summary)
    for scene_mode, scene_summaries in by_scene.items():
        ordered = sorted(scene_summaries, key=lambda item: int(item["catalog_size"]))
        for smaller, larger in zip(ordered, ordered[1:]):
            smaller_poses = smaller["pose_by_index"]
            larger_poses = larger["pose_by_index"]
            for index, pose in smaller_poses.items():
                if index in larger_poses and larger_poses[index] != pose:
                    raise ValueError(
                        f"{scene_mode} catalogs are not nested at camera index {index}"
                    )


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate merged KYC nested-camera scaling views"
    )
    parser.add_argument(
        "--view",
        action="append",
        required=True,
        metavar="CATALOG_SIZE=SCENE_MODE=PATH",
    )
    parser.add_argument("--expected-records", type=int, default=67_392)
    parser.add_argument("--expected-replicas", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    specs = [parse_view_spec(value) for value in parsed.view]
    summaries = [
        validate_view(
            catalog_size=catalog_size,
            scene_mode=scene_mode,
            root=root,
            expected_records=parsed.expected_records,
            expected_replicas=parsed.expected_replicas,
        )
        for catalog_size, scene_mode, root in specs
    ]
    source_hashes = {summary["source_records_sha256"] for summary in summaries}
    if len(source_hashes) != 1:
        raise ValueError("scaling views do not share the same source records")
    validate_nested_catalogs(summaries)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "source_records_sha256": source_hashes.pop(),
        "views": [
            {key: value for key, value in summary.items() if key != "pose_by_index"}
            for summary in summaries
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if parsed.output is not None:
        if parsed.output.exists():
            raise FileExistsError(f"refusing to overwrite validation: {parsed.output}")
        parsed.output.parent.mkdir(parents=True, exist_ok=True)
        parsed.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
