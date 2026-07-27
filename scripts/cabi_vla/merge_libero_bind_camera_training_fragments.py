from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"refusing to replace camera shard: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _safe_relative_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe camera shard path: {value!r}")
    return relative


def _validate_camera_matrices(row: dict, *, index: int) -> None:
    intrinsics = np.asarray(row.get("camera_intrinsics"), dtype=np.float64)
    camera_to_world = np.asarray(
        row.get("camera_to_world_opencv"),
        dtype=np.float64,
    )
    if intrinsics.shape != (3, 3) or camera_to_world.shape != (4, 4):
        raise ValueError(f"record {index} has invalid camera matrix shapes")
    if not np.all(np.isfinite(intrinsics)) or not np.all(
        np.isfinite(camera_to_world)
    ):
        raise ValueError(f"record {index} has non-finite camera matrices")
    rotation = camera_to_world[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
        or not np.allclose(camera_to_world[3], [0.0, 0.0, 0.0, 1.0])
    ):
        raise ValueError(f"record {index} has invalid camera-to-world transform")


def _validate_camera_shard(path: Path, rows: list[dict]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing camera shard: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"agentview", "wrist", "robot_state"}
        if set(archive.files) != required:
            raise ValueError(
                f"camera shard {path} has keys {sorted(archive.files)}, "
                f"expected {sorted(required)}"
            )
        agentview = archive["agentview"]
        wrist = archive["wrist"]
        robot_state = archive["robot_state"]
        count = len(agentview)
        if (
            agentview.shape != (count, 224, 224, 3)
            or wrist.shape != (count, 224, 224, 3)
            or robot_state.shape != (count, 8)
            or agentview.dtype != np.uint8
            or wrist.dtype != np.uint8
            or not np.all(np.isfinite(robot_state))
        ):
            raise ValueError(
                f"camera shard {path} has invalid arrays: "
                f"agent={agentview.shape}/{agentview.dtype}, "
                f"wrist={wrist.shape}/{wrist.dtype}, "
                f"state={robot_state.shape}/{robot_state.dtype}"
            )
        for row in rows:
            camera_index = int(row["camera_view_index"])
            if not 0 <= camera_index < count:
                raise IndexError(
                    f"camera index {camera_index} is outside {path} length {count}"
                )


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge disjoint randomized-camera LIBERO-Bind fragments"
    )
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fragments", type=Path, nargs="+", required=True)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite camera training view: {args.output}")
    source_manifest = json.loads((args.training_view / "manifest.json").read_text())
    source_records = [
        json.loads(line)
        for line in (args.training_view / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    source_records_sha256 = _sha256_file(args.training_view / "records.jsonl")

    manifests = [
        json.loads((fragment / "manifest.json").read_text())
        for fragment in args.fragments
    ]
    if any(manifest.get("status") != "complete" for manifest in manifests):
        raise ValueError("all camera fragments must be complete")
    config_hashes = {
        _canonical_json_sha256(manifest["camera_config"])
        for manifest in manifests
    }
    if len(config_hashes) != 1:
        raise ValueError("camera fragments use different randomization configs")
    camera_config_sha256 = config_hashes.pop()
    for fragment, manifest in zip(args.fragments, manifests):
        if Path(manifest["source_training_view"]).resolve() != args.training_view.resolve():
            raise ValueError(f"fragment uses a different training view: {fragment}")
        if int(manifest["source_record_count"]) != len(source_records):
            raise ValueError(f"fragment source record count mismatch: {fragment}")
        declared_source_hash = manifest.get("source_records_sha256")
        if (
            declared_source_hash is not None
            and declared_source_hash != source_records_sha256
        ):
            raise ValueError(f"fragment source records hash mismatch: {fragment}")
        declared_config_hash = manifest.get("camera_config_sha256")
        if (
            declared_config_hash is not None
            and declared_config_hash != camera_config_sha256
        ):
            raise ValueError(f"fragment camera config hash mismatch: {fragment}")
        baseline_tolerance = manifest.get("baseline_image_mae_tolerance")
        if (
            baseline_tolerance is not None
            and float(manifest["baseline_image_mae_max"])
            > float(baseline_tolerance)
        ):
            raise ValueError(f"fragment failed baseline image gate: {fragment}")

    records_by_index = {}
    shard_rows: dict[tuple[Path, Path], list[dict]] = defaultdict(list)
    fragment_record_hashes = {}
    for fragment, manifest in zip(args.fragments, manifests):
        records_path = fragment / "records.jsonl"
        records_text = records_path.read_text()
        records = [
            json.loads(line)
            for line in records_text.splitlines()
            if line.strip()
        ]
        records_sha256 = hashlib.sha256(records_text.encode()).hexdigest()
        if records_sha256 != manifest["records_sha256"]:
            raise ValueError(f"fragment records hash mismatch: {fragment}")
        fragment_record_hashes[str(fragment)] = records_sha256
        if len(records) != int(manifest["selected_record_count"]):
            raise ValueError(f"fragment record count mismatch: {fragment}")
        for row in records:
            index = int(row["source_record_index"])
            if index in records_by_index:
                raise ValueError(f"duplicate source record index: {index}")
            if not 0 <= index < len(source_records):
                raise IndexError(f"source record index outside source view: {index}")
            source_row = source_records[index]
            identity_keys = (
                "sample_id",
                "edge_id",
                "episode_file",
                "frame_index",
                "canonical_state_index",
                "language_instruction",
                "split",
            )
            mismatched = [
                key
                for key in identity_keys
                if key in source_row and row.get(key) != source_row[key]
            ]
            if mismatched:
                raise ValueError(
                    f"fragment row {index} does not match source fields: {mismatched}"
                )
            _validate_camera_matrices(row, index=index)
            relative = _safe_relative_path(str(row["camera_view_file"]))
            records_by_index[index] = row
            shard_rows[(fragment, relative)].append(row)

    expected_indices = set(range(len(source_records)))
    actual_indices = set(records_by_index)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        extra = sorted(actual_indices - expected_indices)
        raise ValueError(
            f"camera fragments do not cover source records: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    output_shards: dict[Path, tuple[Path, str]] = {}
    for (fragment, relative), rows in sorted(
        shard_rows.items(),
        key=lambda item: (str(item[0][0]), str(item[0][1])),
    ):
        source_path = fragment / relative
        _validate_camera_shard(source_path, rows)
        source_hash = _sha256_file(source_path)
        previous = output_shards.get(relative)
        if previous is not None and previous[1] != source_hash:
            raise ValueError(
                f"camera fragments conflict at {relative}: "
                f"{previous[0]} vs {source_path}"
            )
        output_shards[relative] = (source_path, source_hash)

    staging = args.output.parent / f".{args.output.name}.staging-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for relative, (source_path, _source_hash) in sorted(
            output_shards.items(),
            key=lambda item: str(item[0]),
        ):
            _link_or_copy(source_path, staging / relative)

        ordered_records = []
        for index in range(len(source_records)):
            row = records_by_index[index]
            ordered_records.append(row)

        records_text = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in ordered_records
        )
        (staging / "records.jsonl").write_text(records_text)
        anchors_name = str(source_manifest.get("anchors_file", "anchors.npz"))
        _link_or_copy(
            args.training_view / anchors_name,
            staging / anchors_name,
        )

        visibility_counts = Counter()
        for manifest in manifests:
            visibility_counts.update(manifest.get("visibility_counts", {}))
        baseline_camera = manifests[0]["baseline_camera"]
        for manifest in manifests[1:]:
            if manifest["baseline_camera"] != baseline_camera:
                raise ValueError("camera fragment baseline calibrations differ")

        manifest = {
            **source_manifest,
            "schema_version": max(2, int(source_manifest.get("schema_version", 1))),
            "source_training_view": str(args.training_view),
            "records_file": "records.jsonl",
            "record_count": len(ordered_records),
            "records_sha256": hashlib.sha256(records_text.encode()).hexdigest(),
            "camera_training_view": {
                "schema_version": 1,
                "method": "successful_open_loop_teacher_replay_with_random_camera",
                "camera_config_path": manifests[0]["camera_config_path"],
                "camera_config": manifests[0]["camera_config"],
                "camera_config_sha256": camera_config_sha256,
                "baseline_camera": baseline_camera,
                "source_records_sha256": source_records_sha256,
                "fragment_records_sha256": fragment_record_hashes,
                "camera_shards_sha256": {
                    str(relative): source_hash
                    for relative, (_source_path, source_hash) in sorted(
                        output_shards.items(),
                        key=lambda item: str(item[0]),
                    )
                },
                "fragment_paths": [str(path) for path in args.fragments],
                "fragment_edges": [
                    manifest["selected_edges"] for manifest in manifests
                ],
                "replay_state_max_abs": max(
                    float(manifest["replay_state_max_abs"])
                    for manifest in manifests
                ),
                "replay_pose_max_abs": max(
                    float(manifest["replay_pose_max_abs"])
                    for manifest in manifests
                ),
                "baseline_image_mae_max": max(
                    float(manifest["baseline_image_mae_max"])
                    for manifest in manifests
                ),
                "baseline_image_mae_tolerance": (
                    min(
                        float(manifest["baseline_image_mae_tolerance"])
                        for manifest in manifests
                        if manifest.get("baseline_image_mae_tolerance") is not None
                    )
                    if all(
                        manifest.get("baseline_image_mae_tolerance") is not None
                        for manifest in manifests
                    )
                    else None
                ),
                "visibility_counts": dict(visibility_counts),
                "record_count": len(ordered_records),
                "unique_camera_shards": len(output_shards),
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "record_count": len(ordered_records),
                    "camera_shards": len(output_shards),
                },
                sort_keys=True,
            )
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
