from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from libero_wrist_camera import (
    eef_transform_from_pose,
    eef_transform_from_robot_state,
    wrist_camera_from_eef,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add dynamic wrist calibration to a LIBERO-Bind camera view"
    )
    parser.add_argument("--source-view", type=Path, required=True)
    parser.add_argument("--hand-eye", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _dual_calibration(
    *,
    external_intrinsics,
    external_to_world,
    wrist_intrinsics,
    wrist_to_world,
) -> dict[str, list]:
    return {
        "camera_intrinsics": np.asarray(
            external_intrinsics,
            dtype=np.float64,
        ).tolist(),
        "camera_to_world_opencv": np.asarray(
            external_to_world,
            dtype=np.float64,
        ).tolist(),
        "camera_intrinsics_by_view": [
            np.asarray(external_intrinsics, dtype=np.float64).tolist(),
            np.asarray(wrist_intrinsics, dtype=np.float64).tolist(),
        ],
        "camera_to_world_opencv_by_view": [
            np.asarray(external_to_world, dtype=np.float64).tolist(),
            np.asarray(wrist_to_world, dtype=np.float64).tolist(),
        ],
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite dual-camera view: {args.output}")
    source_manifest = json.loads((args.source_view / "manifest.json").read_text())
    hand_eye_payload = json.loads(args.hand_eye.read_text())
    if hand_eye_payload.get("status") != "validated":
        raise ValueError("hand-eye calibration has not passed validation")
    source_collection = Path(source_manifest["source_collection"])
    hand_eye = np.asarray(hand_eye_payload["eef_to_wrist_opencv"], dtype=np.float64)
    wrist_intrinsics = np.asarray(hand_eye_payload["intrinsics"], dtype=np.float64)
    records_path = args.source_view / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line]

    episode_cache: dict[str, np.lib.npyio.NpzFile] = {}
    output_rows = []
    try:
        for row in records:
            episode_file = str(row["episode_file"])
            if episode_file not in episode_cache:
                episode_cache[episode_file] = np.load(
                    source_collection / episode_file,
                    allow_pickle=False,
                )
            episode = episode_cache[episode_file]
            frame = int(row["frame_index"])
            wrist_to_world = wrist_camera_from_eef(
                eef_transform_from_pose(episode["eef_pose"][frame]),
                hand_eye,
            )
            output_rows.append(
                {
                    **row,
                    **_dual_calibration(
                        external_intrinsics=row["camera_intrinsics"],
                        external_to_world=row["camera_to_world_opencv"],
                        wrist_intrinsics=wrist_intrinsics,
                        wrist_to_world=wrist_to_world,
                    ),
                }
            )
    finally:
        for episode in episode_cache.values():
            episode.close()

    camera_training_view = dict(source_manifest["camera_training_view"])
    baseline_external = camera_training_view["baseline_camera"]
    canonical_wrist = np.asarray(
        hand_eye_payload["canonical_wrist_camera_to_world_opencv"],
        dtype=np.float64,
    )
    camera_training_view["baseline_camera"] = {
        **baseline_external,
        **_dual_calibration(
            external_intrinsics=baseline_external["camera_intrinsics"],
            external_to_world=baseline_external["camera_to_world_opencv"],
            wrist_intrinsics=wrist_intrinsics,
            wrist_to_world=canonical_wrist,
        ),
    }

    with np.load(args.source_view / "anchors.npz", allow_pickle=False) as anchors:
        baseline_cameras_by_anchor = {}
        for tetrad in source_manifest["tetrads"]:
            state_index = int(tetrad["canonical_state_index"])
            decision_point = str(tetrad.get("decision_point", ""))
            for corner in tetrad["corners"].values():
                physical_edge = str(corner["physical_edge"])
                key = f"{physical_edge}__state_{state_index:02d}"
                if decision_point:
                    key += f"__{decision_point}"
                if key in baseline_cameras_by_anchor:
                    continue
                state = np.asarray(anchors[f"{key}__state"], dtype=np.float64)
                wrist_to_world = wrist_camera_from_eef(
                    eef_transform_from_robot_state(state),
                    hand_eye,
                )
                baseline_cameras_by_anchor[key] = _dual_calibration(
                    external_intrinsics=baseline_external["camera_intrinsics"],
                    external_to_world=baseline_external["camera_to_world_opencv"],
                    wrist_intrinsics=wrist_intrinsics,
                    wrist_to_world=wrist_to_world,
                )
    camera_training_view.update(
        {
            "dual_camera_calibration": True,
            "hand_eye_path": str(args.hand_eye),
            "hand_eye_sha256": hashlib.sha256(args.hand_eye.read_bytes()).hexdigest(),
            "baseline_cameras_by_anchor": baseline_cameras_by_anchor,
        }
    )

    records_text = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in output_rows
    )
    output_manifest = {
        **source_manifest,
        "camera_training_view": camera_training_view,
        "dual_camera_source_view": str(args.source_view),
        "dual_camera_records_sha256": hashlib.sha256(records_text.encode()).hexdigest(),
    }
    staging = args.output.with_name(f".{args.output.name}.staging-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "records.jsonl").write_text(records_text)
    (staging / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n"
    )
    os.symlink((args.source_view / "anchors.npz").resolve(), staging / "anchors.npz")
    os.symlink((args.source_view / "camera_views").resolve(), staging / "camera_views")
    staging.rename(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "record_count": len(output_rows),
                "anchor_calibration_count": len(baseline_cameras_by_anchor),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
