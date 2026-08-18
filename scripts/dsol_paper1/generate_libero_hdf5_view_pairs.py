from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _stable_unit(seed: int, identity: str, field: str) -> float:
    digest = hashlib.sha256(f"{seed}::{identity}::{field}".encode()).digest()
    return int.from_bytes(digest[:8], "little") / float(1 << 64)


def _stable_index(seed: int, identity: str, field: str, size: int) -> int:
    return min(int(_stable_unit(seed, identity, field) * size), size - 1)


def _pose_pair(
    poses: list[Mapping[str, Any]], seed: int, identity: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if len(poses) < 2:
        raise ValueError("paired generation requires at least two poses")
    first = _stable_index(seed, identity, "pose_a", len(poses))
    second_raw = _stable_index(seed, identity, "pose_b", len(poses) - 1)
    second = second_raw if second_raw < first else second_raw + 1
    return poses[first], poses[second]


def _split(seed: int, episode_id: str) -> str:
    unit = _stable_unit(seed, episode_id, "split")
    if unit < 0.8:
        return "train"
    if unit < 0.9:
        return "val"
    return "test"


def _action_chunk(actions: np.ndarray, frame: int, horizon: int) -> np.ndarray:
    chunk = np.asarray(actions[frame : frame + horizon], dtype=np.float32)
    if len(chunk) < horizon:
        chunk = np.concatenate(
            [chunk, np.zeros((horizon - len(chunk), actions.shape[1]), np.float32)],
            axis=0,
        )
    return chunk


def _policy_image(value: Any) -> np.ndarray:
    image = np.asarray(value, dtype=np.uint8)
    return np.ascontiguousarray(image[::-1, ::-1])


def _render_observation(env: Any) -> Mapping[str, Any]:
    env.env._update_observables(force=True)
    return env.env._get_observations()


def _restore_reference(env: Any, reference: Mapping[str, Any]) -> None:
    sim = env.env.sim
    camera_id = int(reference["camera_id"])
    sim.model.cam_pos[camera_id] = np.asarray(reference["position"])
    sim.model.cam_quat[camera_id] = np.asarray(reference["quaternion"])
    sim.forward()


def generate(args: argparse.Namespace) -> dict[str, Any]:
    import h5py

    script_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_root))
    sys.path.insert(0, str(script_root.parent / "cabi_vla"))
    from audit_libero_hdf5_restore import _configure_runtime, _decode, _rewrite_model_paths
    from libero_pair_records import (
        IMAGE_ORDER,
        initialize_shard,
        sha256_file,
        write_record,
    )

    hdf5_path = args.hdf5.resolve()
    runtime = args.runtime.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    _configure_runtime(runtime, hdf5_path.parent.parent, args.config_root.resolve())
    from libero.libero.envs import OffScreenRenderEnv
    from libero_camera_pose import (
        capture_camera_reference,
        install_camera_pose,
        mujoco_camera_calibration,
    )

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    pose_ids = catalog["broad_training_sets"][args.pose_set]
    pose_by_id = {
        pose["pose_id"]: pose for pose in catalog["broad_training_64"]
    }
    poses = [pose_by_id[name] for name in pose_ids]
    suite = hdf5_path.parent.name
    acquisition = json.loads(args.acquisition.read_text(encoding="utf-8"))
    relative_hdf5 = f"{suite}/{hdf5_path.name}"
    source_record = next(
        row for row in acquisition["files"] if row["path"] == relative_hdf5
    )
    if not str(source_record["status"]).startswith("VERIFIED"):
        raise ValueError(f"source HDF5 is not verified: {relative_hdf5}")

    records_path = staging / "records.jsonl"
    shard_path = staging / "pairs.bin"
    index_rows = []
    rewrite_totals = {"robosuite": 0, "libero": 0}
    counts_by_split = {"train": 0, "val": 0, "test": 0}
    with h5py.File(hdf5_path, "r") as source:
        data = source["data"]
        bddl_name = Path(_decode(data.attrs["bddl_file_name"])).name
        problem_info = json.loads(_decode(data.attrs["problem_info"]))
        instruction = str(problem_info["language_instruction"])
        bddl = runtime / "libero" / "libero" / "bddl_files" / suite / bddl_name
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_names=("agentview", "robot0_eye_in_hand"),
            camera_heights=args.resolution,
            camera_widths=args.resolution,
            render_gpu_device_id=args.render_gpu,
        )
        record_count = 0
        try:
            env.reset()
            with shard_path.open("wb") as shard:
                initialize_shard(shard)
                for demo_name in sorted(data.keys()):
                    if (
                        args.demo_limit is not None
                        and int(demo_name.split("_")[-1]) >= args.demo_limit
                    ):
                        continue
                    demo = data[demo_name]
                    model_xml, rewrites = _rewrite_model_paths(
                        _decode(demo.attrs["model_file"]), runtime
                    )
                    for name, count in rewrites.items():
                        rewrite_totals[name] += count
                    env.reset_from_xml_string(model_xml)
                    env.set_init_state(np.asarray(demo["states"][0]))
                    reference = capture_camera_reference(
                        env,
                        camera_name="agentview",
                        table_plane_z=float(catalog["table_plane_z"]),
                    )
                    episode_id = f"{suite}::{hdf5_path.stem}::{demo_name}"
                    episode_split = _split(args.seed, episode_id)
                    frame_count = int(demo["states"].shape[0])
                    actions = np.asarray(demo["actions"], dtype=np.float32)
                    for frame in range(0, frame_count, args.frame_stride):
                        if args.record_limit is not None and record_count >= args.record_limit:
                            break
                        sample_id = f"{episode_id}::frame-{frame:05d}"
                        pose_a, pose_b = _pose_pair(poses, args.seed, sample_id)
                        state = np.asarray(demo["states"][frame])
                        _restore_reference(env, reference)
                        canonical = env.set_init_state(state)
                        canonical_calibration = mujoco_camera_calibration(
                            env,
                            camera_name="agentview",
                            height=args.resolution,
                            width=args.resolution,
                        )
                        canonical_image = _policy_image(canonical["agentview_image"])
                        wrist_image = _policy_image(
                            canonical["robot0_eye_in_hand_image"]
                        )

                        camera_a = install_camera_pose(env, reference, pose_a)
                        observation_a = _render_observation(env)
                        calibration_a = mujoco_camera_calibration(
                            env,
                            camera_name="agentview",
                            height=args.resolution,
                            width=args.resolution,
                        )
                        image_a = _policy_image(observation_a["agentview_image"])

                        _restore_reference(env, reference)
                        camera_b = install_camera_pose(env, reference, pose_b)
                        observation_b = _render_observation(env)
                        calibration_b = mujoco_camera_calibration(
                            env,
                            camera_name="agentview",
                            height=args.resolution,
                            width=args.resolution,
                        )
                        image_b = _policy_image(observation_b["agentview_image"])

                        robot_state = np.concatenate(
                            [
                                np.asarray(demo["obs/ee_pos"][frame]),
                                np.asarray(demo["obs/ee_ori"][frame]),
                                np.asarray(demo["obs/gripper_states"][frame]),
                            ]
                        ).astype(np.float32)
                        header = {
                            "schema": "dsol_libero_view_pair_record_v1",
                            "sample_id": sample_id,
                            "episode_id": episode_id,
                            "split": episode_split,
                            "suite": suite,
                            "task": (
                                hdf5_path.stem[:-5]
                                if hdf5_path.stem.endswith("_demo")
                                else hdf5_path.stem
                            ),
                            "demo": demo_name,
                            "frame": frame,
                            "frame_count": frame_count,
                            "language_instruction": instruction,
                            "robot_state": robot_state.tolist(),
                            "action_chunk": _action_chunk(
                                actions, frame, args.action_horizon
                            ).tolist(),
                            "source_state_dim": int(state.shape[0]),
                            "source_state_sha256": hashlib.sha256(
                                state.tobytes()
                            ).hexdigest(),
                            "pose_a": camera_a,
                            "pose_b": camera_b,
                            "camera_intrinsics": np.asarray(
                                canonical_calibration["intrinsics"]
                            ).tolist(),
                            "canonical_camera_to_world_opencv": np.asarray(
                                canonical_calibration["camera_to_world_opencv"]
                            ).tolist(),
                            "camera_a_to_world_opencv": np.asarray(
                                calibration_a["camera_to_world_opencv"]
                            ).tolist(),
                            "camera_b_to_world_opencv": np.asarray(
                                calibration_b["camera_to_world_opencv"]
                            ).tolist(),
                        }
                        location = write_record(
                            shard,
                            header=header,
                            images={
                                IMAGE_ORDER[0]: canonical_image,
                                IMAGE_ORDER[1]: image_a,
                                IMAGE_ORDER[2]: image_b,
                                IMAGE_ORDER[3]: wrist_image,
                            },
                            jpeg_quality=args.jpeg_quality,
                        )
                        index_rows.append(
                            {
                                "sample_id": sample_id,
                                "episode_id": episode_id,
                                "split": episode_split,
                                "frame": frame,
                                "pose_a_id": pose_a["pose_id"],
                                "pose_b_id": pose_b["pose_id"],
                                **location,
                            }
                        )
                        counts_by_split[episode_split] += 1
                        record_count += 1
                    if args.record_limit is not None and record_count >= args.record_limit:
                        break
        finally:
            env.close()

    with records_path.open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema": "dsol_libero_hdf5_view_pair_shard_v1",
        "status": "VERIFIED",
        "source_hdf5": str(hdf5_path),
        "source_hdf5_expected_sha256": source_record["expected_sha256"],
        "source_hdf5_size_bytes": source_record["size_bytes"],
        "source_revision": acquisition["revision"],
        "runtime": str(runtime),
        "catalog": str(args.catalog.resolve()),
        "pose_set": args.pose_set,
        "seed": args.seed,
        "resolution": args.resolution,
        "action_horizon": args.action_horizon,
        "frame_stride": args.frame_stride,
        "record_count": len(index_rows),
        "counts_by_split": counts_by_split,
        "image_order": list(IMAGE_ORDER),
        "image_transform": "rot180_policy_upright",
        "asset_path_rewrites": rewrite_totals,
        "shard": shard_path.name,
        "shard_size_bytes": shard_path.stat().st_size,
        "shard_sha256": sha256_file(shard_path),
        "records": records_path.name,
        "records_sha256": sha256_file(records_path),
    }
    _atomic_json(staging / "manifest.json", manifest)
    staging.replace(output)
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exact-state canonical and broad LIBERO view pairs."
    )
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pose-set",
        choices=("broad_32", "broad_64"),
        default="broad_32",
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--demo-limit", type=int)
    parser.add_argument("--record-limit", type=int)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--render-gpu", type=int, default=0)
    return parser.parse_args(argv)


def main() -> None:
    result = generate(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "record_count": result["record_count"],
                "counts_by_split": result["counts_by_split"],
                "shard_size_bytes": result["shard_size_bytes"],
                "shard_sha256": result["shard_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
