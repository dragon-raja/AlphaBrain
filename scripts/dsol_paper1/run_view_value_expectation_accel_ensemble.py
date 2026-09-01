#!/usr/bin/env python3
"""Render and rank the formal 97-view population with an Accel ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def configure_imports() -> None:
    scripts = Path(__file__).resolve().parents[1]
    for path in (scripts / "dsol_paper1", scripts / "cabi_vla"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(label: str, root_seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{root_seed}::{label}".encode()).digest()[:4], "little")


def load_population(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS":
        raise ValueError("formal population did not PASS")
    states = []
    for split in ("calibration", "heldout_test"):
        states.extend(payload["population"][split]["states"])
    return sorted(states, key=lambda row: row["pair_key"])


def load_scans(root: Path) -> dict[str, dict[str, Any]]:
    scans = {}
    for ledger in sorted(root.glob("shard-*.jsonl")):
        for line in ledger.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "PASS":
                raise ValueError(f"scan failed: {row.get('scan_id')}")
            scans[row["scan_id"]] = json.loads((Path(row["output_dir"]) / "scan.json").read_text())
    return scans


def operational_records(scan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
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
    records.sort(key=lambda row: str(row["pose_id"]))
    if len(records) != 97 or len({row["pose_id"] for row in records}) != 97:
        raise ValueError("formal Accel bank must contain 97 operational candidates")
    return records


def render_state(
    state_spec: Mapping[str, Any],
    scan: Mapping[str, Any],
    *,
    runtime: Path,
    config_root: Path,
    catalog: Path,
    render_gpu: int,
    resize_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    import h5py

    configure_imports()
    from audit_libero_hdf5_restore import _configure_runtime, _decode, _rewrite_model_paths

    hdf5_path = Path(state_spec["hdf5"]).resolve()
    _configure_runtime(runtime, hdf5_path.parent.parent, config_root)
    from evaluate_pi05_libero_plus_views import (
        agentview_camera_calibration,
        physics_state_sha256,
        prepare_policy_observation,
    )
    from libero.libero.envs import OffScreenRenderEnv
    from libero_camera_pose import capture_camera_reference, install_camera_pose
    from libero_constructed_view import inject_static_visual_occluder
    from scan_libero_hdf5_views import _restore_reference

    with h5py.File(hdf5_path, "r") as handle:
        data = handle["data"]
        demo = data[str(state_spec["demo_name"])]
        state = np.asarray(demo["states"][int(state_spec["source_state_index"])])
        model_xml, rewrites = _rewrite_model_paths(_decode(demo.attrs["model_file"]), runtime)
        bddl_name = Path(_decode(data.attrs["bddl_file_name"])).name
        prompt = str(json.loads(_decode(data.attrs["problem_info"]))["language_instruction"])
    model_xml = inject_static_visual_occluder(model_xml, scan["scene_construction"])
    bddl = runtime / "libero" / "libero" / "bddl_files" / state_spec["suite"] / bddl_name
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_names=("agentview", "robot0_eye_in_hand"),
        camera_heights=256,
        camera_widths=256,
        render_gpu_device_id=render_gpu,
    )
    candidates = operational_records(scan)
    external_images = []
    wrist_images = []
    robot_states = []
    intrinsics = []
    extrinsics = []
    try:
        env.seed(int(state_spec["environment_seed"]))
        env.reset()
        env.reset_from_xml_string(model_xml)
        env.set_init_state(state)
        physics_before, physics_size = physics_state_sha256(env)
        table_plane_z = float(json.loads(catalog.read_text())["table_plane_z"])
        reference = capture_camera_reference(
            env,
            camera_name="agentview",
            table_plane_z=table_plane_z,
        )
        for record in candidates:
            _restore_reference(env, reference)
            if record["pose_id"] != "canonical":
                install_camera_pose(env, reference, record["pose"])
            env.env._update_observables(force=True)
            observation = env.env._get_observations()
            calibration = agentview_camera_calibration(env)
            example, _agent, _wrist = prepare_policy_observation(
                observation,
                prompt=prompt,
                resize_size=resize_size,
                eval_seed=int(state_spec["environment_seed"]),
                camera_calibration=calibration,
            )
            external_images.append(np.asarray(example["observation/image"], dtype=np.uint8))
            wrist_images.append(np.asarray(example["observation/wrist_image"], dtype=np.uint8))
            robot_states.append(np.asarray(example["observation/state"], dtype=np.float32))
            intrinsics.append(np.asarray(example["camera_intrinsics"], dtype=np.float64))
            extrinsics.append(np.asarray(example["camera_to_world_opencv"], dtype=np.float64))
        physics_after, physics_after_size = physics_state_sha256(env)
        if physics_before != physics_after or physics_size != physics_after_size:
            raise ValueError("counterfactual camera rendering changed physical state")
    finally:
        env.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "policy_inputs.npz"
    np.savez_compressed(
        artifact,
        candidate_ids=np.asarray([row["pose_id"] for row in candidates]),
        external_images=np.stack(external_images),
        wrist_images=np.stack(wrist_images),
        robot_states=np.stack(robot_states),
        camera_intrinsics=np.stack(intrinsics),
        camera_to_world_opencv=np.stack(extrinsics),
    )
    record = {
        "schema": "dsol_view_value_expectation_accel_render_v1",
        "status": "PASS",
        "pair_key": state_spec["pair_key"],
        "split": state_spec["split"],
        "task_id": state_spec["task_id"],
        "source_group": state_spec["source_group"],
        "language": prompt,
        "candidate_count": 97,
        "physics_state_sha256": physics_before,
        "physics_state_size": physics_size,
        "artifact": str(artifact.resolve()),
        "artifact_sha256": sha256_file(artifact),
        "asset_path_rewrites": rewrites,
    }
    atomic_json(output_dir / "render.json", record)
    return record


def rank_state(
    state_spec: Mapping[str, Any],
    *,
    model: Any,
    render_dir: Path,
    ensemble_size: int,
    root_seed: int,
    batch_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    import torch

    configure_imports()
    from accel_inference import rank_fixed_state_candidates_chunked

    render = json.loads((render_dir / "render.json").read_text())
    artifact = Path(render["artifact"])
    if sha256_file(artifact) != render["artifact_sha256"]:
        raise ValueError("rendered policy input checksum mismatch")
    with np.load(artifact, allow_pickle=False) as values:
        candidate_ids = [str(value) for value in values["candidate_ids"]]
        examples = [
            {
                "image": [values["external_images"][index], values["wrist_images"][index]],
                "lang": render["language"],
                "language": render["language"],
                "state": values["robot_states"][index],
                "camera_intrinsics": values["camera_intrinsics"][index],
                "camera_to_world_opencv": values["camera_to_world_opencv"][index],
            }
            for index in range(len(candidate_ids))
        ]
    scores = {candidate_id: [] for candidate_id in candidate_ids}
    seeds = []
    started = time.perf_counter()
    with torch.inference_mode():
        for member in range(ensemble_size):
            seed = stable_seed(f"accel::{state_spec['pair_key']}::{member}", root_seed)
            seeds.append(seed)
            ranking = rank_fixed_state_candidates_chunked(
                model,
                examples,
                candidate_ids,
                seed=seed,
                action_horizon=int(model.action_horizon),
                action_dim=int(model.action_dim),
                batch_size=batch_size,
                include_trace_artifacts=False,
            )
            for row in ranking["ranking"]:
                scores[str(row["candidate_id"])].append(float(row["accel_3"]))
    summaries = [
        {
            "candidate_id": candidate_id,
            "mean_accel_3": float(np.mean(values)),
            "std_accel_3": float(np.std(values)),
            "member_accel_3": values,
        }
        for candidate_id, values in scores.items()
    ]
    summaries.sort(key=lambda row: (row["mean_accel_3"], row["candidate_id"]))
    record = {
        "schema": "dsol_view_value_expectation_accel_ensemble_v1",
        "status": "PASS",
        "pair_key": state_spec["pair_key"],
        "split": state_spec["split"],
        "task_id": state_spec["task_id"],
        "source_group": state_spec["source_group"],
        "checkpoint_action_horizon": int(model.action_horizon),
        "ensemble_size": ensemble_size,
        "ensemble_seeds": seeds,
        "candidate_count": len(candidate_ids),
        "selected_candidate_id": summaries[0]["candidate_id"],
        "selected_is_canonical": summaries[0]["candidate_id"] == "canonical",
        "top2_margin": summaries[1]["mean_accel_3"] - summaries[0]["mean_accel_3"],
        "ranking": summaries,
        "inference_seconds": time.perf_counter() - started,
        "render_artifact_sha256": render["artifact_sha256"],
        "physics_state_sha256": render["physics_state_sha256"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "ranking.json", record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("render", "rank"), required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render-gpu", type=int, default=0)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--root-seed", type=int, default=20260921)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    states = load_population(args.population)
    scans = load_scans(args.scan_root)
    selected = [state for index, state in enumerate(states) if index % args.num_shards == args.shard_index]
    ledger = args.output_root / f"{args.stage}-shard-{args.shard_index:02d}.jsonl"
    completed = (
        {json.loads(line)["pair_key"] for line in ledger.read_text().splitlines() if line.strip()}
        if ledger.exists()
        else set()
    )
    model = None
    if args.stage == "rank":
        if args.checkpoint is None:
            raise ValueError("rank stage requires --checkpoint")
        import torch

        from AlphaBrain.model.framework.base_framework import BaseFramework

        model = BaseFramework.from_pretrained(str(args.checkpoint), strict_checkpoint=True)
        model = model.to(torch.bfloat16).to(args.device).eval()
        model.gripper_remap = False
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    for state in selected:
        if state["pair_key"] in completed:
            continue
        state_dir = args.output_root / "states" / hashlib.sha256(state["pair_key"].encode()).hexdigest()[:20]
        if args.stage == "render":
            if args.runtime is None or args.config_root is None:
                raise ValueError("render stage requires --runtime and --config-root")
            row = render_state(
                state,
                scans[state["pair_key"]],
                runtime=args.runtime.resolve(),
                config_root=args.config_root.resolve(),
                catalog=args.catalog.resolve(),
                render_gpu=args.render_gpu,
                resize_size=args.resize_size,
                output_dir=state_dir,
            )
        else:
            row = rank_state(
                state,
                model=model,
                render_dir=state_dir,
                ensemble_size=args.ensemble_size,
                root_seed=args.root_seed,
                batch_size=args.batch_size,
                output_dir=state_dir,
            )
        append_jsonl(ledger, row)
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "pair_key": row["pair_key"],
                    "status": row["status"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
