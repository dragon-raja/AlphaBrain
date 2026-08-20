#!/usr/bin/env python3
"""Run fixed-state Accel over the frozen constructed-M0 candidate bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from accel_core import rank_accel_candidates
from accel_inference import rank_fixed_state_candidates_chunked


PROTOCOL_SCHEMA = "dsol_constructed_m1_frozen_closed_loop_protocol_v1"
OPERATIONAL_GROUPS = ("canonical", "broad_training_64", "broad_heldout_32")
SENSOR_CONTROLS = ("external_blackout", "wrist_blackout", "all_camera_blackout")
DIAGNOSTIC_ROLES = (
    "canonical",
    "strong_info",
    "matched_control",
    "blind",
    "look_away",
    *SENSOR_CONTROLS,
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(key: str, *, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}::{key}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def pose_index(catalog: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    poses: dict[str, Mapping[str, Any]] = {}
    groups: dict[str, str] = {}
    for group, values in catalog.items():
        if not isinstance(values, list) or not values or not isinstance(values[0], Mapping):
            continue
        for pose in values:
            pose_id = str(pose["pose_id"])
            if pose_id in poses:
                raise ValueError(f"duplicate catalog pose: {pose_id}")
            poses[pose_id] = pose
            groups[pose_id] = str(group)
    return poses, groups


def build_candidate_bank(
    catalog: Mapping[str, Any], visibility_selection: Mapping[str, Any]
) -> dict[str, Any]:
    poses, groups = pose_index(catalog)
    operational = ["canonical"]
    operational.extend(str(row["pose_id"]) for row in catalog["broad_training_64"])
    operational.extend(str(row["pose_id"]) for row in catalog["broad_heldout_32"])
    if len(operational) != 97 or len(set(operational)) != 97:
        raise ValueError("operational bank must contain canonical + 64 train + 32 held-out")

    role_ids = {
        role: str(visibility_selection[role]["pose_id"])
        for role in ("canonical", "strong_info", "matched_control", "blind", "look_away")
    }
    if role_ids["canonical"] != "canonical":
        raise ValueError("frozen canonical role must use the canonical pose")
    missing = sorted(set(role_ids.values()) - set(poses))
    if missing:
        raise ValueError(f"frozen diagnostic poses are absent from catalog: {missing}")

    physical = list(operational)
    for role in ("blind", "look_away"):
        if role_ids[role] not in physical:
            physical.append(role_ids[role])
    all_candidates = physical + list(SENSOR_CONTROLS)
    if len(all_candidates) != len(set(all_candidates)):
        raise ValueError("constructed Accel candidate IDs must be unique")
    diagnostic = [role_ids[role] for role in DIAGNOSTIC_ROLES[:5]]
    diagnostic.extend(SENSOR_CONTROLS)
    if len(diagnostic) != len(set(diagnostic)):
        raise ValueError("frozen diagnostic shortlist must contain distinct candidates")
    return {
        "all_candidate_ids": all_candidates,
        "all_physical_ids": physical,
        "operational_ids": operational,
        "diagnostic_ids": diagnostic,
        "role_ids": {**role_ids, **{value: value for value in SENSOR_CONTROLS}},
        "pose_index": poses,
        "group_index": groups,
        "training_pose_ids": [
            str(row["pose_id"]) for row in catalog["broad_training_64"]
        ],
        "heldout_pose_ids": [
            str(row["pose_id"]) for row in catalog["broad_heldout_32"]
        ],
    }


def subset_ranking(
    candidate_ids: Sequence[str],
    all_candidate_ids: Sequence[str],
    velocity_trace: np.ndarray,
    initial_noise: np.ndarray,
) -> dict[str, Any]:
    indices = {str(value): index for index, value in enumerate(all_candidate_ids)}
    missing = sorted(set(candidate_ids) - set(indices))
    if missing:
        raise ValueError(f"ranking subset references absent candidates: {missing}")
    selected = np.asarray([indices[str(value)] for value in candidate_ids], dtype=np.int64)
    return rank_accel_candidates(
        candidate_ids,
        np.asarray(velocity_trace)[selected],
        initial_noise=np.asarray(initial_noise)[selected],
    )


def _visibility_sensor_controls(canonical: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    external = float(canonical["per_camera"]["agentview"]["score"])
    wrist = float(canonical["per_camera"]["robot0_eye_in_hand"]["score"])
    return {
        "external_blackout": {
            "score": wrist / 2.0,
            "per_camera_scores": {"agentview": 0.0, "robot0_eye_in_hand": wrist},
        },
        "wrist_blackout": {
            "score": external / 2.0,
            "per_camera_scores": {"agentview": external, "robot0_eye_in_hand": 0.0},
        },
        "all_camera_blackout": {
            "score": 0.0,
            "per_camera_scores": {"agentview": 0.0, "robot0_eye_in_hand": 0.0},
        },
    }


def _save_diagnostic_montage(
    path: Path,
    images: Mapping[str, tuple[np.ndarray, np.ndarray]],
    candidate_ids: Sequence[str],
    roles_by_id: Mapping[str, Sequence[str]],
) -> None:
    width = 224 * 2
    height = 224 + 42
    columns = 4
    rows = int(np.ceil(len(candidate_ids) / columns))
    canvas = Image.new("RGB", (columns * width, rows * height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, candidate_id in enumerate(candidate_ids):
        agent, wrist = images[candidate_id]
        panel = np.concatenate([agent, wrist], axis=1)
        left = (index % columns) * width
        top = (index // columns) * height
        canvas.paste(Image.fromarray(np.ascontiguousarray(panel)), (left, top))
        roles = ",".join(roles_by_id.get(candidate_id, []))
        draw.text((left + 4, top + 226), f"{candidate_id}\n{roles}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_rendered_bundle(
    output_dir: Path,
    *,
    examples: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
    fixed_state_audit: Mapping[str, Any],
    images: Mapping[str, tuple[np.ndarray, np.ndarray]],
    bank: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_ids = list(bank["all_candidate_ids"])
    if [str(row["candidate_id"]) for row in metadata] != candidate_ids:
        raise ValueError("render metadata order differs from candidate bank order")
    if len(examples) != len(candidate_ids):
        raise ValueError("rendered examples and candidate bank counts differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "rendered_policy_inputs.npz",
        candidate_ids=np.asarray(candidate_ids),
        external_images=np.stack([np.asarray(row["image"][0]) for row in examples]),
        wrist_images=np.stack([np.asarray(row["image"][1]) for row in examples]),
        robot_states=np.stack([np.asarray(row["state"]) for row in examples]),
        camera_intrinsics=np.stack(
            [np.asarray(row["camera_intrinsics"]) for row in examples]
        ),
        camera_to_world_opencv=np.stack(
            [np.asarray(row["camera_to_world_opencv"]) for row in examples]
        ),
    )
    atomic_json(output_dir / "candidate_metadata.json", list(metadata))
    bank_manifest = {
        key: bank[key]
        for key in (
            "all_candidate_ids",
            "all_physical_ids",
            "operational_ids",
            "diagnostic_ids",
            "role_ids",
            "training_pose_ids",
            "heldout_pose_ids",
        )
    }
    atomic_json(output_dir / "candidate_bank.json", bank_manifest)
    _save_diagnostic_montage(
        output_dir / "diagnostic_montage.png",
        images,
        bank["diagnostic_ids"],
        {
            candidate_id: [
                role
                for role, value in bank["role_ids"].items()
                if value == candidate_id
            ]
            for candidate_id in bank["diagnostic_ids"]
        },
    )
    record = {
        "schema": "dsol_constructed_accel_render_bundle_v1",
        "status": "PASS",
        "pair_key": str(fixed_state_audit["pair_key"]),
        "fixed_state_audit": dict(fixed_state_audit),
        "candidate_count": len(candidate_ids),
        "policy_input_artifact": "rendered_policy_inputs.npz",
        "policy_input_sha256": sha256(output_dir / "rendered_policy_inputs.npz"),
        "candidate_metadata_sha256": sha256(output_dir / "candidate_metadata.json"),
        "candidate_bank_sha256": sha256(output_dir / "candidate_bank.json"),
    }
    atomic_json(output_dir / "render_record.json", record)
    return record


def load_rendered_bundle(
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    render_record = json.loads(
        (output_dir / "render_record.json").read_text(encoding="utf-8")
    )
    if render_record.get("status") != "PASS":
        raise ValueError("constructed Accel render bundle did not PASS")
    language = str(render_record["fixed_state_audit"]["language"])
    artifact = output_dir / str(render_record["policy_input_artifact"])
    if sha256(artifact) != render_record["policy_input_sha256"]:
        raise ValueError("rendered policy input checksum changed")
    with np.load(artifact, allow_pickle=False) as values:
        arrays = {key: values[key] for key in values.files}
        candidate_ids = [str(value) for value in arrays["candidate_ids"]]
        examples = [
            {
                "image": [
                    arrays["external_images"][index],
                    arrays["wrist_images"][index],
                ],
                "lang": language,
                "language": language,
                "state": arrays["robot_states"][index],
                "camera_intrinsics": arrays["camera_intrinsics"][index],
                "camera_to_world_opencv": arrays["camera_to_world_opencv"][index],
            }
            for index in range(len(candidate_ids))
        ]
    metadata = json.loads(
        (output_dir / "candidate_metadata.json").read_text(encoding="utf-8")
    )
    bank = json.loads((output_dir / "candidate_bank.json").read_text(encoding="utf-8"))
    if candidate_ids != list(bank["all_candidate_ids"]):
        raise ValueError("rendered candidate IDs differ from the frozen bank")
    return examples, metadata, render_record["fixed_state_audit"], bank


def _configure_import_paths() -> None:
    scripts = Path(__file__).resolve().parents[1]
    for path in (scripts / "dsol_paper1", scripts / "cabi_vla"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def render_state_candidates(
    spec: Mapping[str, Any],
    bank: Mapping[str, Any],
    *,
    runtime: Path,
    config_root: Path,
    render_gpu: int,
    resize_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    import h5py

    _configure_import_paths()
    from audit_libero_hdf5_restore import _configure_runtime, _decode, _rewrite_model_paths

    hdf5_path = Path(spec["hdf5"]).resolve()
    _configure_runtime(runtime, hdf5_path.parent.parent, config_root)
    from libero.libero.envs import OffScreenRenderEnv
    from libero_camera_pose import capture_camera_reference, install_camera_pose
    from libero_visibility import task_entity_visibility
    from scan_libero_hdf5_views import _install_look_away, _restore_reference
    from evaluate_pi05_libero_plus_views import (
        agentview_camera_calibration,
        physics_state_sha256,
        prepare_policy_observation,
    )

    with h5py.File(hdf5_path, "r") as handle:
        data = handle["data"]
        demo = data[str(spec["demo_name"])]
        state = np.asarray(demo["states"][int(spec["source_state_index"])])
        model_xml, rewrites = _rewrite_model_paths(_decode(demo.attrs["model_file"]), runtime)
        bddl_name = Path(_decode(data.attrs["bddl_file_name"])).name
        prompt = str(json.loads(_decode(data.attrs["problem_info"]))["language_instruction"])
    suite = str(spec["suite"])
    bddl = runtime / "libero" / "libero" / "bddl_files" / suite / bddl_name
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_names=("agentview", "robot0_eye_in_hand"),
        camera_heights=256,
        camera_widths=256,
        render_gpu_device_id=render_gpu,
    )
    examples: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    images: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    roles_by_id: dict[str, list[str]] = {}
    for role, candidate_id in bank["role_ids"].items():
        roles_by_id.setdefault(str(candidate_id), []).append(str(role))
    try:
        env.seed(seed)
        env.reset()
        env.reset_from_xml_string(model_xml)
        env.set_init_state(state)
        physics_before, physics_size = physics_state_sha256(env)
        entities = list(env.env.obj_of_interest)
        reference = capture_camera_reference(
            env,
            camera_name="agentview",
            table_plane_z=float(bank["table_plane_z"]),
        )
        canonical_visibility: Mapping[str, Any] | None = None
        canonical_policy: Mapping[str, Any] | None = None
        canonical_images: tuple[np.ndarray, np.ndarray] | None = None
        for candidate_id in bank["all_physical_ids"]:
            _restore_reference(env, reference)
            pose = bank["pose_index"][candidate_id]
            if candidate_id != "canonical":
                if pose.get("orientation_mode") == "relative_look_away":
                    _install_look_away(env, reference, pose)
                else:
                    install_camera_pose(env, reference, pose)
            env.env._update_observables(force=True)
            observation = env.env._get_observations()
            calibration = agentview_camera_calibration(env)
            policy_observation, agent, wrist = prepare_policy_observation(
                observation,
                prompt=prompt,
                resize_size=resize_size,
                eval_seed=seed,
                camera_calibration=calibration,
            )
            example = {
                "image": [
                    np.ascontiguousarray(policy_observation["observation/image"]),
                    np.ascontiguousarray(
                        policy_observation["observation/wrist_image"]
                    ),
                ],
                "lang": prompt,
                "language": prompt,
                "state": np.ascontiguousarray(
                    policy_observation["observation/state"], dtype=np.float32
                ),
                "camera_intrinsics": np.ascontiguousarray(
                    policy_observation["camera_intrinsics"]
                ),
                "camera_to_world_opencv": np.ascontiguousarray(
                    policy_observation["camera_to_world_opencv"]
                ),
            }
            visibility = task_entity_visibility(
                env,
                entity_names=entities,
                camera_names=("agentview", "robot0_eye_in_hand"),
                height=resize_size,
                width=resize_size,
            )
            camera_to_world = np.asarray(calibration["camera_to_world_opencv"])
            examples.append(example)
            images[candidate_id] = (agent, wrist)
            metadata.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_kind": "physical_view",
                    "catalog_group": bank["group_index"][candidate_id],
                    "relation_roles": roles_by_id.get(candidate_id, []),
                    "exact_training_pose": candidate_id in bank["training_pose_ids"],
                    "heldout_operational_pose": candidate_id in bank["heldout_pose_ids"],
                    "visibility_score": float(visibility["score"]),
                    "per_camera_visibility": {
                        name: float(value["score"])
                        for name, value in visibility["per_camera"].items()
                    },
                    "camera_position": camera_to_world[:3, 3].tolist(),
                    "camera_rotation_matrix": camera_to_world[:3, :3].tolist(),
                }
            )
            if candidate_id == "canonical":
                canonical_visibility = visibility
                canonical_policy = policy_observation
                canonical_images = (agent, wrist)
        if canonical_visibility is None or canonical_policy is None or canonical_images is None:
            raise RuntimeError("candidate bank did not materialize canonical observation")
        sensor_visibility = _visibility_sensor_controls(canonical_visibility)
        for candidate_id in SENSOR_CONTROLS:
            policy_observation = dict(canonical_policy)
            agent = canonical_images[0].copy()
            wrist = canonical_images[1].copy()
            if candidate_id in {"external_blackout", "all_camera_blackout"}:
                agent.fill(0)
            if candidate_id in {"wrist_blackout", "all_camera_blackout"}:
                wrist.fill(0)
            policy_observation["observation/image"] = agent
            policy_observation["observation/wrist_image"] = wrist
            example = {
                "image": [agent, wrist],
                "lang": prompt,
                "language": prompt,
                "state": np.ascontiguousarray(
                    policy_observation["observation/state"], dtype=np.float32
                ),
                "camera_intrinsics": np.ascontiguousarray(
                    policy_observation["camera_intrinsics"]
                ),
                "camera_to_world_opencv": np.ascontiguousarray(
                    policy_observation["camera_to_world_opencv"]
                ),
            }
            examples.append(example)
            images[candidate_id] = (agent, wrist)
            visibility = sensor_visibility[candidate_id]
            camera_to_world = np.asarray(policy_observation["camera_to_world_opencv"])
            metadata.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_kind": "sensor_control",
                    "catalog_group": "sensor_controls",
                    "relation_roles": roles_by_id.get(candidate_id, []),
                    "exact_training_pose": False,
                    "heldout_operational_pose": False,
                    "visibility_score": float(visibility["score"]),
                    "per_camera_visibility": visibility["per_camera_scores"],
                    "camera_position": camera_to_world[:3, 3].tolist(),
                    "camera_rotation_matrix": camera_to_world[:3, :3].tolist(),
                }
            )
        physics_after, physics_after_size = physics_state_sha256(env)
        if physics_before != physics_after or physics_size != physics_after_size:
            raise ValueError("camera candidate rendering changed the physical simulator state")
        metadata_by_id = {row["candidate_id"]: row for row in metadata}
        canonical_score = float(metadata_by_id["canonical"]["visibility_score"])
        for row in metadata:
            row["delta_visibility"] = float(row["visibility_score"] - canonical_score)
        audit = {
            "pair_key": str(spec["pair_key"]),
            "physics_state_sha256": physics_before,
            "physics_state_size": physics_size,
            "physics_state_preserved_across_all_candidates": True,
            "task_entities": entities,
            "language": prompt,
            "hdf5": str(hdf5_path),
            "demo_name": str(spec["demo_name"]),
            "source_state_index": int(spec["source_state_index"]),
            "asset_path_rewrites": rewrites,
            "candidate_count": len(examples),
            "render_gpu": int(render_gpu),
            "resize_size": int(resize_size),
        }
        return examples, metadata, audit, images
    finally:
        env.close()


def _role_rank(ranking: Mapping[str, Any], candidate_id: str) -> int:
    return next(
        int(row["rank"])
        for row in ranking["ranking"]
        if str(row["candidate_id"]) == str(candidate_id)
    )


def _accel_score(ranking: Mapping[str, Any], candidate_id: str) -> float:
    return next(
        float(row["accel_3"])
        for row in ranking["ranking"]
        if str(row["candidate_id"]) == str(candidate_id)
    )


def _rank_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    left_ranks = np.argsort(np.argsort(left_values, kind="stable"), kind="stable")
    right_ranks = np.argsort(np.argsort(right_values, kind="stable"), kind="stable")
    if float(np.std(left_ranks)) == 0.0 or float(np.std(right_ranks)) == 0.0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _candidate_category(metadata: Mapping[str, Any]) -> str:
    if metadata["candidate_id"] == "canonical":
        return "canonical"
    if metadata["candidate_kind"] == "sensor_control":
        return "sensor_control"
    if bool(metadata["exact_training_pose"]):
        return "broad64_training_support"
    if bool(metadata["heldout_operational_pose"]):
        return "broad32_heldout"
    return "diagnostic_physical"


def _mean_present(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return None if not present else float(np.mean(present))


def evaluate_state(
    spec: Mapping[str, Any],
    *,
    model: Any,
    output_dir: Path,
    device: str,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    state_seed = stable_seed(str(spec["pair_key"]), seed=seed)
    examples, metadata, fixed_state_audit, bank = load_rendered_bundle(output_dir)
    candidate_ids = list(bank["all_candidate_ids"])
    if len(examples) != len(candidate_ids):
        raise ValueError("rendered examples and frozen candidate IDs differ")
    started = time.perf_counter()
    with torch.inference_mode():
        result = rank_fixed_state_candidates_chunked(
            model,
            examples,
            candidate_ids,
            seed=state_seed,
            action_horizon=int(model.action_horizon),
            action_dim=int(model.action_dim),
            batch_size=batch_size,
            include_trace_artifacts=True,
        )
    inference_seconds = time.perf_counter() - started
    trace = np.asarray(result.pop("flow_velocity_trace"), dtype=np.float32)
    noise = np.asarray(result.pop("flow_initial_noise"), dtype=np.float32)
    actions = np.asarray(result.pop("actions"), dtype=np.float32)
    rankings = {
        "complete": result,
        "all_physical": subset_ranking(
            bank["all_physical_ids"], candidate_ids, trace, noise
        ),
        "operational_97": subset_ranking(
            bank["operational_ids"], candidate_ids, trace, noise
        ),
        "diagnostic_shortlist": subset_ranking(
            bank["diagnostic_ids"], candidate_ids, trace, noise
        ),
    }
    metadata_by_id = {row["candidate_id"]: row for row in metadata}
    role_ids = bank["role_ids"]
    role_metrics = {
        role: {
            "candidate_id": candidate_id,
            "complete_rank": _role_rank(rankings["complete"], candidate_id),
            "diagnostic_rank": _role_rank(rankings["diagnostic_shortlist"], candidate_id),
            "visibility_score": metadata_by_id[candidate_id]["visibility_score"],
            "delta_visibility": metadata_by_id[candidate_id]["delta_visibility"],
        }
        for role, candidate_id in role_ids.items()
    }
    complete_scores = {
        str(row["candidate_id"]): float(row["accel_3"])
        for row in rankings["complete"]["ranking"]
    }
    operational_visibility = [
        float(metadata_by_id[candidate_id]["visibility_score"])
        for candidate_id in bank["operational_ids"]
    ]
    operational_accel = [
        complete_scores[candidate_id] for candidate_id in bank["operational_ids"]
    ]
    selected_categories = {
        name: _candidate_category(metadata_by_id[str(ranking["selected_candidate_id"])])
        for name, ranking in rankings.items()
    }
    strong_id = str(role_ids["strong_info"])
    control_id = str(role_ids["matched_control"])
    canonical_id = str(role_ids["canonical"])
    relation_diagnostics = {
        "operational_accel_visibility_spearman": _rank_correlation(
            operational_accel, operational_visibility
        ),
        "strong_info_minus_matched_control_accel_3": (
            complete_scores[strong_id] - complete_scores[control_id]
        ),
        "strong_info_lower_accel_than_matched_control": bool(
            complete_scores[strong_id] < complete_scores[control_id]
        ),
        "strong_info_minus_canonical_accel_3": (
            complete_scores[strong_id] - complete_scores[canonical_id]
        ),
        "strong_info_lower_accel_than_canonical": bool(
            complete_scores[strong_id] < complete_scores[canonical_id]
        ),
    }
    np.savez_compressed(
        output_dir / "flow_trace.npz",
        candidate_ids=np.asarray(candidate_ids),
        velocity_trace=trace,
        initial_noise=noise,
        normalized_actions=actions,
        flow_times=np.asarray(result["flow_times"], dtype=np.float32),
    )
    atomic_json(output_dir / "rankings.json", rankings)
    record = {
        "schema": "dsol_constructed_accel_fixed_state_v1",
        "status": "PASS",
        "pair_key": str(spec["pair_key"]),
        "task_id": str(spec["task_id"]),
        "source_episode_id": str(spec["episode_id_source"]),
        "source_state_index": int(spec["source_state_index"]),
        "state_seed": state_seed,
        "device": device,
        "render_gpu": int(fixed_state_audit["render_gpu"]),
        "fixed_state_audit": fixed_state_audit,
        "candidate_counts": {
            "complete": len(candidate_ids),
            "all_physical": len(bank["all_physical_ids"]),
            "operational_97": len(bank["operational_ids"]),
            "diagnostic_shortlist": len(bank["diagnostic_ids"]),
        },
        "selected_candidates": {
            name: ranking["selected_candidate_id"] for name, ranking in rankings.items()
        },
        "selected_candidate_categories": selected_categories,
        "role_metrics": role_metrics,
        "relation_diagnostics": relation_diagnostics,
        "inference_seconds": inference_seconds,
        "closed_loop_oracle_relation": "DEFERRED_UNTIL_CONSTRUCTED_M1_COMPLETES",
    }
    atomic_json(output_dir / "rank_record.json", record)
    return record


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty Accel run")
    selected = {
        key: Counter(str(row["selected_candidates"][key]) for row in records)
        for key in records[0]["selected_candidates"]
    }
    selected_categories = {
        key: Counter(str(row["selected_candidate_categories"][key]) for row in records)
        for key in records[0]["selected_candidate_categories"]
    }
    roles = records[0]["role_metrics"].keys()
    role_summary = {}
    for role in roles:
        complete_ranks = [int(row["role_metrics"][role]["complete_rank"]) for row in records]
        diagnostic_ranks = [
            int(row["role_metrics"][role]["diagnostic_rank"]) for row in records
        ]
        role_summary[role] = {
            "mean_complete_rank": float(np.mean(complete_ranks)),
            "median_complete_rank": float(np.median(complete_ranks)),
            "mean_diagnostic_rank": float(np.mean(diagnostic_ranks)),
            "diagnostic_top1_rate": float(np.mean(np.asarray(diagnostic_ranks) == 1)),
            "mean_delta_visibility": float(
                np.mean([row["role_metrics"][role]["delta_visibility"] for row in records])
            ),
        }
    return {
        "schema": "dsol_constructed_accel_summary_v1",
        "status": "PASS_DESCRIPTIVE_RELATION_ANALYSIS",
        "state_count": len(records),
        "task_counts": dict(Counter(str(row["task_id"]) for row in records)),
        "fixed_state_audit_pass_count": sum(
            bool(row["fixed_state_audit"]["physics_state_preserved_across_all_candidates"])
            for row in records
        ),
        "selected_candidate_counts": {
            key: dict(value) for key, value in selected.items()
        },
        "selected_candidate_category_counts": {
            key: dict(value) for key, value in selected_categories.items()
        },
        "relation_diagnostics": {
            "mean_operational_accel_visibility_spearman": _mean_present(
                [
                    row["relation_diagnostics"][
                        "operational_accel_visibility_spearman"
                    ]
                    for row in records
                ]
            ),
            "strong_info_lower_accel_than_matched_control_rate": float(
                np.mean(
                    [
                        row["relation_diagnostics"][
                            "strong_info_lower_accel_than_matched_control"
                        ]
                        for row in records
                    ]
                )
            ),
            "strong_info_lower_accel_than_canonical_rate": float(
                np.mean(
                    [
                        row["relation_diagnostics"][
                            "strong_info_lower_accel_than_canonical"
                        ]
                        for row in records
                    ]
                )
            ),
        },
        "role_summary": role_summary,
        "claim_scope": (
            "Accel familiarity/information relation on frozen states; no closed-loop "
            "view-value claim until constructed M1 oracle outcomes are joined."
        ),
        "closed_loop_oracle_relation": "DEFERRED_UNTIL_CONSTRUCTED_M1_COMPLETES",
    }


def selected_state_specs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("status") != "PASS":
        raise ValueError("constructed M1 protocol must be a frozen PASS artifact")
    specs = [row for row in protocol["specs"] if row["condition"] == "canonical_both"]
    if len(specs) != int(protocol["selected_state_count"]):
        raise ValueError("protocol must contain exactly one canonical spec per selected state")
    return sorted((dict(row) for row in specs), key=lambda row: str(row["pair_key"]))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("render", "rank"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--render-gpu", type=int, default=7)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--max-states", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    catalog_path = Path(protocol["catalog"])
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    specs = selected_state_specs(protocol)
    if args.max_states is not None:
        if args.max_states < 1:
            raise ValueError("max-states must be positive")
        specs = specs[: args.max_states]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    states_root = args.output_dir / "states"
    state_dirs = {
        str(spec["pair_key"]): states_root
        / hashlib.sha256(str(spec["pair_key"]).encode()).hexdigest()[:16]
        for spec in specs
    }

    if args.stage == "render":
        if args.runtime is None or args.config_root is None:
            raise ValueError("render stage requires --runtime and --config-root")
        completed = 0
        for index, spec in enumerate(specs, start=1):
            state_dir = state_dirs[str(spec["pair_key"])]
            record_path = state_dir / "render_record.json"
            if record_path.is_file():
                row = json.loads(record_path.read_text(encoding="utf-8"))
                if row.get("status") == "PASS":
                    completed += 1
                    continue
            bank = build_candidate_bank(catalog, spec["visibility_selection"])
            bank["table_plane_z"] = float(catalog["table_plane_z"])
            state_seed = stable_seed(str(spec["pair_key"]), seed=args.seed)
            examples, metadata, fixed_state_audit, images = render_state_candidates(
                spec,
                bank,
                runtime=args.runtime.resolve(),
                config_root=args.config_root.resolve(),
                render_gpu=args.render_gpu,
                resize_size=args.resize_size,
                seed=state_seed,
            )
            row = save_rendered_bundle(
                state_dir,
                examples=examples,
                metadata=metadata,
                fixed_state_audit=fixed_state_audit,
                images=images,
                bank=bank,
            )
            completed += 1
            print(
                json.dumps(
                    {
                        "rendered": index,
                        "state_count": len(specs),
                        "pair_key": row["pair_key"],
                        "candidate_count": row["candidate_count"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        render_summary = {
            "schema": "dsol_constructed_accel_render_summary_v1",
            "status": "PASS" if completed == len(specs) else "FAIL",
            "state_count": len(specs),
            "completed_state_count": completed,
            "candidate_count_per_state": 102,
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": sha256(args.protocol),
            "catalog": str(catalog_path.resolve()),
            "catalog_sha256": sha256(catalog_path),
            "runtime": str(args.runtime.resolve()),
            "config_root": str(args.config_root.resolve()),
            "render_gpu": args.render_gpu,
            "resize_size": args.resize_size,
            "seed": args.seed,
        }
        atomic_json(args.output_dir / "render_summary.json", render_summary)
        print(json.dumps(render_summary, sort_keys=True), flush=True)
        return

    if args.checkpoint is None:
        raise ValueError("rank stage requires --checkpoint")
    existing: dict[str, dict[str, Any]] = {}
    for path in states_root.glob("*/rank_record.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") == "PASS":
            existing[str(row["pair_key"])] = row
    missing_renders = [
        pair_key
        for pair_key, state_dir in state_dirs.items()
        if not (state_dir / "render_record.json").is_file()
    ]
    if missing_renders:
        raise ValueError(
            f"rank stage is missing {len(missing_renders)} rendered state bundles"
        )
    pending = [row for row in specs if str(row["pair_key"]) not in existing]

    load_seconds = 0.0
    peak_memory = None
    if pending:
        import torch
        from AlphaBrain.model.framework.base_framework import BaseFramework

        if args.device.startswith("cuda:"):
            torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        model = BaseFramework.from_pretrained(
            str(args.checkpoint.resolve()), strict_checkpoint=True
        )
        model = model.to(torch.bfloat16).to(args.device).eval()
        model.gripper_remap = False
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        load_seconds = time.perf_counter() - started
        if int(model.action_dim) != 7:
            raise ValueError(f"expected LIBERO action_dim=7, got {model.action_dim}")
        for index, spec in enumerate(pending, start=1):
            state_dir = state_dirs[str(spec["pair_key"])]
            state_dir.mkdir(parents=True, exist_ok=True)
            row = evaluate_state(
                spec,
                model=model,
                output_dir=state_dir,
                device=args.device,
                seed=args.seed,
                batch_size=args.batch_size,
            )
            existing[str(row["pair_key"])] = row
            print(
                json.dumps(
                    {
                        "completed": index,
                        "pending_total": len(pending),
                        "pair_key": row["pair_key"],
                        "selected": row["selected_candidates"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.device.startswith("cuda:"):
            peak_memory = int(torch.cuda.max_memory_allocated(args.device))

    records = [existing[str(spec["pair_key"])] for spec in specs]
    summary = summarize(records)
    summary.update(
        {
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": sha256(args.protocol),
            "catalog": str(catalog_path.resolve()),
            "catalog_sha256": sha256(catalog_path),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_weights_sha256": sha256(args.checkpoint / "model.safetensors"),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "seed": args.seed,
            "device": args.device,
            "batch_size": args.batch_size,
            "model_load_seconds_this_invocation": load_seconds,
            "cuda_peak_memory_bytes_this_invocation": peak_memory,
        }
    )
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "manifest.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
