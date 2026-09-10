from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path("/alphabrain")
SHARE_ROOT = Path("/share/longjunyu/alphabrain")
DEFAULT_OUTPUT = SHARE_ROOT / "handoffs" / "view-active-materials-8gpu-20260809"
ALLOWED_SUFFIXES = {".json", ".jsonl", ".csv", ".yaml", ".yml", ".py", ".sh", ".txt"}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".mp4",
    ".pth",
    ".pt",
    ".safetensors",
    ".tar",
    ".webm",
    ".zip",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class BundleBuilder:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.sources: list[dict[str, Any]] = []

    def copy(self, source: Path, relative: Path | str, *, kind: str) -> Path:
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        suffix = source.suffix.lower()
        if suffix not in ALLOWED_SUFFIXES or suffix in FORBIDDEN_SUFFIXES:
            raise ValueError(f"refusing non-text artifact: {source}")
        destination = self.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copy2(source, destination)
        self.sources.append(
            {
                "kind": kind,
                "source": str(source),
                "bundled_path": str(destination.relative_to(self.output)),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
        return destination

    def copy_run_records(self, source: Path, relative: Path) -> None:
        for path in sorted(source.glob("episodes-shard-*.jsonl")):
            self.copy(path, relative / path.name, kind="raw_episode_records")
        for name in ("run_manifest.json", "metrics.json"):
            path = source / name
            if path.is_file():
                self.copy(path, relative / name, kind="run_metadata")

    def finalize_index(self) -> None:
        rows = sorted(self.sources, key=lambda row: row["bundled_path"])
        write_csv(
            self.output / "source_index.csv",
            rows,
            ["kind", "source", "bundled_path", "bytes", "sha256"],
        )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl_files(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.rstrip("\n")


def budget_rows(episodes: Sequence[Mapping[str, Any]], fraction: float) -> list[Mapping[str, Any]]:
    return [
        row
        for row in episodes
        if row["split"] == "train" and float(row["budget_percentile"]) <= fraction
    ]


def build_multiview_tables(output: Path, manifest: Mapping[str, Any]) -> None:
    inventory = []
    for row in manifest["episodes"]:
        inventory.append(
            {
                "episode_id": row["episode_id"],
                "split": row["split"],
                "language_instruction": row["language_instruction"],
                "step_count": int(row["step_count"]),
                "camera_pose_group_id": row["camera_pose_group_id"],
                "budget_percentile": float(row["budget_percentile"]),
                "in_train_25pct_pool": bool(
                    row["split"] == "train" and float(row["budget_percentile"]) <= 0.25
                ),
                "in_train_100pct_pool": bool(row["split"] == "train"),
            }
        )
    write_csv(
        output / "multiview" / "episode_inventory.csv",
        inventory,
        [
            "episode_id",
            "split",
            "language_instruction",
            "step_count",
            "camera_pose_group_id",
            "budget_percentile",
            "in_train_25pct_pool",
            "in_train_100pct_pool",
        ],
    )

    groups: dict[str, dict[str, Any]] = {}
    for row in manifest["episodes"]:
        group = groups.setdefault(
            str(row["camera_pose_group_id"]),
            {
                "camera_pose_group_id": row["camera_pose_group_id"],
                "matrix": np.asarray(row["camera_to_world_opencv"], dtype=np.float64),
                "episode_count": 0,
                "step_count": 0,
                "train_25_episode_count": 0,
                "train_25_step_count": 0,
                "train_100_episode_count": 0,
                "train_100_step_count": 0,
                "splits": Counter(),
            },
        )
        group["episode_count"] += 1
        group["step_count"] += int(row["step_count"])
        group["splits"][str(row["split"])] += 1
        if row["split"] == "train":
            group["train_100_episode_count"] += 1
            group["train_100_step_count"] += int(row["step_count"])
            if float(row["budget_percentile"]) <= 0.25:
                group["train_25_episode_count"] += 1
                group["train_25_step_count"] += int(row["step_count"])

    pose_rows = []
    for group_id, group in sorted(groups.items()):
        matrix = group.pop("matrix")
        pose_rows.append(
            {
                "camera_pose_group_id": group_id,
                "camera_x": float(matrix[0, 3]),
                "camera_y": float(matrix[1, 3]),
                "camera_z": float(matrix[2, 3]),
                "rotation_flat_json": json.dumps(matrix[:3, :3].reshape(-1).tolist()),
                "episode_count": group["episode_count"],
                "step_count": group["step_count"],
                "train_episode_count": group["splits"].get("train", 0),
                "val_episode_count": group["splits"].get("val", 0),
                "test_episode_count": group["splits"].get("test", 0),
                "train_25_episode_count": group["train_25_episode_count"],
                "train_25_step_count": group["train_25_step_count"],
                "train_100_episode_count": group["train_100_episode_count"],
                "train_100_step_count": group["train_100_step_count"],
            }
        )
    write_csv(
        output / "multiview" / "pose_groups.csv",
        pose_rows,
        list(pose_rows[0]),
    )

    exposure = {}
    total_samples = 33_000 * 2
    for label, fraction in (("25pct", 0.25), ("100pct", 1.0)):
        selected = budget_rows(manifest["episodes"], fraction)
        windows = sum(int(row["step_count"]) for row in selected)
        exposure[label] = {
            "budget_fraction": fraction,
            "train_episode_count": len(selected),
            "train_pose_group_count": len({row["camera_pose_group_id"] for row in selected}),
            "sliding_window_count": windows,
            "action_horizon": 10,
            "optimizer_steps": 33_000,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "training_sample_draws": total_samples,
            "expected_full_window_exposure": total_samples / windows,
        }
    write_json(
        output / "multiview" / "sampling_and_exposure.json",
        {
            "data_relation": "unpaired demonstrations across camera poses",
            "camera_assignment": "one fixed external camera pose per source episode",
            "camera_transition_within_episode": False,
            "window_policy": "all episode frames are indexed as sliding-window anchors; shuffled by the training dataloader",
            "terminal_padding": "zero-pad the action chunk to horizon 10",
            "budget_policy": "budget_fraction filters train episodes by deterministic per-task budget_percentile; validation and test are not training inputs",
            "exposure": exposure,
        },
    )


def last_jsonl_row(path: Path) -> dict[str, Any]:
    last = ""
    with path.open() as stream:
        for line in stream:
            if line.strip():
                last = line
    return json.loads(last) if last else {}


def build_training_run_table(output: Path) -> None:
    roots = [
        SHARE_ROOT / "experiments/libero-plus-mv-rgb-v1/runs",
        SHARE_ROOT / "experiments/libero-plus-kyc-factor-separated-v1/runs",
    ]
    eval_roots = [
        SHARE_ROOT / "experiments/libero-plus-mv-rgb-v1/gate-v1",
        SHARE_ROOT / "experiments/libero-plus-kyc-matched-v1",
        SHARE_ROOT / "experiments/libero-plus-kyc-factor-separated-v1",
    ]
    checkpoint_meta: dict[str, dict[str, Any]] = {}
    for root in eval_roots:
        for path in root.glob("**/run_manifest.json"):
            payload = read_json(path)
            checkpoint = payload.get("checkpoint")
            if checkpoint:
                checkpoint_meta[str(Path(checkpoint))] = payload

    rows = []
    for root in roots:
        for metrics in sorted(root.glob("*/metrics.jsonl")):
            run_id = metrics.parent.name
            if "smoke" in run_id or "calibrate" in run_id or "steps33000" not in run_id:
                continue
            match = re.search(r"_seed(?P<seed>\d+)_steps(?P<steps>\d+)$", run_id)
            if not match:
                continue
            final_model = metrics.parent / "final_model"
            meta = checkpoint_meta.get(str(final_model), {})
            final_metrics = last_jsonl_row(metrics)
            if "factor-separated" in run_id:
                data_view = "pi05-goal-factor-separated-v1"
                budget_fraction = 1.0
            else:
                data_view = "pi05-mv-rgb-v1"
                budget_fraction = 0.25 if "b025" in run_id or "matched" in run_id else 1.0
            method = "unknown"
            for candidate in ("action_only", "visual_lora_control", "visual_lora_kyc", "visual_lora"):
                if candidate in run_id:
                    method = candidate
                    break
            rows.append(
                {
                    "run_id": run_id,
                    "method": method,
                    "training_seed": int(match.group("seed")),
                    "optimizer_steps": int(match.group("steps")),
                    "data_view": data_view,
                    "budget_fraction": budget_fraction,
                    "checkpoint_available_on_8gpu": (final_model / "model.safetensors").is_file(),
                    "checkpoint_source_path_not_bundled": str(final_model),
                    "checkpoint_sha256": meta.get("checkpoint_sha256", "not_recorded"),
                    "last_logged_step": final_metrics.get("step", ""),
                    "last_action_dit_loss": final_metrics.get("action_dit_loss", ""),
                    "metrics_source": str(metrics),
                }
            )
    write_csv(output / "multiview" / "training_runs.csv", rows, list(rows[0]))

    write_json(
        output / "multiview" / "action_only_vs_visual_lora.json",
        {
            "shared_inputs": {
                "external_rgb": True,
                "wrist_rgb": True,
                "third_image_slot_masked": True,
                "robot_state_input": False,
                "language_input": True,
                "action_horizon": 10,
            },
            "action_only": {
                "freeze_modules": "vlm_interface",
                "visual_adapter": "none",
                "trainable_parameters_reported": 693_370_000,
            },
            "visual_lora": {
                "freeze_modules": "language_model,lm_head,multi_modal_projector",
                "visual_adapter": "SigLIP MLP fc1/fc2 low-rank adapter, rank=16, alpha=16, dropout=0",
                "visual_adapter_parameters_reported": 4_713_984,
                "trainable_parameters_reported": 698_080_000,
            },
            "canonical_control": "same RGB, capacity, training order, and camera branch as KYC, but every external image receives the canonical fixed ray map",
            "kyc": "same as canonical control except each external image receives the calibrated ray map aligned to its current camera pose",
            "online_image_augmentation": "no random crop in the recorded runs (joint_crop_min_scale=1.0); multiview variation comes from rendered source episodes rather than per-window camera transitions",
        },
    )


def flatten_composition_tasks(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = [
        "suite",
        "base_task",
        "task_index",
        "task_id",
        "difficulty_level",
        "camera_task_name",
        "camera_difficulty_level",
        "perturbation_family",
        "orbit_yaw_deg",
        "orbit_pitch_deg",
        "radius_percent",
        "look_yaw_deg",
        "look_pitch_deg",
        "background_task_name",
        "background_kind",
        "background_texture_index",
        "background_difficulty_level",
        "camera_background_task_name",
    ]
    return [{field: row.get(field, "") for field in fields} for row in protocol["composition_tasks"]]


def build_composition_tables(output: Path, protocol: Mapping[str, Any], factor_manifest: Mapping[str, Any]) -> None:
    tasks36 = flatten_composition_tasks(protocol)
    write_csv(output / "composition" / "tasks_36.csv", tasks36, list(tasks36[0]))
    tasks9 = []
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for row in factor_manifest["episodes"]:
        by_task[str(row["task_id"])][str(row["factor_class"])] += 1
    for task, counts in sorted(by_task.items()):
        tasks9.append(
            {
                "task_id": task,
                "camera_only_episode_count": counts.get("camera_only", 0),
                "background_only_episode_count": counts.get("background_only", 0),
                "joint_camera_background_episode_count": counts.get("camera_background", 0),
            }
        )
    write_csv(output / "composition" / "tasks_9.csv", tasks9, list(tasks9[0]))
    write_json(
        output / "composition" / "factor_definition.json",
        {
            "camera": "episode-fixed external camera extrinsics; intrinsics, resolution, robot and wrist camera are held fixed",
            "background": "LIBERO-Plus tb_N table/background appearance texture variant",
            "object_identity_changed": False,
            "object_layout_changed": False,
            "lighting_changed": False,
            "new_kitchen_geometry_tested": False,
            "paired_physics_verified": True,
            "joint_ood_36_task_training_pairing": "training contains multiview camera episodes but no explicit background factor; the joint camera+texture condition is evaluation-only",
            "factor_separated_9_task_training_pairing": "camera-only and background-only categories are both present; no joint category is used for training",
            "exact_pose_texture_pair_identity_available": False,
            "strict_exact_held_pair_claim_allowed": False,
        },
    )


def flatten_ray_response(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in payload["rows"]:
        common = {
            "edge_id": row["edge_id"],
            "canonical_state_index": row["canonical_state_index"],
            "visual_pose": row["visual_pose"],
            "mismatched_ray_pose": row["mismatched_ray_pose"],
            "agent_sha256": row["agent_sha256"],
            "wrist_sha256": row["wrist_sha256"],
        }
        for comparison in ("canonical_vs_correct", "mismatched_vs_correct"):
            rows.append({**common, "comparison": comparison, **row[comparison]})
    return rows


def build_kyc_tables(output: Path, ray_payload: Mapping[str, Any]) -> None:
    rows = flatten_ray_response(ray_payload)
    write_csv(output / "kyc" / "fixed_checkpoint_ray_response.csv", rows, list(rows[0]))
    coverage = [
        {
            "intervention": "external_correct",
            "available": True,
            "evidence": "fixed RGB/state/noise action-difference diagnostic",
            "notes": "aggregate action differences retained; raw action vectors and intermediate features were not retained",
        },
        {
            "intervention": "external_canonical",
            "available": True,
            "evidence": "fixed RGB/state/noise action-difference diagnostic",
            "notes": "canonical ray substituted at the same checkpoint",
        },
        {
            "intervention": "external_mismatched",
            "available": True,
            "evidence": "fixed RGB/state/noise action-difference diagnostic",
            "notes": "deterministic next-pose mismatch; not a random shuffle",
        },
        {
            "intervention": "external_shuffled",
            "available": False,
            "evidence": "none",
            "notes": "not run",
        },
        {
            "intervention": "external_delayed",
            "available": False,
            "evidence": "none",
            "notes": "not run for the external camera",
        },
        {
            "intervention": "external_noisy",
            "available": False,
            "evidence": "none",
            "notes": "not run",
        },
        {
            "intervention": "wrist_correct",
            "available": True,
            "evidence": "closed-loop dual-camera causal intervention",
            "notes": "current wrist calibration",
        },
        {
            "intervention": "wrist_initial_fixed",
            "available": True,
            "evidence": "closed-loop dual-camera causal intervention",
            "notes": "episode-initial wrist ray held fixed",
        },
        {
            "intervention": "wrist_delayed",
            "available": True,
            "evidence": "closed-loop dual-camera causal intervention",
            "notes": "previous policy-call wrist ray with K=3 execution",
        },
    ]
    write_csv(output / "kyc" / "intervention_coverage.csv", coverage, list(coverage[0]))
    write_json(
        output / "kyc" / "ray_convention.json",
        {
            "camera_extrinsics": "OpenCV camera-to-world: +x right, +y down, +z forward",
            "plucker_channels": "[unit_world_direction, camera_origin cross unit_world_direction]",
            "tensor_shape": "[batch, 6, height, width]",
            "policy_image_transform": "MuJoCo input is rotated 180 degrees",
            "ray_alignment_transform": "horizontal flip for mujoco_upright after the policy image transform",
            "conditioned_views": {
                "matched_libero_plus": ["external"],
                "dual_camera_screen": ["external", "wrist", "both", "neither controls"],
            },
        },
    )
    write_json(
        output / "kyc" / "act_vs_pi05_conditions.json",
        {
            "official_act_positive_control": {
                "external_camera": True,
                "wrist_camera": False,
                "model_family": "ACT",
                "training_seeds": [0, 1, 2],
                "held_out_camera_image_success": 0.24666666666666667,
                "held_out_camera_kyc_success": 0.6266666666666666,
            },
            "pi05_primary_gate": {
                "external_camera": True,
                "wrist_camera": True,
                "model_family": "Pi0.5 flow-matching VLA",
                "training_seeds": [41, 42, 43],
                "geometry_injection": "separate ray CNN fused into frozen/pretrained visual tokens",
                "primary_comparison": "real aligned ray versus canonical-ray capacity control",
            },
            "interpretation": "ACT is a positive control for the KYC mechanism, not an architecture-matched estimate of Pi0.5 geometry gain",
        },
    )


def flatten_candidate_rows(policy: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        if not str(row["condition"]).startswith("candidate:"):
            continue
        metrics = row["initial_metrics"]
        visibility = metrics["sim_visibility"]
        action_probe = metrics["action_probe"]
        agent = metrics["agent"]
        flattened.append(
            {
                "policy": policy,
                "pair_key": row["pair_key"],
                "suite": row["suite"],
                "base_task": row["base_task"],
                "init_state_index": row["init_state_index"],
                "episode_seed": row["episode_seed"],
                "candidate_view": str(row["condition"]).split(":", 1)[1],
                "success": bool(row["success"]),
                "completion_steps": row["completion_steps"],
                "action_mean_pairwise_rms": action_probe["mean_pairwise_rms"],
                "action_mean_variance": action_probe["mean_variance"],
                "image_entropy_32bin_bits": agent["entropy_32bin_bits"],
                "image_mean_edge_strength": agent["mean_edge_strength"],
                "image_rgb_std": agent["rgb_std"],
                "image_clipped_fraction": agent["clipped_fraction"],
                "all_interest_visible": visibility["all_interest_visible"],
                "all_interest_visible_at_least_16px": visibility[
                    "all_interest_visible_at_least_16px"
                ],
                "any_interest_border_touch": visibility["any_interest_border_touch"],
                "minimum_interest_pixel_count": visibility["minimum_interest_pixel_count"],
                "objects_json": json.dumps(visibility["objects"], sort_keys=True),
            }
        )
    return flattened


def build_active_view_tables(
    output: Path,
    protocol: Mapping[str, Any],
    official_rows: Sequence[Mapping[str, Any]],
    strong_rows: Sequence[Mapping[str, Any]],
) -> None:
    rows = flatten_candidate_rows("official_pi05", official_rows)
    rows += flatten_candidate_rows("strong_multiview_visual_lora_25pct", strong_rows)
    write_csv(output / "active_view" / "state_candidate_policy_matrix.csv", rows, list(rows[0]))

    declared_ranges = {
        "orbit_yaw_deg": (-60.0, 60.0),
        "orbit_pitch_deg": (-25.0, 25.0),
        "radius_percent": (90.0, 125.0),
    }
    support_rows = []
    for view in protocol["candidate_views"]:
        in_range = all(
            declared_ranges[key][0] <= float(view[key]) <= declared_ranges[key][1]
            for key in declared_ranges
        )
        support_rows.append(
            {
                "candidate_view": view["name"],
                "orbit_yaw_deg": view["orbit_yaw_deg"],
                "orbit_pitch_deg": view["orbit_pitch_deg"],
                "radius_percent": view["radius_percent"],
                "inside_declared_training_parameter_range": in_range,
                "exact_pose_identity_available": False,
                "support_interpretation": "range-supported; task-specific exact pose matching is not established",
            }
        )
    write_csv(output / "active_view" / "candidate_support.csv", support_rows, list(support_rows[0]))
    write_json(
        output / "active_view" / "selector_and_label_coverage.json",
        {
            "selector_scores_available": [
                "initial RGB entropy/edge/std/clipping heuristic",
                "three-sample flow-action mean pairwise RMS and variance",
            ],
            "oracle_definition": "post-hoc success if any available candidate view succeeded for the same task and init-state group",
            "visibility_scope": "initial observation only",
            "dynamic_camera_motion": False,
            "camera_transition_within_episode": False,
            "same_state_multiple_camera_candidates": True,
            "same_state_different_task_labels": False,
            "same_state_different_stage_labels": False,
            "stage_conditioned_view_utility_available": False,
            "candidate_execution": "each candidate is a separate full closed-loop episode from a paired initial state; the selected camera remains fixed",
        },
    )


def copy_inputs(builder: BundleBuilder) -> dict[str, Any]:
    dataset_views = SHARE_ROOT / "datasets/libero-plus/views"
    mv_manifest = dataset_views / "pi05-mv-rgb-v1/manifest.json"
    factor_manifest = dataset_views / "pi05-goal-factor-separated-v1/manifest.json"
    builder.copy(mv_manifest, "multiview/manifests/pi05-mv-rgb-v1.json", kind="dataset_manifest")
    builder.copy(
        factor_manifest,
        "composition/manifests/pi05-goal-factor-separated-v1.json",
        kind="dataset_manifest",
    )

    config_files = [
        REPO_ROOT / "configs/experiments/pi05_libero_plus_multiview.yaml",
        REPO_ROOT / "configs/experiments/kyc_libero_bind.yaml",
        REPO_ROOT / "docs/cabi_vla/configs/camera_pose_train_random_v2.json",
        REPO_ROOT / "docs/cabi_vla/configs/camera_pose_train_global_scaling_v3.json",
        REPO_ROOT / "docs/cabi_vla/configs/camera_pose_policy_gate_v6.json",
        REPO_ROOT / "docs/cabi_vla/configs/libero_wrist_hand_eye_v1.json",
    ]
    for path in config_files:
        builder.copy(path, Path("configs") / path.name, kind="config")

    script_files = [
        "AlphaBrain/dataloader/paligemma_datasets.py",
        "AlphaBrain/model/framework/PaliGemmaPi.py",
        "AlphaBrain/model/modules/vlm/camera_conditioning.py",
        "scripts/cabi_vla/build_libero_plus_training_view.py",
        "scripts/cabi_vla/build_libero_plus_factor_separated_view.py",
        "scripts/cabi_vla/build_libero_plus_view_protocol.py",
        "scripts/cabi_vla/build_libero_plus_composition_protocol.py",
        "scripts/cabi_vla/evaluate_pi05_libero_plus_views.py",
        "scripts/cabi_vla/analyze_pi05_libero_plus_views.py",
        "scripts/cabi_vla/analyze_pi05_libero_plus_composition.py",
        "scripts/cabi_vla/analyze_pi05_libero_plus_kyc_matched.py",
        "scripts/cabi_vla/audit_pi05_libero_plus_composition_isolation.py",
        "scripts/cabi_vla/diagnose_kyc_ray_use.py",
        "scripts/cabi_vla/validate_libero_plus_ray_alignment.py",
        "scripts/cabi_vla/evaluate_libero_bind_camera_viewpoints.py",
        "scripts/cabi_vla/summarize_kyc_dual_camera_diagnostics.py",
        "scripts/cabi_vla/run_pi05_libero_plus_multiview_train.sh",
        "scripts/cabi_vla/run_kyc_train.sh",
        "scripts/cabi_vla/build_8gpu_view_handoff_bundle.py",
    ]
    for relative in script_files:
        builder.copy(REPO_ROOT / relative, Path("scripts") / relative, kind="code")

    view_root = SHARE_ROOT / "experiments/libero-plus-view-gap-v1"
    mv_root = SHARE_ROOT / "experiments/libero-plus-mv-rgb-v1"
    composition_root = SHARE_ROOT / "experiments/libero-plus-camera-background-v1"
    matched_root = SHARE_ROOT / "experiments/libero-plus-kyc-matched-v1"
    factor_root = SHARE_ROOT / "experiments/libero-plus-kyc-factor-separated-v1"

    metadata_files = [
        (view_root / "protocol-v3.json", "active_view/protocol-v3.json"),
        (composition_root / "protocol-v1.json", "composition/protocol-v1.json"),
        (composition_root / "final/isolation_audit.json", "composition/isolation_audit.json"),
        (composition_root / "final-joint-ood-v2/metrics.json", "composition/joint_ood_metrics.json"),
        (matched_root / "final/metrics.json", "composition/kyc_matched_metrics.json"),
        (factor_root / "final/metrics.json", "composition/kyc_factor_separated_metrics.json"),
        (mv_root / "gate-v1/comparison/metrics.json", "multiview/gate_comparison_metrics.json"),
        (mv_root / "gate-v1/final-best/metrics.json", "active_view/strong_multiview_active_metrics.json"),
        (view_root / "pi05-libero-official-v1/metrics.json", "active_view/official_pi05_active_metrics.json"),
        (matched_root / "diagnostics/ray_alignment_v1.json", "kyc/ray_alignment_v1.json"),
        (
            Path("/share/longjunyu/cabi-vla/kyc-scaling-v3/diagnostics/kyc_seed41_ray_use_v1.json"),
            "kyc/kyc_seed41_ray_use_v1.json",
        ),
        (
            Path("/share/longjunyu/cabi-vla/kyc-scaling-v3/diagnostics/camera_pose_leakage_v2.json"),
            "kyc/camera_pose_leakage_v2.json",
        ),
        (
            Path("/share/longjunyu/cabi-vla/kyc-scaling-v3/eval/factorial/n10/analysis/confirmed/summary.json"),
            "kyc/scene_cue_wrist_factorial_summary.json",
        ),
        (
            Path("/share/longjunyu/kyc-official-data/runs/analysis/official_act_summary.json"),
            "kyc/official_act_positive_control.json",
        ),
        (
            Path("/share/longjunyu/cabi-vla/dual-camera-kyc-screen-v1/summary.json"),
            "kyc/dual_camera_summary.json",
        ),
        (
            Path("/share/longjunyu/cabi-vla/dual-camera-kyc-screen-v1/diagnostics.json"),
            "kyc/dual_camera_diagnostics.json",
        ),
    ]
    for source, relative in metadata_files:
        builder.copy(source, relative, kind="analysis_result")

    multiview_runs = [
        "action_b025-gap",
        "action_b100-gap",
        "visual_b025-gap",
        "visual_b100-gap",
        "visual_b025-candidates",
    ]
    for name in multiview_runs:
        builder.copy_run_records(
            mv_root / "gate-v1" / name,
            Path("multiview/raw") / name,
        )

    for name in ("official_pi05", "visual_b025"):
        builder.copy_run_records(
            composition_root / name,
            Path("composition/raw/joint_ood") / name,
        )
    for method in ("control", "kyc"):
        for seed in (41, 42, 43):
            name = f"{method}_seed{seed}"
            builder.copy_run_records(
                matched_root / name,
                Path("composition/raw/kyc_matched") / name,
            )
            builder.copy_run_records(
                factor_root / name,
                Path("composition/raw/factor_separated") / name,
            )

    builder.copy_run_records(
        view_root / "pi05-libero-official-v1",
        Path("active_view/raw/official_pi05"),
    )
    builder.copy_run_records(
        mv_root / "gate-v1/visual_b025-candidates",
        Path("active_view/raw/strong_multiview_visual_lora_25pct"),
    )

    dual_root = Path("/share/longjunyu/cabi-vla/dual-camera-kyc-screen-v1")
    for name in (
        "dual-rgb-s41-u2000",
        "dual-control-s41-u2000",
        "external-s41-u2000",
        "wrist-s41-u2000",
        "dual-s41-u2000",
        "dual-wrist-initial-s41-u2000",
        "dual-wrist-lagged-s41-u2000",
    ):
        builder.copy(
            dual_root / name / "camera_sweep_test.json",
            Path("kyc/raw/dual_camera") / f"{name}.json",
            kind="raw_episode_records",
        )

    for metrics in sorted((mv_root / "runs").glob("*/metrics.jsonl")):
        if "steps33000" in metrics.parent.name and "smoke" not in metrics.parent.name:
            builder.copy(
                metrics,
                Path("multiview/training_metrics") / f"{metrics.parent.name}.jsonl",
                kind="training_metrics",
            )
            resolved = metrics.parent / "final_model/framework_config.yaml"
            if resolved.is_file():
                builder.copy(
                    resolved,
                    Path("configs/resolved_training") / f"{metrics.parent.name}.yaml",
                    kind="resolved_config",
                )
    for metrics in sorted((factor_root / "runs").glob("*/metrics.jsonl")):
        if "steps33000" in metrics.parent.name and "smoke" not in metrics.parent.name:
            builder.copy(
                metrics,
                Path("multiview/training_metrics") / f"{metrics.parent.name}.jsonl",
                kind="training_metrics",
            )
            resolved = metrics.parent / "final_model/framework_config.yaml"
            if resolved.is_file():
                builder.copy(
                    resolved,
                    Path("configs/resolved_training") / f"{metrics.parent.name}.yaml",
                    kind="resolved_config",
                )

    official_rows = read_jsonl_files((view_root / "pi05-libero-official-v1").glob("episodes-shard-*.jsonl"))
    strong_rows = read_jsonl_files(
        (mv_root / "gate-v1/visual_b025-candidates").glob("episodes-shard-*.jsonl")
    )
    return {
        "mv_manifest": read_json(mv_manifest),
        "factor_manifest": read_json(factor_manifest),
        "protocol": read_json(composition_root / "protocol-v1.json"),
        "active_protocol": read_json(view_root / "protocol-v3.json"),
        "ray_payload": read_json(
            Path("/share/longjunyu/cabi-vla/kyc-scaling-v3/diagnostics/kyc_seed41_ray_use_v1.json")
        ),
        "official_candidate_rows": official_rows,
        "strong_candidate_rows": strong_rows,
    }


def build_provenance(output: Path, builder: BundleBuilder) -> None:
    historical = {
        "current_alphabrain_head": git_output("rev-parse", "HEAD"),
        "current_branch": git_output("branch", "--show-current"),
        "current_worktree_status": git_output("status", "--short").splitlines(),
        "official_pi05_active_view_alphabrain_commit": read_json(
            SHARE_ROOT
            / "experiments/libero-plus-view-gap-v1/pi05-libero-official-v1/run_manifest.json"
        ).get("alphabrain_commit"),
        "official_pi05_active_view_openpi_commit": read_json(
            SHARE_ROOT
            / "experiments/libero-plus-view-gap-v1/pi05-libero-official-v1/run_manifest.json"
        ).get("openpi_commit"),
        "official_pi05_composition_alphabrain_commit": read_json(
            SHARE_ROOT
            / "experiments/libero-plus-camera-background-v1/official_pi05/run_manifest.json"
        ).get("alphabrain_commit"),
        "official_kyc_repository_commit": read_json(
            Path("/share/longjunyu/kyc-official-data/runs/analysis/official_act_summary.json")
        ).get("official_repository_commit"),
        "official_kyc_robosuite_commit": read_json(
            Path("/share/longjunyu/kyc-official-data/runs/analysis/official_act_summary.json")
        ).get("official_robosuite_commit"),
        "artifact_source_limit": "several trained AlphaBrain runs recorded evaluator SHA256 and checkpoint SHA256 but not an AlphaBrain Git commit; do not infer an exact commit for those runs",
    }
    write_json(output / "provenance" / "commit_hashes.json", historical)
    write_json(
        output / "coverage_matrix.json",
        {
            "multiview": {
                "episode_ids_25_100": "complete",
                "pose_groups_and_histogram": "complete",
                "window_and_exposure_accounting": "complete",
                "pairing_and_camera_transition_semantics": "complete",
                "canonical_control_and_image_augmentation": "complete",
                "per_seed_checkpoints": "metadata_only_no_weights",
                "per_seed_results": "complete_for_existing_runs",
                "action_only_vs_visual_lora": "complete",
            },
            "composition": {
                "four_cell_raw_records": "complete",
                "background_definition": "complete_with_texture_only_scope",
                "training_camera_appearance_pairing": "category_level_only_exact_pair_ids_unavailable",
                "task_lists_36_and_9": "complete",
                "per_seed_interaction": "complete",
                "object_layout_light_changes": "verified_unchanged",
            },
            "kyc_geometry": {
                "correct_canonical_mismatched": "complete_aggregate_action_response",
                "wrist_correct_initial_delayed": "complete_closed_loop",
                "shuffled_pose": "not_run",
                "noisy_pose": "not_run",
                "external_delayed_pose": "not_run",
                "raw_action_vectors": "not_retained",
                "intermediate_feature_outputs": "not_retained",
                "ray_convention": "complete",
                "act_vs_pi05_conditions": "complete",
                "external_wrist_settings": "complete",
            },
            "active_view": {
                "state_candidate_policy_matrix": "complete_for_official_and_seed41_strong_multiview",
                "candidate_support": "parameter_range_only_exact_pose_identity_unavailable",
                "visibility_pixel_area_occlusion": "initial_observation_complete",
                "selector_scores_and_oracle": "complete_post_hoc",
                "same_state_different_task_labels": "not_constructed",
                "same_state_different_stage_labels": "not_constructed",
                "dynamic_active_camera": "not_tested",
            },
        },
    )
    builder.finalize_index()
    write_json(
        output / "bundle_manifest.json",
        {
            "schema_version": 1,
            "bundle_name": output.name,
            "source_machine_role": "8x RTX 5090 AlphaBrain camera/view experiments",
            "content_policy": "JSON/JSONL/CSV/configs/source scripts and hashes only",
            "excluded": [
                "checkpoint directories",
                "model weights",
                "datasets and TFRecord shards",
                "videos and images",
                "launcher logs",
                "secrets, auth, tokens, SSH keys and proxy configuration",
            ],
            "file_count": sum(1 for path in output.rglob("*") if path.is_file()) + 2,
            "copied_source_file_count": len(builder.sources),
            "copied_source_bytes": sum(int(row["bytes"]) for row in builder.sources),
            "generated_tables": [
                "multiview/episode_inventory.csv",
                "multiview/pose_groups.csv",
                "multiview/sampling_and_exposure.json",
                "multiview/training_runs.csv",
                "composition/tasks_36.csv",
                "composition/tasks_9.csv",
                "kyc/fixed_checkpoint_ray_response.csv",
                "kyc/intervention_coverage.csv",
                "active_view/state_candidate_policy_matrix.csv",
                "active_view/candidate_support.csv",
            ],
        },
    )


def validate_bundle(output: Path) -> dict[str, Any]:
    validation_path = output / "validation.json"
    files = [
        path for path in output.rglob("*") if path.is_file() and path != validation_path
    ]
    forbidden = [path for path in files if path.suffix.lower() in FORBIDDEN_SUFFIXES]
    unsupported = [path for path in files if path.suffix.lower() not in ALLOWED_SUFFIXES]
    sensitive_names = [
        path
        for path in files
        if any(token in path.name.lower() for token in ("auth.json", "api_key", "token", "id_rsa"))
    ]
    checkpoint_components = [
        path for path in files if any(part in {"checkpoint", "checkpoints"} for part in path.parts)
    ]
    result = {
        "status": "pass"
        if not forbidden and not unsupported and not sensitive_names and not checkpoint_components
        else "fail",
        "file_count": len(files) + 1,
        "payload_bytes_excluding_validation": sum(path.stat().st_size for path in files),
        "forbidden_files": [str(path.relative_to(output)) for path in forbidden],
        "unsupported_files": [str(path.relative_to(output)) for path in unsupported],
        "sensitive_filename_matches": [str(path.relative_to(output)) for path in sensitive_names],
        "checkpoint_directory_matches": [
            str(path.relative_to(output)) for path in checkpoint_components
        ],
    }
    write_json(validation_path, result)
    if result["status"] != "pass":
        raise RuntimeError(f"bundle validation failed: {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the 8-GPU camera/view evidence handoff")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite handoff bundle: {output}")
    output.mkdir(parents=True)
    builder = BundleBuilder(output)
    inputs = copy_inputs(builder)
    build_multiview_tables(output, inputs["mv_manifest"])
    build_training_run_table(output)
    build_composition_tables(output, inputs["protocol"], inputs["factor_manifest"])
    build_kyc_tables(output, inputs["ray_payload"])
    build_active_view_tables(
        output,
        inputs["active_protocol"],
        inputs["official_candidate_rows"],
        inputs["strong_candidate_rows"],
    )
    build_provenance(output, builder)
    validation = validate_bundle(output)
    print(json.dumps({"output": str(output), **validation}, sort_keys=True))


if __name__ == "__main__":
    main()
