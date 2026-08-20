from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from accel_inference import rank_fixed_state_candidates
from accel_core import shared_flow_noise


CANDIDATE_IDS = (
    "canonical",
    "broad_a",
    "broad_b",
    "external_blackout",
    "all_blackout",
)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_array(left: Any, right: Any) -> bool:
    return np.array_equal(np.asarray(left), np.asarray(right))


def build_fixed_state_candidates(
    canonical_sample: Mapping[str, Any],
    paired_samples: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build evaluation-only view candidates while preserving one physical state."""

    if len(paired_samples) != 2:
        raise ValueError("paired_samples must contain broad_a and broad_b")
    broad_by_pose = {str(sample["camera_pose"]): sample for sample in paired_samples}
    if set(broad_by_pose) != {"broad_a", "broad_b"}:
        raise ValueError("paired_samples must have broad_a and broad_b camera poses")

    sources = [canonical_sample, broad_by_pose["broad_a"], broad_by_pose["broad_b"]]
    pair_ids = {str(sample["dsol_pair_id"]) for sample in sources}
    episode_ids = {str(sample["episode_id"]) for sample in sources}
    frame_indices = {int(sample["frame_index"]) for sample in sources}
    languages = {str(sample["lang"]) for sample in sources}
    if len(pair_ids) != 1 or len(episode_ids) != 1 or len(frame_indices) != 1:
        raise ValueError("candidate sources do not identify the same simulator state")
    if len(languages) != 1:
        raise ValueError("candidate sources do not share one language instruction")
    if not all(_same_array(sources[0]["state"], sample["state"]) for sample in sources[1:]):
        raise ValueError("candidate sources have different robot states")
    if not all(_same_array(sources[0]["action"], sample["action"]) for sample in sources[1:]):
        raise ValueError("candidate sources have different action supervision")

    candidates = [copy.deepcopy(dict(sample)) for sample in sources]
    canonical_external = np.asarray(candidates[0]["image"][0], dtype=np.uint8)
    canonical_wrist = np.asarray(candidates[0]["image"][1], dtype=np.uint8)

    external_blackout = copy.deepcopy(candidates[0])
    external_blackout["image"] = [
        np.zeros_like(canonical_external),
        canonical_wrist.copy(),
    ]
    external_blackout["sample_id"] = f"{sources[0]['dsol_pair_id']}::external_blackout"
    external_blackout["camera_pose"] = "external_blackout"

    all_blackout = copy.deepcopy(candidates[0])
    all_blackout["image"] = [
        np.zeros_like(canonical_external),
        np.zeros_like(canonical_wrist),
    ]
    all_blackout["sample_id"] = f"{sources[0]['dsol_pair_id']}::all_blackout"
    all_blackout["camera_pose"] = "all_blackout"
    candidates.extend([external_blackout, all_blackout])

    metadata = []
    for candidate_id, sample in zip(CANDIDATE_IDS, candidates):
        matrix = np.asarray(sample["camera_to_world_opencv"], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"invalid camera matrix for {candidate_id}: {matrix.shape}")
        metadata.append(
            {
                "candidate_id": candidate_id,
                "candidate_kind": (
                    "physical_view" if "blackout" not in candidate_id else "sensor_control"
                ),
                "camera_position": matrix[:3, 3].tolist(),
                "camera_rotation_matrix": matrix[:3, :3].tolist(),
                "external_blackout": candidate_id in {"external_blackout", "all_blackout"},
                "wrist_blackout": candidate_id == "all_blackout",
            }
        )

    audit = {
        "same_pair_id": len(pair_ids) == 1,
        "same_episode_id": len(episode_ids) == 1,
        "same_frame_index": len(frame_indices) == 1,
        "same_language": len(languages) == 1,
        "same_robot_state": True,
        "same_action_supervision": True,
        "pair_id": next(iter(pair_ids)),
        "episode_id": next(iter(episode_ids)),
        "frame_index": next(iter(frame_indices)),
    }
    return candidates, metadata, audit


def _save_montage(path: Path, candidates: Sequence[Mapping[str, Any]]) -> None:
    panels = []
    for candidate_id, sample in zip(CANDIDATE_IDS, candidates):
        external = Image.fromarray(np.asarray(sample["image"][0], dtype=np.uint8))
        wrist = Image.fromarray(np.asarray(sample["image"][1], dtype=np.uint8))
        panel = Image.new("RGB", (224, 224 * 2 + 28), "white")
        panel.paste(external, (0, 28))
        panel.paste(wrist, (0, 252))
        ImageDraw.Draw(panel).text((6, 7), candidate_id, fill="black")
        panels.append(panel)
    montage = Image.new("RGB", (224 * len(panels), panels[0].height), "white")
    for index, panel in enumerate(panels):
        montage.paste(panel, (224 * index, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real AlphaBrain Pi0.5 fixed-state Accel engineering smoke."
    )
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.record_index < 0:
        raise ValueError("record-index must be non-negative")
    if not (args.pair_root / "manifest.json").is_file():
        raise FileNotFoundError(f"missing pair collection: {args.pair_root}")
    if not (args.checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"missing checkpoint weights: {args.checkpoint}")

    import torch
    from AlphaBrain.dataloader.paligemma_datasets import DsolLiberoPairDataset
    from AlphaBrain.model.framework.base_framework import BaseFramework

    canonical_dataset = DsolLiberoPairDataset(
        args.pair_root, split=args.split, arm="canonical_unique"
    )
    paired_dataset = DsolLiberoPairDataset(
        args.pair_root, split=args.split, arm="broad_paired_fm"
    )
    if args.record_index >= len(canonical_dataset):
        raise IndexError(
            f"record-index {args.record_index} >= split size {len(canonical_dataset)}"
        )
    candidates, metadata, fixed_state_audit = build_fixed_state_candidates(
        canonical_dataset[args.record_index], paired_dataset[args.record_index]
    )

    started = time.perf_counter()
    model = BaseFramework.from_pretrained(str(args.checkpoint), strict_checkpoint=True)
    model = model.to(torch.bfloat16).to(args.device).eval()
    model.gripper_remap = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    load_seconds = time.perf_counter() - started

    horizon = int(model.action_horizon)
    action_dim = int(model.action_dim)
    if action_dim != 7:
        raise ValueError(f"expected LIBERO action_dim=7, got {action_dim}")

    infer_started = time.perf_counter()
    with torch.inference_mode():
        result = rank_fixed_state_candidates(
            model,
            candidates,
            CANDIDATE_IDS,
            seed=args.seed,
            action_horizon=horizon,
            action_dim=action_dim,
            include_trace_artifacts=True,
        )
        shared_noise = shared_flow_noise(
            seed=args.seed,
            candidate_count=len(candidates),
            action_horizon=horizon,
            action_dim=action_dim,
        )
        default_output = model.predict_action(examples=candidates, noise=shared_noise)
    infer_seconds = time.perf_counter() - infer_started

    trace = np.asarray(result.pop("flow_velocity_trace"), dtype=np.float32)
    returned_noise = np.asarray(result.pop("flow_initial_noise"), dtype=np.float32)
    traced_actions = np.asarray(result["actions"], dtype=np.float32)
    default_actions = np.asarray(default_output["normalized_actions"], dtype=np.float32)
    if traced_actions.shape != default_actions.shape:
        raise ValueError("default and trace action shapes differ")
    max_action_difference = float(np.max(np.abs(traced_actions - default_actions)))
    exact_action_match = bool(np.array_equal(traced_actions, default_actions))
    if not exact_action_match:
        raise ValueError(
            f"flow tracing changed deterministic actions (max_abs={max_action_difference})"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "flow_trace.npz",
        candidate_ids=np.asarray(CANDIDATE_IDS),
        velocity_trace=trace,
        initial_noise=returned_noise,
        flow_times=np.asarray(result["flow_times"], dtype=np.float32),
        traced_actions=traced_actions,
        default_actions=default_actions,
    )
    _atomic_json(args.output_dir / "ranking.json", result)
    _atomic_json(args.output_dir / "candidate_metadata.json", metadata)
    _save_montage(args.output_dir / "candidate_montage.png", candidates)

    weights = args.checkpoint / "model.safetensors"
    manifest = {
        "schema": "dsol_accel_real_checkpoint_smoke_v1",
        "status": "PASS",
        "scope": "engineering_smoke_only",
        "scientific_claim_allowed": False,
        "candidate_ids": list(CANDIDATE_IDS),
        "pair_root": str(args.pair_root.resolve()),
        "pair_manifest_sha256": _sha256(args.pair_root / "manifest.json"),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_weights_bytes": weights.stat().st_size,
        "checkpoint_config_sha256": _sha256(args.checkpoint / "framework_config.yaml"),
        "device": args.device,
        "seed": args.seed,
        "split": args.split,
        "record_index": args.record_index,
        "fixed_state_audit": fixed_state_audit,
        "shared_noise_audit": result["shared_flow_noise_audit"],
        "default_trace_action_exact_match": exact_action_match,
        "default_trace_action_max_abs_difference": max_action_difference,
        "trace_shape": list(trace.shape),
        "action_shape": list(traced_actions.shape),
        "selected_candidate_id": result["selected_candidate_id"],
        "model_load_seconds": load_seconds,
        "inference_seconds": infer_seconds,
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(args.device))
            if str(args.device).startswith("cuda")
            else None
        ),
        "relation_analysis": {
            "status": "DEFERRED_UNTIL_M0_REFERENCES_ARE_FROZEN",
            "reason": (
                "This engineering record has no validated strong-info, reveal, or "
                "closed-loop oracle labels."
            ),
        },
        "artifacts": {
            "ranking": "ranking.json",
            "trace": "flow_trace.npz",
            "candidate_metadata": "candidate_metadata.json",
            "montage": "candidate_montage.png",
        },
    }
    _atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
