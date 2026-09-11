from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from collect_libero_bind_teacher import load_state_bank, upright_image
from libero_camera_pose import (
    capture_camera_reference,
    install_camera_pose,
    load_camera_sweep_config,
)
from libero_scene_cues import (
    capture_scene_cue_reference,
    install_scene_cues,
    render_background_only,
)


def downsample_rgb(image: np.ndarray, *, size: int) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("image must have shape [H, W, 3]")
    height, width = values.shape[:2]
    if height % size or width % size:
        raise ValueError("feature size must divide image height and width")
    reduced = values.reshape(
        size,
        height // size,
        size,
        width // size,
        3,
    ).mean(axis=(1, 3))
    return (reduced / 255.0).reshape(-1)


def ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    x_train = np.asarray(train_x, dtype=np.float64)
    x_test = np.asarray(test_x, dtype=np.float64)
    y_train = np.asarray(train_y, dtype=np.float64)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale
    x_train = np.concatenate(
        [x_train, np.ones((len(x_train), 1), dtype=np.float64)],
        axis=1,
    )
    x_test = np.concatenate(
        [x_test, np.ones((len(x_test), 1), dtype=np.float64)],
        axis=1,
    )
    dual = np.linalg.solve(
        x_train @ x_train.T + ridge * np.eye(len(x_train)),
        y_train,
    )
    return x_test @ x_train.T @ dual


def probe_metrics(
    features: np.ndarray,
    pose_indices: np.ndarray,
    pose_parameters: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    ridge: float,
) -> dict[str, Any]:
    labels = np.asarray(pose_indices, dtype=np.int64)
    num_classes = int(labels.max()) + 1
    one_hot = np.eye(num_classes, dtype=np.float64)[labels]
    class_scores = ridge_fit_predict(
        features[train_mask],
        one_hot[train_mask],
        features[test_mask],
        ridge=ridge,
    )
    prediction = np.argmax(class_scores, axis=1)
    truth = labels[test_mask]
    accuracy = float(np.mean(prediction == truth))
    recalls = [
        float(np.mean(prediction[truth == label] == label))
        for label in range(num_classes)
        if np.any(truth == label)
    ]

    regression = ridge_fit_predict(
        features[train_mask],
        pose_parameters[train_mask],
        features[test_mask],
        ridge=ridge,
    )
    regression_truth = pose_parameters[test_mask]
    r2 = []
    for axis in range(regression_truth.shape[1]):
        residual = float(
            np.sum(np.square(regression_truth[:, axis] - regression[:, axis]))
        )
        total = float(
            np.sum(
                np.square(
                    regression_truth[:, axis]
                    - np.mean(regression_truth[:, axis])
                )
            )
        )
        r2.append(1.0 - residual / total if total > 1e-12 else 0.0)
    return {
        "train_count": int(np.sum(train_mask)),
        "test_count": int(np.sum(test_mask)),
        "pose_count": num_classes,
        "chance_accuracy": 1.0 / num_classes,
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "r2": {
            "azimuth_deg": float(r2[0]),
            "elevation_deg": float(r2[1]),
            "radius_scale": float(r2[2]),
        },
        "mean_positive_r2": float(np.mean(np.maximum(r2, 0.0))),
    }


def relative_reduction(value: float, reference: float) -> float:
    return float((reference - value) / reference) if reference > 1e-12 else 0.0


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure camera-pose leakage from LIBERO background cues"
    )
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edge", default="red-left")
    parser.add_argument("--state-indices", required=True)
    parser.add_argument("--train-state-count", type=int, default=7)
    parser.add_argument("--scene-seeds", type=int, default=8)
    parser.add_argument("--train-scene-seeds", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--feature-size", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--contact-sheet", type=Path)
    return parser.parse_args(args)


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite probe: {args.output}")
    state_indices = [
        int(part.strip())
        for part in args.state_indices.split(",")
        if part.strip()
    ]
    if len(state_indices) != len(set(state_indices)) or len(state_indices) < 2:
        raise ValueError("state-indices must be a unique list of at least two")
    if not 0 < args.train_state_count < len(state_indices):
        raise ValueError("train-state-count must leave held-out states")
    if not 0 < args.train_scene_seeds < args.scene_seeds:
        raise ValueError("train-scene-seeds must leave held-out scene seeds")
    if args.resolution % args.feature_size:
        raise ValueError("feature-size must divide resolution")

    manifest = json.loads((args.suite_root / "manifest.json").read_text())
    edge_by_name = {str(edge["edge_id"]): edge for edge in manifest["edges"]}
    edge = edge_by_name[args.edge]
    states = load_state_bank(Path(manifest["canonical_init_states"]))
    config = load_camera_sweep_config(args.camera_config)
    poses = list(config["poses"])
    pose_parameters = np.asarray(
        [
            [
                float(pose.get("azimuth_deg", 0.0)),
                float(pose.get("elevation_deg", 0.0)),
                float(pose.get("radius_scale", 1.0)),
            ]
            for pose in poses
        ],
        dtype=np.float64,
    )

    env = OffScreenRenderEnv(
        bddl_file_name=edge["bddl"],
        camera_heights=args.resolution,
        camera_widths=args.resolution,
        horizon=64,
        ignore_done=True,
    )
    feature_rows = []
    labels = []
    parameters = []
    train_flags = []
    test_flags = []
    records = []
    contact_images: dict[tuple[str, int], np.ndarray] = {}
    physics_max_abs = 0.0
    try:
        env.seed(args.seed)
        camera_reference = capture_camera_reference(
            env,
            camera_name=config["camera_name"],
            table_plane_z=config["table_plane_z"],
        )
        scene_reference = capture_scene_cue_reference(env)
        for state_position, state_index in enumerate(state_indices):
            env.reset()
            observation = env.set_init_state(np.asarray(states[state_index]))
            for _ in range(8):
                observation, _, _, _ = env.step(
                    np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
                )
            state_before = env.env.sim.get_state().flatten().copy()
            for mode in ("fixed", "cue_randomized"):
                for scene_index in range(args.scene_seeds):
                    scene_metadata = install_scene_cues(
                        env,
                        scene_reference,
                        mode=mode,
                        seed=args.seed,
                        sample_id=(
                            f"{args.edge}::state-{state_index}::"
                            f"scene-{scene_index}"
                        ),
                    )
                    for pose_index, pose in enumerate(poses):
                        install_camera_pose(env, camera_reference, pose)
                        raw = render_background_only(
                            env,
                            camera_name=config["camera_name"],
                            height=args.resolution,
                            width=args.resolution,
                            background_geom_ids=scene_reference[
                                "background_geom_ids"
                            ],
                        )
                        image = upright_image(raw)
                        feature_rows.append(
                            downsample_rgb(image, size=args.feature_size)
                        )
                        labels.append(pose_index)
                        parameters.append(pose_parameters[pose_index])
                        train_flags.append(
                            state_position < args.train_state_count
                            and scene_index < args.train_scene_seeds
                        )
                        test_flags.append(
                            state_position >= args.train_state_count
                            and scene_index >= args.train_scene_seeds
                        )
                        records.append(
                            {
                                "scene_mode": mode,
                                "state_index": state_index,
                                "scene_index": scene_index,
                                "pose_name": str(pose["name"]),
                                "image_sha256": hashlib.sha256(
                                    image.tobytes()
                                ).hexdigest(),
                                "scene_cue_seed": scene_metadata[
                                    "scene_cue_seed"
                                ],
                            }
                        )
                        if (
                            state_position == len(state_indices) - 1
                            and scene_index == args.scene_seeds - 1
                        ):
                            contact_images[(mode, pose_index)] = image
                    current_state = env.env.sim.get_state().flatten()
                    physics_max_abs = max(
                        physics_max_abs,
                        float(np.max(np.abs(current_state - state_before))),
                    )
    finally:
        env.close()

    features = np.asarray(feature_rows, dtype=np.float64)
    pose_labels = np.asarray(labels, dtype=np.int64)
    pose_values = np.asarray(parameters, dtype=np.float64)
    train_flags_array = np.asarray(train_flags, dtype=bool)
    test_flags_array = np.asarray(test_flags, dtype=bool)
    scene_modes = np.asarray([row["scene_mode"] for row in records])
    metrics = {}
    for mode in ("fixed", "cue_randomized"):
        mode_mask = scene_modes == mode
        metrics[mode] = probe_metrics(
            features,
            pose_labels,
            pose_values,
            train_flags_array & mode_mask,
            test_flags_array & mode_mask,
            ridge=args.ridge,
        )

    chance = metrics["fixed"]["chance_accuracy"]
    fixed_advantage = max(metrics["fixed"]["accuracy"] - chance, 0.0)
    randomized_advantage = max(
        metrics["cue_randomized"]["accuracy"] - chance,
        0.0,
    )
    classification_reduction = relative_reduction(
        randomized_advantage,
        fixed_advantage,
    )
    r2_reduction = relative_reduction(
        metrics["cue_randomized"]["mean_positive_r2"],
        metrics["fixed"]["mean_positive_r2"],
    )
    gate_passed = bool(
        physics_max_abs <= 1e-12
        and max(classification_reduction, r2_reduction) >= 0.25
    )

    if args.contact_sheet is not None:
        from PIL import Image

        rows = []
        for mode in ("fixed", "cue_randomized"):
            rows.append(
                np.concatenate(
                    [
                        contact_images[(mode, pose_index)]
                        for pose_index in range(len(poses))
                    ],
                    axis=1,
                )
            )
        args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.concatenate(rows, axis=0)).save(args.contact_sheet)

    payload = {
        "schema_version": 1,
        "study": "libero_background_camera_pose_leakage",
        "suite_root": str(args.suite_root),
        "camera_config": str(args.camera_config),
        "edge": args.edge,
        "state_indices": state_indices,
        "train_state_count": args.train_state_count,
        "scene_seeds": args.scene_seeds,
        "train_scene_seeds": args.train_scene_seeds,
        "resolution": args.resolution,
        "feature_size": args.feature_size,
        "ridge": args.ridge,
        "metrics": metrics,
        "classification_advantage_reduction": classification_reduction,
        "mean_positive_r2_reduction": r2_reduction,
        "physics_state_max_abs": physics_max_abs,
        "gate_passed": gate_passed,
        "contact_sheet": (
            None if args.contact_sheet is None else str(args.contact_sheet)
        ),
        "render_count": len(records),
    }
    _atomic_write(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

