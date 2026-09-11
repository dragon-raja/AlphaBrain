#!/usr/bin/env python3
"""Train frozen small-data view rankers without opening oracle-v2 test outcomes."""

from __future__ import annotations

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[3]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))


import argparse
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.dsol_paper1.protocols.build_statewise_view_oracle_v2 import freeze_json, load_scan_assets, operational_candidates
from scripts.dsol_paper1.protocols.build_statewise_view_oracle_v2_stage import matched_paths, summarize_ledgers, validate_input_audits
from AlphaBrain.research.dsol.data.flow_noise import sha256_file


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def tensor_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_render_index(legacy_root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted((legacy_root / "accel-ensemble" / "states").glob("*/render.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "PASS" or int(row.get("candidate_count", 0)) != 97:
            raise ValueError(f"invalid render receipt: {path}")
        result[str(row["pair_key"])] = row
    if len(result) != 64:
        raise ValueError(f"expected 64 render receipts, found {len(result)}")
    return result


def load_accel_index(legacy_root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted((legacy_root / "accel-ensemble" / "states").glob("*/ranking.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if int(row.get("candidate_count", 0)) != 97:
            raise ValueError(f"invalid Accel ranking: {path}")
        result[str(row["pair_key"])] = row
    if len(result) != 64:
        raise ValueError(f"expected 64 Accel rankings, found {len(result)}")
    return result


def load_resnet18(weights_path: Path, device: str):
    import torch
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


def encode_images(model: Any, images: np.ndarray, *, device: str, batch_size: int) -> np.ndarray:
    import torch

    array = np.asarray(images)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"expected NHWC RGB image batch, got {array.shape}")
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device)[None, :, None, None]
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(array), batch_size):
            value = torch.from_numpy(array[start : start + batch_size]).to(device=device)
            value = value.permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
            value = (value - mean) / std
            outputs.append(model(value).to(torch.float32).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def extract_embedding_cache(
    *,
    states: Sequence[Mapping[str, Any]],
    render_index: Mapping[str, Mapping[str, Any]],
    weights_path: Path,
    device: str,
    batch_size: int,
    output: Path,
) -> dict[str, Any]:
    model = load_resnet18(weights_path, device)
    pair_keys = []
    candidate_ids_reference = None
    candidate_embeddings = []
    canonical_embeddings = []
    wrist_embeddings = []
    robot_states = []
    source_artifacts = []
    for state in states:
        source_key = str(state["asset_source_pair_key"])
        render = render_index[source_key]
        artifact = Path(render["artifact"])
        if sha256_file(artifact) != render["artifact_sha256"]:
            raise ValueError(f"rendered policy-input checksum mismatch: {artifact}")
        with np.load(artifact, allow_pickle=False) as values:
            candidate_ids = [str(value) for value in values["candidate_ids"]]
            if len(candidate_ids) != 97 or len(set(candidate_ids)) != 97:
                raise ValueError(f"invalid candidate IDs in {artifact}")
            if candidate_ids_reference is None:
                candidate_ids_reference = candidate_ids
            elif candidate_ids_reference != candidate_ids:
                raise ValueError("rendered candidate ordering differs across states")
            external = np.asarray(values["external_images"], dtype=np.uint8)
            wrist = np.asarray(values["wrist_images"][0:1], dtype=np.uint8)
            state_vector = np.asarray(values["robot_states"][0], dtype=np.float32)
        embeddings = encode_images(model, external, device=device, batch_size=batch_size)
        wrist_embedding = encode_images(model, wrist, device=device, batch_size=1)[0]
        canonical_index = candidate_ids.index("canonical")
        pair_keys.append(str(state["pair_key"]))
        candidate_embeddings.append(embeddings)
        canonical_embeddings.append(embeddings[canonical_index])
        wrist_embeddings.append(wrist_embedding)
        robot_states.append(state_vector)
        source_artifacts.append(
            {
                "pair_key": str(state["pair_key"]),
                "artifact": str(artifact.resolve()),
                "artifact_sha256": render["artifact_sha256"],
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        pair_keys=np.asarray(pair_keys),
        candidate_ids=np.asarray(candidate_ids_reference),
        candidate_embeddings=np.asarray(candidate_embeddings, dtype=np.float32),
        canonical_embeddings=np.asarray(canonical_embeddings, dtype=np.float32),
        wrist_embeddings=np.asarray(wrist_embeddings, dtype=np.float32),
        robot_states=np.asarray(robot_states, dtype=np.float32),
    )
    return {
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "pair_key_count": len(pair_keys),
        "candidate_count": len(candidate_ids_reference),
        "candidate_embedding_shape": [len(pair_keys), len(candidate_ids_reference), 512],
        "source_artifacts": source_artifacts,
    }


def load_or_build_embedding_cache(
    *,
    states: Sequence[Mapping[str, Any]],
    render_index: Mapping[str, Mapping[str, Any]],
    weights_path: Path,
    device: str,
    batch_size: int,
    output: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if output.exists():
        with np.load(output, allow_pickle=False) as values:
            cache = {key: np.asarray(values[key]) for key in values.files}
        expected_keys = [str(state["pair_key"]) for state in states]
        if [str(value) for value in cache["pair_keys"]] != expected_keys:
            raise ValueError("existing embedding cache has different state identities")
        receipt = {
            "path": str(output.resolve()),
            "sha256": sha256_file(output),
            "pair_key_count": len(expected_keys),
            "candidate_count": int(len(cache["candidate_ids"])),
            "reused_verified_cache": True,
        }
        return cache, receipt
    receipt = extract_embedding_cache(
        states=states,
        render_index=render_index,
        weights_path=weights_path,
        device=device,
        batch_size=batch_size,
        output=output,
    )
    with np.load(output, allow_pickle=False) as values:
        cache = {key: np.asarray(values[key]) for key in values.files}
    receipt["reused_verified_cache"] = False
    return cache, receipt


def geometry_vector(candidate: Mapping[str, Any]) -> np.ndarray:
    candidate_id = str(candidate["selected_candidate_id"])
    pose = candidate.get("pose") or {}
    azimuth = np.deg2rad(float(pose.get("azimuth_deg", 0.0)))
    elevation = float(pose.get("elevation_deg", 0.0)) / 45.0
    radius = float(pose.get("radius_scale", 1.0)) - 1.0
    return np.asarray(
        [
            candidate_id == "canonical",
            candidate_id.startswith("broad_train_"),
            candidate_id.startswith("broad_heldout_"),
            np.sin(azimuth),
            np.cos(azimuth),
            elevation,
            radius,
        ],
        dtype=np.float64,
    )


def fit_pca(values: np.ndarray, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    centered = values - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return mean, vt[: min(dimensions, len(vt))]


def apply_pca(values: np.ndarray, pca: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, components = pca
    return (values - mean) @ components.T


@dataclass
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    beta: np.ndarray
    intercept: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.scale) @ self.beta + self.intercept


def fit_ridge(features: np.ndarray, targets: np.ndarray, alpha: float) -> RidgeModel:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1e-10] = 1.0
    standardized = (features - mean) / scale
    intercept = float(np.mean(targets))
    centered_target = targets - intercept
    gram = standardized.T @ standardized
    beta = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0]), standardized.T @ centered_target
    )
    return RidgeModel(mean=mean, scale=scale, beta=beta, intercept=intercept)


def task_one_hot(task_id: str, tasks: Sequence[str]) -> np.ndarray:
    value = np.zeros(len(tasks), dtype=np.float64)
    value[tasks.index(task_id)] = 1.0
    return value


def build_feature_rows(
    *,
    state_indices: Sequence[int],
    states: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    geometry: np.ndarray,
    cache: Mapping[str, np.ndarray],
    tasks: Sequence[str],
    context_pca: tuple[np.ndarray, np.ndarray],
    candidate_pca: tuple[np.ndarray, np.ndarray],
    family: str,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    raw_context = np.concatenate(
        [cache["canonical_embeddings"], cache["wrist_embeddings"], cache["robot_states"]],
        axis=1,
    )
    context = apply_pca(raw_context, context_pca)
    candidate_embedding = apply_pca(
        cache["candidate_embeddings"].reshape(-1, 512), candidate_pca
    ).reshape(len(states), len(candidate_ids), -1)
    rows = []
    identities = []
    canonical_index = candidate_ids.index("canonical")
    for state_index in state_indices:
        task = task_one_hot(str(states[state_index]["task_id"]), tasks)
        state_context = context[state_index]
        for candidate_index, _candidate_id in enumerate(candidate_ids):
            geo = geometry[state_index, candidate_index]
            base = [
                task,
                geo,
                np.outer(task, geo).reshape(-1),
                np.outer(state_context, geo).reshape(-1),
            ]
            if family == "image_queryable_ridge":
                candidate_value = candidate_embedding[state_index, candidate_index]
                canonical_value = candidate_embedding[state_index, canonical_index]
                base.extend(
                    [
                        candidate_value,
                        candidate_value - canonical_value,
                        np.outer(task, candidate_value).reshape(-1),
                    ]
                )
            elif family != "geometry_context_ridge":
                raise ValueError(f"unsupported model family: {family}")
            rows.append(np.concatenate(base))
            identities.append((state_index, candidate_index))
    return np.asarray(rows, dtype=np.float64), identities


def select_predictions(
    predictions: np.ndarray,
    identities: Sequence[tuple[int, int]],
    candidate_ids: Sequence[str],
) -> dict[int, int]:
    grouped: dict[int, list[tuple[float, str, int]]] = {}
    for prediction, (state_index, candidate_index) in zip(predictions, identities):
        grouped.setdefault(state_index, []).append(
            (float(prediction), str(candidate_ids[candidate_index]), candidate_index)
        )
    return {
        state_index: sorted(values, key=lambda row: (-row[0], row[1]))[0][2]
        for state_index, values in grouped.items()
    }


def select_fixed_baselines(
    *,
    development_states: Sequence[Mapping[str, Any]],
    test_states: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    success: np.ndarray,
    progress: np.ndarray,
    scan_assets: Mapping[str, Mapping[str, Any]],
    accel_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    candidate_ids = list(candidate_ids)
    global_scores = [
        (float(np.mean(success[:, index])), float(np.mean(progress[:, index])), candidate_id)
        for index, candidate_id in enumerate(candidate_ids)
    ]
    global_candidate = sorted(global_scores, key=lambda row: (-row[0], -row[1], row[2]))[0][2]
    task_candidates = {}
    for task in sorted({str(state["task_id"]) for state in development_states}):
        state_indices = [
            index for index, state in enumerate(development_states) if state["task_id"] == task
        ]
        scores = [
            (
                float(np.mean(success[state_indices, index])),
                float(np.mean(progress[state_indices, index])),
                candidate_id,
            )
            for index, candidate_id in enumerate(candidate_ids)
        ]
        task_candidates[task] = sorted(scores, key=lambda row: (-row[0], -row[1], row[2]))[0][2]
    methods = {
        "global_fixed": {
            "deployment_class": "fixed_without_state_observation",
            "development_choice": global_candidate,
            "state_selections": {state["pair_key"]: global_candidate for state in test_states},
        },
        "task_fixed": {
            "deployment_class": "task_language_only",
            "development_choices_by_task": task_candidates,
            "state_selections": {
                state["pair_key"]: task_candidates[str(state["task_id"])] for state in test_states
            },
        },
        "visibility": {
            "deployment_class": "privileged_simulator_segmentation_baseline",
            "state_selections": {},
        },
        "accel": {
            "deployment_class": "candidate_image_query_and_policy_internal_baseline",
            "state_selections": {},
        },
    }
    for state in test_states:
        source_key = str(state["asset_source_pair_key"])
        candidates = operational_candidates(scan_assets[source_key]["scan"])
        visibility = sorted(
            candidates,
            key=lambda row: (
                -float(row["candidate_features"]["visibility_score"]),
                str(row["selected_candidate_id"]),
            ),
        )[0]["selected_candidate_id"]
        accel = str(accel_index[source_key]["ranking"][0]["candidate_id"])
        methods["visibility"]["state_selections"][state["pair_key"]] = visibility
        methods["accel"]["state_selections"][state["pair_key"]] = accel
    return methods


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.learner_config.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_DENSE_O_OUTCOME_ANALYSIS":
        raise ValueError("learner config is not frozen")
    if sha256_file(Path(config["frozen_visual_encoder"]["weights"])) != config[
        "frozen_visual_encoder"
    ]["weights_sha256"]:
        raise ValueError("frozen ResNet18 weights checksum mismatch")
    population = json.loads(args.population.read_text(encoding="utf-8"))
    if population.get("status") != "PASS":
        raise ValueError("v2 population did not PASS")
    development = population["population"]["development"]["states"]
    test = population["population"]["test"]["states"]
    validate_input_audits(args.input_audits, expected_count=8)
    summaries = summarize_ledgers(
        matched_paths(args.input_ledgers), expected_bank="O", expected_repeats=32
    )
    if set(summaries) != {state["pair_key"] for state in development}:
        raise ValueError("development dense summaries have the wrong state set")
    legacy_root = Path(args.legacy_root)
    scan_assets = load_scan_assets(legacy_root)
    render_index = load_render_index(legacy_root)
    accel_index = load_accel_index(legacy_root)
    all_states = [*development, *test]
    cache, cache_receipt = load_or_build_embedding_cache(
        states=all_states,
        render_index=render_index,
        weights_path=Path(config["frozen_visual_encoder"]["weights"]),
        device=args.device,
        batch_size=args.batch_size,
        output=args.output_dir / "features" / "resnet18-static-assets.npz",
    )
    candidate_ids = [str(value) for value in cache["candidate_ids"]]
    state_index = {str(value): index for index, value in enumerate(cache["pair_keys"])}
    if [state_index[state["pair_key"]] for state in all_states] != list(range(len(all_states))):
        raise ValueError("embedding cache ordering differs from population ordering")
    success = np.asarray(
        [
            [summaries[state["pair_key"]][candidate]["mean_success"] for candidate in candidate_ids]
            for state in development
        ],
        dtype=np.float64,
    )
    progress = np.asarray(
        [
            [summaries[state["pair_key"]][candidate]["mean_progress"] for candidate in candidate_ids]
            for state in development
        ],
        dtype=np.float64,
    )
    geometry = []
    for state in all_states:
        candidates = operational_candidates(scan_assets[state["asset_source_pair_key"]]["scan"])
        lookup = {row["selected_candidate_id"]: row for row in candidates}
        geometry.append([geometry_vector(lookup[candidate]) for candidate in candidate_ids])
    geometry_array = np.asarray(geometry, dtype=np.float64)
    tasks = sorted({str(state["task_id"]) for state in all_states})
    context_raw = np.concatenate(
        [cache["canonical_embeddings"], cache["wrist_embeddings"], cache["robot_states"]], axis=1
    )
    alpha_grid = [float(value) for value in config["estimator"]["alpha_grid"]]
    family_results = {}
    learned_methods = {}
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    development_indices = list(range(len(development)))
    test_indices = list(range(len(development), len(all_states)))
    for family in ("geometry_context_ridge", "image_queryable_ridge"):
        cv_by_alpha = []
        for alpha_index, alpha in enumerate(alpha_grid):
            selected_records = []
            for fold in range(6):
                train_indices = [index for index in development_indices if development[index]["cv_fold"] != fold]
                validation_indices = [index for index in development_indices if development[index]["cv_fold"] == fold]
                context_pca = fit_pca(
                    context_raw[train_indices],
                    int(config["feature_reduction"]["context_embedding_pca_dimensions"]),
                )
                candidate_pca = fit_pca(
                    cache["candidate_embeddings"][train_indices].reshape(-1, 512),
                    int(config["feature_reduction"]["candidate_embedding_pca_dimensions"]),
                )
                train_features, train_identity = build_feature_rows(
                    state_indices=train_indices,
                    states=all_states,
                    candidate_ids=candidate_ids,
                    geometry=geometry_array,
                    cache=cache,
                    tasks=tasks,
                    context_pca=context_pca,
                    candidate_pca=candidate_pca,
                    family=family,
                )
                train_targets = np.asarray(
                    [success[state, candidate] for state, candidate in train_identity]
                )
                ridge = fit_ridge(train_features, train_targets, alpha)
                validation_features, validation_identity = build_feature_rows(
                    state_indices=validation_indices,
                    states=all_states,
                    candidate_ids=candidate_ids,
                    geometry=geometry_array,
                    cache=cache,
                    tasks=tasks,
                    context_pca=context_pca,
                    candidate_pca=candidate_pca,
                    family=family,
                )
                selected = select_predictions(
                    ridge.predict(validation_features), validation_identity, candidate_ids
                )
                for state, candidate in selected.items():
                    selected_records.append(
                        {
                            "state_index": state,
                            "candidate_index": candidate,
                            "success": float(success[state, candidate]),
                            "progress": float(progress[state, candidate]),
                        }
                    )
            cv_by_alpha.append(
                {
                    "alpha": alpha,
                    "alpha_index": alpha_index,
                    "mean_selected_success": float(
                        np.mean([row["success"] for row in selected_records])
                    ),
                    "mean_selected_progress": float(
                        np.mean([row["progress"] for row in selected_records])
                    ),
                    "state_count": len(selected_records),
                }
            )
        chosen = sorted(
            cv_by_alpha,
            key=lambda row: (
                -row["mean_selected_success"],
                -row["mean_selected_progress"],
                row["alpha_index"],
            ),
        )[0]
        context_pca = fit_pca(
            context_raw[development_indices],
            int(config["feature_reduction"]["context_embedding_pca_dimensions"]),
        )
        candidate_pca = fit_pca(
            cache["candidate_embeddings"][development_indices].reshape(-1, 512),
            int(config["feature_reduction"]["candidate_embedding_pca_dimensions"]),
        )
        train_features, train_identity = build_feature_rows(
            state_indices=development_indices,
            states=all_states,
            candidate_ids=candidate_ids,
            geometry=geometry_array,
            cache=cache,
            tasks=tasks,
            context_pca=context_pca,
            candidate_pca=candidate_pca,
            family=family,
        )
        train_targets = np.asarray([success[state, candidate] for state, candidate in train_identity])
        ridge = fit_ridge(train_features, train_targets, float(chosen["alpha"]))
        test_features, test_identity = build_feature_rows(
            state_indices=test_indices,
            states=all_states,
            candidate_ids=candidate_ids,
            geometry=geometry_array,
            cache=cache,
            tasks=tasks,
            context_pca=context_pca,
            candidate_pca=candidate_pca,
            family=family,
        )
        test_selected = select_predictions(ridge.predict(test_features), test_identity, candidate_ids)
        model_path = model_dir / f"{family}.npz"
        np.savez_compressed(
            model_path,
            feature_mean=ridge.mean,
            feature_scale=ridge.scale,
            beta=ridge.beta,
            intercept=np.asarray([ridge.intercept]),
            context_pca_mean=context_pca[0],
            context_pca_components=context_pca[1],
            candidate_pca_mean=candidate_pca[0],
            candidate_pca_components=candidate_pca[1],
            candidate_ids=np.asarray(candidate_ids),
            tasks=np.asarray(tasks),
        )
        selections = {
            all_states[state]["pair_key"]: candidate_ids[candidate]
            for state, candidate in test_selected.items()
        }
        deployment_class = config["model_families"][family]["deployment_class"]
        learned_methods[family] = {
            "deployment_class": deployment_class,
            "state_selections": selections,
            "model_artifact": str(model_path.resolve()),
            "model_artifact_sha256": sha256_file(model_path),
        }
        family_results[family] = {
            "cv_by_alpha": cv_by_alpha,
            "chosen_alpha": chosen["alpha"],
            "chosen_cv_success": chosen["mean_selected_success"],
            "chosen_cv_progress": chosen["mean_selected_progress"],
            "feature_dimension": int(train_features.shape[1]),
            "model_artifact": str(model_path.resolve()),
            "model_artifact_sha256": sha256_file(model_path),
        }
    methods = select_fixed_baselines(
        development_states=development,
        test_states=test,
        candidate_ids=candidate_ids,
        success=success,
        progress=progress,
        scan_assets=scan_assets,
        accel_index=accel_index,
    )
    methods.update(learned_methods)
    receipt = {
        "schema": "dsol_statewise_view_selector_freeze_v1",
        "status": "FROZEN_BEFORE_TEST_OUTCOMES_OPENED",
        "test_outcomes_read": False,
        "learner_config": str(args.learner_config.resolve()),
        "learner_config_sha256": sha256_file(args.learner_config),
        "population": str(args.population.resolve()),
        "population_sha256": sha256_file(args.population),
        "development_input_integrity_audits": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in matched_paths(args.input_audits)
        ],
        "embedding_cache": cache_receipt,
        "families": family_results,
        "methods": methods,
        "method_count": len(methods),
        "test_state_count": len(test),
        "manual_candidate_override": False,
    }
    freeze_json(args.output_dir / "selector-freeze-receipt.json", receipt)
    atomic_json(
        args.output_dir / "development-cv-report.json",
        {
            "schema": "dsol_statewise_view_selector_development_cv_v1",
            "status": "PASS",
            "test_outcomes_read": False,
            "families": family_results,
        },
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--learner-config", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--input-ledgers", nargs="+", required=True)
    parser.add_argument("--input-audits", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    for field in ("population", "learner_config", "legacy_root", "output_dir"):
        setattr(args, field, getattr(args, field).resolve())
    receipt = train(args)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "methods": receipt["method_count"],
                "test_states": receipt["test_state_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
