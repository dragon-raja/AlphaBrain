from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    from scripts.fresh_vla.paired_evaluation import bootstrap_summary
except ModuleNotFoundError:
    from paired_evaluation import bootstrap_summary


MODALITIES = ("agentview", "wrist", "vision", "robot_state", "vision_state")
LABELS = {"attached": 1.0, "slipped": -1.0}


def block_average(image: np.ndarray, size: int) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got {value.shape}")
    if size <= 0 or value.shape[0] % size or value.shape[1] % size:
        raise ValueError(f"image shape {value.shape[:2]} is not divisible by output size {size}")
    row_block = value.shape[0] // size
    column_block = value.shape[1] // size
    pooled = value.reshape(size, row_block, size, column_block, 3).mean(axis=(1, 3))
    return (pooled / 255.0).reshape(-1)


def feature_vector(arrays: Mapping[str, np.ndarray], index: int, modality: str, image_size: int) -> np.ndarray:
    agent = block_average(arrays["agentview"][index], image_size)
    wrist = block_average(arrays["wrist"][index], image_size)
    state = np.asarray(arrays["robot_state"][index], dtype=np.float32).reshape(-1)
    if modality == "agentview":
        return agent
    if modality == "wrist":
        return wrist
    if modality == "vision":
        return np.concatenate((agent, wrist))
    if modality == "robot_state":
        return state
    if modality == "vision_state":
        return np.concatenate((agent, wrist, state))
    raise ValueError(f"unknown modality: {modality}")


def standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_value = (train - mean) / scale
    other_value = (other - mean) / scale
    return train_value, other_value


def ridge_scores(
    train: np.ndarray,
    labels: np.ndarray,
    query: np.ndarray,
    regularization: float,
) -> np.ndarray:
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    train_value, query_value = standardize(train, query)
    train_value = np.concatenate((train_value, np.ones((len(train_value), 1), dtype=np.float64)), axis=1)
    query_value = np.concatenate((query_value, np.ones((len(query_value), 1), dtype=np.float64)), axis=1)
    kernel = train_value @ train_value.T
    alpha = np.linalg.solve(kernel + regularization * np.eye(len(kernel)), labels)
    return query_value @ train_value.T @ alpha


def accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    predictions = np.where(scores >= 0, 1.0, -1.0)
    return float(np.mean(predictions == labels))


def summarize_predictions(
    metadata: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    predictions = np.where(scores >= 0, 1.0, -1.0)
    groups: dict[str, list[float]] = defaultdict(list)
    sources: dict[str, list[float]] = defaultdict(list)
    branch_correct: dict[str, list[float]] = defaultdict(list)
    paired_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for row, label, prediction, score in zip(metadata, labels, predictions, scores, strict=True):
        correct = float(label == prediction)
        pair_id = str(row["pair_id"])
        outcome = str(row["outcome"])
        groups[pair_id].append(correct)
        sources[str(row["source_initial_state_index"])].append(correct)
        branch_correct[outcome].append(correct)
        paired_scores[pair_id][outcome] = float(score)
    group_values = [float(np.mean(values)) for values in groups.values()]
    source_values = [float(np.mean(values)) for values in sources.values()]
    ranking = [
        float(values["attached"] > values["slipped"])
        for values in paired_scores.values()
        if values.keys() >= {"attached", "slipped"}
    ]
    return {
        "sample_accuracy": accuracy(labels, scores),
        "attached_accuracy": float(np.mean(branch_correct["attached"])),
        "slipped_accuracy": float(np.mean(branch_correct["slipped"])),
        "pair_ranking_accuracy": float(np.mean(ranking)),
        "group_bootstrap_95": bootstrap_summary(group_values, seed=seed),
        "source_state_bootstrap_95": bootstrap_summary(source_values, seed=seed + 1),
        "group_count": len(groups),
        "source_state_count": len(sources),
    }


def load_examples(
    episode_root: Path,
    groups: Sequence[Mapping[str, object]],
    *,
    offset: int,
    image_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, object]]]:
    features: dict[str, list[np.ndarray]] = {modality: [] for modality in MODALITIES}
    labels = []
    metadata = []
    for group in groups:
        feedback_time = int(group["feedback_reveal_time"])
        for outcome in ("attached", "slipped"):
            episode_path = episode_root / str(group["episode_files"][outcome])
            with np.load(episode_path, allow_pickle=False) as arrays:
                index = feedback_time + offset
                if not 0 <= index < len(arrays["robot_state"]):
                    raise ValueError(f"offset {offset} is out of range for {episode_path}")
                for modality in MODALITIES:
                    features[modality].append(feature_vector(arrays, index, modality, image_size))
            labels.append(LABELS[outcome])
            metadata.append(
                {
                    "pair_id": group["pair_id"],
                    "outcome": outcome,
                    "source_initial_state_index": group["source_initial_state_index"],
                }
            )
    return (
        {modality: np.asarray(values, dtype=np.float64) for modality, values in features.items()},
        np.asarray(labels, dtype=np.float64),
        metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test whether branch outcome is observable at feedback reveal")
    parser.add_argument(
        "--episode-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/libero-full-episode-v2-128"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/research-reset/feedback_observability.json"),
    )
    parser.add_argument("--offsets", nargs="+", type=int, default=(-1, 0, 1, 2, 3, 5))
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--shuffle-repeats", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    groups_by_split = {
        split: sorted(
            (group for group in manifest["groups"] if group["split"] == split),
            key=lambda group: str(group["pair_id"]),
        )
        for split in ("train", "val", "test")
    }
    source_sets = {
        split: {int(group["source_initial_state_index"]) for group in groups}
        for split, groups in groups_by_split.items()
    }
    if any(source_sets[left] & source_sets[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("source initial states overlap across splits")

    regularizations = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    results: dict[str, dict[str, object]] = {}
    cached: dict[tuple[int, str], tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, object]]]] = {}
    for offset_index, offset in enumerate(args.offsets):
        results[str(offset)] = {}
        for split in groups_by_split:
            cached[(offset, split)] = load_examples(
                args.episode_root,
                groups_by_split[split],
                offset=offset,
                image_size=args.image_size,
            )
        for modality_index, modality in enumerate(MODALITIES):
            train_features, train_labels, _ = cached[(offset, "train")]
            val_features, val_labels, _ = cached[(offset, "val")]
            train = train_features[modality]
            val = val_features[modality]
            validation = {
                str(value): accuracy(val_labels, ridge_scores(train, train_labels, val, value))
                for value in regularizations
            }
            selected = max(regularizations, key=lambda value: (validation[str(value)], -value))
            fit = np.concatenate((train, val), axis=0)
            fit_labels = np.concatenate((train_labels, val_labels), axis=0)
            test_features, test_labels, test_metadata = cached[(offset, "test")]
            test = test_features[modality]
            scores = ridge_scores(fit, fit_labels, test, selected)
            results[str(offset)][modality] = {
                "selected_regularization": selected,
                "validation_accuracy_by_regularization": validation,
                **summarize_predictions(
                    test_metadata,
                    test_labels,
                    scores,
                    seed=args.seed + offset_index * 100 + modality_index * 2,
                ),
            }

    train_features, train_labels, _ = cached[(0, "train")]
    val_features, val_labels, _ = cached[(0, "val")]
    train = train_features["vision_state"]
    val = val_features["vision_state"]
    fit = np.concatenate((train, val), axis=0)
    fit_labels = np.concatenate((train_labels, val_labels), axis=0)
    test_features, test_labels, _ = cached[(0, "test")]
    test = test_features["vision_state"]
    regularization = float(results["0"]["vision_state"]["selected_regularization"])
    rng = np.random.default_rng(args.seed)
    shuffled_accuracies = []
    for _ in range(args.shuffle_repeats):
        shuffled = rng.permutation(fit_labels)
        shuffled_accuracies.append(accuracy(test_labels, ridge_scores(fit, shuffled, test, regularization)))

    post = results["0"]["vision_state"]
    pre = results["-1"]["vision_state"]
    shuffle_mean = float(np.mean(shuffled_accuracies))
    supports_single_frame_observability = bool(
        float(post["sample_accuracy"]) >= 0.85
        and float(post["group_bootstrap_95"]["bootstrap_95_low"]) > 0.70
        and float(pre["sample_accuracy"]) <= 0.60
        and shuffle_mean <= 0.60
    )
    payload = {
        "experiment": "feedback_observability_probe",
        "hypothesis": "post-feedback policy inputs make attached versus slipped outcomes linearly identifiable",
        "go_criteria": {
            "post_feedback_vision_state_accuracy_min": 0.85,
            "post_feedback_group_ci_low_strictly_above": 0.70,
            "pre_feedback_accuracy_max": 0.60,
            "shuffled_label_mean_accuracy_max": 0.60,
        },
        "supports_single_frame_observability": supports_single_frame_observability,
        "split_group_counts": {split: len(groups) for split, groups in groups_by_split.items()},
        "split_source_state_counts": {split: len(values) for split, values in source_sets.items()},
        "source_state_disjoint": True,
        "offset_definition": "frame_index = feedback_reveal_time + offset",
        "image_feature": f"{args.image_size}x{args.image_size} RGB block averages per view",
        "modalities": list(MODALITIES),
        "results": results,
        "shuffled_label_control": {
            "offset": 0,
            "modality": "vision_state",
            "repeats": args.shuffle_repeats,
            "mean_accuracy": shuffle_mean,
            "min_accuracy": float(np.min(shuffled_accuracies)),
            "max_accuracy": float(np.max(shuffled_accuracies)),
            "accuracies": shuffled_accuracies,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "supports_single_frame_observability": supports_single_frame_observability,
                "pre_feedback_accuracy": pre["sample_accuracy"],
                "post_feedback_accuracy": post["sample_accuracy"],
                "shuffled_label_mean_accuracy": shuffle_mean,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
