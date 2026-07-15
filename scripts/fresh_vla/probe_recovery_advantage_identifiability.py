from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_libero_closed_loop import _atomic_write_json
from evaluate_recovery_segment_oracle import (
    BINARY_METRICS,
    METRICS,
    aggregate_recovery_preference_key,
)
from paired_evaluation import bootstrap_summary
from probe_feedback_observability import block_average
from summarize_recovery_segment_oracle import load_and_validate


FIXED_REGULARIZATION = 1.0
FEATURE_CONFIGS = {
    "action_all": ("action", False),
    "current_all": ("current", False),
    "action_post0": ("action", True),
    "current_post0": ("current", True),
    "oac_history_post0": ("history", True),
}


@dataclass(frozen=True)
class DecisionRecord:
    seed: int
    pair_id: str
    source: str
    replan_index: int
    current_context: np.ndarray
    candidate_actions: np.ndarray
    selection_top_set: tuple[int, ...]
    heldout_top_set: tuple[int, ...]
    previous_context: np.ndarray | None = None
    previous_selected_action: np.ndarray | None = None


def metric_summary(vector: np.ndarray) -> dict[str, float]:
    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (len(METRICS),) or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid recovery metric vector shape/value: {values.shape}")
    return {
        (f"{metric}_rate" if metric in BINARY_METRICS else metric): float(value)
        for metric, value in zip(METRICS, values, strict=True)
    }


def top_set(metric_matrix: np.ndarray) -> tuple[int, ...]:
    values = np.asarray(metric_matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 4 or values.shape[1] != len(METRICS):
        raise ValueError(f"candidate metric matrix must be [4,{len(METRICS)}]")
    keys = [aggregate_recovery_preference_key(metric_summary(row)) for row in values]
    best = max(keys)
    return tuple(index for index, key in enumerate(keys) if key == best)


def current_context(images: np.ndarray, robot_state: np.ndarray, image_size: int) -> np.ndarray:
    views = np.asarray(images)
    if views.shape != (2, 224, 224, 3):
        raise ValueError(f"expected two 224x224 RGB views, found {views.shape}")
    return np.concatenate(
        (
            block_average(views[0], image_size),
            block_average(views[1], image_size),
            np.asarray(robot_state, dtype=np.float32).reshape(-1),
        )
    ).astype(np.float64)


def _load_bank_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    image_size: int,
) -> list[DecisionRecord]:
    base: list[dict[str, Any]] = []
    for row in rows:
        training_path = Path(row["training_bank_file"])
        audit_path = Path(row["privileged_audit_bank_file"])
        with (
            np.load(training_path, allow_pickle=False) as training,
            np.load(audit_path, allow_pickle=False) as audit,
        ):
            required_training = {
                "images",
                "robot_state",
                "candidate_action_prefix",
                "candidate_action_mask",
                "oracle_index",
                "replan_index",
                "decision_uid",
                "source_initial_state_index",
                "model_seed",
                "candidate_selection_metrics",
            }
            if not required_training <= set(training.files):
                raise ValueError(f"training bank is missing required fields: {training_path}")
            if "candidate_decision_heldout_metrics" not in audit.files:
                raise ValueError(f"audit bank lacks heldout metrics: {audit_path}")
            if "decision_heldout_oracle_index" not in audit.files:
                raise ValueError(f"audit bank lacks heldout Oracle indices: {audit_path}")
            decision_count = len(training["replan_index"])
            if decision_count != 4:
                raise ValueError(f"formal bank must contain four decisions: {training_path}")
            if not np.all(training["candidate_action_mask"]):
                raise ValueError(
                    f"candidate_action_mask is not all true and may leak early termination: {training_path}"
                )
            selection_metrics = np.asarray(
                training["candidate_selection_metrics"], dtype=np.float64
            )
            heldout_metrics = np.asarray(
                audit["candidate_decision_heldout_metrics"], dtype=np.float64
            )
            if selection_metrics.shape != (4, 4, len(METRICS)):
                raise ValueError(f"invalid selection metrics in {training_path}")
            if heldout_metrics.shape != selection_metrics.shape:
                raise ValueError(f"selection/heldout metric shape mismatch in {audit_path}")

            for decision_index in range(decision_count):
                seed = int(training["model_seed"][decision_index])
                source = str(int(training["source_initial_state_index"][decision_index]))
                if seed != int(row["seed"]) or source != str(
                    int(row["source_initial_state_index"])
                ):
                    raise ValueError(f"bank provenance disagrees with JSON row: {training_path}")
                selection_top = top_set(selection_metrics[decision_index])
                heldout_top = top_set(heldout_metrics[decision_index])
                selected_index = int(training["oracle_index"][decision_index])
                heldout_index = int(audit["decision_heldout_oracle_index"][decision_index])
                if selected_index != selection_top[0] or heldout_index != heldout_top[0]:
                    raise ValueError(
                        f"bank Oracle index disagrees with recomputed top set: {training_path}"
                    )
                base.append(
                    {
                        "seed": seed,
                        "pair_id": str(row["pair_id"]),
                        "source": source,
                        "replan_index": int(training["replan_index"][decision_index]),
                        "context": current_context(
                            training["images"][decision_index],
                            training["robot_state"][decision_index],
                            image_size,
                        ),
                        "actions": np.asarray(
                            training["candidate_action_prefix"][decision_index],
                            dtype=np.float64,
                        ).reshape(4, -1),
                        "selected_index": selected_index,
                        "selection_top_set": selection_top,
                        "heldout_top_set": heldout_top,
                    }
                )

    by_episode: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in base:
        by_episode.setdefault((row["seed"], row["pair_id"]), []).append(row)
    records = []
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda value: value["replan_index"])
        if [value["replan_index"] for value in episode_rows] != [0, 1, 2, 3]:
            raise ValueError("each episode must contain ordered replans 0..3")
        for index, value in enumerate(episode_rows):
            previous = episode_rows[index - 1] if index else None
            records.append(
                DecisionRecord(
                    seed=value["seed"],
                    pair_id=value["pair_id"],
                    source=value["source"],
                    replan_index=value["replan_index"],
                    current_context=value["context"],
                    candidate_actions=value["actions"],
                    selection_top_set=value["selection_top_set"],
                    heldout_top_set=value["heldout_top_set"],
                    previous_context=(previous["context"] if previous else None),
                    previous_selected_action=(
                        previous["actions"][previous["selected_index"]]
                        if previous
                        else None
                    ),
                )
            )
    return records


def candidate_feature(record: DecisionRecord, candidate_index: int, mode: str) -> np.ndarray:
    action = np.asarray(record.candidate_actions[candidate_index], dtype=np.float64)
    if mode == "action":
        return action
    if mode == "current":
        context = record.current_context
    elif mode == "history":
        if record.previous_context is None or record.previous_selected_action is None:
            raise ValueError("history feature requires a previous O-A-C step")
        context = np.concatenate(
            (
                record.current_context,
                record.current_context - record.previous_context,
                record.previous_selected_action,
            )
        )
    else:
        raise ValueError(f"unknown feature mode: {mode}")
    return np.concatenate((action, np.multiply.outer(context, action).reshape(-1)))


def strict_pairs(top: Sequence[int], candidate_count: int = 4) -> list[tuple[int, int]]:
    winners = set(int(value) for value in top)
    return [
        (winner, loser)
        for winner in sorted(winners)
        for loser in range(candidate_count)
        if loser not in winners
    ]


def training_matrix(
    records: Sequence[DecisionRecord],
    *,
    mode: str,
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for record in records:
        top = getattr(record, f"{target}_top_set")
        for winner, loser in strict_pairs(top):
            left, right = sorted((winner, loser))
            difference = candidate_feature(record, left, mode) - candidate_feature(
                record, right, mode
            )
            features.append(difference)
            labels.append(1.0 if left == winner else -1.0)
    if not features:
        raise ValueError("no strict preference pairs remain after tie abstention")
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.float64)


@dataclass(frozen=True)
class RidgeRanker:
    train_scaled: np.ndarray
    alpha: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    def scores(self, record: DecisionRecord, mode: str) -> np.ndarray:
        features = np.stack(
            [candidate_feature(record, index, mode) for index in range(4)]
        )
        scaled = (features - self.mean) / self.scale
        return scaled @ self.train_scaled.T @ self.alpha


def fit_ranker(
    records: Sequence[DecisionRecord],
    *,
    mode: str,
    target: str,
    regularization: float,
) -> RidgeRanker:
    features, labels = training_matrix(records, mode=mode, target=target)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    scaled = (features - mean) / scale
    kernel = scaled @ scaled.T
    alpha = np.linalg.solve(
        kernel + regularization * np.eye(len(kernel), dtype=np.float64),
        labels,
    )
    return RidgeRanker(scaled, alpha, mean, scale)


def evaluate_ranker(
    ranker: RidgeRanker,
    records: Sequence[DecisionRecord],
    *,
    mode: str,
    target: str,
) -> list[dict[str, float | str | int | None]]:
    rows = []
    for record in records:
        scores = ranker.scores(record, mode)
        top = getattr(record, f"{target}_top_set")
        pairs = strict_pairs(top)
        pair_values = [
            float(scores[winner] > scores[loser])
            + 0.5 * float(scores[winner] == scores[loser])
            for winner, loser in pairs
        ]
        rows.append(
            {
                "seed": record.seed,
                "pair_id": record.pair_id,
                "source": record.source,
                "replan_index": record.replan_index,
                "strict_pair_accuracy": (
                    float(np.mean(pair_values)) if pair_values else None
                ),
                "unique_best_accuracy": (
                    float(int(np.argmax(scores)) == top[0]) if len(top) == 1 else None
                ),
                "top_set_hit": float(int(np.argmax(scores)) in top),
                "heldout_top_set_size": len(top),
            }
        )
    return rows


def mean_available(rows: Sequence[Mapping[str, Any]], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if row[metric] is not None]
    return float(np.mean(values)) if values else None


def summarize_cross_validated(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "decision_count": len(rows),
        "strict_pair_decision_count": sum(
            row["strict_pair_accuracy"] is not None for row in rows
        ),
        "unique_best_decision_count": sum(
            row["unique_best_accuracy"] is not None for row in rows
        ),
        "raw_top_set_hit": mean_available(rows, "top_set_hit"),
    }
    for metric_index, metric in enumerate(
        ("strict_pair_accuracy", "unique_best_accuracy", "top_set_hit")
    ):
        by_source: dict[str, list[float]] = {}
        for row in rows:
            value = row[metric]
            if value is not None:
                by_source.setdefault(str(row["source"]), []).append(float(value))
        source_values = {
            source: float(np.mean(values)) for source, values in sorted(by_source.items())
        }
        result[metric] = {
            "source_cluster_level": (
                bootstrap_summary(
                    list(source_values.values()),
                    seed=seed + metric_index,
                )
                if source_values
                else None
            ),
            "per_source_cluster": source_values,
        }
    return result


def source_loso_probe(
    records: Sequence[DecisionRecord],
    *,
    mode: str,
    require_history: bool,
    train_target: str,
    seed: int,
) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if not require_history or record.previous_context is not None
    ]
    sources = sorted({record.source for record in selected})
    predictions = []
    for source in sources:
        train = [record for record in selected if record.source != source]
        query = [record for record in selected if record.source == source]
        ranker = fit_ranker(
            train,
            mode=mode,
            target=train_target,
            regularization=FIXED_REGULARIZATION,
        )
        predictions.extend(
            evaluate_ranker(ranker, query, mode=mode, target="heldout")
        )
    return {
        "feature_mode": mode,
        "requires_previous_oac": require_history,
        "training_label": train_target,
        "evaluation_label": "decision_heldout",
        "source_disjoint_loso": True,
        "tie_policy": "abstain from strict pairs; unique-best metric excludes tied decisions",
        "regularization": FIXED_REGULARIZATION,
        "regularization_selection": "fixed before evaluation; no target-dependent tuning",
        "summary": summarize_cross_validated(predictions, seed=seed),
    }


def baseline_summary(records: Sequence[DecisionRecord], *, require_history: bool) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if not require_history or record.previous_context is not None
    ]
    unique = [record for record in selected if len(record.heldout_top_set) == 1]
    return {
        "decision_count": len(selected),
        "unique_best_decision_count": len(unique),
        "candidate0_unique_best_accuracy": (
            float(np.mean([record.heldout_top_set == (0,) for record in unique]))
            if unique
            else None
        ),
        "uniform_random_unique_best_accuracy": 0.25 if unique else None,
        "uniform_random_expected_top_set_hit": float(
            np.mean([len(record.heldout_top_set) / 4.0 for record in selected])
        ),
        "strict_pair_chance": 0.5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether deployable O-A-C information predicts recovery action advantage"
    )
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20_260_715)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, validation = load_and_validate(args.inputs)
    if not validation["decision_eligible"]:
        raise ValueError("advantage identifiability probe requires a complete decision grid")
    records = _load_bank_records(rows, image_size=args.image_size)
    experiments = {}
    experiment_index = 0
    for train_target in ("selection", "heldout"):
        for name, (mode, require_history) in FEATURE_CONFIGS.items():
            experiments[f"{train_target}_to_heldout__{name}"] = source_loso_probe(
                records,
                mode=mode,
                require_history=require_history,
                train_target=train_target,
                seed=args.seed + experiment_index * 10,
            )
            experiment_index += 1
    payload = {
        "schema_version": 1,
        "experiment": "recovery_advantage_identifiability",
        "research_question": (
            "Can deployable current or one-step O-A-C information predict a recovery "
            "action ranking that is stable under held-out continuations?"
        ),
        "policy_input_only": True,
        "privileged_fields_used_only_as_labels": [
            "candidate_selection_metrics",
            "candidate_decision_heldout_metrics",
        ],
        "metric_order": list(METRICS),
        "image_feature": f"{args.image_size}x{args.image_size} block averages per view",
        "action_feature": "executed K3 candidate action prefix",
        "history_feature": (
            "current context, current-minus-previous context, previous executed selected action"
        ),
        "source_disjoint_loso": True,
        "decision_count": len(records),
        "source_cluster_count": len({record.source for record in records}),
        "baselines": {
            "all_decisions": baseline_summary(records, require_history=False),
            "post_replan0": baseline_summary(records, require_history=True),
        },
        "experiments": experiments,
        "interpretation_guard": (
            "This low-capacity probe can falsify gross deployable identifiability. A positive "
            "result is not sufficient for policy learning and must still pass N=1 closed loop."
        ),
    }
    _atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision_count": len(records),
                "selection_current_pair_accuracy": experiments[
                    "selection_to_heldout__current_post0"
                ]["summary"]["strict_pair_accuracy"]["source_cluster_level"]["mean"],
                "selection_history_pair_accuracy": experiments[
                    "selection_to_heldout__oac_history_post0"
                ]["summary"]["strict_pair_accuracy"]["source_cluster_level"]["mean"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
