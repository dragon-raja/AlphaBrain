from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
METHODS = ("candidate", "direct", "response", "candidate_response")
BOOTSTRAP_SAMPLES = 20_000
UTILITY_WEIGHTS = np.asarray([8.0**2, 8.0**3, 8.0**4, 8.0**5, 8.0, 1.0])
UTILITY_DENOMINATOR = float(sum(8.0**power for power in range(6)))


@dataclass(frozen=True)
class Record:
    record_id: str
    source_id: int
    candidate: np.ndarray
    direct: np.ndarray
    response: np.ndarray
    profiles: np.ndarray

    @property
    def utilities(self) -> np.ndarray:
        return (self.profiles @ UTILITY_WEIGHTS) / UTILITY_DENOMINATOR


def frozen_source_split(source_ids: Sequence[int], evaluation_count: int = 5) -> tuple[tuple[int, ...], tuple[int, ...]]:
    unique = sorted({int(value) for value in source_ids})
    if len(unique) <= evaluation_count:
        raise ValueError("source split leaves no fitting sources")
    ranked = sorted(
        unique,
        key=lambda value: (
            hashlib.sha256(f"policy-response-gate-minus1-v1::{value}".encode("ascii")).digest(),
            value,
        ),
    )
    evaluation = tuple(sorted(ranked[:evaluation_count]))
    fitting = tuple(sorted(set(unique) - set(evaluation)))
    return fitting, evaluation


def feature(record: Record, method: str) -> np.ndarray:
    if method == "candidate":
        value = record.candidate
    elif method == "direct":
        value = record.direct
    elif method == "response":
        value = record.response
    elif method == "candidate_response":
        value = np.concatenate([record.candidate, record.response], axis=1)
    else:
        raise ValueError(f"unknown method: {method}")
    return value - value.mean(axis=0, keepdims=True)


def load_records(root: Path) -> tuple[list[Record], Mapping[str, object]]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("source_partition") != "fit":
        raise ValueError("requires a complete fit-only response collection")
    records = []
    for relative in manifest["records"]:
        path = root / str(relative)
        lowered = {part.lower() for part in path.parts}
        if lowered & {"holdout", "test", "confirmation", "sealed"}:
            raise ValueError(f"refusing non-fit record path: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            candidate = np.asarray(arrays["candidate_prefixes"], dtype=np.float64).reshape(16, -1)
            direct = np.asarray(arrays["direct_signatures"], dtype=np.float64)
            response = np.asarray(arrays["response_actions"], dtype=np.float64)[:, :, :8].reshape(16, -1)
            profiles = np.asarray(arrays["continuation_profiles"], dtype=np.float64)
            source_id = int(arrays["source_id"])
        if candidate.shape != (16, 14) or direct.shape != (16, 6) or response.shape != (16, 112):
            raise ValueError(f"unexpected feature shape in {path}")
        if profiles.shape != (16, 6):
            raise ValueError(f"unexpected profile shape in {path}")
        records.append(Record(str(relative), source_id, candidate, direct, response, profiles))
    return records, manifest


def grouped_folds(source_ids: Sequence[int], fold_count: int = 5) -> list[set[int]]:
    ranked = sorted(
        {int(value) for value in source_ids},
        key=lambda value: (
            hashlib.sha256(f"policy-response-cv-v1::{value}".encode("ascii")).digest(),
            value,
        ),
    )
    return [set(ranked[index::fold_count]) for index in range(fold_count)]


def design(records: Sequence[Record], method: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([feature(record, method) for record in records], axis=0)
    y = np.concatenate(
        [record.utilities - record.utilities.mean() for record in records], axis=0
    )
    return x, y


@dataclass(frozen=True)
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def predict(self, value: np.ndarray) -> np.ndarray:
        return ((value - self.mean) / self.scale) @ self.weights


def fit_ridge(records: Sequence[Record], method: str, alpha: float) -> RidgeModel:
    x, y = design(records, method)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    normalized = (x - mean) / scale
    gram = normalized.T @ normalized
    weights = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), normalized.T @ y)
    return RidgeModel(mean, scale, weights)


def choose_alpha(records: Sequence[Record], method: str) -> tuple[float, dict[str, float]]:
    folds = grouped_folds([record.source_id for record in records])
    losses = {alpha: [] for alpha in ALPHAS}
    for held_sources in folds:
        train = [record for record in records if record.source_id not in held_sources]
        held = [record for record in records if record.source_id in held_sources]
        for alpha in ALPHAS:
            model = fit_ridge(train, method, alpha)
            for record in held:
                prediction = model.predict(feature(record, method))
                target = record.utilities - record.utilities.mean()
                losses[alpha].append(float(np.square(prediction - target).mean()))
    means = {str(alpha): float(np.mean(values)) for alpha, values in losses.items()}
    selected = min(ALPHAS, key=lambda alpha: (means[str(alpha)], alpha))
    return selected, means


def stable_grasp_harm(record: Record, selected: int) -> bool:
    chosen, base = record.profiles[selected], record.profiles[0]
    deeper_tied = bool(np.allclose(chosen[[3, 2, 1]], base[[3, 2, 1]], atol=1e-8))
    return deeper_tied and bool(chosen[0] + 1e-8 < base[0])


def record_metrics(record: Record, scores: np.ndarray) -> dict[str, float]:
    utility = record.utilities
    selected = int(np.argmax(scores))
    oracle = float(utility.max() - utility[0])
    pairs_total = 0
    pairs_correct = 0.0
    for left in range(16):
        for right in range(left + 1, 16):
            target_delta = float(utility[left] - utility[right])
            if abs(target_delta) <= 1e-10:
                continue
            score_delta = float(scores[left] - scores[right])
            pairs_total += 1
            pairs_correct += float(np.sign(score_delta) == np.sign(target_delta))
    return {
        "gain": float(utility[selected] - utility[0]),
        "oracle_gain": oracle,
        "concordant": pairs_correct,
        "comparable": float(pairs_total),
        "top_hit": float(abs(float(utility[selected] - utility.max())) <= 1e-10),
        "stable_grasp_harm": float(stable_grasp_harm(record, selected)),
    }


def aggregate(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    gain = float(np.mean([row["gain"] for row in rows]))
    oracle = float(np.mean([row["oracle_gain"] for row in rows]))
    comparable = float(sum(row["comparable"] for row in rows))
    return {
        "utility_gain": gain,
        "oracle_gain": oracle,
        "oracle_gain_recovered": gain / oracle if oracle > 0 else 0.0,
        "pairwise_concordance": float(sum(row["concordant"] for row in rows) / comparable) if comparable else 0.5,
        "oracle_top_set_hit_rate": float(np.mean([row["top_hit"] for row in rows])),
        "stable_grasp_harm_rate": float(np.mean([row["stable_grasp_harm"] for row in rows])),
    }


def source_rows(records: Sequence[Record], models: Mapping[str, RidgeModel]) -> dict[str, dict[int, dict[str, float]]]:
    output = {method: {} for method in METHODS}
    for method, model in models.items():
        by_source: dict[int, list[dict[str, float]]] = {}
        for record in records:
            scores = model.predict(feature(record, method))
            by_source.setdefault(record.source_id, []).append(record_metrics(record, scores))
        output[method] = {source: aggregate(rows) for source, rows in by_source.items()}
    return output


def aggregate_sources(rows: Mapping[int, Mapping[str, float]]) -> dict[str, float]:
    values = list(rows.values())
    gain = float(np.mean([row["utility_gain"] for row in values]))
    oracle = float(np.mean([row["oracle_gain"] for row in values]))
    return {
        "utility_gain": gain,
        "oracle_gain": oracle,
        "oracle_gain_recovered": gain / oracle if oracle > 0 else 0.0,
        "pairwise_concordance": float(np.mean([row["pairwise_concordance"] for row in values])),
        "oracle_top_set_hit_rate": float(np.mean([row["oracle_top_set_hit_rate"] for row in values])),
        "stable_grasp_harm_rate": float(np.mean([row["stable_grasp_harm_rate"] for row in values])),
    }


def bootstrap_intervals(
    grouped: Mapping[str, Mapping[int, Mapping[str, float]]], seed: int = 260719
) -> dict[str, object]:
    sources = sorted(next(iter(grouped.values())))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sources), size=(BOOTSTRAP_SAMPLES, len(sources)))
    oracle = np.asarray([grouped["candidate"][source]["oracle_gain"] for source in sources])
    ratios = {}
    for method in METHODS:
        gains = np.asarray([grouped[method][source]["utility_gain"] for source in sources])
        sample_gain = gains[draws].mean(axis=1)
        sample_oracle = oracle[draws].mean(axis=1)
        ratio = np.divide(sample_gain, sample_oracle, out=np.zeros_like(sample_gain), where=sample_oracle > 0)
        ratios[method] = ratio
    best_baseline = np.maximum(ratios["candidate"], ratios["direct"])

    def interval(values: np.ndarray) -> list[float]:
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

    return {
        "oracle_gain_95": interval(oracle[draws].mean(axis=1)),
        "recovered_gain_95": {method: interval(values) for method, values in ratios.items()},
        "candidate_response_minus_best_baseline_95": interval(
            ratios["candidate_response"] - best_baseline
        ),
    }


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate policy-response surrogate Gate -1")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, manifest = load_records(args.root)
    fitting_sources, evaluation_sources = frozen_source_split([record.source_id for record in records])
    fitting = [record for record in records if record.source_id in fitting_sources]
    evaluation = [record for record in records if record.source_id in evaluation_sources]
    models = {}
    alpha_rows = {}
    for method in METHODS:
        alpha, losses = choose_alpha(fitting, method)
        models[method] = fit_ridge(fitting, method, alpha)
        alpha_rows[method] = {"selected": alpha, "cv_mse": losses}
    grouped = source_rows(evaluation, models)
    metrics = {method: aggregate_sources(grouped[method]) for method in METHODS}
    intervals = bootstrap_intervals(grouped)
    candidate_response = metrics["candidate_response"]
    best_baseline = max(metrics["candidate"]["oracle_gain_recovered"], metrics["direct"]["oracle_gain_recovered"])
    checks = {
        "oracle_gain_ci_low_above_zero": intervals["oracle_gain_95"][0] > 0,
        "candidate_response_recovers_35pct": candidate_response["oracle_gain_recovered"] >= 0.35,
        "beats_candidate_and_direct_by_10pp": candidate_response["oracle_gain_recovered"] - best_baseline >= 0.10,
        "paired_improvement_ci_low_above_zero": intervals["candidate_response_minus_best_baseline_95"][0] > 0,
        "stable_grasp_harm_at_most_5pct": candidate_response["stable_grasp_harm_rate"] <= 0.05,
        "response_concordance_above_055": metrics["response"]["pairwise_concordance"] > 0.55,
    }
    decision = (
        "PROCEED_POLICY_RESPONSE_MODEL_GATE0"
        if all(checks.values())
        else "STOP_POLICY_RESPONSE_SURROGATE"
    )
    payload = {
        "experiment": "policy_response_surrogate_gate_minus1",
        "decision": decision,
        "collection_preregistration_sha256": manifest["preregistration_sha256"],
        "fit_source_ids": list(fitting_sources),
        "evaluation_source_ids": list(evaluation_sources),
        "fit_state_count": len(fitting),
        "evaluation_state_count": len(evaluation),
        "alpha_selection": alpha_rows,
        "metrics": metrics,
        "bootstrap": intervals,
        "checks": checks,
        "ccv_holdout_states_opened": 0,
        "test_or_confirmation_states_opened": 0,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"decision": decision, "checks": checks, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
