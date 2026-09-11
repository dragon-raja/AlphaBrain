from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ccv import PROFILE_NAMES, scalar_viability_utility, stable_seed


METHODS = ("terminal_scalar", "dense_scalar", "ccv_profile")
SEEDS = (41, 42, 43)
THRESHOLDS = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)
TRAIN_STEPS = 3000
BATCH_STATES = 16


@dataclass(frozen=True)
class StateRecord:
    pair_id: str
    state_id: str
    source_id: int
    partition: str
    observation: np.ndarray
    actions: np.ndarray
    profiles: np.ndarray

    @property
    def utilities(self) -> np.ndarray:
        return np.asarray([scalar_viability_utility(row) for row in self.profiles])


def calibration_split(source_ids: Sequence[int], count: int = 5) -> tuple[tuple[int, ...], tuple[int, ...]]:
    unique = sorted({int(value) for value in source_ids})
    ranked = sorted(
        unique,
        key=lambda source_id: (
            hashlib.sha256(f"ccv-vla-calibration-v1::{source_id}".encode("ascii")).digest(),
            source_id,
        ),
    )
    calibration = tuple(sorted(ranked[:count]))
    optimizer = tuple(sorted(set(unique) - set(calibration)))
    return optimizer, calibration


def load_records(
    root: Path,
    *,
    partitions: Sequence[str],
) -> tuple[list[StateRecord], Mapping[str, object]]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("split") != "train":
        raise ValueError("ranker fitting requires a complete train-only CCV manifest")
    records = []
    for group in manifest["groups"]:
        if group["source_partition"] not in partitions:
            continue
        for state in group["states"]:
            deployable_path = root / state["deployable_file"]
            labels_path = root / state["labels_file"]
            if "audit" in str(deployable_path).lower() or "audit" in str(labels_path).lower():
                raise ValueError("critic loader may not open an audit path")
            with np.load(deployable_path, allow_pickle=False) as deployable:
                observation = np.concatenate(
                    [
                        np.asarray(deployable["vla_feature"], dtype=np.float32),
                        np.asarray(deployable["robot_state"], dtype=np.float32),
                    ]
                )
                actions = np.asarray(deployable["candidates"], dtype=np.float32)[:, :2].reshape(16, -1)
            with np.load(labels_path, allow_pickle=False) as labels:
                profiles = np.asarray(labels["continuation_profiles"], dtype=np.float32)
            if observation.shape != (4104,) or actions.shape != (16, 14) or profiles.shape != (16, 6):
                raise ValueError(
                    f"unexpected CCV shapes: observation={observation.shape}, actions={actions.shape}, profiles={profiles.shape}"
                )
            records.append(
                StateRecord(
                    pair_id=str(group["pair_id"]),
                    state_id=str(state["state_id"]),
                    source_id=int(group["source_initial_state_index"]),
                    partition=str(group["source_partition"]),
                    observation=observation,
                    actions=actions,
                    profiles=profiles,
                )
            )
    return records, manifest


@dataclass
class Normalizer:
    observation_mean: np.ndarray
    observation_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @classmethod
    def fit(cls, records: Sequence[StateRecord]) -> "Normalizer":
        observations = np.stack([record.observation for record in records])
        actions = np.concatenate([record.actions for record in records], axis=0)
        observation_std = observations.std(axis=0)
        action_std = actions.std(axis=0)
        return cls(
            observation_mean=observations.mean(axis=0),
            observation_std=np.where(observation_std < 1e-6, 1.0, observation_std),
            action_mean=actions.mean(axis=0),
            action_std=np.where(action_std < 1e-6, 1.0, action_std),
        )

    def arrays(self, records: Sequence[StateRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observations = np.stack(
            [(record.observation - self.observation_mean) / self.observation_std for record in records]
        ).astype(np.float32)
        actions = np.stack(
            [(record.actions - self.action_mean) / self.action_std for record in records]
        ).astype(np.float32)
        profiles = np.stack([record.profiles for record in records]).astype(np.float32)
        return observations, actions, profiles


class CandidateCritic(nn.Module):
    def __init__(self, method: str) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"unknown critic method: {method}")
        self.method = method
        self.observation_encoder = nn.Sequential(
            nn.Linear(4104, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(14, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Linear(128, 6 if method == "ccv_profile" else 1)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, candidates, _ = actions.shape
        obs = self.observation_encoder(observations)[:, None].expand(-1, candidates, -1)
        act = self.action_encoder(actions)
        raw = self.head(self.fusion(torch.cat([obs, act], dim=-1)))
        if self.method == "ccv_profile":
            conditional = torch.sigmoid(raw[..., :4])
            milestones = torch.cumprod(conditional, dim=-1)
            profile = torch.cat([milestones, torch.sigmoid(raw[..., 4:])], dim=-1)
            return profile, utility_tensor(profile)
        score = torch.sigmoid(raw[..., 0])
        return raw, score


def utility_tensor(profiles: torch.Tensor) -> torch.Tensor:
    weights = profiles.new_tensor([8.0**2, 8.0**3, 8.0**4, 8.0**5, 8.0, 1.0])
    return (profiles * weights).sum(dim=-1) / float(sum(8.0**power for power in range(6)))


def pairwise_ranking_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_diff = targets[:, :, None] - targets[:, None, :]
    score_diff = scores[:, :, None] - scores[:, None, :]
    upper = torch.triu(torch.ones_like(target_diff, dtype=torch.bool), diagonal=1)
    mask = upper & (target_diff.abs() > 1e-6)
    if not bool(mask.any()):
        return scores.sum() * 0.0
    signs = target_diff[mask].sign()
    return F.softplus(-10.0 * signs * score_diff[mask]).mean()


def critic_loss(method: str, predictions: torch.Tensor, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_utility = utility_tensor(targets)
    if method == "terminal_scalar":
        target = targets[..., 3]
        base = F.binary_cross_entropy(scores, target)
        ranking_target = target
    elif method == "dense_scalar":
        base = F.mse_loss(scores, target_utility)
        ranking_target = target_utility
    else:
        milestone = F.binary_cross_entropy(predictions[..., :4], targets[..., :4])
        no_regress = F.binary_cross_entropy(predictions[..., 4], targets[..., 4])
        progress = F.mse_loss(predictions[..., 5], targets[..., 5])
        base = milestone + no_regress + progress
        ranking_target = target_utility
    return base + 0.2 * pairwise_ranking_loss(scores, ranking_target)


def train_model(
    method: str,
    seed: int,
    records: Sequence[StateRecord],
    normalizer: Normalizer,
    *,
    device: torch.device,
) -> CandidateCritic:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = CandidateCritic(method).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    observations, actions, profiles = normalizer.arrays(records)
    rng = np.random.default_rng(stable_seed("ccv-training-order", seed))
    model.train()
    for step in range(TRAIN_STEPS):
        indices = rng.integers(0, len(records), size=BATCH_STATES)
        obs = torch.from_numpy(observations[indices]).to(device)
        act = torch.from_numpy(actions[indices]).to(device)
        target = torch.from_numpy(profiles[indices]).to(device)
        predictions, scores = model(obs, act)
        loss = critic_loss(method, predictions, scores, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 500 == 0:
            print(json.dumps({"method": method, "seed": seed, "step": step + 1, "loss": float(loss)}), flush=True)
    return model.eval()


@torch.no_grad()
def predict_scores(
    model: CandidateCritic,
    records: Sequence[StateRecord],
    normalizer: Normalizer,
    device: torch.device,
) -> list[np.ndarray]:
    observations, actions, _ = normalizer.arrays(records)
    outputs = []
    for start in range(0, len(records), 64):
        obs = torch.from_numpy(observations[start : start + 64]).to(device)
        act = torch.from_numpy(actions[start : start + 64]).to(device)
        _, score = model(obs, act)
        outputs.extend(score.cpu().numpy())
    return outputs


def selected_index(scores: np.ndarray, threshold: float) -> int:
    best = int(np.argmax(scores))
    return best if float(scores[best] - scores[0]) >= threshold else 0


def stable_grasp_harm(record: StateRecord, selected: int) -> bool:
    selected_profile = record.profiles[selected]
    sample0 = record.profiles[0]
    deeper_tied = bool(np.allclose(selected_profile[[3, 2, 1]], sample0[[3, 2, 1]], atol=1e-8))
    return deeper_tied and bool(selected_profile[0] + 1e-8 < sample0[0])


def evaluate_threshold(
    records: Sequence[StateRecord], scores: Sequence[np.ndarray], threshold: float
) -> dict[str, float]:
    selected = [selected_index(score, threshold) for score in scores]
    gains = [record.utilities[index] - record.utilities[0] for record, index in zip(records, selected)]
    harms = [stable_grasp_harm(record, index) for record, index in zip(records, selected)]
    return {
        "threshold": threshold,
        "utility_gain": float(np.mean(gains)),
        "stable_grasp_harm_rate": float(np.mean(harms)),
        "override_rate": float(np.mean([index != 0 for index in selected])),
    }


def calibrate_threshold(records: Sequence[StateRecord], scores: Sequence[np.ndarray]) -> tuple[float, list[dict[str, float]]]:
    rows = [evaluate_threshold(records, scores, threshold) for threshold in THRESHOLDS]
    eligible = [row for row in rows if row["stable_grasp_harm_rate"] <= 0.05]
    if not eligible:
        return max(THRESHOLDS), rows
    chosen = max(eligible, key=lambda row: (row["utility_gain"], row["threshold"]))
    return float(chosen["threshold"]), rows


def medoid_index(actions: np.ndarray) -> int:
    distances = np.linalg.norm(actions[:, None] - actions[None, :], axis=-1)
    return int(np.argmin(distances.mean(axis=1)))


def method_rows(
    records: Sequence[StateRecord],
    method: str,
    *,
    scores: Sequence[np.ndarray] | None = None,
    threshold: float = 0.0,
    seed: int = 41,
) -> list[dict[str, float | int | str | bool]]:
    rows = []
    for index, record in enumerate(records):
        if method == "sample0":
            selected = 0
        elif method == "oracle":
            selected = int(np.argmax(record.utilities))
        elif method == "random":
            selected = stable_seed("ccv-random", seed, record.pair_id, record.state_id) % 16
        elif method == "self_consistency":
            selected = medoid_index(record.actions)
        else:
            selected = selected_index(scores[index], threshold)
        utility = record.utilities
        rows.append(
            {
                "pair_id": record.pair_id,
                "state_id": record.state_id,
                "source_id": record.source_id,
                "method": method,
                "selected_index": int(selected),
                "utility": float(utility[selected]),
                "sample0_utility": float(utility[0]),
                "oracle_utility": float(np.max(utility)),
                "regret": float(np.max(utility) - utility[selected]),
                "stable_grasp_harm": stable_grasp_harm(record, selected),
            }
        )
    return rows


def source_metric(rows: Sequence[Mapping[str, object]], field: str) -> dict[int, float]:
    result = {}
    for source in sorted({int(row["source_id"]) for row in rows}):
        result[source] = float(np.mean([float(row[field]) for row in rows if int(row["source_id"]) == source]))
    return result


def bootstrap_ci(values: Sequence[float], seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(10_000, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
    }


def aggregate_seed_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    utility = source_metric(rows, "utility")
    sample0 = source_metric(rows, "sample0_utility")
    oracle = source_metric(rows, "oracle_utility")
    regret = source_metric(rows, "regret")
    harm = source_metric(rows, "stable_grasp_harm")
    sources = sorted(utility)
    gains = [utility[source] - sample0[source] for source in sources]
    available = [oracle[source] - sample0[source] for source in sources]
    return {
        "utility": float(np.mean(list(utility.values()))),
        "sample0_utility": float(np.mean(list(sample0.values()))),
        "oracle_utility": float(np.mean(list(oracle.values()))),
        "oracle_gain_recovered": float(np.sum(gains) / max(np.sum(available), 1e-12)),
        "regret": float(np.mean(list(regret.values()))),
        "stable_grasp_harm_rate": float(np.mean(list(harm.values()))),
        "utility_gain_95ci": bootstrap_ci(gains, stable_seed("ccv-heldout-gain", rows[0]["method"])),
    }


def save_model(
    path: Path,
    model: CandidateCritic,
    normalizer: Normalizer,
    *,
    method: str,
    seed: int,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "method": method,
            "seed": seed,
            "threshold": threshold,
            "normalizer": {
                "observation_mean": normalizer.observation_mean,
                "observation_std": normalizer.observation_std,
                "action_mean": normalizer.action_mean,
                "action_std": normalizer.action_std,
            },
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate CCV Gate 0B rankers")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--gate0a-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate0a = json.loads(args.gate0a_json.read_text())
    if gate0a.get("decision") != "PASS_CCV_GATE0A":
        raise ValueError("Gate 0B is locked until Gate 0A passes")
    fit_records, manifest = load_records(args.dataset_root, partitions=("fit",))
    fit_sources = [int(value) for value in manifest["fit_source_ids"]]
    optimizer_sources, calibration_sources = calibration_split(fit_sources)
    optimizer_records = [record for record in fit_records if record.source_id in optimizer_sources]
    calibration_records = [record for record in fit_records if record.source_id in calibration_sources]
    if not optimizer_records or not calibration_records:
        raise ValueError("optimizer and calibration records must both be non-empty")
    normalizer = Normalizer.fit(optimizer_records)
    device = torch.device(args.device)
    all_results = {}
    for method in METHODS:
        all_results[method] = {}
        for seed in SEEDS:
            model = train_model(method, seed, optimizer_records, normalizer, device=device)
            calibration_scores = predict_scores(model, calibration_records, normalizer, device)
            threshold, threshold_rows = calibrate_threshold(calibration_records, calibration_scores)
            all_results[method][str(seed)] = {
                "threshold": threshold,
                "calibration": threshold_rows,
            }
            save_model(
                args.output_root / "models" / f"{method}-seed{seed}.pt",
                model,
                normalizer,
                method=method,
                seed=seed,
                threshold=threshold,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Holdout files are opened only after all models and thresholds are fixed on disk.
    holdout_records, _ = load_records(args.dataset_root, partitions=("holdout",))
    if not holdout_records:
        raise ValueError("holdout records must be non-empty")
    holdout_rows_by_method_seed = {}
    for method in METHODS:
        for seed in SEEDS:
            model_path = args.output_root / "models" / f"{method}-seed{seed}.pt"
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model = CandidateCritic(method).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            threshold = float(all_results[method][str(seed)]["threshold"])
            holdout_scores = predict_scores(model, holdout_records, normalizer, device)
            rows = method_rows(
                holdout_records,
                method,
                scores=holdout_scores,
                threshold=threshold,
                seed=seed,
            )
            holdout_rows_by_method_seed[(method, seed)] = rows
            all_results[method][str(seed)]["heldout"] = aggregate_seed_rows(rows)
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()

    baselines = {}
    for method in ("sample0", "random", "self_consistency", "oracle"):
        rows = method_rows(holdout_records, method)
        baselines[method] = aggregate_seed_rows(rows)

    cross_seed = {}
    for method in METHODS:
        seed_summaries = [all_results[method][str(seed)]["heldout"] for seed in SEEDS]
        cross_seed[method] = {
            field: float(np.mean([summary[field] for summary in seed_summaries]))
            for field in ("utility", "oracle_gain_recovered", "regret", "stable_grasp_harm_rate")
        }
    ccv = cross_seed["ccv_profile"]
    available = baselines["oracle"]["utility"] - baselines["sample0"]["utility"]
    ccv_gain = ccv["utility"] - baselines["sample0"]["utility"]
    ccv_source_seed_rows = [
        row
        for seed in SEEDS
        for row in holdout_rows_by_method_seed[("ccv_profile", seed)]
    ]
    source_seed_gains = source_metric(ccv_source_seed_rows, "utility")
    source_seed_sample0 = source_metric(ccv_source_seed_rows, "sample0_utility")
    gain_ci = bootstrap_ci(
        [source_seed_gains[source] - source_seed_sample0[source] for source in source_seed_gains],
        stable_seed("ccv-gate0b-final"),
    )
    regret_reduction_terminal = 1.0 - ccv["regret"] / max(cross_seed["terminal_scalar"]["regret"], 1e-12)
    regret_reduction_dense = 1.0 - ccv["regret"] / max(cross_seed["dense_scalar"]["regret"], 1e-12)
    conditions = {
        "oracle_gain_recovered_at_least_35pct": ccv_gain / max(available, 1e-12) >= 0.35,
        "gain_point_positive": ccv_gain > 0.0,
        "gain_ci_above_negative_10pct_available": gain_ci["low"] >= -0.10 * available,
        "regret_10pct_better_than_terminal": regret_reduction_terminal >= 0.10,
        "regret_10pct_better_than_dense": regret_reduction_dense >= 0.10,
        "stable_grasp_harm_at_most_5pct": ccv["stable_grasp_harm_rate"] <= 0.05,
    }
    result = {
        "experiment": "ccv_vla_gate0b_rankers",
        "decision": "PASS_CCV_GATE0B" if all(conditions.values()) else "STOP_CCV_GATE0B",
        "optimizer_source_ids": list(optimizer_sources),
        "calibration_source_ids": list(calibration_sources),
        "holdout_source_ids": list(manifest["holdout_source_ids"]),
        "training_steps": TRAIN_STEPS,
        "seeds": list(SEEDS),
        "baselines": baselines,
        "learned_per_seed": all_results,
        "cross_seed": cross_seed,
        "ccv_gain_95ci": gain_ci,
        "regret_reduction_vs_terminal": regret_reduction_terminal,
        "regret_reduction_vs_dense": regret_reduction_dense,
        "conditions": conditions,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "gate0b_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "conditions": conditions}, sort_keys=True))


if __name__ == "__main__":
    main()
