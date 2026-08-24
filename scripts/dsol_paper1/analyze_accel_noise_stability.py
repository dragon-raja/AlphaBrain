#!/usr/bin/env python3
"""Aggregate fixed-state Accel rankings across shared flow-noise seeds."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MODEL_DIRS = {
    "broad64_practical": "broad64-practical",
    "broad64_state_matched": "broad64-state-matched",
    "broad64_paired_fm": "broad64-paired-fm",
    "broad64_paired_consistency": "broad64-paired-consistency",
}
MODEL_LABELS = {
    "broad64_practical": "Practical",
    "broad64_state_matched": "State-matched",
    "broad64_paired_fm": "Paired FM",
    "broad64_paired_consistency": "Consistency",
}
CATEGORY_ORDER = ("canonical", "broad64_training_support", "broad32_heldout")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def spearman_from_ranks(left: dict[str, int], right: dict[str, int]) -> float:
    candidate_ids = sorted(left)
    if candidate_ids != sorted(right):
        raise ValueError("rankings do not identify the same candidate bank")
    x = np.asarray([left[value] for value in candidate_ids], dtype=np.float64)
    y = np.asarray([right[value] for value in candidate_ids], dtype=np.float64)
    return float(np.corrcoef(x, y)[0, 1])


def jaccard_top_k(left: dict[str, int], right: dict[str, int], k: int) -> float:
    a = {candidate_id for candidate_id, rank in left.items() if rank <= k}
    b = {candidate_id for candidate_id, rank in right.items() if rank <= k}
    return len(a & b) / len(a | b)


def run_dir(
    *, model_key: str, seed: int, reference_seed: int, reference_root: Path, run_root: Path
) -> Path:
    stem = MODEL_DIRS[model_key]
    if seed == reference_seed:
        legacy_reference = reference_root / f"{stem}-seed41-full"
        if legacy_reference.is_dir():
            return legacy_reference
    return run_root / f"{stem}-flow-seed{seed}"


def load_run(path: Path) -> dict[str, dict[str, Any]]:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not str(summary.get("status", "")).startswith("PASS"):
        raise ValueError(f"non-PASS Accel run: {summary_path}")
    records: dict[str, dict[str, Any]] = {}
    for rank_path in sorted((path / "states").glob("*/rank_record.json")):
        record = json.loads(rank_path.read_text(encoding="utf-8"))
        ranking_path = rank_path.with_name("rankings.json")
        ranking = json.loads(ranking_path.read_text(encoding="utf-8"))[
            "operational_97"
        ]["ranking"]
        ranks = {str(row["candidate_id"]): int(row["rank"]) for row in ranking}
        scores = {
            str(row["candidate_id"]): float(row["accel_3"]) for row in ranking
        }
        pair_key = str(record["pair_key"])
        records[pair_key] = {
            "task_id": str(record["task_id"]),
            "top1": str(record["selected_candidates"]["operational_97"]),
            "category": str(
                record["selected_candidate_categories"]["operational_97"]
            ),
            "ranks": ranks,
            "scores": scores,
            "role_ranks": {
                role: int(values["complete_rank"])
                for role, values in record["role_metrics"].items()
            },
        }
    if len(records) != int(summary["state_count"]):
        raise ValueError(f"state count mismatch in {path}")
    return records


def aggregate_model(seed_runs: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    seeds = sorted(seed_runs)
    pair_keys = sorted(seed_runs[seeds[0]])
    for seed in seeds[1:]:
        if sorted(seed_runs[seed]) != pair_keys:
            raise ValueError("flow-noise runs do not contain identical states")
    top1_modal_fractions = []
    category_modal_fractions = []
    top1_all_seed_agreement = []
    category_all_seed_agreement = []
    pairwise_spearman = []
    pairwise_top5 = []
    pairwise_top10 = []
    category_counts: Counter[str] = Counter()
    role_ranks: dict[str, list[int]] = {}
    relative_spreads = []
    state_rows = []
    ensemble_rankings = {}
    ensemble_category_counts: Counter[str] = Counter()
    ensemble_score_rank_agreement = []
    ensemble_relative_margins = []
    for pair_key in pair_keys:
        rows = [seed_runs[seed][pair_key] for seed in seeds]
        top1_counts = Counter(row["top1"] for row in rows)
        category_count = Counter(row["category"] for row in rows)
        top1_modal = max(top1_counts.values()) / len(seeds)
        category_modal = max(category_count.values()) / len(seeds)
        top1_modal_fractions.append(top1_modal)
        category_modal_fractions.append(category_modal)
        top1_all_seed_agreement.append(len(top1_counts) == 1)
        category_all_seed_agreement.append(len(category_count) == 1)
        category_counts.update(row["category"] for row in rows)
        for row in rows:
            values = np.asarray(list(row["scores"].values()), dtype=np.float64)
            relative_spreads.append(float(np.std(values) / np.mean(values)))
            for role, rank in row["role_ranks"].items():
                role_ranks.setdefault(role, []).append(rank)
        for left, right in combinations(rows, 2):
            pairwise_spearman.append(spearman_from_ranks(left["ranks"], right["ranks"]))
            pairwise_top5.append(jaccard_top_k(left["ranks"], right["ranks"], 5))
            pairwise_top10.append(jaccard_top_k(left["ranks"], right["ranks"], 10))
        candidate_ids = sorted(rows[0]["scores"])
        mean_scores = {
            candidate_id: float(
                np.mean([row["scores"][candidate_id] for row in rows])
            )
            for candidate_id in candidate_ids
        }
        mean_ranks = {
            candidate_id: float(
                np.mean([row["ranks"][candidate_id] for row in rows])
            )
            for candidate_id in candidate_ids
        }
        score_order = sorted(candidate_ids, key=lambda value: (mean_scores[value], value))
        rank_order = sorted(candidate_ids, key=lambda value: (mean_ranks[value], value))
        ensemble_ranking = {
            candidate_id: rank for rank, candidate_id in enumerate(score_order, start=1)
        }
        ensemble_rankings[pair_key] = ensemble_ranking
        ensemble_top1 = score_order[0]
        ensemble_category = (
            "canonical"
            if ensemble_top1 == "canonical"
            else (
                "broad64_training_support"
                if ensemble_top1.startswith("broad_train_")
                else "broad32_heldout"
            )
        )
        ensemble_category_counts[ensemble_category] += 1
        ensemble_score_rank_agreement.append(score_order[0] == rank_order[0])
        ensemble_relative_margins.append(
            (mean_scores[score_order[1]] - mean_scores[score_order[0]])
            / max(abs(mean_scores[score_order[0]]), 1e-12)
        )
        state_rows.append(
            {
                "pair_key": pair_key,
                "task_id": rows[0]["task_id"],
                "top1_modal_fraction": top1_modal,
                "category_modal_fraction": category_modal,
                "top1_unique_count": len(top1_counts),
                "category_unique_count": len(category_count),
                "modal_top1": top1_counts.most_common(1)[0][0],
                "modal_category": category_count.most_common(1)[0][0],
                "ensemble_top1": ensemble_top1,
                "ensemble_category": ensemble_category,
                "ensemble_top1_relative_margin": ensemble_relative_margins[-1],
            }
        )
    denominator = len(pair_keys) * len(seeds)
    return {
        "state_count": len(pair_keys),
        "noise_seed_count": len(seeds),
        "mean_top1_modal_fraction": float(np.mean(top1_modal_fractions)),
        "all_seed_top1_agreement_rate": float(np.mean(top1_all_seed_agreement)),
        "mean_category_modal_fraction": float(np.mean(category_modal_fractions)),
        "all_seed_category_agreement_rate": float(
            np.mean(category_all_seed_agreement)
        ),
        "mean_pairwise_rank_spearman": float(np.mean(pairwise_spearman)),
        "mean_pairwise_top5_jaccard": float(np.mean(pairwise_top5)),
        "mean_pairwise_top10_jaccard": float(np.mean(pairwise_top10)),
        "selected_category_rates": {
            category: category_counts[category] / denominator
            for category in CATEGORY_ORDER
        },
        "ensemble_selected_category_rates": {
            category: ensemble_category_counts[category] / len(pair_keys)
            for category in CATEGORY_ORDER
        },
        "ensemble_score_vs_mean_rank_top1_agreement_rate": float(
            np.mean(ensemble_score_rank_agreement)
        ),
        "mean_ensemble_top1_relative_margin": float(
            np.mean(ensemble_relative_margins)
        ),
        "mean_role_ranks": {
            role: float(np.mean(values)) for role, values in role_ranks.items()
        },
        "mean_accel_relative_spread": float(np.mean(relative_spreads)),
        "state_rows": state_rows,
        "ensemble_rankings": ensemble_rankings,
    }


def plot_summary(path: Path, models: dict[str, dict[str, Any]]) -> None:
    keys = list(MODEL_DIRS)
    labels = [MODEL_LABELS[key] for key in keys]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    axes[0].bar(
        labels,
        [models[key]["mean_pairwise_rank_spearman"] for key in keys],
        color="#3B7EA1",
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Across-noise rank stability")
    axes[0].set_ylabel("Mean pairwise Spearman")
    axes[0].tick_params(axis="x", rotation=20)

    bottom = np.zeros(len(keys))
    colors = ("#4C78A8", "#59A14F", "#F28E2B")
    category_labels = ("Canonical", "Train support", "Held-out")
    for category, label, color in zip(CATEGORY_ORDER, category_labels, colors):
        values = np.asarray(
            [models[key]["ensemble_selected_category_rates"][category] for key in keys]
        )
        axes[1].bar(labels, values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Top-1 support category")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].tick_params(axis="x", rotation=20)

    width = 0.36
    x = np.arange(len(keys))
    axes[2].bar(
        x - width / 2,
        [models[key]["mean_role_ranks"]["canonical"] for key in keys],
        width,
        label="Canonical",
        color="#4C78A8",
    )
    axes[2].bar(
        x + width / 2,
        [models[key]["mean_role_ranks"]["strong_info"] for key in keys],
        width,
        label="Strong-info",
        color="#E15759",
    )
    axes[2].set_xticks(x, labels, rotation=20)
    axes[2].set_title("Role rank (lower is better)")
    axes[2].set_ylabel("Mean rank in complete bank")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-seed", type=int, default=20260820)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    seeds = sorted(set(args.seeds))
    if args.reference_seed not in seeds:
        raise ValueError("seed list must include the reference seed")

    model_payloads = {}
    state_rows = []
    ensemble_rankings = {}
    for model_key in MODEL_DIRS:
        seed_runs = {
            seed: load_run(
                run_dir(
                    model_key=model_key,
                    seed=seed,
                    reference_seed=args.reference_seed,
                    reference_root=args.reference_root,
                    run_root=args.run_root,
                )
            )
            for seed in seeds
        }
        aggregate = aggregate_model(seed_runs)
        state_rows.extend(
            {"model": model_key, **row} for row in aggregate.pop("state_rows")
        )
        ensemble_rankings[model_key] = aggregate.pop("ensemble_rankings")
        model_payloads[model_key] = aggregate

    cross_model = {}
    for left, right in combinations(MODEL_DIRS, 2):
        pair_keys = sorted(ensemble_rankings[left])
        values = [
            spearman_from_ranks(
                ensemble_rankings[left][pair_key],
                ensemble_rankings[right][pair_key],
            )
            for pair_key in pair_keys
        ]
        cross_model[f"{left}__vs__{right}"] = {
            "mean_ensemble_rank_spearman": float(np.mean(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }

    output = {
        "schema": "dsol_accel_flow_noise_stability_v1",
        "status": "PASS",
        "reference_seed": args.reference_seed,
        "flow_noise_seeds": seeds,
        "models": model_payloads,
        "cross_model_ensemble_relations": cross_model,
        "claim_scope": (
            "Fixed-state rank stability and view-support diagnostics only; no "
            "closed-loop view-value claim."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "summary.json", output)
    with (args.output_dir / "state_stability.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(state_rows[0]))
        writer.writeheader()
        writer.writerows(state_rows)
    plot_summary(args.output_dir / "noise_stability.png", model_payloads)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
