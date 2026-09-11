#!/usr/bin/env python3
"""Analyze exhaustive same-state view sweeps as post-hoc view-value discovery data.

The analysis is deliberately state conditional. A view score is useful only if it
ranks successful views above failed views for the same task state; pooled
correlations can otherwise be dominated by task difficulty.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_rows(patterns: Sequence[str]) -> list[dict[str, Any]]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError("no episode JSONL matched")
    rows: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    episode_ids = [str(row["episode_id"]) for row in rows]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("episode IDs are not unique")
    if any(row.get("status") != "complete" for row in rows):
        raise ValueError("all episode rows must be complete")
    return rows


def mean_or_none(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(values))


def pairwise_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return float(wins / (len(positives) * len(negatives)))


def task_visibility_features(row: Mapping[str, Any]) -> dict[str, float]:
    visibility = row["initial_metrics"]["task_entity_visibility"]
    per_camera = visibility["per_camera"]
    camera_names = list(visibility["camera_names"])
    entity_names = list(visibility["entity_names"])

    entity_means = []
    visible_entities = 0
    total_visible_pixels = 0
    for entity in entity_names:
        fractions = []
        entity_pixels = 0
        for camera in camera_names:
            entry = per_camera[camera]["entities"][entity]
            fractions.append(float(entry["visible_fraction"]))
            entity_pixels += int(entry["visible_pixels"])
        entity_means.append(float(np.mean(fractions)))
        total_visible_pixels += entity_pixels
        visible_entities += int(entity_pixels > 0)

    agent = per_camera.get("agentview", {}).get("score", 0.0)
    wrist = per_camera.get("robot0_eye_in_hand", {}).get("score", 0.0)
    minimum = min(entity_means) if entity_means else 0.0
    harmonic = (
        0.0
        if not entity_means or any(value <= 0 for value in entity_means)
        else float(len(entity_means) / sum(1.0 / value for value in entity_means))
    )
    return {
        "visibility_score": float(visibility["score"]),
        "visibility_agent": float(agent),
        "visibility_wrist": float(wrist),
        "visibility_entity_min": float(minimum),
        "visibility_entity_hmean": float(harmonic),
        "visible_entity_fraction": float(visible_entities / max(len(entity_names), 1)),
        "total_visible_pixels": float(total_visible_pixels),
    }


def optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def candidate_group(candidate_id: str, selection: Mapping[str, Any]) -> str:
    if selection.get("catalog_group") is not None:
        return str(selection["catalog_group"])
    if candidate_id == "canonical":
        return "canonical"
    if candidate_id.startswith("broad_train_"):
        return "broad_training_64"
    if candidate_id.startswith("broad_heldout_"):
        return "broad_heldout_32"
    return "unknown"


def flatten_row(row: Mapping[str, Any]) -> dict[str, Any]:
    pose = row.get("pose") or {}
    selection = row.get("selection_metadata") or {}
    visibility = task_visibility_features(row)
    candidate_id = str(row["selected_candidate_id"])
    return {
        "pair_key": str(row["pair_key"]),
        "task_id": str(row["task_id"]),
        "source_episode_id": str(row["episode_id_source"]),
        "source_state_index": int(row["source_state_index"]),
        "stage_fraction": float(row["stage_fraction"]),
        "candidate_id": candidate_id,
        "candidate_group": candidate_group(candidate_id, selection),
        "azimuth_deg": float(pose.get("azimuth_deg", 0.0)),
        "elevation_deg": float(pose.get("elevation_deg", 0.0)),
        "radius_scale": float(pose.get("radius_scale", 1.0)),
        "delta_visibility": optional_float(selection.get("delta_visibility")),
        "ensemble_accel_3": optional_float(selection.get("ensemble_accel_3")),
        "ensemble_accel_rank": selection.get("ensemble_accel_rank"),
        "single_noise_accel_3": optional_float(selection.get("single_noise_accel_3")),
        "success": int(bool(row["success"])),
        "completion_steps": int(row["completion_steps"]),
        **visibility,
    }


def random_hit_probability(candidate_count: int, success_count: int, budget: int) -> float:
    budget = min(budget, candidate_count)
    if success_count == 0:
        return 0.0
    if candidate_count - success_count < budget:
        return 1.0
    return float(
        1.0
        - math.comb(candidate_count - success_count, budget)
        / math.comb(candidate_count, budget)
    )


def top_k_hit(
    rows: Sequence[Mapping[str, Any]],
    score: Callable[[Mapping[str, Any]], float],
    budget: int,
    *,
    reverse: bool,
) -> float:
    ranked = sorted(
        rows,
        key=lambda row: (score(row), str(row["candidate_id"])),
        reverse=reverse,
    )
    return float(any(bool(row["success"]) for row in ranked[:budget]))


SCORE_SPECS: dict[str, tuple[str, bool]] = {
    "visibility_mean": ("visibility_score", True),
    "visibility_external": ("visibility_agent", True),
    "visibility_min_entity": ("visibility_entity_min", True),
    "visibility_hmean_entity": ("visibility_entity_hmean", True),
    "visible_entity_fraction": ("visible_entity_fraction", True),
    "accel_ensemble": ("ensemble_accel_3", False),
}


CONTRAST_FIELDS = (
    "visibility_score",
    "visibility_agent",
    "visibility_wrist",
    "visibility_entity_min",
    "visibility_entity_hmean",
    "visible_entity_fraction",
    "total_visible_pixels",
    "ensemble_accel_3",
    "azimuth_deg",
    "elevation_deg",
    "radius_scale",
)


def analyze(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    flat = [flatten_row(row) for row in rows]
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flat:
        by_state[str(row["pair_key"])].append(row)
    candidate_counts = {len(values) for values in by_state.values()}
    if len(candidate_counts) != 1:
        raise ValueError(f"states have inconsistent candidate counts: {candidate_counts}")
    candidate_count = next(iter(candidate_counts))
    if any("canonical" not in {row["candidate_id"] for row in values} for values in by_state.values()):
        raise ValueError("every state must contain canonical")
    for values in by_state.values():
        canonical_visibility = next(
            float(row["visibility_score"])
            for row in values
            if row["candidate_id"] == "canonical"
        )
        for row in values:
            if row["delta_visibility"] is None:
                row["delta_visibility"] = float(row["visibility_score"]) - canonical_visibility

    active_score_specs = {
        name: (field, higher_is_better)
        for name, (field, higher_is_better) in SCORE_SPECS.items()
        if all(row.get(field) is not None for row in flat)
    }

    budgets = sorted({min(value, candidate_count) for value in (1, 2, 4, 8, 16, 32, 64, candidate_count)})
    curves: dict[str, list[dict[str, float]]] = defaultdict(list)
    for budget in budgets:
        random_hits = []
        for values in by_state.values():
            random_hits.append(
                random_hit_probability(
                    candidate_count,
                    sum(bool(row["success"]) for row in values),
                    budget,
                )
            )
        curves["random_pool_oracle"].append(
            {"budget": budget, "state_hit_rate": float(np.mean(random_hits))}
        )
        for name, (field, higher_is_better) in active_score_specs.items():
            hits = [
                top_k_hit(
                    values,
                    lambda row, field=field: float(row[field]),
                    budget,
                    reverse=higher_is_better,
                )
                for values in by_state.values()
            ]
            curves[name].append(
                {"budget": budget, "state_hit_rate": float(np.mean(hits))}
            )

    score_diagnostics = {}
    for name, (field, higher_is_better) in active_score_specs.items():
        aucs = []
        for values in by_state.values():
            scores = [float(row[field]) * (1.0 if higher_is_better else -1.0) for row in values]
            auc = pairwise_auc(scores, [bool(row["success"]) for row in values])
            if auc is not None:
                aucs.append(auc)
        score_diagnostics[name] = {
            "state_conditional_auc": mean_or_none(aucs),
            "informative_states": len(aucs),
            "top1_success_rate": curves[name][0]["state_hit_rate"],
        }

    feature_contrasts = {}
    for field in CONTRAST_FIELDS:
        differences = []
        for values in by_state.values():
            positive = [
                float(row[field])
                for row in values
                if row["success"] and row.get(field) is not None
            ]
            negative = [
                float(row[field])
                for row in values
                if not row["success"] and row.get(field) is not None
            ]
            if positive and negative:
                differences.append(float(np.mean(positive) - np.mean(negative)))
        feature_contrasts[field] = {
            "success_minus_failure_state_macro": mean_or_none(differences),
            "informative_states": len(differences),
        }

    state_rows = []
    for pair_key, values in sorted(by_state.items()):
        successes = [row for row in values if row["success"]]
        canonical = next(row for row in values if row["candidate_id"] == "canonical")
        fastest = min(successes, key=lambda row: row["completion_steps"]) if successes else None
        per_score = {}
        for name, (field, higher_is_better) in active_score_specs.items():
            ranked = sorted(
                values,
                key=lambda row, field=field: (float(row[field]), str(row["candidate_id"])),
                reverse=higher_is_better,
            )
            oriented_scores = [
                float(row[field]) * (1.0 if higher_is_better else -1.0)
                for row in values
            ]
            per_score[name] = {
                "auc": pairwise_auc(
                    oriented_scores, [bool(row["success"]) for row in values]
                ),
                "top1_candidate_id": ranked[0]["candidate_id"],
                "top1_success": bool(ranked[0]["success"]),
            }
        state_rows.append(
            {
                "pair_key": pair_key,
                "task_id": values[0]["task_id"],
                "source_episode_id": values[0]["source_episode_id"],
                "canonical_success": bool(canonical["success"]),
                "successful_candidate_count": len(successes),
                "successful_candidate_fraction": len(successes) / candidate_count,
                "oracle_at_all_success": bool(successes),
                "fastest_success_candidate_id": None if fastest is None else fastest["candidate_id"],
                "fastest_success_steps": None if fastest is None else fastest["completion_steps"],
                "score_diagnostics": per_score,
            }
        )

    group_rates = {}
    groups = sorted({str(row["candidate_group"]) for row in flat})
    for group in groups:
        per_state = []
        for values in by_state.values():
            selected = [row for row in values if row["candidate_group"] == group]
            if selected:
                per_state.append(float(np.mean([row["success"] for row in selected])))
        group_rates[group] = {
            "state_macro_success_fraction": mean_or_none(per_state),
            "states": len(per_state),
        }


    candidate_ids = sorted({str(row["candidate_id"]) for row in flat})
    candidate_summary = []
    for candidate_id in candidate_ids:
        selected = [row for row in flat if row["candidate_id"] == candidate_id]
        successful_steps = [row["completion_steps"] for row in selected if row["success"]]
        candidate_summary.append(
            {
                "candidate_id": candidate_id,
                "candidate_group": selected[0]["candidate_group"],
                "state_success_rate": float(np.mean([row["success"] for row in selected])),
                "successful_states": int(sum(row["success"] for row in selected)),
                "mean_success_steps": mean_or_none(successful_steps),
                "azimuth_deg": selected[0]["azimuth_deg"],
                "elevation_deg": selected[0]["elevation_deg"],
                "radius_scale": selected[0]["radius_scale"],
            }
        )
    candidate_summary.sort(
        key=lambda row: (-row["state_success_rate"], row["candidate_id"])
    )

    source_ids = sorted({str(row["source_episode_id"]) for row in flat})
    loso_rows = []
    for heldout_source in source_ids:
        train = [row for row in flat if row["source_episode_id"] != heldout_source]
        heldout = [row for row in flat if row["source_episode_id"] == heldout_source]
        train_by_candidate: dict[str, list[int]] = defaultdict(list)
        for row in train:
            train_by_candidate[str(row["candidate_id"])].append(int(row["success"]))
        selected_candidate = sorted(
            train_by_candidate,
            key=lambda candidate: (
                -float(np.mean(train_by_candidate[candidate])),
                candidate,
            ),
        )[0]
        selected_heldout = [
            row for row in heldout if row["candidate_id"] == selected_candidate
        ]
        canonical_heldout = [row for row in heldout if row["candidate_id"] == "canonical"]
        loso_rows.append(
            {
                "heldout_source_episode_id": heldout_source,
                "selected_candidate_id": selected_candidate,
                "states": len(selected_heldout),
                "selected_success_rate": float(
                    np.mean([row["success"] for row in selected_heldout])
                ),
                "canonical_success_rate": float(
                    np.mean([row["success"] for row in canonical_heldout])
                ),
            }
        )

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in state_rows:
        by_task[str(row["task_id"])].append(row)
    task_summary = {
        task: {
            "states": len(values),
            "source_episodes": len({row["source_episode_id"] for row in values}),
            "canonical_success_rate": float(np.mean([row["canonical_success"] for row in values])),
            "oracle_success_rate": float(np.mean([row["oracle_at_all_success"] for row in values])),
            "mean_successful_candidate_fraction": float(
                np.mean([row["successful_candidate_fraction"] for row in values])
            ),
        }
        for task, values in sorted(by_task.items())
    }

    payload = {
        "schema": "dsol_view_value_discovery_analysis_v1",
        "status": "PASS",
        "evidence_role": "post_hoc_discovery_only",
        "confirmatory_test_eligible": False,
        "warning": (
            "These outcomes were previously inspected and each state-view has one rollout. "
            "Use them to design hypotheses, not to report a learned selector's final test result."
        ),
        "rows": len(flat),
        "states": len(by_state),
        "source_episodes": len({row["source_episode_id"] for row in flat}),
        "tasks": len({row["task_id"] for row in flat}),
        "candidate_count_per_state": candidate_count,
        "canonical_success_rate": float(np.mean([row["canonical_success"] for row in state_rows])),
        "oracle_at_all_success_rate": float(np.mean([row["oracle_at_all_success"] for row in state_rows])),
        "mean_successful_candidate_fraction": float(
            np.mean([row["successful_candidate_fraction"] for row in state_rows])
        ),
        "score_diagnostics": score_diagnostics,
        "feature_contrasts": feature_contrasts,
        "candidate_budget_curves": dict(curves),
        "candidate_group_success": group_rates,
        "top_global_candidates": candidate_summary[:10],
        "leave_one_source_out_global_pose": {
            "source_folds": len(loso_rows),
            "selected_success_rate": float(
                np.average(
                    [row["selected_success_rate"] for row in loso_rows],
                    weights=[row["states"] for row in loso_rows],
                )
            ),
            "canonical_success_rate": float(
                np.average(
                    [row["canonical_success_rate"] for row in loso_rows],
                    weights=[row["states"] for row in loso_rows],
                )
            ),
            "folds": loso_rows,
        },
        "task_summary": task_summary,
        "state_rows": state_rows,
    }
    return payload, flat


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload, flat = analyze(load_rows(args.episodes))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "analysis.json", payload)
    write_csv(args.output_dir / "state_view_outcomes.csv", flat)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
