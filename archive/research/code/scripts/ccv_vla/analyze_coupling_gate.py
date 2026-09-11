from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ccv import (
    best_candidate_index,
    profile_from_signatures,
    scalar_viability_utility,
    stable_seed,
    viability_key,
)


INDEPENDENT_DRAWS = 32
BOOTSTRAP_REPLICATES = 10_000


def independent_permutations(
    candidate_count: int,
    repeats: int,
    *,
    state_id: str,
    draw: int,
) -> np.ndarray:
    permutations = []
    for candidate in range(candidate_count):
        rng = np.random.default_rng(
            stable_seed("ccv-independent", state_id, draw, candidate)
        )
        permutations.append(rng.permutation(repeats))
    return np.stack(permutations)


def profile_rows(signatures: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(signatures, dtype=np.float64)
    if indices is None:
        return np.stack([profile_from_signatures(rows) for rows in values])
    return np.stack(
        [profile_from_signatures(values[candidate, candidate_indices]) for candidate, candidate_indices in enumerate(indices)]
    )


def selection_regret(reference_profiles: np.ndarray, estimate_profiles: np.ndarray) -> float:
    best = best_candidate_index(reference_profiles)
    selected = best_candidate_index(estimate_profiles)
    return scalar_viability_utility(reference_profiles[best]) - scalar_viability_utility(
        reference_profiles[selected]
    )


def complement_indices(repeats: int, selected: np.ndarray) -> np.ndarray:
    chosen = {int(index) for index in np.asarray(selected).reshape(-1)}
    remaining = [index for index in range(repeats) if index not in chosen]
    if not remaining:
        raise ValueError("leave-out target needs at least one unused repeat")
    return np.asarray(remaining, dtype=np.int64)


def leave_out_profiles(signatures: np.ndarray, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(signatures, dtype=np.float64)
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 2 or selected.shape[0] != values.shape[0]:
        raise ValueError("indices must have shape [candidate, budget]")
    targets = np.stack(
        [
            complement_indices(values.shape[1], selected[candidate])
            for candidate in range(values.shape[0])
        ]
    )
    return profile_rows(values, targets)


def leave_out_selection_regret(signatures: np.ndarray, indices: np.ndarray) -> float:
    return selection_regret(
        leave_out_profiles(signatures, indices),
        profile_rows(signatures, indices),
    )


def pairwise_mse(signatures: np.ndarray, permutations: np.ndarray | None = None) -> float:
    values = np.asarray(signatures, dtype=np.float64)
    candidate_count, repeats, _ = values.shape
    errors = []
    selected_by_trial = (
        np.broadcast_to(np.arange(repeats), (candidate_count, repeats)).T
        if permutations is None
        else np.asarray(permutations, dtype=np.int64).T
    )
    for selected in selected_by_trial:
        estimates = values[np.arange(candidate_count), selected]
        targets = np.stack(
            [
                values[candidate, complement_indices(repeats, [selected[candidate]])].mean(
                    axis=0
                )
                for candidate in range(candidate_count)
            ]
        )
        for left, right in itertools.combinations(range(candidate_count), 2):
            estimate_difference = estimates[left] - estimates[right]
            target_difference = targets[left] - targets[right]
            errors.append(float(np.square(estimate_difference - target_difference).mean()))
    return float(np.mean(errors))


def state_metrics(signatures: np.ndarray, *, state_id: str) -> dict[str, float | bool]:
    values = np.asarray(signatures, dtype=np.float64)
    candidate_count, repeats, dimensions = values.shape
    if candidate_count != 16 or repeats != 6 or dimensions != 6:
        raise ValueError(f"formal label shape must be [16, 6, 6], got {values.shape}")
    reference = profile_rows(values)
    reference_keys = {viability_key(row) for row in reference}
    best = best_candidate_index(reference)
    oracle_utility = scalar_viability_utility(reference[best])
    sample0_utility = scalar_viability_utility(reference[0])

    coupled_mse = pairwise_mse(values)
    independent_mses = []
    independent_regrets_1 = []
    independent_regrets_2 = []
    for draw in range(INDEPENDENT_DRAWS):
        permutations = independent_permutations(
            candidate_count,
            repeats,
            state_id=state_id,
            draw=draw,
        )
        independent_mses.append(pairwise_mse(values, permutations))
        independent_regrets_1.append(
            leave_out_selection_regret(values, permutations[:, :1])
        )
        independent_regrets_2.append(
            leave_out_selection_regret(values, permutations[:, :2])
        )

    coupled_regrets_1 = []
    coupled_regrets_2 = []
    for repeat in range(repeats):
        one = np.full((candidate_count, 1), repeat, dtype=np.int64)
        two = np.stack(
            [
                np.full(candidate_count, repeat, dtype=np.int64),
                np.full(candidate_count, (repeat + 1) % repeats, dtype=np.int64),
            ],
            axis=1,
        )
        coupled_regrets_1.append(leave_out_selection_regret(values, one))
        coupled_regrets_2.append(leave_out_selection_regret(values, two))

    independent_mse = float(np.mean(independent_mses))
    return {
        "action_leverage": len(reference_keys) > 1,
        "unique_profiles": float(len(reference_keys)),
        "coupled_pairwise_mse": coupled_mse,
        "independent_pairwise_mse": independent_mse,
        "mse_reduction_fraction": (
            0.0 if independent_mse <= 0 else 1.0 - coupled_mse / independent_mse
        ),
        "available_oracle_gain": oracle_utility - sample0_utility,
        "coupled_regret_b1": float(np.mean(coupled_regrets_1)),
        "independent_regret_b1": float(np.mean(independent_regrets_1)),
        "regret_improvement_b1": float(np.mean(independent_regrets_1))
        - float(np.mean(coupled_regrets_1)),
        "coupled_regret_b2": float(np.mean(coupled_regrets_2)),
        "independent_regret_b2": float(np.mean(independent_regrets_2)),
        "regret_improvement_b2": float(np.mean(independent_regrets_2))
        - float(np.mean(coupled_regrets_2)),
    }


def source_means(rows: Sequence[Mapping[str, float | int | bool]]) -> list[dict[str, float | int]]:
    metrics = (
        "action_leverage",
        "coupled_pairwise_mse",
        "independent_pairwise_mse",
        "mse_reduction_fraction",
        "available_oracle_gain",
        "coupled_regret_b1",
        "independent_regret_b1",
        "regret_improvement_b1",
        "coupled_regret_b2",
        "independent_regret_b2",
        "regret_improvement_b2",
    )
    sources = []
    for source_id in sorted({int(row["source_initial_state_index"]) for row in rows}):
        selected = [row for row in rows if int(row["source_initial_state_index"]) == source_id]
        source = {"source_initial_state_index": source_id, "state_count": len(selected)}
        source.update(
            {
                metric: float(np.mean([float(row[metric]) for row in selected]))
                for metric in metrics
            }
        )
        sources.append(source)
    return sources


def bootstrap_mean_ci(values: Sequence[float], *, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        return {"mean": float(array.mean()), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
    }


def summarize(rows: Sequence[Mapping[str, float | int | bool]]) -> dict[str, object]:
    sources = source_means(rows)
    intervals = {
        metric: bootstrap_mean_ci(
            [float(source[metric]) for source in sources],
            seed=stable_seed("ccv-bootstrap", metric),
        )
        for metric in (
            "action_leverage",
            "mse_reduction_fraction",
            "available_oracle_gain",
            "regret_improvement_b1",
            "regret_improvement_b2",
        )
    }
    available = intervals["available_oracle_gain"]["mean"]
    passes = {
        "action_leverage_at_least_30pct": intervals["action_leverage"]["mean"] >= 0.30,
        "mse_reduction_at_least_20pct": intervals["mse_reduction_fraction"]["mean"] >= 0.20,
        "one_repeat_regret_point_positive": intervals["regret_improvement_b1"]["mean"] > 0.0,
        "one_repeat_regret_ci_above_negative_5pct_gain": intervals["regret_improvement_b1"]["low"]
        >= -0.05 * available,
    }
    return {
        "state_count": len(rows),
        "source_count": len(sources),
        "source_rows": sources,
        "source_bootstrap_95ci": intervals,
        "conditions": passes,
        "decision": "PASS_CCV_GATE0A" if all(passes.values()) else "STOP_CCV_GATE0A",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CCV coupled-label Gate 0A")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def markdown_report(result: Mapping[str, object]) -> str:
    intervals = result["source_bootstrap_95ci"]
    lines = [
        "# CCV-VLA Gate 0A Coupling Result",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        f"Independent units: {result['source_count']} source IDs; {result['state_count']} states.",
        "",
        "| Metric | Mean | 95% source-bootstrap CI |",
        "|---|---:|---:|",
    ]
    for name, row in intervals.items():
        lines.append(f"| {name} | {row['mean']:.6f} | [{row['low']:.6f}, {row['high']:.6f}] |")
    lines.extend(["", "## Conditions", ""])
    for name, passed in result["conditions"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.dataset_root / "manifest.json").read_text())
    if manifest.get("split") != "train":
        raise ValueError("Gate 0A may only consume a train-only CCV manifest")
    if manifest.get("status") != "complete" and not args.allow_partial:
        raise ValueError("formal Gate 0A requires a complete collection")
    rows = []
    for group in manifest["groups"]:
        if group["source_partition"] != "fit":
            continue
        for state in group["states"]:
            if state["source_partition"] != "fit":
                continue
            with np.load(args.dataset_root / state["labels_file"], allow_pickle=False) as labels:
                signatures = np.asarray(labels["continuation_signatures"])
            metrics = state_metrics(
                signatures,
                state_id=f"{group['pair_id']}::{state['state_id']}",
            )
            rows.append(
                {
                    "pair_id": group["pair_id"],
                    "state_id": state["state_id"],
                    "source_initial_state_index": int(group["source_initial_state_index"]),
                    **metrics,
                }
            )
    result = {
        "experiment": "ccv_vla_gate0a_coupling",
        "dataset_root": str(args.dataset_root.resolve()),
        "independent_permutation_draws": INDEPENDENT_DRAWS,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        **summarize(rows),
        "state_rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown_report(result))
    print(json.dumps({key: result[key] for key in ("decision", "state_count", "source_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
