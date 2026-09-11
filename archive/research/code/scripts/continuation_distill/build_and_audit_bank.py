from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


UTILITY_WEIGHTS = np.asarray([8.0**2, 8.0**3, 8.0**4, 8.0**5, 8.0, 1.0])
UTILITY_DENOMINATOR = float(sum(8.0**power for power in range(6)))
BOOTSTRAP_SAMPLES = 20_000


def source_split(source_ids: Sequence[int], evaluation_count: int = 5) -> tuple[tuple[int, ...], tuple[int, ...]]:
    unique = sorted({int(value) for value in source_ids})
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


def utilities(signatures: np.ndarray) -> np.ndarray:
    value = np.asarray(signatures, dtype=np.float64)
    if value.shape != (16, 6, 6):
        raise ValueError(f"expected [16, 6, 6] signatures, got {value.shape}")
    return np.einsum("nrf,f->nr", value, UTILITY_WEIGHTS) / UTILITY_DENOMINATOR


def stable_grasp_harm(profiles: np.ndarray, selected: int) -> bool:
    chosen, base = profiles[selected], profiles[0]
    deeper_tied = bool(np.allclose(chosen[[3, 2, 1]], base[[3, 2, 1]], atol=1e-8))
    return deeper_tied and bool(chosen[0] + 1e-8 < base[0])


def robust_winner(signatures: np.ndarray, profiles: np.ndarray) -> dict[str, Any]:
    repeat_utility = utilities(signatures)
    means = repeat_utility.mean(axis=1)
    winner = int(np.argmax(means))
    leave_one_out = []
    for held in range(6):
        leave_one_out.append(int(np.argmax(np.delete(repeat_utility, held, axis=1).mean(axis=1))))
    strict_repeat_wins = int(np.count_nonzero(repeat_utility[winner] > repeat_utility[0] + 1e-12))
    loo_agreement = int(sum(index == winner for index in leave_one_out))
    harm = stable_grasp_harm(profiles, winner)
    accepted = bool(
        winner != 0
        and means[winner] > means[0] + 1e-12
        and strict_repeat_wins >= 5
        and loo_agreement >= 5
        and not harm
    )
    return {
        "winner_index": winner,
        "mean_gain": float(means[winner] - means[0]),
        "strict_repeat_wins": strict_repeat_wins,
        "leave_one_out_agreement": loo_agreement,
        "stable_grasp_harm": harm,
        "accepted": accepted,
        "available_oracle_gain": float(means.max() - means[0]),
    }


def direct_winner(direct_signatures: np.ndarray) -> int:
    value = np.asarray(direct_signatures, dtype=np.float64)
    if value.shape != (16, 6):
        raise ValueError(f"expected direct signatures [16, 6], got {value.shape}")
    return int(np.argmax((value @ UTILITY_WEIGHTS) / UTILITY_DENOMINATOR))


def deterministic_random_nonzero(pair_id: str, state_id: str) -> int:
    digest = hashlib.sha256(f"continuation-distill-random-v1::{pair_id}::{state_id}".encode()).digest()
    return 1 + int.from_bytes(digest[:4], "big") % 15


def source_bootstrap(rows: Sequence[Mapping[str, Any]], field: str, seed: int) -> list[float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["source_id"])].append(float(row[field]))
    source_values = np.asarray([np.mean(grouped[source]) for source in sorted(grouped)])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(source_values), size=(BOOTSTRAP_SAMPLES, len(source_values)))
    means = source_values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(rows: Sequence[Mapping[str, Any]], expected_sources: int) -> dict[str, Any]:
    accepted = [row for row in rows if row["accepted"]]
    gain_ci = source_bootstrap(accepted, "mean_gain", 260719) if accepted else [0.0, 0.0]
    direct_gap_ci = source_bootstrap(accepted, "winner_minus_direct_gain", 260720) if accepted else [0.0, 0.0]
    return {
        "state_count": len(rows),
        "accepted_state_count": len(accepted),
        "accepted_rate": len(accepted) / len(rows) if rows else 0.0,
        "accepted_source_count": len({row["source_id"] for row in accepted}),
        "expected_source_count": expected_sources,
        "accepted_stage_counts": dict(sorted(Counter(row["stage"] for row in accepted).items())),
        "mean_gain": float(np.mean([row["mean_gain"] for row in accepted])) if accepted else 0.0,
        "mean_gain_bootstrap_95": gain_ci,
        "winner_minus_direct_gain": float(np.mean([row["winner_minus_direct_gain"] for row in accepted])) if accepted else 0.0,
        "winner_minus_direct_gain_bootstrap_95": direct_gap_ci,
        "available_oracle_gain": float(np.mean([row["available_oracle_gain"] for row in accepted])) if accepted else 0.0,
        "stable_grasp_harm_rate": float(np.mean([row["stable_grasp_harm"] for row in accepted])) if accepted else 0.0,
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and audit robust continuation-winner bank")
    parser.add_argument("--ccv-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.ccv_root / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("split") != "train":
        raise ValueError("requires complete train-only CCV collection")
    fitting_sources, evaluation_sources = source_split(manifest["fit_source_ids"])
    rows = []
    for group in manifest["groups"]:
        if group["source_partition"] != "fit":
            continue
        source_id = int(group["source_initial_state_index"])
        partition = "fitting" if source_id in fitting_sources else "evaluation"
        for state in group["states"]:
            labels_path = args.ccv_root / state["labels_file"]
            if any(token in str(labels_path).lower() for token in ("holdout", "test", "confirmation", "sealed")):
                raise ValueError(f"refusing forbidden path: {labels_path}")
            with np.load(labels_path, allow_pickle=False) as labels:
                signatures = np.asarray(labels["continuation_signatures"], dtype=np.float64)
                profiles = np.asarray(labels["continuation_profiles"], dtype=np.float64)
                direct = np.asarray(labels["direct_signatures"], dtype=np.float64)
            winner = robust_winner(signatures, profiles)
            direct_index = direct_winner(direct)
            mean_utility = utilities(signatures).mean(axis=1)
            row = {
                "pair_id": str(group["pair_id"]),
                "state_id": str(state["state_id"]),
                "source_id": source_id,
                "partition": partition,
                "stage": str(state["stage"]),
                "deployable_file": str(state["deployable_file"]),
                "labels_file": str(state["labels_file"]),
                "sample0_index": 0,
                "random_nonzero_index": deterministic_random_nonzero(str(group["pair_id"]), str(state["state_id"])),
                "direct_physical_index": direct_index,
                "continuation_winner_index": winner["winner_index"],
                "winner_minus_direct_gain": float(mean_utility[winner["winner_index"]] - mean_utility[direct_index]),
                **winner,
            }
            rows.append(row)

    fitting = [row for row in rows if row["partition"] == "fitting"]
    evaluation = [row for row in rows if row["partition"] == "evaluation"]
    summaries = {
        "fitting": summarize(fitting, len(fitting_sources)),
        "evaluation": summarize(evaluation, len(evaluation_sources)),
    }
    fit_summary, eval_summary = summaries["fitting"], summaries["evaluation"]
    eval_relative_direct = (
        eval_summary["winner_minus_direct_gain"] / eval_summary["available_oracle_gain"]
        if eval_summary["available_oracle_gain"] > 0
        else 0.0
    )
    checks = {
        "fitting_accepted_at_least_120": fit_summary["accepted_state_count"] >= 120,
        "evaluation_accepted_at_least_30": eval_summary["accepted_state_count"] >= 30,
        "fitting_sources_at_least_15": fit_summary["accepted_source_count"] >= 15,
        "evaluation_sources_at_least_4": eval_summary["accepted_source_count"] >= 4,
        "three_fitting_stages_at_least_20": sum(count >= 20 for count in fit_summary["accepted_stage_counts"].values()) >= 3,
        "fitting_gain_ci_low_above_zero": fit_summary["mean_gain_bootstrap_95"][0] > 0,
        "evaluation_gain_ci_low_above_zero": eval_summary["mean_gain_bootstrap_95"][0] > 0,
        "evaluation_beats_direct_by_20pct_oracle": eval_relative_direct >= 0.20,
        "fitting_stable_grasp_harm_at_most_5pct": fit_summary["stable_grasp_harm_rate"] <= 0.05,
        "evaluation_stable_grasp_harm_at_most_5pct": eval_summary["stable_grasp_harm_rate"] <= 0.05,
    }
    decision = (
        "PROCEED_CONTINUATION_DISTILL_SEED41_PILOT"
        if all(checks.values())
        else "STOP_CONTINUATION_DISTILL_LABEL_BANK"
    )
    payload = {
        "experiment": "continuation_selected_self_distillation_gate0",
        "decision": decision,
        "preregistration": str(args.preregistration.resolve()),
        "preregistration_sha256": file_sha256(args.preregistration),
        "fit_source_ids": list(fitting_sources),
        "evaluation_source_ids": list(evaluation_sources),
        "summaries": summaries,
        "evaluation_winner_minus_direct_fraction_of_oracle": eval_relative_direct,
        "checks": checks,
        "accepted_rows": [row for row in rows if row["accepted"]],
        "ccv_holdout_states_opened": 0,
        "test_or_confirmation_states_opened": 0,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"decision": decision, "summaries": summaries, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
