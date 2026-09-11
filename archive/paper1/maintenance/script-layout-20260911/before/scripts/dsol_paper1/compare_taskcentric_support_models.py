from __future__ import annotations

import argparse
import csv
import glob
import json
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = (
    "canonical_both",
    "strong_info_both",
    "matched_control_both",
    "canonical_wrist_only",
    "canonical_external_only",
    "strong_info_external_only",
    "matched_control_external_only",
    "all_camera_blackout",
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_root(root: Path) -> list[dict[str, Any]]:
    rows = []
    for filename in sorted(glob.glob(str(root / "episodes-shard-*.jsonl"))):
        with Path(filename).open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError(f"no evaluation rows in {root}")
    return rows


def source_group(row: Mapping[str, Any]) -> str:
    return f"{row['task_id']}::{row['episode_id_source']}"


def condition_scores(
    rows: Sequence[Mapping[str, Any]], condition: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["condition"] == condition:
            grouped[source_group(row)].append(float(bool(row["success"])))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def subtract(
    first: Mapping[str, float], second: Mapping[str, float]
) -> dict[str, float]:
    keys = sorted(set(first) & set(second))
    return {key: float(first[key] - second[key]) for key in keys}


def paired_stratified_bootstrap(
    values: Mapping[str, float], *, seed: int, samples: int
) -> tuple[float, float, float]:
    tasks: dict[str, list[str]] = defaultdict(list)
    for key in values:
        tasks[key.split("::", 1)[0]].append(key)
    observed = float(np.mean(list(values.values())))
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_values = []
        for keys in tasks.values():
            sampled = generator.choice(keys, size=len(keys), replace=True)
            sampled_values.extend(values[key] for key in sampled)
        draws[index] = np.mean(sampled_values)
    low, high = np.quantile(draws, (0.025, 0.975))
    return observed, float(low), float(high)


def effect_row(
    *,
    name: str,
    values: Mapping[str, float],
    seed: int,
    samples: int,
) -> dict[str, Any]:
    estimate, low, high = paired_stratified_bootstrap(
        values, seed=seed, samples=samples
    )
    return {
        "effect": name,
        "source_episode_count": len(values),
        "difference_pp": 100.0 * estimate,
        "paired_stratified_bootstrap_95_ci_pp": [100.0 * low, 100.0 * high],
    }


def compare(
    models: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    identities = {
        label: {(str(row["pair_key"]), str(row["condition"])) for row in rows}
        for label, rows in models.items()
    }
    reference = identities["baseline"]
    if any(value != reference for value in identities.values()):
        raise ValueError("model evaluations do not contain identical state-condition pairs")

    scores = {
        label: {
            condition: condition_scores(rows, condition) for condition in CONDITIONS
        }
        for label, rows in models.items()
    }
    condition_table = []
    for label in models:
        for condition in CONDITIONS:
            state_rows = [row for row in models[label] if row["condition"] == condition]
            condition_table.append(
                {
                    "model": label,
                    "condition": condition,
                    "state_count": len(state_rows),
                    "state_success_rate": float(
                        np.mean([bool(row["success"]) for row in state_rows])
                    ),
                    "source_episode_macro_success_rate": float(
                        np.mean(list(scores[label][condition].values()))
                    ),
                }
            )

    effects = []
    effect_index = 0
    for first, second in (("info_support", "control_support"), ("info_support", "baseline")):
        for condition in CONDITIONS:
            effects.append(
                effect_row(
                    name=f"{first}_minus_{second}::{condition}",
                    values=subtract(scores[first][condition], scores[second][condition]),
                    seed=seed + effect_index,
                    samples=samples,
                )
            )
            effect_index += 1

    within_model = {}
    for label in models:
        within_model[label] = {
            "strong_minus_canonical": subtract(
                scores[label]["strong_info_both"], scores[label]["canonical_both"]
            ),
            "strong_minus_control": subtract(
                scores[label]["strong_info_both"],
                scores[label]["matched_control_both"],
            ),
        }
    for reference_label in ("control_support", "baseline"):
        for contrast in ("strong_minus_canonical", "strong_minus_control"):
            effects.append(
                effect_row(
                    name=(
                        f"interaction::info_support_minus_{reference_label}::"
                        f"{contrast}"
                    ),
                    values=subtract(
                        within_model["info_support"][contrast],
                        within_model[reference_label][contrast],
                    ),
                    seed=seed + effect_index,
                    samples=samples,
                )
            )
            effect_index += 1

    return {
        "schema": "dsol_taskcentric_support_model_comparison_v1",
        "status": "PASS",
        "statistical_unit": "source_episode",
        "model_count": len(models),
        "paired_state_condition_count": len(reference),
        "source_episode_count": len(scores["baseline"]["canonical_both"]),
        "bootstrap": {
            "method": "paired_stratified_source_episode_bootstrap",
            "samples": samples,
            "seed": seed,
        },
        "conditions": condition_table,
        "effects": effects,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    selected = ("canonical_both", "strong_info_both", "matched_control_both")
    models = ("baseline", "control_support", "info_support")
    lookup = {
        (row["model"], row["condition"]): 100.0
        * float(row["state_success_rate"])
        for row in rows
    }
    x = np.arange(len(selected), dtype=np.float64)
    width = 0.24
    colors = ("#7B8794", "#3B82A0", "#C85C70")
    figure, axis = plt.subplots(figsize=(9.5, 5.4))
    for index, (model, color) in enumerate(zip(models, colors)):
        values = [lookup[(model, condition)] for condition in selected]
        bars = axis.bar(x + (index - 1) * width, values, width, label=model, color=color)
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axis.set_xticks(x, [value.replace("_both", "").replace("_", "\n") for value in selected])
    axis.set_ylabel("Closed-loop success rate (%)")
    axis.set_ylim(0, 105)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--control-support", type=Path, required=True)
    parser.add_argument("--info-support", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    result = compare(
        {
            "baseline": load_root(args.baseline),
            "control_support": load_root(args.control_support),
            "info_support": load_root(args.info_support),
        },
        seed=args.seed,
        samples=args.bootstrap_samples,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "metrics.json", result)
    write_csv(args.output_dir / "condition_success.csv", result["conditions"])
    write_csv(args.output_dir / "paired_effects.csv", result["effects"])
    plot(args.output_dir / "main_conditions.png", result["conditions"])
    print(
        json.dumps(
            {
                "status": result["status"],
                "source_episode_count": result["source_episode_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
