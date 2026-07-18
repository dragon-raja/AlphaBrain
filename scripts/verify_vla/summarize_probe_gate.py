from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.verify_vla.probe_gate_common import (
    LABELS,
    MODALITIES,
    NON_RELEASE_PROBES,
    OUTCOMES,
    PROBE_NAMES,
    bootstrap_mean,
    feature_vector,
    ridge_scores,
    summarize_predictions,
)


DEFAULT_EPISODE_ROOT = Path("/share/longjunyu/fresh-vla/libero-full-episode-v2-128")
DEFAULT_INPUT_ROOT = Path("/share/longjunyu/verify-vla/gate0-probe-v1")
MIN_VALID_TRAIN = 80
MIN_VALID_VAL = 10
MIN_VALID_FRACTION = 0.75


def _assert_unsealed_path(path: Path) -> None:
    forbidden = {"test", "tests", "confirmation", "confirm", "sealed"}
    lowered = {part.lower() for part in path.parts}
    if lowered & forbidden or any("confirmation" in part for part in lowered):
        raise ValueError(f"refusing to access sealed path: {path}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_record_metadata(input_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((input_root / "records").glob("*.json")):
        row = json.loads(path.read_text())
        row["metadata_path"] = str(path)
        row["arrays_path"] = str(path.with_suffix(".npz"))
        rows.append(row)
    return rows


def _condition_examples(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    condition: str,
    modality: str,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    features = []
    labels = []
    metadata = []
    for row in rows:
        if row.get("status") != "valid" or row.get("split") != split:
            continue
        with np.load(str(row["arrays_path"]), allow_pickle=False) as arrays:
            probe_names = [str(value) for value in arrays["probe_names"]]
            outcome_names = [str(value) for value in arrays["outcome_names"]]
            if outcome_names != list(OUTCOMES):
                raise ValueError(f"unexpected outcome order for {row['pair_id']}: {outcome_names}")
            if condition == "pre":
                probe_index = 0
                time_index = 0
            else:
                probe_index = probe_names.index(condition)
                time_index = -1
            for outcome_index, outcome in enumerate(OUTCOMES):
                features.append(
                    feature_vector(
                        arrays["agentview"][probe_index, outcome_index, time_index],
                        arrays["wrist"][probe_index, outcome_index, time_index],
                        arrays["robot_state"][probe_index, outcome_index, time_index],
                        modality,
                        image_size,
                    )
                )
                labels.append(LABELS[outcome])
                metadata.append(
                    {
                        "pair_id": row["pair_id"],
                        "outcome": outcome,
                        "source_initial_state_index": row["source_initial_state_index"],
                    }
                )
    if not features:
        raise ValueError(f"no valid {split} examples for {condition}/{modality}")
    return np.asarray(features), np.asarray(labels), metadata


def classifier_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    image_size: int,
    regularization: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for condition_index, condition in enumerate(("pre", *PROBE_NAMES)):
        results[condition] = {}
        for modality_index, modality in enumerate(MODALITIES):
            train, train_labels, _ = _condition_examples(
                rows,
                split="train",
                condition=condition,
                modality=modality,
                image_size=image_size,
            )
            val, val_labels, val_metadata = _condition_examples(
                rows,
                split="val",
                condition=condition,
                modality=modality,
                image_size=image_size,
            )
            scores = ridge_scores(train, train_labels, val, regularization)
            results[condition][modality] = {
                "regularization": regularization,
                **summarize_predictions(
                    val_metadata,
                    val_labels,
                    scores,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + condition_index * 20 + modality_index * 2,
                ),
            }
    return results


def viability_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    val_rows = [row for row in rows if row.get("status") == "valid" and row.get("split") == "val"]
    if not val_rows:
        raise ValueError("no valid val records for viability analysis")
    result: dict[str, dict[str, Any]] = {}
    for condition_index, condition in enumerate(("no_probe", *PROBE_NAMES)):
        result[condition] = {}
        for outcome_index, outcome in enumerate(OUTCOMES):
            successes = []
            completion_steps = []
            displacements = []
            newly_irreversible = []
            for row in val_rows:
                endpoint = row["conditions"][condition][outcome]
                viability = endpoint.get("teacher_viability")
                if not isinstance(viability, Mapping):
                    raise ValueError(
                        f"missing teacher viability for {row['pair_id']} {condition}/{outcome}"
                    )
                success = float(bool(viability["success"]))
                baseline = row["conditions"]["no_probe"][outcome]["teacher_viability"]
                successes.append(success)
                completion_steps.append(float(viability["completion_steps"]))
                displacements.append(float(endpoint["object_displacement_m"]))
                newly_irreversible.append(float(bool(baseline["success"]) and not bool(success)))
            result[condition][outcome] = {
                "teacher_success": bootstrap_mean(
                    successes,
                    samples=bootstrap_samples,
                    seed=seed + condition_index * 20 + outcome_index * 3,
                ),
                "completion_steps": bootstrap_mean(
                    completion_steps,
                    samples=bootstrap_samples,
                    seed=seed + condition_index * 20 + outcome_index * 3 + 1,
                ),
                "object_displacement_median_m": float(np.median(displacements)),
                "object_displacement_max_m": float(np.max(displacements)),
                "new_irreversible_failure_count": int(np.sum(newly_irreversible)),
                "group_count": len(successes),
            }
        all_displacements = [
            float(row["conditions"][condition][outcome]["object_displacement_m"])
            for row in val_rows
            for outcome in OUTCOMES
        ]
        result[condition]["combined_object_displacement_median_m"] = float(
            np.median(all_displacements)
        )
        result[condition]["combined_new_irreversible_failure_count"] = int(
            sum(result[condition][outcome]["new_irreversible_failure_count"] for outcome in OUTCOMES)
        )
    return result


def decide_gate(
    classifier: Mapping[str, Mapping[str, Mapping[str, Any]]],
    viability: Mapping[str, Mapping[str, Any]],
    counts: Mapping[str, Mapping[str, int]],
) -> tuple[str, dict[str, Any]]:
    twin_valid = bool(
        counts["train"]["valid"] >= MIN_VALID_TRAIN
        and counts["val"]["valid"] >= MIN_VALID_VAL
        and counts["train"]["valid_fraction"] >= MIN_VALID_FRACTION
        and counts["val"]["valid_fraction"] >= MIN_VALID_FRACTION
    )
    baseline_attached = float(viability["no_probe"]["attached"]["teacher_success"]["mean"])
    baseline_detached = float(viability["no_probe"]["detached"]["teacher_success"]["mean"])
    baseline_teacher_valid = baseline_attached >= 0.90 and baseline_detached >= 0.80
    candidate_checks: dict[str, Any] = {}
    for probe in NON_RELEASE_PROBES:
        checks: dict[str, Any] = {}
        for modality in ("vision_state", "vision"):
            pre_accuracy = float(classifier["pre"][modality]["sample_accuracy"])
            post_accuracy = float(classifier[probe][modality]["sample_accuracy"])
            hold_accuracy = float(classifier["hold_closed"][modality]["sample_accuracy"])
            ranking = float(classifier[probe][modality]["pair_ranking_accuracy"])
            checks[modality] = {
                "pre_accuracy": pre_accuracy,
                "post_accuracy": post_accuracy,
                "hold_gain_pp": 100.0 * (post_accuracy - hold_accuracy),
                "pair_ranking_accuracy": ranking,
                "passes": bool(
                    pre_accuracy <= 0.65
                    and post_accuracy >= 0.85
                    and post_accuracy - hold_accuracy >= 0.20 - 1e-12
                    and ranking >= 0.85
                ),
            }
        attached_success = float(viability[probe]["attached"]["teacher_success"]["mean"])
        detached_success = float(viability[probe]["detached"]["teacher_success"]["mean"])
        displacement = float(viability[probe]["combined_object_displacement_median_m"])
        irreversible = int(viability[probe]["combined_new_irreversible_failure_count"])
        checks["safety"] = {
            "attached_teacher_success": attached_success,
            "detached_teacher_success": detached_success,
            "combined_object_displacement_median_m": displacement,
            "new_irreversible_failure_count": irreversible,
            "passes": bool(
                attached_success >= 0.90
                and detached_success >= 0.80
                and displacement <= 0.03
                and irreversible == 0
            ),
        }
        checks["passes"] = bool(
            checks["vision_state"]["passes"]
            and checks["vision"]["passes"]
            and checks["safety"]["passes"]
        )
        candidate_checks[probe] = checks
    passing = [probe for probe, checks in candidate_checks.items() if checks["passes"]]
    if not twin_valid or not baseline_teacher_valid:
        decision = "GATE0_INVALID"
    elif passing:
        decision = "PROCEED_TO_LEARNED_DVOV"
    else:
        decision = "STOP_VERIFY_VLA"
    return decision, {
        "twin_construction_valid": twin_valid,
        "baseline_teacher_valid": baseline_teacher_valid,
        "baseline_no_probe_attached_success": baseline_attached,
        "baseline_no_probe_detached_success": baseline_detached,
        "candidate_checks": candidate_checks,
        "passing_probes": passing,
    }


def _counts(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    expected = defaultdict(int)
    for group in manifest["groups"]:
        if group["split"] in {"train", "val"}:
            expected[str(group["split"])] += 1
    result = {}
    for split in ("train", "val"):
        split_rows = [row for row in rows if row.get("split") == split]
        valid = sum(row.get("status") == "valid" for row in split_rows)
        invalid = sum(row.get("status") == "invalid" for row in split_rows)
        result[split] = {
            "expected": expected[split],
            "recorded": len(split_rows),
            "valid": valid,
            "invalid": invalid,
            "missing": expected[split] - len(split_rows),
            "valid_fraction": valid / expected[split],
        }
    return result


def render_markdown(payload: Mapping[str, Any]) -> str:
    classifier = payload["classifier"]
    viability = payload["viability"]
    lines = [
        "# VERIFY-VLA Gate 0 Results",
        "",
        f"Decision: **{payload['decision']}**",
        "",
        "This report uses only the frozen train/val latent-contact twins. No sealed test or confirmation episode was opened.",
        "",
        "## Data validity",
        "",
        "| Split | Expected | Valid | Invalid | Missing |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "val"):
        row = payload["counts"][split]
        lines.append(
            f"| {split} | {row['expected']} | {row['valid']} | {row['invalid']} | {row['missing']} |"
        )
    lines.extend(
        [
            "",
            "## Outcome identifiability",
            "",
            "Fixed 16x16 block-average ridge (regularization 1.0), fit on train and evaluated on val.",
            "",
            "| Condition | Modality | Accuracy | Group 95% CI | Pair ranking | Gain over hold |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    hold = {
        modality: float(classifier["hold_closed"][modality]["sample_accuracy"])
        for modality in MODALITIES
    }
    for condition in ("pre", *PROBE_NAMES):
        for modality in MODALITIES:
            row = classifier[condition][modality]
            interval = row["group_bootstrap_95"]
            gain = 100.0 * (float(row["sample_accuracy"]) - hold[modality])
            lines.append(
                f"| `{condition}` | `{modality}` | {100*row['sample_accuracy']:.1f}% | "
                f"[{100*interval['bootstrap_95_low']:.1f}, {100*interval['bootstrap_95_high']:.1f}]% | "
                f"{100*row['pair_ranking_accuracy']:.1f}% | {gain:+.1f} pp |"
            )
    lines.extend(
        [
            "",
            "## Privileged teacher viability",
            "",
            "Teacher success is a recoverability upper bound, not a deployment result.",
            "",
            "| Condition | Attached success | Detached recovery | Median displacement | New irreversible failures |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition in ("no_probe", *PROBE_NAMES):
        row = viability[condition]
        lines.append(
            f"| `{condition}` | {100*row['attached']['teacher_success']['mean']:.1f}% | "
            f"{100*row['detached']['teacher_success']['mean']:.1f}% | "
            f"{100*row['combined_object_displacement_median_m']:.2f} cm | "
            f"{row['combined_new_irreversible_failure_count']} |"
        )
    lines.extend(
        [
            "",
            "## Gate checks",
            "",
            "| Probe | Vision-state | Vision-only | Safety | Overall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for probe, checks in payload["gate"]["candidate_checks"].items():
        lines.append(
            f"| `{probe}` | {checks['vision_state']['passes']} | {checks['vision']['passes']} | "
            f"{checks['safety']['passes']} | {checks['passes']} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "The prior disclosed visible-slip control reached 100% single-frame vision accuracy. It is referenced only as a control and is not part of this gate. Gate 0 asks a different question: can a low-risk physical tail reveal an initially latent contact outcome without destroying task recoverability?",
            "",
            f"Final Gate 0 decision: **{payload['decision']}**",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize preregistered VERIFY-VLA Gate 0 probes")
    parser.add_argument("--episode-root", type=Path, default=DEFAULT_EPISODE_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=260718)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _assert_unsealed_path(args.episode_root)
    _assert_unsealed_path(args.input_root)
    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    rows = load_record_metadata(args.input_root)
    counts = _counts(manifest, rows)
    if not args.allow_incomplete and any(counts[split]["missing"] for split in ("train", "val")):
        raise ValueError(f"Gate 0 collection is incomplete: {counts}")
    classifier = classifier_results(
        rows,
        image_size=args.image_size,
        regularization=args.regularization,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    viability = viability_results(rows, bootstrap_samples=args.bootstrap_samples, seed=args.seed + 1000)
    decision, gate = decide_gate(classifier, viability, counts)
    payload = {
        "schema_version": 1,
        "experiment": "verify_vla_latent_contact_probe_gate0",
        "data_policy": "train/val only; sealed test and confirmation episodes not opened",
        "episode_root": str(args.episode_root),
        "input_root": str(args.input_root),
        "fixed_classifier": {
            "features": "agent+wrist 16x16 block-average RGB plus optional 8D robot state",
            "regularization": args.regularization,
            "fit_split": "train",
            "evaluation_split": "val",
        },
        "statistical_unit": "snapshot group; outcome frames remain paired within group",
        "counts": counts,
        "classifier": classifier,
        "viability": viability,
        "gate": gate,
        "visible_feedback_control": {
            "status": "prior disclosed result; not reopened or used for Gate 0 selection",
            "artifact": "/share/longjunyu/fresh-vla/research-reset/feedback_observability.json",
            "single_frame_vision_accuracy": 1.0,
        },
        "decision": decision,
    }
    output_json = args.output_json or args.input_root / "gate0_results.json"
    output_markdown = args.output_markdown or args.input_root / "gate0_results.md"
    _atomic_json(output_json, payload)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(render_markdown(payload))
    print(json.dumps({"decision": decision, "counts": counts, "passing_probes": gate["passing_probes"]}, indent=2))


if __name__ == "__main__":
    main()
