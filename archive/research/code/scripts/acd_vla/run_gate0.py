from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.branch_vla.gate0_common import (
    ALPHAS,
    bootstrap_mean,
    fit_ridge,
    mean_by_source,
    select_alpha_classifier,
    select_alpha_regression,
)
from scripts.branch_vla.run_gate0 import reduction_summary


HORIZON = 8
EXPECTED_SPLITS = {"train": 102, "val": 13}
FORBIDDEN_PARTS = {"test", "tests", "confirmation", "confirm", "sealed"}


@dataclass(frozen=True)
class ResponseRecord:
    pair_id: str
    source_id: int
    split: str
    pre_feature: np.ndarray
    attached_post_feature: np.ndarray
    slipped_post_feature: np.ndarray
    pre_actions: np.ndarray
    attached_actions: np.ndarray
    slipped_actions: np.ndarray


def assert_unsealed(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    if lowered & FORBIDDEN_PARTS or any("confirmation" in part for part in lowered):
        raise ValueError(f"refusing sealed path: {path}")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _validate_actions(value: np.ndarray, name: str) -> np.ndarray:
    actions = np.asarray(value, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < HORIZON + 1:
        raise ValueError(f"invalid {name}: {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError(f"non-finite {name}")
    return actions


def load_records(collection_root: Path) -> tuple[list[ResponseRecord], dict[str, Any]]:
    assert_unsealed(collection_root)
    manifest = json.loads((collection_root / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("policy-response collection is incomplete")
    if manifest.get("test_episode_files_opened") != 0 or manifest.get("confirmation_paths_opened") != 0:
        raise RuntimeError("sealed-data audit failed")
    if manifest.get("split_group_counts") != EXPECTED_SPLITS:
        raise RuntimeError(f"unexpected development split counts: {manifest.get('split_group_counts')}")
    if manifest.get("train_val_source_overlap") != 0:
        raise RuntimeError("train/val source overlap")

    records = []
    for metadata in manifest["records"]:
        if metadata["split"] not in EXPECTED_SPLITS:
            raise RuntimeError(f"non-development record: {metadata['pair_id']}")
        path = collection_root / str(metadata["record_file"])
        assert_unsealed(path)
        with np.load(path, allow_pickle=False) as handle:
            records.append(
                ResponseRecord(
                    pair_id=str(metadata["pair_id"]),
                    source_id=int(metadata["source_id"]),
                    split=str(metadata["split"]),
                    pre_feature=np.asarray(handle["pre_feature"], dtype=np.float64),
                    attached_post_feature=np.asarray(handle["attached_post_feature"], dtype=np.float64),
                    slipped_post_feature=np.asarray(handle["slipped_post_feature"], dtype=np.float64),
                    pre_actions=_validate_actions(handle["pre_actions"], "pre_actions"),
                    attached_actions=_validate_actions(handle["attached_actions"], "attached_actions"),
                    slipped_actions=_validate_actions(handle["slipped_actions"], "slipped_actions"),
                )
            )
    return records, manifest


def action_statistics(records: Sequence[ResponseRecord]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate(
        [
            np.stack([record.attached_actions[:HORIZON] for record in records]),
            np.stack([record.slipped_actions[:HORIZON] for record in records]),
        ],
        axis=0,
    )
    mean = values.reshape(-1, 7).mean(axis=0)
    scale = values.reshape(-1, 7).std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def normalize_actions(value: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=np.float64) - mean) / scale


def guard_metrics(
    train: Sequence[ResponseRecord],
    val: Sequence[ResponseRecord],
    *,
    seed: int,
    shuffle_repeats: int,
) -> tuple[dict[str, Any], dict[tuple[str, str], float]]:
    train_x = np.stack(
        [feature for record in train for feature in (record.attached_post_feature, record.slipped_post_feature)]
    )
    train_y = np.tile(np.asarray([1.0, -1.0]), len(train))
    train_sources = [record.source_id for record in train for _ in range(2)]
    alpha, cv = select_alpha_classifier(train_x, train_y, train_sources)
    model = fit_ridge(train_x, train_y[:, None], alpha)

    val_post = np.stack(
        [feature for record in val for feature in (record.attached_post_feature, record.slipped_post_feature)]
    )
    val_pre = np.stack([record.pre_feature for record in val for _ in range(2)])
    labels = np.tile(np.asarray([1.0, -1.0]), len(val))
    post_scores = model.predict(val_post)[:, 0]
    pre_scores = model.predict(val_pre)[:, 0]
    post_prediction = np.where(post_scores >= 0, 1.0, -1.0)
    pre_prediction = np.where(pre_scores >= 0, 1.0, -1.0)

    route_scores = {}
    source_post: dict[int, list[float]] = defaultdict(list)
    source_pre: dict[int, list[float]] = defaultdict(list)
    pair_ranking = []
    for index, record in enumerate(val):
        attached_score = float(post_scores[2 * index])
        slipped_score = float(post_scores[2 * index + 1])
        route_scores[(record.pair_id, "attached")] = attached_score
        route_scores[(record.pair_id, "slipped")] = slipped_score
        source_post[record.source_id].extend(
            [float(post_prediction[2 * index] == 1.0), float(post_prediction[2 * index + 1] == -1.0)]
        )
        source_pre[record.source_id].extend(
            [float(pre_prediction[2 * index] == 1.0), float(pre_prediction[2 * index + 1] == -1.0)]
        )
        pair_ranking.append(float(attached_score > slipped_score))

    rng = np.random.default_rng(seed)
    shuffled_accuracies = []
    for _ in range(shuffle_repeats):
        shuffled = rng.permutation(train_y)
        shuffled_model = fit_ridge(train_x, shuffled[:, None], alpha)
        prediction = np.where(shuffled_model.predict(val_post)[:, 0] >= 0, 1.0, -1.0)
        shuffled_accuracies.append(float(np.mean(prediction == labels)))

    post_source_values = [float(np.mean(values)) for values in source_post.values()]
    pre_source_values = [float(np.mean(values)) for values in source_pre.values()]
    return {
        "selected_alpha": alpha,
        "train_source_cv_accuracy": cv,
        "post_feedback_accuracy": float(np.mean(post_prediction == labels)),
        "pre_feedback_accuracy": float(np.mean(pre_prediction == labels)),
        "post_feedback_source_bootstrap": bootstrap_mean(post_source_values, seed=seed + 1),
        "pre_feedback_source_bootstrap": bootstrap_mean(pre_source_values, seed=seed + 2),
        "pair_ranking_accuracy": float(np.mean(pair_ranking)),
        "shuffle_repeats": shuffle_repeats,
        "shuffled_label_mean_accuracy": float(np.mean(shuffled_accuracies)),
        "shuffled_label_accuracies": shuffled_accuracies,
    }, route_scores


def source_median(rows: Sequence[Mapping[str, float | int]], key: str) -> float:
    values = mean_by_source(rows, key)
    return float(np.median(list(values.values())))


def evaluate(
    records: Sequence[ResponseRecord],
    manifest: Mapping[str, Any],
    *,
    seed: int,
    shuffle_repeats: int,
) -> dict[str, Any]:
    train = [record for record in records if record.split == "train"]
    val = [record for record in records if record.split == "val"]
    action_mean, action_scale = action_statistics(train)
    train_x = np.stack([record.pre_feature for record in train])
    train_attached = np.stack(
        [normalize_actions(record.attached_actions[:HORIZON], action_mean, action_scale) for record in train]
    )
    train_slipped = np.stack(
        [normalize_actions(record.slipped_actions[:HORIZON], action_mean, action_scale) for record in train]
    )
    branch_targets = np.concatenate(
        (train_attached.reshape(len(train), -1), train_slipped.reshape(len(train), -1)), axis=1
    )
    branch_alpha, branch_cv = select_alpha_regression(
        train_x, branch_targets, [record.source_id for record in train]
    )
    branch_model = fit_ridge(train_x, branch_targets, branch_alpha)

    merged_targets = 0.5 * (
        train_attached.reshape(len(train), -1) + train_slipped.reshape(len(train), -1)
    )
    merged_alpha, merged_cv = select_alpha_regression(
        train_x, merged_targets, [record.source_id for record in train]
    )
    merged_model = fit_ridge(train_x, merged_targets, merged_alpha)
    constants = {
        "attached": train_attached.mean(axis=0),
        "slipped": train_slipped.mean(axis=0),
    }
    guard, route_scores = guard_metrics(
        train, val, seed=seed + 100, shuffle_repeats=shuffle_repeats
    )

    val_x = np.stack([record.pre_feature for record in val])
    branch_prediction = branch_model.predict(val_x)
    width = HORIZON * 7
    predicted_attached = branch_prediction[:, :width].reshape(-1, HORIZON, 7)
    predicted_slipped = branch_prediction[:, width:].reshape(-1, HORIZON, 7)
    predicted_merged = merged_model.predict(val_x).reshape(-1, HORIZON, 7)

    rows = []
    group_separation = []
    for index, record in enumerate(val):
        targets = {
            "attached": normalize_actions(record.attached_actions[:HORIZON], action_mean, action_scale),
            "slipped": normalize_actions(record.slipped_actions[:HORIZON], action_mean, action_scale),
        }
        predictions = {
            "attached": predicted_attached[index],
            "slipped": predicted_slipped[index],
        }
        stale = normalize_actions(record.pre_actions[1 : HORIZON + 1], action_mean, action_scale)
        target_separation = float(np.sqrt(np.mean(np.square(targets["attached"] - targets["slipped"]))))
        predicted_separation = float(
            np.sqrt(np.mean(np.square(predictions["attached"] - predictions["slipped"])))
        )
        group_separation.append(
            {
                "source_id": record.source_id,
                "target_branch_rms": target_separation,
                "separation_ratio": predicted_separation / max(target_separation, 1e-8),
            }
        )
        for outcome in ("attached", "slipped"):
            opposite = "slipped" if outcome == "attached" else "attached"
            own = predictions[outcome]
            other = predictions[opposite]
            routed = predictions["attached"] if route_scores[(record.pair_id, outcome)] >= 0 else predictions["slipped"]
            target = targets[outcome]
            mse = lambda prediction: float(np.mean(np.square(prediction - target)))
            rows.append(
                {
                    "pair_id": record.pair_id,
                    "source_id": record.source_id,
                    "outcome": outcome,
                    "stale_pi05_mse": mse(stale),
                    "linear_merged_mse": mse(predicted_merged[index]),
                    "branch_constant_mse": mse(constants[outcome]),
                    "oracle_branch_mse": mse(own),
                    "learned_route_mse": mse(routed),
                    "random_precommit_mse": 0.5 * (mse(own) + mse(other)),
                }
            )

    reductions = {
        "oracle_vs_constant": reduction_summary(
            rows, "branch_constant_mse", "oracle_branch_mse", seed=seed + 200
        ),
        "learned_vs_stale_pi05": reduction_summary(
            rows, "stale_pi05_mse", "learned_route_mse", seed=seed + 201
        ),
        "learned_vs_linear_merged": reduction_summary(
            rows, "linear_merged_mse", "learned_route_mse", seed=seed + 202
        ),
        "oracle_vs_stale_pi05": reduction_summary(
            rows, "stale_pi05_mse", "oracle_branch_mse", seed=seed + 203
        ),
        "random_precommit_vs_stale_pi05": reduction_summary(
            rows, "stale_pi05_mse", "random_precommit_mse", seed=seed + 204
        ),
    }
    outcome_reductions = {
        outcome: reduction_summary(
            [row for row in rows if row["outcome"] == outcome],
            "branch_constant_mse",
            "oracle_branch_mse",
            seed=seed + 210 + index,
        )
        for index, outcome in enumerate(("attached", "slipped"))
    }
    means = {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "stale_pi05_mse",
            "linear_merged_mse",
            "branch_constant_mse",
            "oracle_branch_mse",
            "learned_route_mse",
            "random_precommit_mse",
        )
    }
    target_rms_by_source = mean_by_source(group_separation, "target_branch_rms")
    separation_by_source = mean_by_source(group_separation, "separation_ratio")
    mean_target_rms = float(np.mean(list(target_rms_by_source.values())))
    median_separation_ratio = float(np.median(list(separation_by_source.values())))

    data_valid = bool(
        len(train) == EXPECTED_SPLITS["train"]
        and len(val) == EXPECTED_SPLITS["val"]
        and manifest["split_group_counts"] == EXPECTED_SPLITS
        and manifest["train_val_source_overlap"] == 0
        and manifest["test_episode_files_opened"] == 0
        and manifest["confirmation_paths_opened"] == 0
    )
    checks = {
        "data_valid": data_valid,
        "post_guard_at_least_85pct": guard["post_feedback_accuracy"] >= 0.85,
        "pre_guard_at_most_60pct": guard["pre_feedback_accuracy"] <= 0.60,
        "shuffle_control_at_most_60pct": guard["shuffled_label_mean_accuracy"] <= 0.60,
        "teacher_response_rms_at_least_0_10": mean_target_rms >= 0.10,
        "oracle_vs_constant_at_least_20pct": reductions["oracle_vs_constant"]["mean"] >= 0.20,
        "oracle_vs_constant_ci_low_above_5pct": reductions["oracle_vs_constant"]["bootstrap_95_low"] > 0.05,
        "both_outcomes_vs_constant_at_least_10pct": all(
            value["mean"] >= 0.10 for value in outcome_reductions.values()
        ),
        "learned_vs_stale_at_least_25pct": reductions["learned_vs_stale_pi05"]["mean"] >= 0.25,
        "learned_vs_stale_ci_low_above_5pct": reductions["learned_vs_stale_pi05"]["bootstrap_95_low"] > 0.05,
        "median_separation_ratio_at_least_0_5": median_separation_ratio >= 0.5,
    }
    if not data_valid:
        decision = "ACD_GATE0_INVALID"
    elif all(checks.values()):
        decision = "PROCEED_ACD_CLOSED_LOOP_GATE"
    else:
        decision = "STOP_ACD_CURRENT_TASK_GATE0"

    return {
        "schema_version": 1,
        "experiment": "acd_vla_gate0_policy_response",
        "decision": decision,
        "data_policy": "train/val policy-response records only; test and confirmation remain sealed",
        "statistical_unit": "source initial state; outcomes are paired within group",
        "configuration": {
            "horizon": HORIZON,
            "alphas": list(ALPHAS),
            "shuffle_repeats": shuffle_repeats,
        },
        "audit": {
            "split_group_counts": manifest["split_group_counts"],
            "split_source_counts": manifest["split_source_counts"],
            "train_val_source_overlap": manifest["train_val_source_overlap"],
            "test_episode_files_opened": manifest["test_episode_files_opened"],
            "confirmation_paths_opened": manifest["confirmation_paths_opened"],
        },
        "policy_identity": manifest["policy_identity"],
        "action_normalization": {"mean": action_mean.tolist(), "scale": action_scale.tolist()},
        "branch_predictor": {"selected_alpha": branch_alpha, "source_cv_mse": branch_cv},
        "merged_predictor": {"selected_alpha": merged_alpha, "source_cv_mse": merged_cv},
        "guard": guard,
        "mean_teacher_response_branch_rms": mean_target_rms,
        "median_source_separation_ratio": median_separation_ratio,
        "metric_means": means,
        "reductions": reductions,
        "outcome_reductions_vs_constant": outcome_reductions,
        "gate_checks": checks,
        "rows": rows,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    guard = payload["guard"]
    constant = payload["reductions"]["oracle_vs_constant"]
    stale = payload["reductions"]["learned_vs_stale_pi05"]
    merged = payload["reductions"]["learned_vs_linear_merged"]
    lines = [
        "# ACD-VLA Gate 0 results",
        "",
        f"Decision: **{payload['decision']}**",
        "",
        "Only train/val policy-response records were evaluated. Original test and confirmation data remain sealed.",
        "",
        "## Guard and branch signal",
        "",
        f"- post-feedback guard accuracy: {100*guard['post_feedback_accuracy']:.1f}%",
        f"- pre-feedback guard accuracy: {100*guard['pre_feedback_accuracy']:.1f}%",
        f"- shuffled-label mean accuracy: {100*guard['shuffled_label_mean_accuracy']:.1f}%",
        f"- mean normalized teacher-response branch RMS: {payload['mean_teacher_response_branch_rms']:.3f}",
        f"- median predicted/target separation ratio: {payload['median_source_separation_ratio']:.3f}",
        "",
        "## Response imitation",
        "",
        f"- oracle branch vs per-outcome constant: {100*constant['mean']:.1f}% MSE reduction, "
        f"95% CI [{100*constant['bootstrap_95_low']:.1f}, {100*constant['bootstrap_95_high']:.1f}]",
        f"- learned route vs stale Pi0.5 tail: {100*stale['mean']:.1f}% MSE reduction, "
        f"95% CI [{100*stale['bootstrap_95_low']:.1f}, {100*stale['bootstrap_95_high']:.1f}]",
        f"- learned route vs learned linear merged continuation: {100*merged['mean']:.1f}% MSE reduction",
        "",
        "| Outcome | Oracle branch vs constant | 95% CI |",
        "|---|---:|---:|",
    ]
    for outcome, values in payload["outcome_reductions_vs_constant"].items():
        lines.append(
            f"| `{outcome}` | {100*values['mean']:.1f}% | "
            f"[{100*values['bootstrap_95_low']:.1f}, {100*values['bootstrap_95_high']:.1f}] |"
        )
    lines.extend(["", "## Frozen checks", ""])
    for key, value in payload["gate_checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "This Gate measures whether a compact pre-feedback predictor can imitate future frozen-policy responses. It does not establish simulator success, latency benefit, or multi-task generality.",
            "",
            f"Decision: **{payload['decision']}**",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACD-VLA policy-response Gate 0")
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=260718)
    parser.add_argument("--shuffle-repeats", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_unsealed(args.collection_root)
    assert_unsealed(args.output_root)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite Gate result: {args.output_root}")
    records, manifest = load_records(args.collection_root)
    payload = evaluate(records, manifest, seed=args.seed, shuffle_repeats=args.shuffle_repeats)
    payload["collection_root"] = str(args.collection_root.resolve())
    payload["output_root"] = str(args.output_root.resolve())
    args.output_root.mkdir(parents=True)
    atomic_json(args.output_root / "gate0_results.json", payload)
    (args.output_root / "gate0_results.md").write_text(render_markdown(payload))
    print(json.dumps({"decision": payload["decision"], "gate_checks": payload["gate_checks"], "output_root": str(args.output_root)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
