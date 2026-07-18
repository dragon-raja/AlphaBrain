from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.branch_vla.gate0_common import (
    ALPHAS,
    HORIZON,
    LEADS,
    BranchExample,
    bootstrap_mean,
    fit_ridge,
    masked_mse,
    mean_by_source,
    observation_feature,
    prefix_mse,
    select_alpha_classifier,
    select_alpha_regression,
)


DEFAULT_EPISODE_ROOT = Path("/share/longjunyu/fresh-vla/libero-full-episode-v2-128")
DEFAULT_OUTPUT_ROOT = Path("/share/longjunyu/branch-vla/gate0-representation-v1")
FORBIDDEN_PARTS = {"test", "tests", "confirmation", "confirm", "sealed"}


def assert_unsealed(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    if lowered & FORBIDDEN_PARTS or any("confirmation" in part for part in lowered):
        raise ValueError(f"refusing sealed path: {path}")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _episode_path(root: Path, group: Mapping[str, Any], outcome: str) -> Path:
    if group["split"] not in {"train", "val"}:
        raise ValueError(f"refusing non-development split: {group['split']}")
    path = root / str(group["episode_files"][outcome])
    assert_unsealed(path)
    return path


def load_data(
    episode_root: Path,
    *,
    image_size: int,
) -> tuple[
    list[BranchExample],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    manifest = json.loads((episode_root / "manifest.json").read_text())
    groups = [group for group in manifest["groups"] if group["split"] in {"train", "val"}]
    examples: list[BranchExample] = []
    guards: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    parity_failures = []
    split_groups: dict[str, set[str]] = defaultdict(set)
    split_sources: dict[str, set[int]] = defaultdict(set)
    required = ("agentview", "wrist", "robot_state", "actions")

    for group in sorted(groups, key=lambda item: str(item["pair_id"])):
        split = str(group["split"])
        pair_id = str(group["pair_id"])
        source_id = int(group["source_initial_state_index"])
        reveal = int(group["feedback_reveal_time"])
        split_groups[split].add(pair_id)
        split_sources[split].add(source_id)
        with np.load(_episode_path(episode_root, group, "attached"), allow_pickle=False) as handle:
            attached = {key: handle[key] for key in required}
        with np.load(_episode_path(episode_root, group, "slipped"), allow_pickle=False) as handle:
            slipped = {key: handle[key] for key in required}

        parity = {
            key: bool(np.array_equal(attached[key][:reveal], slipped[key][:reveal]))
            for key in required
        }
        if not all(parity.values()):
            parity_failures.append({"pair_id": pair_id, "checks": parity})
            continue
        if min(len(attached["actions"]), len(slipped["actions"])) < reveal + HORIZON:
            parity_failures.append({"pair_id": pair_id, "checks": {"horizon_available": False}})
            continue

        for lead in LEADS:
            index = reveal - lead
            if index < 0:
                parity_failures.append({"pair_id": pair_id, "checks": {"lead_available": False}})
                continue
            feature = observation_feature(
                attached["agentview"][index],
                attached["wrist"][index],
                attached["robot_state"][index],
                lead=lead,
                image_size=image_size,
            )
            examples.append(
                BranchExample(
                    pair_id=pair_id,
                    source_id=source_id,
                    split=split,
                    lead=lead,
                    feature=feature,
                    attached=np.asarray(attached["actions"][index : index + HORIZON], dtype=np.float64),
                    slipped=np.asarray(slipped["actions"][index : index + HORIZON], dtype=np.float64),
                )
            )

        for outcome, arrays, label in (("attached", attached, 1.0), ("slipped", slipped, -1.0)):
            guards[split].append(
                {
                    "pair_id": pair_id,
                    "source_id": source_id,
                    "outcome": outcome,
                    "label": label,
                    "post": observation_feature(
                        arrays["agentview"][reveal],
                        arrays["wrist"][reveal],
                        arrays["robot_state"][reveal],
                        image_size=image_size,
                    ),
                    "pre": observation_feature(
                        arrays["agentview"][reveal - 1],
                        arrays["wrist"][reveal - 1],
                        arrays["robot_state"][reveal - 1],
                        image_size=image_size,
                    ),
                }
            )

    audit = {
        "split_group_counts": {split: len(values) for split, values in split_groups.items()},
        "split_source_counts": {split: len(values) for split, values in split_sources.items()},
        "example_counts": {
            split: sum(example.split == split for example in examples) for split in ("train", "val")
        },
        "parity_failure_count": len(parity_failures),
        "parity_failures": parity_failures,
        "test_episode_files_opened": 0,
        "confirmation_paths_opened": 0,
    }
    return examples, guards, audit


def action_statistics(examples: Sequence[BranchExample]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate(
        [
            np.stack([example.attached for example in examples]),
            np.stack([example.slipped for example in examples]),
        ],
        axis=0,
    )
    mean = values.reshape(-1, values.shape[-1]).mean(axis=0)
    scale = values.reshape(-1, values.shape[-1]).std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def normalize_actions(value: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=np.float64) - mean) / scale


def reduction_summary(
    rows: Sequence[Mapping[str, float | int]],
    baseline_key: str,
    method_key: str,
    *,
    seed: int,
    samples: int = 20_000,
) -> dict[str, float | int]:
    baseline = mean_by_source(rows, baseline_key)
    method = mean_by_source(rows, method_key)
    sources = sorted(set(baseline) & set(method))
    base_array = np.asarray([baseline[source] for source in sources], dtype=np.float64)
    method_array = np.asarray([method[source] for source in sources], dtype=np.float64)
    if np.any(base_array <= 0):
        raise ValueError(f"baseline {baseline_key} must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sources), size=(samples, len(sources)))
    draws = 1.0 - method_array[indices].mean(axis=1) / base_array[indices].mean(axis=1)
    return {
        "count": len(sources),
        "mean": float(1.0 - method_array.mean() / base_array.mean()),
        "bootstrap_95_low": float(np.quantile(draws, 0.025)),
        "bootstrap_95_high": float(np.quantile(draws, 0.975)),
        "baseline_source_mean": float(base_array.mean()),
        "method_source_mean": float(method_array.mean()),
    }


def guard_metrics(
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    shuffle_repeats: int,
) -> tuple[dict[str, Any], dict[tuple[str, str], float]]:
    train_x = np.stack([row["post"] for row in train_rows])
    train_y = np.asarray([row["label"] for row in train_rows], dtype=np.float64)
    train_sources = [int(row["source_id"]) for row in train_rows]
    alpha, cv = select_alpha_classifier(train_x, train_y, train_sources)
    model = fit_ridge(train_x, train_y[:, None], alpha)
    val_post = np.stack([row["post"] for row in val_rows])
    val_pre = np.stack([row["pre"] for row in val_rows])
    labels = np.asarray([row["label"] for row in val_rows], dtype=np.float64)
    post_scores = model.predict(val_post)[:, 0]
    pre_scores = model.predict(val_pre)[:, 0]
    post_prediction = np.where(post_scores >= 0, 1.0, -1.0)
    pre_prediction = np.where(pre_scores >= 0, 1.0, -1.0)

    sample_rows = []
    paired_scores: dict[str, dict[str, float]] = defaultdict(dict)
    route_scores = {}
    for row, post, pre, post_pred, pre_pred in zip(
        val_rows, post_scores, pre_scores, post_prediction, pre_prediction, strict=True
    ):
        route_scores[(str(row["pair_id"]), str(row["outcome"]))] = float(post)
        paired_scores[str(row["pair_id"])][str(row["outcome"])] = float(post)
        sample_rows.append(
            {
                "source_id": int(row["source_id"]),
                "post_correct": float(post_pred == row["label"]),
                "pre_correct": float(pre_pred == row["label"]),
            }
        )
    source_post = mean_by_source(sample_rows, "post_correct")
    source_pre = mean_by_source(sample_rows, "pre_correct")
    ranking_rows = []
    pair_to_source = {str(row["pair_id"]): int(row["source_id"]) for row in val_rows}
    for pair_id, values in paired_scores.items():
        if values.keys() >= {"attached", "slipped"}:
            ranking_rows.append(
                {
                    "source_id": pair_to_source[pair_id],
                    "correct": float(values["attached"] > values["slipped"]),
                }
            )
    source_ranking = mean_by_source(ranking_rows, "correct")

    rng = np.random.default_rng(seed)
    shuffled_accuracies = []
    for _ in range(shuffle_repeats):
        shuffled = rng.permutation(train_y)
        shuffled_model = fit_ridge(train_x, shuffled[:, None], alpha)
        prediction = np.where(shuffled_model.predict(val_post)[:, 0] >= 0, 1.0, -1.0)
        shuffled_accuracies.append(float(np.mean(prediction == labels)))

    payload = {
        "selected_alpha": alpha,
        "train_source_cv_accuracy": cv,
        "post_feedback_accuracy": float(np.mean(post_prediction == labels)),
        "pre_feedback_accuracy": float(np.mean(pre_prediction == labels)),
        "post_feedback_source_bootstrap": bootstrap_mean(
            list(source_post.values()), seed=seed + 1
        ),
        "pre_feedback_source_bootstrap": bootstrap_mean(
            list(source_pre.values()), seed=seed + 2
        ),
        "pair_ranking_accuracy": float(np.mean(list(source_ranking.values()))),
        "shuffle_repeats": shuffle_repeats,
        "shuffled_label_mean_accuracy": float(np.mean(shuffled_accuracies)),
        "shuffled_label_accuracies": shuffled_accuracies,
    }
    return payload, route_scores


def evaluate(
    examples: Sequence[BranchExample],
    guard_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    audit: Mapping[str, Any],
    *,
    seed: int,
    shuffle_repeats: int,
) -> dict[str, Any]:
    train = [example for example in examples if example.split == "train"]
    val = [example for example in examples if example.split == "val"]
    action_mean, action_scale = action_statistics(train)
    train_x = np.stack([example.feature for example in train])
    train_attached = np.stack(
        [normalize_actions(example.attached, action_mean, action_scale) for example in train]
    )
    train_slipped = np.stack(
        [normalize_actions(example.slipped, action_mean, action_scale) for example in train]
    )
    output_shape = train_attached.shape[1:]
    branch_targets = np.concatenate(
        (train_attached.reshape(len(train), -1), train_slipped.reshape(len(train), -1)), axis=1
    )
    alpha, cv = select_alpha_regression(
        train_x,
        branch_targets,
        [example.source_id for example in train],
    )
    branch_model = fit_ridge(train_x, branch_targets, alpha)
    merged_targets = 0.5 * (
        train_attached.reshape(len(train), -1) + train_slipped.reshape(len(train), -1)
    )
    merged_model = fit_ridge(train_x, merged_targets, alpha)

    constant: dict[tuple[str, int], np.ndarray] = {}
    for outcome, values in (("attached", train_attached), ("slipped", train_slipped)):
        for lead in LEADS:
            mask = np.asarray([example.lead == lead for example in train])
            constant[(outcome, lead)] = values[mask].mean(axis=0)

    guard, route_scores = guard_metrics(
        guard_rows["train"],
        guard_rows["val"],
        seed=seed + 100,
        shuffle_repeats=shuffle_repeats,
    )
    val_x = np.stack([example.feature for example in val])
    branch_prediction = branch_model.predict(val_x)
    split = int(np.prod(output_shape))
    pred_attached = branch_prediction[:, :split].reshape((-1, *output_shape))
    pred_slipped = branch_prediction[:, split:].reshape((-1, *output_shape))
    pred_merged = merged_model.predict(val_x).reshape((-1, *output_shape))

    rows = []
    for index, example in enumerate(val):
        target = {
            "attached": normalize_actions(example.attached, action_mean, action_scale),
            "slipped": normalize_actions(example.slipped, action_mean, action_scale),
        }
        predictions = {"attached": pred_attached[index], "slipped": pred_slipped[index]}
        separation_target = float(
            np.sqrt(np.mean(np.square(target["attached"][example.lead :] - target["slipped"][example.lead :])))
        )
        separation_predicted = float(
            np.sqrt(
                np.mean(
                    np.square(
                        predictions["attached"][example.lead :]
                        - predictions["slipped"][example.lead :]
                    )
                )
            )
        )
        for outcome in ("attached", "slipped"):
            opposite = "slipped" if outcome == "attached" else "attached"
            route_score = route_scores[(example.pair_id, outcome)]
            routed = predictions["attached"] if route_score >= 0 else predictions["slipped"]
            own = predictions[outcome]
            other = predictions[opposite]
            own_mse = masked_mse(own, target[outcome], example.lead)
            other_mse = masked_mse(other, target[outcome], example.lead)
            rows.append(
                {
                    "pair_id": example.pair_id,
                    "source_id": example.source_id,
                    "lead": example.lead,
                    "outcome": outcome,
                    "linear_suffix_mse": masked_mse(pred_merged[index], target[outcome], example.lead),
                    "oracle_suffix_mse": own_mse,
                    "learned_suffix_mse": masked_mse(routed, target[outcome], example.lead),
                    "random_precommit_suffix_mse": 0.5 * (own_mse + other_mse),
                    "constant_suffix_mse": masked_mse(
                        constant[(outcome, example.lead)], target[outcome], example.lead
                    ),
                    "set_coverage_suffix_mse": min(own_mse, other_mse),
                    "linear_prefix_mse": prefix_mse(pred_merged[index], target[outcome], example.lead),
                    "oracle_prefix_mse": prefix_mse(own, target[outcome], example.lead),
                    "separation_ratio": separation_predicted / max(separation_target, 1e-8),
                }
            )

    reductions = {
        "learned_vs_linear": reduction_summary(
            rows, "linear_suffix_mse", "learned_suffix_mse", seed=seed + 200
        ),
        "oracle_vs_linear": reduction_summary(
            rows, "linear_suffix_mse", "oracle_suffix_mse", seed=seed + 201
        ),
        "oracle_vs_constant": reduction_summary(
            rows, "constant_suffix_mse", "oracle_suffix_mse", seed=seed + 202
        ),
        "random_precommit_vs_linear": reduction_summary(
            rows, "linear_suffix_mse", "random_precommit_suffix_mse", seed=seed + 203
        ),
    }
    branch_reductions = {
        outcome: reduction_summary(
            [row for row in rows if row["outcome"] == outcome],
            "linear_suffix_mse",
            "learned_suffix_mse",
            seed=seed + 210 + index,
        )
        for index, outcome in enumerate(("attached", "slipped"))
    }
    metric_means = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in (
            "linear_suffix_mse",
            "oracle_suffix_mse",
            "learned_suffix_mse",
            "random_precommit_suffix_mse",
            "constant_suffix_mse",
            "set_coverage_suffix_mse",
            "linear_prefix_mse",
            "oracle_prefix_mse",
        )
    }
    separation_by_source = mean_by_source(rows, "separation_ratio")
    median_separation = float(np.median(list(separation_by_source.values())))

    data_valid = bool(
        audit["split_group_counts"] == {"train": 102, "val": 13}
        and int(audit["split_source_counts"].get("val", 0)) >= 8
        and audit["parity_failure_count"] == 0
        and len(train) == 102 * len(LEADS)
        and len(val) == 13 * len(LEADS)
    )
    checks = {
        "data_valid": data_valid,
        "post_guard_at_least_85pct": guard["post_feedback_accuracy"] >= 0.85,
        "pre_guard_at_most_60pct": guard["pre_feedback_accuracy"] <= 0.60,
        "shuffle_control_at_most_60pct": guard["shuffled_label_mean_accuracy"] <= 0.60,
        "learned_reduction_at_least_25pct": reductions["learned_vs_linear"]["mean"] >= 0.25,
        "learned_reduction_ci_low_above_10pct": reductions["learned_vs_linear"][
            "bootstrap_95_low"
        ]
        > 0.10,
        "oracle_reduction_at_least_30pct": reductions["oracle_vs_linear"]["mean"] >= 0.30,
        "both_branches_reduction_at_least_15pct": all(
            row["mean"] >= 0.15 for row in branch_reductions.values()
        ),
        "oracle_beats_constant_by_20pct": reductions["oracle_vs_constant"]["mean"] >= 0.20,
        "median_separation_ratio_at_least_0_5": median_separation >= 0.5,
    }
    if not data_valid:
        decision = "GATE0_INVALID"
    elif all(checks.values()):
        decision = "BRANCH_REPRESENTATION_FEASIBLE"
    else:
        decision = "STOP_BRANCH_ACTION_CHUNK"
    return {
        "schema_version": 1,
        "experiment": "branch_vla_gate0_representation",
        "decision": decision,
        "data_policy": "train/val episode arrays only; test and confirmation episode files not opened",
        "statistical_unit": "source initial state; leads and outcomes are nested",
        "configuration": {
            "leads": list(LEADS),
            "horizon": HORIZON,
            "image_size": 8,
            "alphas": list(ALPHAS),
            "action_dimension": int(action_mean.shape[0]),
        },
        "audit": dict(audit),
        "action_normalization": {"mean": action_mean.tolist(), "scale": action_scale.tolist()},
        "action_predictor": {
            "selected_alpha": alpha,
            "train_source_cv_mse": cv,
        },
        "guard": guard,
        "metric_means": metric_means,
        "reductions": reductions,
        "branch_reductions": branch_reductions,
        "median_source_separation_ratio": median_separation,
        "gate_checks": checks,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    reduction = payload["reductions"]["learned_vs_linear"]
    oracle = payload["reductions"]["oracle_vs_linear"]
    constant = payload["reductions"]["oracle_vs_constant"]
    guard = payload["guard"]
    lines = [
        "# Branch-VLA Gate 0：Contingent Chunk 表示可学性",
        "",
        f"正式裁决：**{payload['decision']}**",
        "",
        "本 Gate 只打开 train/val episode arrays；没有打开 original test 或 confirmation episode。",
        "",
        "## 数据与 parity",
        "",
        f"- groups：`{json.dumps(payload['audit']['split_group_counts'], sort_keys=True)}`；"
        f"source：`{json.dumps(payload['audit']['split_source_counts'], sort_keys=True)}`。",
        f"- pre-feedback parity failures：{payload['audit']['parity_failure_count']}。",
        "",
        "## Guard",
        "",
        f"- post-feedback accuracy：{100*guard['post_feedback_accuracy']:.1f}%。",
        f"- pre-feedback accuracy：{100*guard['pre_feedback_accuracy']:.1f}%。",
        f"- shuffled-label mean：{100*guard['shuffled_label_mean_accuracy']:.1f}%。",
        f"- paired ranking：{100*guard['pair_ranking_accuracy']:.1f}%。",
        "",
        "## Action branches",
        "",
        f"- learned route 相对 linear chunk：{100*reduction['mean']:.1f}% 风险下降，"
        f"source bootstrap 95% CI `[{100*reduction['bootstrap_95_low']:.1f}, "
        f"{100*reduction['bootstrap_95_high']:.1f}]`。",
        f"- oracle route 相对 linear chunk：{100*oracle['mean']:.1f}% 风险下降。",
        f"- oracle route 相对 per-lead constant：{100*constant['mean']:.1f}% 风险下降。",
        f"- source-median branch separation ratio：{payload['median_source_separation_ratio']:.3f}。",
        "",
        "| Branch | Learned route vs linear | 95% CI |",
        "|---|---:|---:|",
    ]
    for outcome, row in payload["branch_reductions"].items():
        lines.append(
            f"| `{outcome}` | {100*row['mean']:.1f}% | "
            f"[{100*row['bootstrap_95_low']:.1f}, {100*row['bootstrap_95_high']:.1f}] |"
        )
    lines.extend(["", "## Gate checks", ""])
    for key, value in payload["gate_checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "即使通过，本结果也只证明 paired action branches 可由 pre-feedback conditioning 预测并在"
            "feedback 后路由；没有证明真实闭环、policy-call 效率或 Pi0.5 branch head 有收益。",
            "",
            f"最终裁决：**{payload['decision']}**",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Branch-VLA representation Gate 0")
    parser.add_argument("--episode-root", type=Path, default=DEFAULT_EPISODE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=260718)
    parser.add_argument("--shuffle-repeats", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_unsealed(args.episode_root)
    assert_unsealed(args.output_root)
    if args.image_size != 8:
        raise ValueError("Gate 0 image_size is frozen at 8")
    examples, guards, audit = load_data(args.episode_root, image_size=args.image_size)
    payload = evaluate(
        examples,
        guards,
        audit,
        seed=args.seed,
        shuffle_repeats=args.shuffle_repeats,
    )
    payload["episode_root"] = str(args.episode_root)
    payload["output_root"] = str(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / "gate0_results.json", payload)
    (args.output_root / "gate0_results.md").write_text(render_markdown(payload))
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "gate_checks": payload["gate_checks"],
                "output_root": str(args.output_root),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
