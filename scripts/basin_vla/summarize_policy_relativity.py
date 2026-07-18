from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.basin_vla.policy_relativity import (
    STAGES,
    TARGET_POLICIES,
    bootstrap_mean,
    leave_one_policy_out_choice,
    lexicographic_percentiles,
    pairwise_policy_metrics,
)


DEFAULT_INPUT_ROOT = Path("/share/longjunyu/basin-vla/policy-relativity-gate0-v1")
POLICY_PAIRS = ((41, 42), (41, 43), (42, 43))


def _assert_unsealed_path(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    forbidden = {"test", "tests", "confirmation", "confirm", "sealed"}
    if lowered & forbidden or any("confirmation" in part for part in lowered):
        raise ValueError(f"refusing sealed path: {path}")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_records(input_root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    by_state: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for policy in TARGET_POLICIES:
        for path in sorted((input_root / f"policy-{policy}" / "records").glob("*.json")):
            row = json.loads(path.read_text())
            state_id = str(row["state_id"])
            if policy in by_state[state_id]:
                raise ValueError(f"duplicate record for state={state_id}, policy={policy}")
            by_state[state_id][policy] = row
    return dict(by_state)


def mean_by_source(rows: Sequence[Mapping[str, Any]], key: str) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        values[int(row["source_initial_state_index"])].append(float(row[key]))
    return {source: float(np.mean(items)) for source, items in values.items()}


def summarize(
    by_state: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    invalid_states = {
        state_id: {
            str(policy): row.get("error", "missing")
            for policy, row in policies.items()
            if row.get("status") != "valid"
        }
        for state_id, policies in by_state.items()
        if set(policies) != set(TARGET_POLICIES)
        or any(row.get("status") != "valid" for row in policies.values())
    }
    valid = {
        state_id: policies
        for state_id, policies in by_state.items()
        if set(policies) == set(TARGET_POLICIES)
        and all(row.get("status") == "valid" for row in policies.values())
    }
    state_rows = []
    for state_id, policies in sorted(valid.items()):
        reference = policies[41]
        fingerprints = {str(row["direct_endpoint_sha256"]) for row in policies.values()}
        candidate_paths = {str(row["candidate_cache"]) for row in policies.values()}
        if len(fingerprints) != 1 or len(candidate_paths) != 1:
            raise ValueError(f"same-state intervention parity failed for {state_id}")
        ranks = {
            policy: lexicographic_percentiles(row["continuation_keys"])
            for policy, row in policies.items()
        }
        pair_metrics = {
            f"{first}-{second}": pairwise_policy_metrics(ranks[first], ranks[second])
            for first, second in POLICY_PAIRS
        }
        loo = {
            str(policy): leave_one_policy_out_choice(ranks, policy)
            for policy in TARGET_POLICIES
        }
        state_rows.append(
            {
                "state_id": state_id,
                "pair_id": reference["pair_id"],
                "source_initial_state_index": int(reference["source_initial_state_index"]),
                "stage": reference["stage"],
                "replan_index": int(reference["replan_index"]),
                "direct_endpoint_sha256": next(iter(fingerprints)),
                "pair_metrics": pair_metrics,
                "leave_one_policy_out": loo,
            }
        )

    pair_summary = {}
    for pair_index, pair in enumerate(POLICY_PAIRS):
        label = f"{pair[0]}-{pair[1]}"
        rows = [
            {
                "source_initial_state_index": row["source_initial_state_index"],
                **row["pair_metrics"][label],
            }
            for row in state_rows
        ]
        source_flip = mean_by_source(rows, "preference_flip_rate")
        source_comparable = mean_by_source(rows, "comparable_fraction")
        source_jaccard = mean_by_source(rows, "top_tier_jaccard")
        pair_summary[label] = {
            "state_median_comparable_fraction": float(
                np.median([row["comparable_fraction"] for row in rows])
            ),
            "source_flip_rate": bootstrap_mean(
                list(source_flip.values()),
                samples=bootstrap_samples,
                seed=seed + pair_index * 10,
            ),
            "source_comparable_fraction": bootstrap_mean(
                list(source_comparable.values()),
                samples=bootstrap_samples,
                seed=seed + pair_index * 10 + 1,
            ),
            "source_top_tier_jaccard": bootstrap_mean(
                list(source_jaccard.values()),
                samples=bootstrap_samples,
                seed=seed + pair_index * 10 + 2,
            ),
        }

    loo_rows = []
    for row in state_rows:
        for policy in TARGET_POLICIES:
            loo_rows.append(
                {
                    "source_initial_state_index": row["source_initial_state_index"],
                    "stage": row["stage"],
                    "target_policy": policy,
                    **row["leave_one_policy_out"][str(policy)],
                }
            )
    source_gap = mean_by_source(loo_rows, "oracle_minus_loo")
    source_candidate0_gap = mean_by_source(loo_rows, "oracle_minus_candidate0")
    loo_summary = {
        "oracle_minus_leave_one_policy_out": bootstrap_mean(
            list(source_gap.values()), samples=bootstrap_samples, seed=seed + 100
        ),
        "oracle_minus_candidate0": bootstrap_mean(
            list(source_candidate0_gap.values()), samples=bootstrap_samples, seed=seed + 101
        ),
        "target_policy_mean": {
            str(policy): float(
                np.mean(
                    [
                        row["oracle_minus_loo"]
                        for row in loo_rows
                        if row["target_policy"] == policy
                    ]
                )
            )
            for policy in TARGET_POLICIES
        },
    }

    stage_summary = {}
    for stage in STAGES:
        stage_states = [row for row in state_rows if row["stage"] == stage]
        flip_values = [
            row["pair_metrics"][f"{first}-{second}"]["preference_flip_rate"]
            for row in stage_states
            for first, second in POLICY_PAIRS
        ]
        gaps = [
            row["leave_one_policy_out"][str(policy)]["oracle_minus_loo"]
            for row in stage_states
            for policy in TARGET_POLICIES
        ]
        stage_summary[stage] = {
            "state_count": len(stage_states),
            "mean_preference_flip_rate": float(np.mean(flip_values)) if flip_values else None,
            "mean_oracle_minus_loo": float(np.mean(gaps)) if gaps else None,
        }

    valid_sources = {int(row["source_initial_state_index"]) for row in state_rows}
    valid_stages = {str(row["stage"]) for row in state_rows}
    data_valid = len(state_rows) >= 18 and valid_stages == set(STAGES)
    comparable_pass = all(
        row["state_median_comparable_fraction"] >= 0.25 for row in pair_summary.values()
    )
    flip_pair_count = sum(
        row["source_flip_rate"]["mean"] >= 0.15 for row in pair_summary.values()
    )
    gap = loo_summary["oracle_minus_leave_one_policy_out"]
    broad_stage_count = sum(
        row["mean_preference_flip_rate"] is not None
        and row["mean_preference_flip_rate"] >= 0.15
        for row in stage_summary.values()
    )
    source_flip_values = []
    for source in valid_sources:
        values = [
            row["pair_metrics"][f"{first}-{second}"]["preference_flip_rate"]
            for row in state_rows
            if int(row["source_initial_state_index"]) == source
            for first, second in POLICY_PAIRS
        ]
        source_flip_values.append(float(np.mean(values)))
    broad_source_count = sum(value >= 0.15 for value in source_flip_values)
    broad_pass = broad_stage_count >= 3 and broad_source_count >= 3
    gate_checks = {
        "data_valid": data_valid,
        "comparable_fraction_pass": comparable_pass,
        "policy_pairs_with_flip_rate_at_least_15pp": flip_pair_count,
        "at_least_two_policy_pairs_pass": flip_pair_count >= 2,
        "oracle_minus_loo_mean_at_least_10pp": gap["mean"] >= 0.10,
        "oracle_minus_loo_ci_low_above_2pp": gap["bootstrap_95_low"] > 0.02,
        "broad_across_at_least_three_stages_and_sources": broad_pass,
        "stages_with_flip_rate_at_least_15pp": broad_stage_count,
        "sources_with_flip_rate_at_least_15pp": broad_source_count,
    }
    if not data_valid:
        decision = "GATE0_INVALID"
    elif all(
        (
            comparable_pass,
            flip_pair_count >= 2,
            gap["mean"] >= 0.10,
            gap["bootstrap_95_low"] > 0.02,
            broad_pass,
        )
    ):
        decision = "POLICY_RELATIVITY_EXISTS"
    else:
        decision = "STOP_POLICY_RELATIVE_COMMITTOR"
    return {
        "valid_state_count": len(state_rows),
        "invalid_states": invalid_states,
        "source_state_count": len(valid_sources),
        "stage_counts": {stage: sum(row["stage"] == stage for row in state_rows) for stage in STAGES},
        "state_rows": state_rows,
        "policy_pair_summary": pair_summary,
        "leave_one_policy_out_summary": loo_summary,
        "stage_summary": stage_summary,
        "gate_checks": gate_checks,
        "decision": decision,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# BASIN-VLA Gate 0：Policy Relativity",
        "",
        f"正式裁决：**{payload['decision']}**",
        "",
        "本实验仅使用既有 validation on-policy cache；没有打开 test 或 confirmation。",
        "",
        "## 数据",
        "",
        f"- 有效 states：{payload['valid_state_count']}；source states：{payload['source_state_count']}。",
        f"- 阶段：`{json.dumps(payload['stage_counts'], sort_keys=True)}`。",
        "- 所有 target policies 使用相同 simulator snapshot、相同 16 candidates 与 matched rollout seed。",
        "",
        "## Cross-policy preference",
        "",
        "| Policy pair | Comparable median | Flip rate (source mean, 95% CI) | Top-tier Jaccard |",
        "|---|---:|---:|---:|",
    ]
    for pair, row in payload["policy_pair_summary"].items():
        flip = row["source_flip_rate"]
        jaccard = row["source_top_tier_jaccard"]
        lines.append(
            f"| `{pair}` | {100*row['state_median_comparable_fraction']:.1f}% | "
            f"{100*flip['mean']:.1f}% [{100*flip['bootstrap_95_low']:.1f}, {100*flip['bootstrap_95_high']:.1f}] | "
            f"{100*jaccard['mean']:.1f}% |"
        )
    gap = payload["leave_one_policy_out_summary"]["oracle_minus_leave_one_policy_out"]
    candidate0 = payload["leave_one_policy_out_summary"]["oracle_minus_candidate0"]
    lines.extend(
        [
            "",
            "## Selection regret",
            "",
            f"- Target-policy Oracle - leave-one-policy-out selector：{100*gap['mean']:.1f} pp，"
            f"source bootstrap 95% CI `[{100*gap['bootstrap_95_low']:.1f}, {100*gap['bootstrap_95_high']:.1f}]`。",
            f"- Target-policy Oracle - candidate0：{100*candidate0['mean']:.1f} pp。",
            "",
            "## Gate checks",
            "",
        ]
    )
    for key, value in payload["gate_checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "本 Gate 只判断 policy conditioning 是否必要。即使通过，也没有证明 learned committor 可预测、"
            "可泛化或能改善闭环成功率。",
            "",
            f"最终裁决：**{payload['decision']}**",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize BASIN-VLA policy-relativity Gate 0")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=260718)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _assert_unsealed_path(args.input_root)
    by_state = load_records(args.input_root)
    if not args.allow_incomplete:
        expected = 21
        counts = {
            policy: len(list((args.input_root / f"policy-{policy}" / "records").glob("*.json")))
            for policy in TARGET_POLICIES
        }
        if any(count != expected for count in counts.values()):
            raise ValueError(f"policy-relativity collection incomplete: {counts}")
    payload = {
        "schema_version": 1,
        "experiment": "basin_vla_policy_relativity_gate0",
        "data_policy": "existing validation cache only; test and confirmation not accessed",
        "input_root": str(args.input_root),
        "statistical_unit": "source initial state; candidates and replanning states are nested",
        **summarize(
            by_state,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    atomic_json(args.input_root / "gate0_results.json", payload)
    (args.input_root / "gate0_results.md").write_text(render_markdown(payload))
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "valid_state_count": payload["valid_state_count"],
                "gate_checks": payload["gate_checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
