#!/usr/bin/env python3
"""Prepare, validate and analyze full-bank searches on screened failure states."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__:
    from .analyze_view_value_discovery import atomic_json, flatten_row, pairwise_auc
    from .build_view_repeatability_protocols import normalized_pose_distance
    from .expand_protocol_noise_repeats import build as repeat_protocol
else:
    from analyze_view_value_discovery import atomic_json, flatten_row, pairwise_auc
    from build_view_repeatability_protocols import normalized_pose_distance
    from expand_protocol_noise_repeats import build as repeat_protocol


DATA = Path("/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-all8-v1")
DEFAULT_ROOT = DATA.parent / "failure-pool-v1-20260831"
DISCOVERY_SEEDS = [20260861, 20260862, 20260863]
CONFIRMATION_SEEDS = [20260871, 20260872, 20260873, 20260874, 20260875]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_write(path, payload):
    if path.exists():
        if read(path) != payload:
            raise ValueError(f"refusing to change frozen protocol: {path}")
    else:
        atomic_json(path, payload)


def prepare_payload(combined, protocols, *, minimum_failures=3):
    if combined.get("status") != "PASS" or combined["expected_repeats"] != 5:
        raise ValueError("screening results must have five complete repeats")
    canonical = [r for r in combined["state_method_rows"] if r["selector_method"] == "canonical"]
    if len({r["pair_key"] for r in canonical}) != len(canonical):
        raise ValueError("duplicate screening state")
    selected = {r["pair_key"]: r for r in canonical if 5 - r["repeat_successes"] >= minimum_failures}
    if not selected:
        raise ValueError("no eligible failure states")
    catalogs = {p["catalog"] for p in protocols}
    if len(catalogs) != 1 or any(p["status"] != "PASS" for p in protocols):
        raise ValueError("base protocols must PASS and share a catalog")
    specs = []
    states = []
    seen = set()
    candidate_sets = {}
    for p in protocols:
        state_meta = {s["pair_key"]: s for s in p["selected_states"]}
        grouped = defaultdict(list)
        for spec in p["specs"]:
            if spec["pair_key"] in selected:
                grouped[spec["pair_key"]].append(spec)
        for key, rows in sorted(grouped.items()):
            if key in seen:
                raise ValueError("state occurs in more than one source protocol")
            seen.add(key)
            candidates = {r["selected_candidate_id"] for r in rows}
            if len(rows) != 97 or len(candidates) != 97 or "canonical" not in candidates:
                raise ValueError("each failure state must keep the full 97-view bank")
            candidate_sets[key] = candidates
            screen = selected[key]
            states.append(
                {
                    **state_meta[key],
                    "source_group": screen["source_group"],
                    "screening_successes_of_5": screen["repeat_successes"],
                }
            )
            for original in sorted(rows, key=lambda r: r["selected_candidate_id"]):
                spec = dict(original)
                spec.update(
                    episode_id=hashlib.sha256(
                        f"failure-bank-v1::{key}::{spec['selected_candidate_id']}".encode()
                    ).hexdigest()[:20],
                    condition=f"candidate__{spec['selected_candidate_id']}",
                    diagnostic_role="posthoc_failure_full_bank_discovery",
                )
                specs.append(spec)
    if seen != set(selected) or len({tuple(sorted(v)) for v in candidate_sets.values()}) != 1:
        raise ValueError("missing failure state or unequal candidate bank")
    states.sort(key=lambda s: s["pair_key"])
    specs.sort(key=lambda s: (s["pair_key"], s["selected_candidate_id"]))
    return {
        "schema": "dsol_failure_full_bank_protocol_v1",
        "status": "PASS",
        "analysis_role": "posthoc_discovery_on_previously_viewed_sources",
        "confirmatory_test_eligible": False,
        "state_selection_uses_screening_policy_outcomes": True,
        "candidate_prefilter_by_visibility_or_accel": False,
        "screening_rule": f"at least {minimum_failures} failures in five canonical screening repeats",
        "screening_noise_seeds": [20260841, 20260842, 20260843, 20260844, 20260845],
        "catalog": next(iter(catalogs)),
        "candidate_count": 97,
        "selected_state_count": len(states),
        "source_episode_count": len({s["source_group"] for s in states}),
        "task_count": len({s["task_id"] for s in states}),
        "episode_count": len(specs),
        "selected_states": states,
        "specs": specs,
        "discovery_noise_seeds": DISCOVERY_SEEDS,
        "confirmation_noise_seeds": CONFIRMATION_SEEDS,
        "confirmation_selection": {
            "top_noncanonical": 4,
            "bottom_noncanonical": 1,
            "pose_diverse_controls": 2,
            "canonical": 1,
            "primary_rule": "discovery_rank_top1_frozen",
        },
        "positive_label_rule": {
            "candidate_success_at_least": 0.8,
            "paired_advantage_at_least": 0.4,
            "role": "exploratory_label_not_familywise_significance",
        },
    }


def prepare(root):
    source_paths = [
        DATA / "independent-source-extension/combined-analysis/analysis.json",
        DATA / "dense-test-protocol.json",
        DATA / "independent-source-extension/dense-test-protocol.json",
    ]
    payload = prepare_payload(read(source_paths[0]), [read(p) for p in source_paths[1:]])
    payload["input_sha256"] = {str(p): sha(p) for p in source_paths}
    payload["catalog_sha256"] = sha(payload["catalog"])
    if (payload["selected_state_count"], payload["source_episode_count"], payload["task_count"]) != (20, 17, 6):
        raise ValueError("screening inventory changed; review instead of silently changing cohort")
    base_path = root / "protocols/discovery-base.json"
    frozen_write(base_path, payload)
    repeated = repeat_protocol(payload, DISCOVERY_SEEDS, source_path=base_path)
    frozen_write(root / "protocols/discovery-three-noise.json", repeated)
    return payload


def load_run(protocol, run_dir, *, require_complete=True):
    expected = {r["episode_id"]: r for r in protocol["specs"]}
    found = {}
    for path in sorted(run_dir.glob("episodes-shard-*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            eid = row["episode_id"]
            if eid not in expected or eid in found or row["status"] != "complete":
                raise ValueError("unexpected, duplicate, or incomplete episode")
            spec = expected[eid]
            for field in ["pair_key", "selected_candidate_id", "evaluation_seed", "sensor_control"]:
                if row[field] != spec[field]:
                    raise ValueError(f"episode identity mismatch: {field}")
            if row["replan_steps"] != 5 or row["wait_steps"] != 0 or row["initial_metrics"]["initial_task_success"]:
                raise ValueError("rollout contract mismatch or initially successful state")
            found[eid] = row
    if require_complete and set(found) != set(expected):
        raise ValueError(f"incomplete matrix: {len(found)}/{len(expected)}")
    pairs = defaultdict(list)
    for row in found.values():
        pairs[row["pair_key"], row["evaluation_seed"]].append(row)
    for rows in pairs.values():
        if len({r["policy_noise_seed"] for r in rows}) != 1:
            raise ValueError("candidate noise mismatch")
        if len({r["initial_metrics"]["physics_state_sha256"] for r in rows}) != 1:
            raise ValueError("candidate physical state mismatch")
    return list(found.values())


def candidate_summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["pair_key"], row["selected_candidate_id"]].append(row)
    result = defaultdict(dict)
    for (state, candidate), values in sorted(grouped.items()):
        if len({v["evaluation_seed"] for v in values}) != len(values):
            raise ValueError("duplicate candidate repeat")
        flat = flatten_row(values[0])
        result[state][candidate] = {
            **flat,
            "pose": values[0].get("pose"),
            "mean_success": float(np.mean([v["success"] for v in values])),
            "mean_steps": float(np.mean([v["completion_steps"] for v in values])),
            "success_by_noise": {str(v["evaluation_seed"]): int(v["success"]) for v in values},
        }
    for candidates in result.values():
        baseline = candidates["canonical"]["mean_success"]
        for row in candidates.values():
            row["advantage"] = row["mean_success"] - baseline
            row["delta_visibility"] = row["visibility_score"] - candidates["canonical"]["visibility_score"]
    return dict(result)


def ranked_noncanonical(candidates):
    return sorted(
        (c for c in candidates if c != "canonical"),
        key=lambda c: (-candidates[c]["mean_success"], candidates[c]["mean_steps"], c),
    )


def choose_confirmation(candidates):
    ranked = ranked_noncanonical(candidates)
    if len(ranked) < 7:
        raise ValueError("not enough candidates for frozen top and control views")
    chosen = {c: ["discovery_top4"] for c in ranked[:4]}
    chosen[ranked[0]].append("primary_frozen_top1")
    chosen["canonical"] = ["canonical"]
    worst = next(c for c in reversed(ranked) if c not in chosen)
    chosen[worst] = ["discovery_bottom_control"]
    while len(chosen) < 8:
        remaining = [c for c in sorted(candidates) if c not in chosen]
        candidate = max(
            remaining, key=lambda c: (min(normalized_pose_distance(candidates[c], candidates[s]) for s in chosen), c)
        )
        chosen[candidate] = ["pose_diverse_control"]
    return chosen


def freeze_confirmation(root):
    protocol = read(root / "protocols/discovery-three-noise.json")
    rows = load_run(protocol, root / "discovery")
    summaries = candidate_summaries(rows)
    base = read(root / "protocols/discovery-base.json")
    specs_by_key = {(r["pair_key"], r["selected_candidate_id"]): r for r in base["specs"]}
    selections = {state: choose_confirmation(candidates) for state, candidates in summaries.items()}
    specs = []
    for state, choices in sorted(selections.items()):
        for candidate, roles in choices.items():
            spec = dict(specs_by_key[state, candidate])
            spec.update(
                episode_id=hashlib.sha256(f"confirmation-v1::{state}::{candidate}".encode()).hexdigest()[:20],
                condition=f"candidate__{candidate}",
                candidate_roles=roles,
                diagnostic_role="independent_noise_confirmation_frozen_candidates",
            )
            specs.append(spec)
    payload = {
        **base,
        "schema": "dsol_failure_frozen_candidates_v1",
        "specs": specs,
        "episode_count": len(specs),
        "candidate_count": 8,
        "selected_candidates": selections,
        "candidate_selection_uses_discovery_outcomes": True,
        "candidate_selection_uses_confirmation_outcomes": False,
        "discovery_protocol_sha256": sha(root / "protocols/discovery-three-noise.json"),
        "discovery_results_sha256": {
            str(p): sha(p) for p in sorted((root / "discovery").glob("episodes-shard-*.jsonl"))
        },
    }
    path = root / "protocols/confirmation-base.json"
    frozen_write(path, payload)
    frozen_write(
        root / "protocols/confirmation-five-noise.json", repeat_protocol(payload, CONFIRMATION_SEEDS, source_path=path)
    )
    atomic_json(
        root / "analysis/discovery.json", {"status": "COMPLETE", "states": summaries, "selected_candidates": selections}
    )
    return payload


def paired_source_summary(state_rows):
    grouped = defaultdict(list)
    for row in state_rows:
        grouped[row["source_group"]].append(row["advantage"])
    means = np.asarray([np.mean(values) for values in grouped.values()])
    draws = np.random.default_rng(20260831).choice(means, (10000, len(means)), replace=True).mean(axis=1)
    return {
        "source_groups": len(means),
        "source_equal_advantage_pp": float(means.mean() * 100),
        "source_bootstrap_ci95_pp": (100 * np.quantile(draws, [0.025, 0.975])).tolist(),
        "state_equal_advantage_pp": float(np.mean([r["advantage"] for r in state_rows]) * 100),
    }


def summarize(root):
    discovery = read(root / "analysis/discovery.json")["states"]
    protocol = read(root / "protocols/confirmation-five-noise.json")
    confirmation = candidate_summaries(load_run(protocol, root / "confirmation"))
    metadata = {s["pair_key"]: s for s in protocol["selected_states"]}
    state_rows = []
    candidate_rows = []
    for state, views in confirmation.items():
        roles = protocol["selected_candidates"][state]
        primary = next(c for c, r in roles.items() if "primary_frozen_top1" in r)
        base = views["canonical"]
        oracle_candidate, oracle_view = max(
            views.items(), key=lambda item: (item[1]["mean_success"], -item[1]["mean_steps"], item[0])
        )
        discovery_oracle_candidate, discovery_oracle_view = max(
            discovery[state].items(),
            key=lambda item: (item[1]["mean_success"], -item[1]["mean_steps"], item[0]),
        )
        positives = [
            c for c, v in views.items() if c != "canonical" and v["mean_success"] >= 0.8 and v["advantage"] >= 0.4
        ]
        d = discovery[state]
        positive_discovery = [
            v for c, v in d.items() if c != "canonical" and v["mean_success"] >= 2 / 3 and v["advantage"] >= 1 / 3 - 1e-9
        ]
        state_rows.append(
            {
                "pair_key": state,
                "task_id": base["task_id"],
                "source_group": metadata[state]["source_group"],
                "screening_successes_of_5": metadata[state]["screening_successes_of_5"],
                "canonical_confirmation_success": base["mean_success"],
                "primary_candidate": primary,
                "primary_confirmation_success": views[primary]["mean_success"],
                "advantage": views[primary]["advantage"],
                "posthoc_oracle8_candidate": oracle_candidate,
                "posthoc_oracle8_success": oracle_view["mean_success"],
                "discovery_oracle97_candidate": discovery_oracle_candidate,
                "discovery_oracle97_success": discovery_oracle_view["mean_success"],
                "primary_confirmation_rank": 1
                + sum(
                    view["mean_success"] > views[primary]["mean_success"]
                    for candidate, view in views.items()
                    if candidate != primary
                ),
                "discovery_positive_candidate_count": len(positive_discovery),
                "confirmation_positive_candidates": positives,
                "canonical_failure_persists": base["mean_success"] <= 0.4,
            }
        )
        for candidate, view in d.items():
            c = views.get(candidate)
            candidate_rows.append(
                {
                    **{k: v for k, v in view.items() if k not in ["pose", "success_by_noise"]},
                    "confirmation_success": None if c is None else c["mean_success"],
                    "confirmation_advantage": None if c is None else c["advantage"],
                    "confirmation_roles": ";".join(roles.get(candidate, [])),
                }
            )
    tasks = {}
    for task in sorted({r["task_id"] for r in state_rows}):
        r = [s for s in state_rows if s["task_id"] == task]
        tasks[task] = {
            "states": len(r),
            "canonical_success": float(np.mean([s["canonical_confirmation_success"] for s in r])),
            "primary_success": float(np.mean([s["primary_confirmation_success"] for s in r])),
            "states_with_exploratory_positive": sum(bool(s["confirmation_positive_candidates"]) for s in r),
            **paired_source_summary(r),
        }
    contrasts = {}
    for field in [
        "visibility_score",
        "visibility_entity_min",
        "visibility_entity_hmean",
        "visibility_agent",
        "azimuth_deg",
        "elevation_deg",
        "radius_scale",
    ]:
        differences = []
        aucs = []
        for views in discovery.values():
            pool = [v for c, v in views.items() if c != "canonical"]
            positive = [v for v in pool if v["mean_success"] >= 2 / 3 and v["advantage"] >= 1 / 3 - 1e-9]
            negative = [v for v in pool if v["mean_success"] <= 1 / 3]
            if positive and negative:
                differences.append(float(np.mean([v[field] for v in positive]) - np.mean([v[field] for v in negative])))
                aucs.append(
                    pairwise_auc(
                        [v[field] for v in positive + negative], [True] * len(positive) + [False] * len(negative)
                    )
                )
        contrasts[field] = {
            "eligible_states": len(differences),
            "state_equal_positive_minus_negative": None if not differences else float(np.mean(differences)),
            "within_state_auc": None if not aucs else float(np.mean(aucs)),
        }
    summary = {
        "schema": "dsol_failure_view_search_summary_v1",
        "status": "COMPLETE",
        "evidence_role": "posthoc_discovery_with_independent_noise_confirmation_not_new_source_test",
        "states": len(state_rows),
        "tasks": tasks,
        "state_rows": state_rows,
        "canonical_success": float(np.mean([s["canonical_confirmation_success"] for s in state_rows])),
        "frozen_top1_success": float(np.mean([s["primary_confirmation_success"] for s in state_rows])),
        "posthoc_oracle8_success": float(np.mean([s["posthoc_oracle8_success"] for s in state_rows])),
        "discovery_oracle97_success": float(np.mean([s["discovery_oracle97_success"] for s in state_rows])),
        "primary_state_outcomes": {
            "improved": sum(s["advantage"] > 0 for s in state_rows),
            "tied": sum(s["advantage"] == 0 for s in state_rows),
            "harmed": sum(s["advantage"] < 0 for s in state_rows),
        },
        "states_posthoc_oracle8_strictly_beats_canonical": sum(
            s["posthoc_oracle8_success"] > s["canonical_confirmation_success"] for s in state_rows
        ),
        "mean_primary_confirmation_rank": float(np.mean([s["primary_confirmation_rank"] for s in state_rows])),
        "states_with_exploratory_positive": sum(bool(s["confirmation_positive_candidates"]) for s in state_rows),
        "failure_screening_not_repeated_states": sum(not s["canonical_failure_persists"] for s in state_rows),
        "primary_paired_comparison": paired_source_summary(state_rows),
        "discovery_feature_contrasts": contrasts,
        "limitations": [
            "one training checkpoint",
            "failure-enriched previously inspected sources",
            "97-view bank boundary",
            "only frozen subset confirmed",
            "per-candidate positive labels are exploratory, not multiplicity-corrected tests",
            "feature contrasts are descriptive, not causal evidence",
            "no training or independent source generalization claim",
        ],
    }
    atomic_json(root / "analysis/final_analysis.json", summary)
    with (root / "analysis/candidate_matrix.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "freeze-confirmation", "analyze", "status"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.phase == "prepare":
        p = prepare(args.root)
        result = {
            k: p[k] for k in ["status", "selected_state_count", "source_episode_count", "task_count", "episode_count"]
        }
    elif args.phase == "freeze-confirmation":
        p = freeze_confirmation(args.root)
        result = {"status": p["status"], "frozen_candidates": p["episode_count"]}
    elif args.phase == "analyze":
        p = summarize(args.root)
        result = {
            k: p[k]
            for k in ["status", "states", "canonical_success", "frozen_top1_success", "primary_paired_comparison"]
        }
    else:
        result = {}
        for phase, name in [("discovery", "discovery-three-noise"), ("confirmation", "confirmation-five-noise")]:
            path = args.root / f"protocols/{name}.json"
            if path.exists():
                p = read(path)
                rows = load_run(p, args.root / phase, require_complete=False)
                result[phase] = {"completed": len(rows), "expected": p["episode_count"]}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
