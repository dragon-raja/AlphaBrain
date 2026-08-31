#!/usr/bin/env python3
"""Audit frozen selector failures without changing labels or selecting new views."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-all8-v1")
METHOD = "visibility_gain_gated"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def visible_area(record, entity, camera=None):
    vis = record["visibility"]
    cameras = [camera] if camera else vis["camera_names"]
    return float(np.mean([vis["per_camera"][name]["entities"][entity]["visible_fraction"] for name in cameras]))


def entity_deltas(canonical, selected):
    return {
        entity: visible_area(selected, entity) - visible_area(canonical, entity)
        for entity in canonical["visibility"]["entity_names"]
    }


def evaluation_seed(row, path):
    if "evaluation_seed" in row:
        return int(row["evaluation_seed"])
    if path.parent.name.startswith("seed-"):
        return int(path.parent.name.removeprefix("seed-"))
    raise ValueError("missing evaluation-repeat identity")


def grouped_summary(rows):
    if not rows:
        return {"states": 0}
    sources = defaultdict(list)
    for row in rows:
        sources[row["source_group"]].append(row["difference_pp"])
    values = np.asarray([np.mean(v) for v in sources.values()])
    draws = np.random.default_rng(20260831).choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return {
        "states": len(rows),
        "sources": len(sources),
        "canonical_success": float(np.mean([r["canonical_success"] for r in rows])),
        "selected_success": float(np.mean([r["selected_success"] for r in rows])),
        "difference_pp": float(np.mean([r["difference_pp"] for r in rows])),
        "source_equal_difference_pp": float(values.mean()),
        "source_bootstrap_ci95_pp": np.quantile(draws, [0.025, 0.975]).tolist(),
        "stable_rescue": sum(r["stable_rescue"] for r in rows),
        "stable_harm": sum(r["stable_harm"] for r in rows),
        "nonpositive_target_gain": sum(r["target_delta_pp"] <= 0 for r in rows),
    }


def audit(root):
    combined_path = root / "independent-source-extension/combined-analysis/analysis.json"
    combined = read_json(combined_path)
    outcomes = {(r["pair_key"], r["selector_method"]): r for r in combined["state_method_rows"]}
    result_rows, examples, mappings, inputs = [], [], {}, [combined_path, Path(__file__)]
    for cohort, subroot in [("initial", root), ("extension", root / "independent-source-extension")]:
        protocol_path = subroot / "dense-test-selector-protocol.json"
        protocol = read_json(protocol_path)
        if protocol["selection_uses_test_policy_outcomes"]:
            raise ValueError("test outcomes were used to select views")
        inputs.append(protocol_path)
        specs = {(r["pair_key"], r["condition"].removeprefix("selector__")): r for r in protocol["specs"]}
        scans = {}
        for ledger in sorted((subroot / "test-feature-scan").glob("shard-*.jsonl")):
            for line in ledger.read_text().splitlines():
                entry = json.loads(line)
                scan_path = Path(entry["output_dir"]) / "scan.json"
                scan = read_json(scan_path)
                inputs.append(scan_path)
                scans[entry["scan_id"]] = (scan, scan_path)
        for state in protocol["selected_states"]:
            key = state["pair_key"]
            spec = specs[key, METHOD]
            scan, scan_path = scans[key]
            records = {r["pose_id"]: r for r in scan["records"]}
            canonical, selected = records["canonical"], records[spec["selected_candidate_id"]]
            deltas = entity_deltas(canonical, selected)
            sources = selected["visibility"]["entity_geom_sources"]
            fallback = {e: g for e, g in sources.items() if e != g}
            target = spec["scene_construction"]["target_entity"]
            target_delta = deltas.get(target)
            if target_delta is None:
                raise ValueError(f"construction target absent from visibility: {target}")
            positive = sum(max(0.0, v) for v in deltas.values())
            fallback_share = sum(max(0.0, deltas[e]) for e in fallback) / positive if positive else 0.0
            c, s = outcomes[key, "canonical"], outcomes[key, METHOD]
            moved = spec["selected_candidate_id"] != "canonical"
            row = {
                "pair_key": key,
                "source_group": c["source_group"],
                "task_id": spec["task_id"],
                "cohort": cohort,
                "stage_fraction": spec["stage_fraction"],
                "stage_bin": "early_le_0.35" if spec["stage_fraction"] <= 0.35 else "later_gt_0.35",
                "selected_candidate_id": spec["selected_candidate_id"],
                "moved": moved,
                "catalog_group": spec["selection_metadata"]["catalog_group"],
                "visibility_gain_pp": 100 * (selected["visibility"]["score"] - canonical["visibility"]["score"]),
                "target_entity": target,
                "target_delta_pp": 100 * target_delta,
                "fallback_entity_count": len(fallback),
                "fallback_positive_gain_share": fallback_share,
                "nontarget_positive_gain_share": sum(max(0.0, v) for e, v in deltas.items() if e != target) / positive
                if positive
                else 0.0,
                "dominant_gain_entity": max(deltas, key=deltas.get),
                "external_delta_pp": 100
                * (selected["per_camera_scores"]["agentview"] - canonical["per_camera_scores"]["agentview"]),
                "wrist_delta_pp": 100
                * (
                    selected["per_camera_scores"]["robot0_eye_in_hand"]
                    - canonical["per_camera_scores"]["robot0_eye_in_hand"]
                ),
                "translation_m": selected["camera_displacement_from_canonical"]["translation_m"],
                "rotation_deg": selected["camera_displacement_from_canonical"]["rotation_geodesic_deg"],
                "canonical_success": c["repeat_success_rate"],
                "selected_success": s["repeat_success_rate"],
                "difference_pp": 100 * (s["repeat_success_rate"] - c["repeat_success_rate"]),
                "stable_rescue": int(not c["stable_success"] and s["stable_success"]),
                "stable_harm": int(c["stable_success"] and not s["stable_success"]),
                "manual_audit_verified": spec["manual_audit_verified"],
            }
            result_rows.append(row)
            mappings[spec["task_id"]] = sources
            if moved and target_delta <= 0 and fallback_share > 0.5:
                examples.append(
                    {
                        **row,
                        "entity_delta_pp": {e: 100 * v for e, v in deltas.items()},
                        "scan_path": str(scan_path),
                        "montage_path": str(scan_path.parent / "visibility_extremes.png"),
                    }
                )
    if len(result_rows) != combined["states"] or len({r["pair_key"] for r in result_rows}) != len(result_rows):
        raise ValueError("missing or duplicate states")
    moved = [r for r in result_rows if r["moved"]]
    breakdown = {}
    for field in ["cohort", "task_id", "catalog_group", "stage_bin"]:
        groups = defaultdict(list)
        for row in result_rows:
            groups[row[field]].append(row)
        breakdown[field] = {key: grouped_summary(rows) for key, rows in sorted(groups.items())}
    raw_paths = sorted((root / "dense-test-selector-eval/runs").glob("seed-*/episodes-shard-*.jsonl"))
    raw_paths += sorted(
        (root / "independent-source-extension/dense-test-selector-eval/run-five-noise").glob("episodes-shard-*.jsonl")
    )
    paired = defaultdict(dict)
    for path in raw_paths:
        inputs.append(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") != "complete":
                raise ValueError("incomplete evaluation row")
            key = (row["pair_key"], evaluation_seed(row, path))
            method = row["condition"].removeprefix("selector__")
            if method in paired[key]:
                raise ValueError("duplicate method/noise record")
            paired[key][method] = row
    if len(paired) != len(result_rows) * 5 or any(len(v) != 6 for v in paired.values()):
        raise ValueError("unexpected paired state/noise/method counts")
    if any(len({r["policy_noise_seed"] for r in rows.values()}) != 1 for rows in paired.values()):
        raise ValueError("selectors did not share the same state-specific flow noise")
    physics_mismatch = sum(
        len({r["initial_metrics"]["physics_state_sha256"] for r in rows.values()}) != 1 for rows in paired.values()
    )
    same_view = [rows for rows in paired.values() if rows[METHOD]["selected_candidate_id"] == "canonical"]
    same_view_disagreements = sum(r["canonical"]["success"] != r[METHOD]["success"] for r in same_view)
    per_noise = defaultdict(list)
    for (_, noise), rows in paired.items():
        per_noise[noise].append(100 * (int(rows[METHOD]["success"]) - int(rows["canonical"]["success"])))
    if len(per_noise) != 5 or any(len(values) != len(result_rows) for values in per_noise.values()):
        raise ValueError("noise repeats do not cover the same state set")
    return {
        "schema": "dsol_frozen_selector_common_failure_audit_v1",
        "status": "PASS" if physics_mismatch == 0 and same_view_disagreements == 0 else "HOLD",
        "evidence_role": "posthoc_descriptive_audit_not_causal_or_new_selector",
        "method": METHOD,
        "states": len(result_rows),
        "sources": combined["source_groups"],
        "policy_checkpoint_count": 1,
        "training_seed_count": 1,
        "policy_noise_repeats": 5,
        "moved_states": len(moved),
        "canonical_states": len(result_rows) - len(moved),
        "selected_noncanonical_pose_ids": sorted({r["selected_candidate_id"] for r in moved}),
        "moved_target_nonpositive_gain_states": sum(r["target_delta_pp"] <= 0 for r in moved),
        "moved_fallback_majority_positive_gain_states": sum(r["fallback_positive_gain_share"] > 0.5 for r in moved),
        "moved_no_target_gain_but_fallback_majority_states": len(examples),
        "wrist_absolute_delta_pp_max": max(abs(r["wrist_delta_pp"]) for r in result_rows),
        "manual_audit_verified_states": sum(r["manual_audit_verified"] for r in result_rows),
        "physics_mismatch_state_noise_groups": physics_mismatch,
        "same_view_state_noise_comparisons": len(same_view),
        "same_view_success_disagreements": same_view_disagreements,
        "per_noise_difference_pp": {str(k): float(np.mean(v)) for k, v in sorted(per_noise.items())},
        "metric_entity_mappings": mappings,
        "breakdown": breakdown,
        "moved_summary": grouped_summary(moved),
        "examples": sorted(examples, key=lambda r: (r["difference_pp"], -r["visibility_gain_pp"]))[:8],
        "rows": result_rows,
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "limitations": [
            "fixed constructed task set",
            "one policy training seed",
            "no displacement-matched control in this selector test",
            "no unoccluded object-area denominator",
            "no new scoring rule fitted or tested",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "common_failure_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with (args.output_dir / "state_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["rows"][0]))
        writer.writeheader()
        writer.writerows(payload["rows"])
    print(
        json.dumps(
            {k: v for k, v in payload.items() if k not in ["rows", "examples", "input_sha256"]}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
