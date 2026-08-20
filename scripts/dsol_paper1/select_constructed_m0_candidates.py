#!/usr/bin/env python3
"""Freeze constructed-M0 selection rules on validation and apply once to test.

This selector deliberately separates automated candidate selection from manual
visual admission.  A PASS result means only that the frozen, episode-level
selection protocol has enough test coverage.  M1 remains on hold until the
selected montages are reviewed by a person.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any


SCHEMA = "dsol_constructed_m0_candidate_selection_v1"
OPERATIONAL_GROUPS = ("broad_heldout_32", "wide_extrapolation_24")
BLIND_GROUPS = ("diagnostic_extreme_orbit", "diagnostic_crossed_orbit")
LOOK_AWAY_GROUPS = ("diagnostic_look_away",)

DEFAULT_PROTOCOL: dict[str, Any] = {
    "strong_info": {
        "episode_quantile": 0.25,
        "minimum_delta": 0.005,
        "maximum_frozen_delta": 0.05,
    },
    "matched_control": {
        "episode_quantile": 0.75,
        "minimum_abs_delta": 0.001,
        "maximum_abs_delta": 0.01,
        "minimum_translation_tolerance_m": 0.02,
        "maximum_translation_tolerance_m": 0.15,
        "minimum_rotation_tolerance_deg": 3.0,
        "maximum_rotation_tolerance_deg": 30.0,
        "geometry_translation_scale_m": 0.05,
        "geometry_rotation_scale_deg": 10.0,
    },
    "population": {
        "minimum_task_count": 1,
        "minimum_val_episodes_per_task": 2,
        "minimum_test_episodes_per_task": 2,
        "minimum_selected_test_snapshots_per_episode": 1,
    },
    "manual_visual_audit_required": True,
}


def _deep_update(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("quantile values must be finite")
    location = probability * (len(ordered) - 1)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _record_is_valid(record: Mapping[str, Any]) -> bool:
    if record.get("status") == "INVALID":
        return False
    try:
        values = (
            float(record["visibility_score"]),
            float(record["delta_visibility"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isfinite(value) for value in values)


def _operational_records(snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        record
        for record in snapshot["records"]
        if _record_is_valid(record) and str(record.get("group")) in OPERATIONAL_GROUPS
    ]


def _records_in_groups(
    snapshot: Mapping[str, Any], groups: Sequence[str]
) -> list[Mapping[str, Any]]:
    allowed = set(groups)
    return [
        record
        for record in snapshot["records"]
        if _record_is_valid(record) and str(record.get("group")) in allowed
    ]


def _canonical(snapshot: Mapping[str, Any]) -> Mapping[str, Any] | None:
    records = [
        record
        for record in snapshot["records"]
        if _record_is_valid(record)
        and str(record.get("group")) == "canonical"
        and str(record.get("pose_id")) == "canonical"
    ]
    return records[0] if len(records) == 1 else None


def _geometry(record: Mapping[str, Any]) -> tuple[float, float] | None:
    displacement = record.get("camera_displacement_from_canonical", {})
    try:
        translation = float(displacement["translation_m"])
        rotation = float(displacement["rotation_geodesic_deg"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(translation) or not math.isfinite(rotation):
        return None
    return translation, rotation


def _raw_strong(snapshot: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates = _operational_records(snapshot)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda record: (float(record["delta_visibility"]), str(record["pose_id"])),
    )


def _raw_control_pair(
    snapshot: Mapping[str, Any],
    strong: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[Mapping[str, Any], float, float] | None:
    strong_geometry = _geometry(strong)
    if strong_geometry is None:
        return None
    translation_scale = float(
        protocol["matched_control"]["geometry_translation_scale_m"]
    )
    rotation_scale = float(
        protocol["matched_control"]["geometry_rotation_scale_deg"]
    )
    maximum_abs_delta = float(
        protocol["matched_control"]["maximum_abs_delta"]
    )
    candidates = []
    for record in _operational_records(snapshot):
        if str(record["pose_id"]) == str(strong["pose_id"]):
            continue
        absolute_delta = abs(float(record["delta_visibility"]))
        if absolute_delta > maximum_abs_delta:
            continue
        geometry = _geometry(record)
        if geometry is None:
            continue
        translation_gap = abs(geometry[0] - strong_geometry[0])
        rotation_gap = abs(geometry[1] - strong_geometry[1])
        score = (
            translation_gap / translation_scale
            + rotation_gap / rotation_scale
        )
        candidates.append(
            (
                score,
                absolute_delta,
                str(record["pose_id"]),
                record,
                translation_gap,
                rotation_gap,
            )
        )
    if not candidates:
        return None
    _, _, _, record, translation_gap, rotation_gap = min(candidates)
    return record, translation_gap, rotation_gap


def _episode_values(
    rows: Sequence[tuple[str, float]],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for episode_id, value in rows:
        grouped[str(episode_id)].append(float(value))
    return {
        episode_id: float(median(values))
        for episode_id, values in sorted(grouped.items())
    }


def freeze_validation_rules(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    task_ids: Sequence[str],
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze per-task thresholds using validation episodes only."""

    resolved = _deep_update(DEFAULT_PROTOCOL, protocol or {})
    validation = [snapshot for snapshot in snapshots if snapshot["split"] == "val"]
    task_rules = {}
    for task_id in sorted(set(str(value) for value in task_ids)):
        task_snapshots = [
            snapshot for snapshot in validation if snapshot["task_id"] == task_id
        ]
        strong_rows: list[tuple[str, float]] = []
        control_delta_rows: list[tuple[str, float]] = []
        translation_rows: list[tuple[str, float]] = []
        rotation_rows: list[tuple[str, float]] = []
        missing_raw_strong = []
        missing_raw_control = []
        for snapshot in task_snapshots:
            strong = _raw_strong(snapshot)
            if strong is None:
                missing_raw_strong.append(snapshot["scan_id"])
                continue
            episode_id = str(snapshot["episode_id"])
            strong_rows.append((episode_id, float(strong["delta_visibility"])))
            control = _raw_control_pair(snapshot, strong, resolved)
            if control is None:
                missing_raw_control.append(snapshot["scan_id"])
                continue
            control_record, translation_gap, rotation_gap = control
            control_delta_rows.append(
                (episode_id, abs(float(control_record["delta_visibility"])))
            )
            translation_rows.append((episode_id, translation_gap))
            rotation_rows.append((episode_id, rotation_gap))

        strong_by_episode = _episode_values(strong_rows)
        control_delta_by_episode = _episode_values(control_delta_rows)
        translation_by_episode = _episode_values(translation_rows)
        rotation_by_episode = _episode_values(rotation_rows)
        reasons = []
        clamp_events = []
        minimum_episodes = int(
            resolved["population"]["minimum_val_episodes_per_task"]
        )
        if len(strong_by_episode) < minimum_episodes:
            reasons.append("insufficient_validation_episodes_with_operational_views")
        if len(control_delta_by_episode) < minimum_episodes:
            reasons.append("insufficient_validation_episodes_with_control_candidates")

        strong_threshold = None
        control_abs_delta = None
        translation_tolerance = None
        rotation_tolerance = None
        if strong_by_episode:
            raw = _quantile(
                list(strong_by_episode.values()),
                float(resolved["strong_info"]["episode_quantile"]),
            )
            strong_threshold = max(
                float(resolved["strong_info"]["minimum_delta"]), raw
            )
            if strong_threshold > float(
                resolved["strong_info"]["maximum_frozen_delta"]
            ):
                clamp_events.append("strong_threshold_clamped_to_protocol_cap")
                strong_threshold = float(
                    resolved["strong_info"]["maximum_frozen_delta"]
                )
        if control_delta_by_episode:
            raw = _quantile(
                list(control_delta_by_episode.values()),
                float(resolved["matched_control"]["episode_quantile"]),
            )
            control_abs_delta = max(
                float(resolved["matched_control"]["minimum_abs_delta"]), raw
            )
            if control_abs_delta > float(
                resolved["matched_control"]["maximum_abs_delta"]
            ):
                clamp_events.append(
                    "matched_control_delta_clamped_to_protocol_cap"
                )
                control_abs_delta = float(
                    resolved["matched_control"]["maximum_abs_delta"]
                )
        if translation_by_episode:
            translation_tolerance = max(
                float(
                    resolved["matched_control"][
                        "minimum_translation_tolerance_m"
                    ]
                ),
                _quantile(
                    list(translation_by_episode.values()),
                    float(resolved["matched_control"]["episode_quantile"]),
                ),
            )
            if translation_tolerance > float(
                resolved["matched_control"]["maximum_translation_tolerance_m"]
            ):
                clamp_events.append(
                    "matched_translation_tolerance_clamped_to_protocol_cap"
                )
                translation_tolerance = float(
                    resolved["matched_control"]["maximum_translation_tolerance_m"]
                )
        if rotation_by_episode:
            rotation_tolerance = max(
                float(
                    resolved["matched_control"]["minimum_rotation_tolerance_deg"]
                ),
                _quantile(
                    list(rotation_by_episode.values()),
                    float(resolved["matched_control"]["episode_quantile"]),
                ),
            )
            if rotation_tolerance > float(
                resolved["matched_control"]["maximum_rotation_tolerance_deg"]
            ):
                clamp_events.append(
                    "matched_rotation_tolerance_clamped_to_protocol_cap"
                )
                rotation_tolerance = float(
                    resolved["matched_control"]["maximum_rotation_tolerance_deg"]
                )

        task_rules[task_id] = {
            "status": "PASS" if not reasons else "HOLD",
            "insufficient_reasons": sorted(set(reasons)),
            "protocol_clamp_events": sorted(set(clamp_events)),
            "strong_info_min_delta": strong_threshold,
            "matched_control_max_abs_delta": control_abs_delta,
            "matched_translation_tolerance_m": translation_tolerance,
            "matched_rotation_tolerance_deg": rotation_tolerance,
            "validation_episode_count": len(
                {str(snapshot["episode_id"]) for snapshot in task_snapshots}
            ),
            "validation_snapshot_count": len(task_snapshots),
            "episode_aggregates": {
                "best_operational_delta": strong_by_episode,
                "raw_control_abs_delta": control_delta_by_episode,
                "raw_control_translation_gap_m": translation_by_episode,
                "raw_control_rotation_gap_deg": rotation_by_episode,
            },
            "missing_raw_strong_scan_ids": sorted(missing_raw_strong),
            "missing_raw_control_scan_ids": sorted(missing_raw_control),
        }

    return {
        "schema": "dsol_constructed_m0_frozen_rules_v1",
        "source_split": "val",
        "test_values_used_for_freeze": False,
        "statistical_unit": "source_episode",
        "within_episode_summary": "median_over_snapshot_groups",
        "operational_strong_info_groups": list(OPERATIONAL_GROUPS),
        "extreme_and_look_away_forbidden_as_strong_info": True,
        "validation_scan_ids": sorted(snapshot["scan_id"] for snapshot in validation),
        "validation_episode_ids": sorted(
            {str(snapshot["episode_id"]) for snapshot in validation}
        ),
        "task_rules": task_rules,
    }


def _select_matched_control(
    snapshot: Mapping[str, Any],
    strong: Mapping[str, Any],
    rule: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    strong_geometry = _geometry(strong)
    thresholds = (
        rule.get("matched_control_max_abs_delta"),
        rule.get("matched_translation_tolerance_m"),
        rule.get("matched_rotation_tolerance_deg"),
    )
    if strong_geometry is None or any(value is None for value in thresholds):
        return None
    candidates = []
    for record in _operational_records(snapshot):
        if str(record["pose_id"]) == str(strong["pose_id"]):
            continue
        geometry = _geometry(record)
        if geometry is None:
            continue
        translation_gap = abs(geometry[0] - strong_geometry[0])
        rotation_gap = abs(geometry[1] - strong_geometry[1])
        if (
            abs(float(record["delta_visibility"])) <= float(thresholds[0])
            and translation_gap <= float(thresholds[1])
            and rotation_gap <= float(thresholds[2])
        ):
            normalized_gap = (
                translation_gap / max(float(thresholds[1]), 1e-12)
                + rotation_gap / max(float(thresholds[2]), 1e-12)
            )
            candidates.append(
                (
                    normalized_gap,
                    abs(float(record["delta_visibility"])),
                    str(record["pose_id"]),
                    record,
                    translation_gap,
                    rotation_gap,
                )
            )
    if not candidates:
        return None
    chosen = min(candidates)
    return {
        **dict(chosen[3]),
        "matched_translation_gap_m": chosen[4],
        "matched_rotation_gap_deg": chosen[5],
    }


def _condition(
    record: Mapping[str, Any],
    *,
    role: str,
    operational: bool,
    evaluation_only: bool,
) -> dict[str, Any]:
    return {
        "condition_role": role,
        "source_pose_id": str(record["pose_id"]),
        "source_group": str(record["group"]),
        "visibility_score": float(record["visibility_score"]),
        "delta_visibility": float(record["delta_visibility"]),
        "camera_displacement_from_canonical": record.get(
            "camera_displacement_from_canonical"
        ),
        "operational": operational,
        "evaluation_only": evaluation_only,
        "training_eligible": False,
        "synthetic": False,
    }


def select_snapshot(
    snapshot: Mapping[str, Any], rule: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons = []
    canonical = _canonical(snapshot)
    if canonical is None:
        reasons.append("missing_unique_canonical")
    strong = _raw_strong(snapshot)
    if strong is None:
        reasons.append("no_operational_strong_info_candidate")
    elif rule.get("strong_info_min_delta") is None or float(
        strong["delta_visibility"]
    ) < float(rule["strong_info_min_delta"]):
        reasons.append("strong_info_below_frozen_validation_threshold")
    control = (
        _select_matched_control(snapshot, strong, rule) if strong is not None else None
    )
    if control is None:
        reasons.append("no_matched_control_under_frozen_validation_rule")
    blind_candidates = _records_in_groups(snapshot, BLIND_GROUPS)
    blind = (
        min(
            blind_candidates,
            key=lambda record: (
                float(record["visibility_score"]),
                str(record["pose_id"]),
            ),
        )
        if blind_candidates
        else None
    )
    if blind is None:
        reasons.append("no_blind_candidate")
    look_away_candidates = _records_in_groups(snapshot, LOOK_AWAY_GROUPS)
    look_away = (
        min(
            look_away_candidates,
            key=lambda record: (
                float(record["visibility_score"]),
                str(record["pose_id"]),
            ),
        )
        if look_away_candidates
        else None
    )
    if look_away is None:
        reasons.append("no_look_away_candidate")
    if reasons:
        return None, sorted(set(reasons))

    assert canonical is not None and strong is not None
    assert control is not None and blind is not None and look_away is not None
    conditions = {
        "canonical": _condition(
            canonical, role="canonical", operational=True, evaluation_only=False
        ),
        "strong_info": _condition(
            strong, role="strong_info", operational=True, evaluation_only=False
        ),
        "matched_control": _condition(
            control,
            role="matched_control",
            operational=True,
            evaluation_only=False,
        ),
        "blind": _condition(
            blind, role="blind", operational=False, evaluation_only=True
        ),
        "look_away": _condition(
            look_away,
            role="look_away",
            operational=False,
            evaluation_only=True,
        ),
        "all_camera_blackout": {
            "condition_role": "all_camera_blackout",
            "source_pose_id": "all_camera_blackout",
            "source_group": "sensor_controls",
            "visibility_score": 0.0,
            "delta_visibility": -float(canonical["visibility_score"]),
            "operational": False,
            "evaluation_only": True,
            "training_eligible": False,
            "synthetic": True,
            "construction": "zero_all_camera_pixels_from_same_snapshot",
        },
    }
    conditions["matched_control"]["matched_translation_gap_m"] = control[
        "matched_translation_gap_m"
    ]
    conditions["matched_control"]["matched_rotation_gap_deg"] = control[
        "matched_rotation_gap_deg"
    ]
    return {
        "snapshot_group_id": str(snapshot["scan_id"]),
        "scan_id": str(snapshot["scan_id"]),
        "task_id": str(snapshot["task_id"]),
        "split": str(snapshot["split"]),
        "source_episode_id": str(snapshot["episode_id"]),
        "source_frame": int(snapshot["frame"]),
        "source_scan_path": str(snapshot.get("scan_path", "")),
        "montage_path": str(snapshot.get("montage_path", "")),
        "conditions": conditions,
        "manual_visual_audit": {
            "required": True,
            "status": "PENDING",
            "automatically_promoted": False,
        },
    }, []


def _episode_coverage(
    snapshots: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    planned: dict[tuple[str, str], set[str]] = defaultdict(set)
    selected_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    for snapshot in snapshots:
        planned[(str(snapshot["task_id"]), str(snapshot["episode_id"]))].add(
            str(snapshot["scan_id"])
        )
    for snapshot in selected:
        selected_ids[
            (str(snapshot["task_id"]), str(snapshot["source_episode_id"]))
        ].add(str(snapshot["scan_id"]))
    rows = []
    for (task_id, episode_id), scan_ids in sorted(planned.items()):
        selected_scans = selected_ids.get((task_id, episode_id), set())
        rows.append(
            {
                "task_id": task_id,
                "source_episode_id": episode_id,
                "snapshot_group_count": len(scan_ids),
                "selected_snapshot_group_count": len(selected_scans),
                "selected_fraction": len(selected_scans) / len(scan_ids),
            }
        )
    return {
        "statistical_unit": "source_episode",
        "episode_count": len(rows),
        "snapshot_groups_are_not_independent_samples": True,
        "episodes": rows,
    }


def build_selection(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    task_ids: Sequence[str] | None = None,
    protocol: Mapping[str, Any] | None = None,
    input_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _deep_update(DEFAULT_PROTOCOL, protocol or {})
    tasks = sorted(
        set(str(value) for value in (task_ids or [row["task_id"] for row in snapshots]))
    )
    frozen = freeze_validation_rules(
        snapshots, task_ids=tasks, protocol=resolved
    )
    test_snapshots = [snapshot for snapshot in snapshots if snapshot["split"] == "test"]
    selected = []
    insufficient = []
    for snapshot in sorted(test_snapshots, key=lambda row: str(row["scan_id"])):
        rule = frozen["task_rules"].get(str(snapshot["task_id"]), {})
        if rule.get("status") != "PASS":
            insufficient.append(
                {
                    "scan_id": snapshot["scan_id"],
                    "task_id": snapshot["task_id"],
                    "source_episode_id": snapshot["episode_id"],
                    "reasons": ["task_validation_rule_not_frozen"],
                }
            )
            continue
        result, reasons = select_snapshot(snapshot, rule)
        if result is None:
            insufficient.append(
                {
                    "scan_id": snapshot["scan_id"],
                    "task_id": snapshot["task_id"],
                    "source_episode_id": snapshot["episode_id"],
                    "reasons": reasons,
                }
            )
        else:
            selected.append(result)

    validation_snapshots = [
        snapshot for snapshot in snapshots if snapshot["split"] == "val"
    ]
    val_coverage = _episode_coverage(validation_snapshots, [])
    test_coverage = _episode_coverage(test_snapshots, selected)
    minimum_val = int(resolved["population"]["minimum_val_episodes_per_task"])
    minimum_test = int(resolved["population"]["minimum_test_episodes_per_task"])
    minimum_selected = int(
        resolved["population"]["minimum_selected_test_snapshots_per_episode"]
    )
    task_coverage = {}
    population_reasons = []
    for task_id in tasks:
        val_episodes = {
            str(snapshot["episode_id"])
            for snapshot in validation_snapshots
            if snapshot["task_id"] == task_id
        }
        test_rows = [
            row for row in test_coverage["episodes"] if row["task_id"] == task_id
        ]
        test_episode_count = len(test_rows)
        selected_episode_count = sum(
            int(row["selected_snapshot_group_count"] >= minimum_selected)
            for row in test_rows
        )
        reasons = []
        if len(val_episodes) < minimum_val:
            reasons.append("insufficient_validation_episode_count")
        if test_episode_count < minimum_test:
            reasons.append("insufficient_test_episode_count")
        if selected_episode_count < minimum_test:
            reasons.append("insufficient_test_episodes_with_complete_candidates")
        rule = frozen["task_rules"].get(task_id)
        if not rule or rule["status"] != "PASS":
            reasons.append("validation_rule_hold")
        task_coverage[task_id] = {
            "status": "PASS" if not reasons else "HOLD",
            "validation_episode_count": len(val_episodes),
            "test_episode_count": test_episode_count,
            "test_episodes_with_complete_candidates": selected_episode_count,
            "selected_test_snapshot_group_count": sum(
                int(row["selected_snapshot_group_count"]) for row in test_rows
            ),
            "insufficient_reasons": sorted(set(reasons)),
        }
        population_reasons.extend(f"{task_id}:{reason}" for reason in reasons)

    if len(tasks) < int(resolved["population"]["minimum_task_count"]):
        population_reasons.append("insufficient_task_count")
    audit = dict(input_audit or {"status": "PASS", "issues": []})
    if audit.get("status") != "PASS":
        population_reasons.append("input_audit_hold")
    split_leakage = sorted(
        set(snapshot["episode_id"] for snapshot in validation_snapshots).intersection(
            snapshot["episode_id"] for snapshot in test_snapshots
        )
    )
    if split_leakage:
        population_reasons.append("source_episode_crosses_val_test")
    status = "PASS" if not population_reasons else "HOLD"
    return {
        "schema": SCHEMA,
        "status": status,
        "automated_selection_status": status,
        "selection_scope": "automated_candidate_selection_only",
        "m1_admission": False,
        "m1_admission_status": (
            "HOLD_MANUAL_AUDIT" if status == "PASS" else "HOLD_CANDIDATE_SELECTION"
        ),
        "manual_visual_audit": {
            "required": bool(resolved["manual_visual_audit_required"]),
            "status": "PENDING",
            "automatically_promoted": False,
            "note": "PASS is never synthesized by this selector.",
        },
        "protocol": resolved,
        "input_audit": audit,
        "frozen_rules": frozen,
        "test_application": {
            "application_count": 1,
            "threshold_retuning_on_test": False,
            "test_scan_ids": sorted(snapshot["scan_id"] for snapshot in test_snapshots),
        },
        "task_coverage": task_coverage,
        "validation_episode_coverage": val_coverage,
        "test_episode_coverage": test_coverage,
        "selected_snapshot_group_count": len(selected),
        "selected_snapshot_groups": selected,
        "insufficient_snapshot_group_count": len(insufficient),
        "insufficient_snapshot_groups": insufficient,
        "population_insufficient_reasons": sorted(set(population_reasons)),
        "val_test_episode_leakage": split_leakage,
        "statistical_unit": "source_episode",
        "frames_are_not_independent_samples": True,
    }


def load_scan_inputs(
    scan_plan_path: Path, ledger_paths: Sequence[Path]
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    plan = json.loads(scan_plan_path.read_text(encoding="utf-8"))
    plan_records = {}
    issues = []
    for row in plan.get("records", []):
        scan_id = str(row["scan_id"])
        if scan_id in plan_records:
            issues.append(f"duplicate_plan_scan_id:{scan_id}")
        plan_records[scan_id] = row
    task_ids = sorted({str(row["task_id"]) for row in plan_records.values()})

    ledger_by_id = {}
    for ledger_path in ledger_paths:
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            scan_id = str(row["scan_id"])
            ledger_by_id[scan_id] = row

    snapshots = []
    for scan_id, plan_row in sorted(plan_records.items()):
        ledger = ledger_by_id.get(scan_id)
        if ledger is None:
            issues.append(f"missing_ledger_record:{scan_id}")
            continue
        if ledger.get("status") != "PASS":
            issues.append(f"non_pass_ledger_record:{scan_id}:{ledger.get('status')}")
            continue
        for key in ("split", "task_id", "episode_id", "frame"):
            if ledger.get(key) != plan_row.get(key):
                issues.append(f"plan_ledger_mismatch:{scan_id}:{key}")
        scan_path = Path(ledger["output_dir"]) / "scan.json"
        if not scan_path.is_file():
            issues.append(f"missing_scan_json:{scan_id}")
            continue
        scan = json.loads(scan_path.read_text(encoding="utf-8"))
        if scan.get("schema") != "dsol_libero_hdf5_view_scan_v1":
            issues.append(f"unexpected_scan_schema:{scan_id}")
            continue
        if scan.get("status") != "PASS" or int(scan.get("invalid_records", 0)):
            issues.append(f"invalid_scan_result:{scan_id}")
            continue
        if (
            str(scan.get("hdf5")) != str(plan_row.get("hdf5"))
            or str(scan.get("demo")) != str(plan_row.get("demo_name"))
            or int(scan.get("frame", -1)) != int(plan_row.get("frame", -2))
        ):
            issues.append(f"plan_scan_identity_mismatch:{scan_id}")
            continue
        snapshots.append(
            {
                "scan_id": scan_id,
                "split": str(plan_row["split"]),
                "task_id": str(plan_row["task_id"]),
                "episode_id": str(plan_row["episode_id"]),
                "frame": int(plan_row["frame"]),
                "stage_fraction": float(plan_row["stage_fraction"]),
                "scan_path": str(scan_path.resolve()),
                "montage_path": str(
                    scan_path.with_name("visibility_extremes.png").resolve()
                ),
                "records": scan["records"],
            }
        )
    extra_ledgers = sorted(set(ledger_by_id).difference(plan_records))
    issues.extend(f"ledger_scan_not_in_plan:{scan_id}" for scan_id in extra_ledgers)
    audit = {
        "status": "PASS" if not issues else "HOLD",
        "issues": sorted(issues),
        "planned_scan_count": len(plan_records),
        "loaded_pass_scan_count": len(snapshots),
        "ledger_record_count": len(ledger_by_id),
        "scan_plan": str(scan_plan_path.resolve()),
        "ledgers": [str(path.resolve()) for path in ledger_paths],
    }
    return snapshots, task_ids, audit


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select strict constructed-M0 candidates after scan completion."
    )
    parser.add_argument("--scan-plan", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol-json",
        type=Path,
        help="Optional selector-only overrides; test measurements must not appear here.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    protocol = (
        json.loads(args.protocol_json.read_text(encoding="utf-8"))
        if args.protocol_json
        else None
    )
    snapshots, task_ids, audit = load_scan_inputs(args.scan_plan, args.ledger)
    result = build_selection(
        snapshots,
        task_ids=task_ids,
        protocol=protocol,
        input_audit=audit,
    )
    _atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "m1_admission_status": result["m1_admission_status"],
                "selected_snapshot_group_count": result[
                    "selected_snapshot_group_count"
                ],
                "insufficient_snapshot_group_count": result[
                    "insufficient_snapshot_group_count"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
