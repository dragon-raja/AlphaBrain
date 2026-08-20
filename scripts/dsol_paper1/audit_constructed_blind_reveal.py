#!/usr/bin/env python3
"""Audit constructed Blind-Reveal states before they are admitted to M1."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from scripts.dsol_paper1.constructed_blind_reveal import (
        REQUIRED_ROLES,
        VISIBILITY_DEFINITION,
        build_snapshot_identity,
        check_record_visibility,
        is_sha256,
        role_index,
        sha256_file,
    )
except ModuleNotFoundError:
    from constructed_blind_reveal import (
        REQUIRED_ROLES,
        VISIBILITY_DEFINITION,
        build_snapshot_identity,
        check_record_visibility,
        is_sha256,
        role_index,
        sha256_file,
    )


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", **details}


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


def _verify_snapshot_identity(scan: Mapping[str, Any]) -> dict[str, Any]:
    identity = scan["snapshot"]
    components = {
        name: Path(record["path"])
        for name, record in identity.get("components", {}).items()
    }
    try:
        recomputed = build_snapshot_identity(
            task_id=str(scan["task_id"]), components=components
        )
    except (FileNotFoundError, ValueError) as error:
        return _check("snapshot_identity", False, error=str(error))

    component_matches = {
        name: (
            recomputed["components"][name]["sha256"] == record.get("sha256")
            and recomputed["components"][name]["size_bytes"]
            == record.get("size_bytes")
        )
        for name, record in identity["components"].items()
    }
    expected = identity.get("snapshot_sha256")
    passed = (
        is_sha256(expected)
        and recomputed["snapshot_sha256"] == expected
        and all(component_matches.values())
    )
    return _check(
        "snapshot_identity",
        passed,
        snapshot_sha256=expected,
        recomputed_snapshot_sha256=recomputed["snapshot_sha256"],
        component_matches=component_matches,
    )


def _geometry(record: Mapping[str, Any]) -> tuple[float, float]:
    displacement = record.get("camera_displacement_from_canonical", {})
    return (
        float(displacement.get("translation_m", 0.0)),
        float(displacement.get("rotation_geodesic_deg", 0.0)),
    )


def audit_snapshot(
    scan: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    checks = []
    checks.append(
        _check(
            "schema",
            scan.get("schema") == "dsol_constructed_blind_reveal_scan_v1",
            observed=scan.get("schema"),
        )
    )
    checks.append(_verify_snapshot_identity(scan))

    records = list(scan.get("records", ()))
    snapshot_sha = scan.get("snapshot", {}).get("snapshot_sha256")
    checks.append(
        _check(
            "same_snapshot_across_conditions",
            bool(records)
            and all(record.get("snapshot_sha256") == snapshot_sha for record in records),
            snapshot_sha256=snapshot_sha,
            condition_count=len(records),
        )
    )

    expected_entities = tuple(str(value) for value in scan.get("task_entities", ()))
    expected_cameras = tuple(str(value) for value in scan.get("camera_names", ()))
    contract_failures = []
    for record in records:
        visibility = record.get("visibility", {})
        if (
            visibility.get("definition") != VISIBILITY_DEFINITION
            or tuple(str(value) for value in visibility.get("entity_names", ()))
            != expected_entities
            or tuple(str(value) for value in visibility.get("camera_names", ()))
            != expected_cameras
        ):
            contract_failures.append(str(record.get("condition_id", "MISSING")))
    checks.append(
        _check(
            "visibility_contract_consistent",
            bool(expected_entities)
            and bool(expected_cameras)
            and scan.get("visibility_definition") == VISIBILITY_DEFINITION
            and not contract_failures,
            task_entities=list(expected_entities),
            camera_names=list(expected_cameras),
            failed_conditions=contract_failures,
        )
    )

    try:
        roles = role_index(records)
        role_error = None
    except (KeyError, ValueError) as error:
        roles = {}
        role_error = str(error)
    missing_roles = sorted(set(REQUIRED_ROLES).difference(roles))
    checks.append(
        _check(
            "required_condition_roles",
            role_error is None and not missing_roles,
            missing_roles=missing_roles,
            error=role_error,
        )
    )

    tolerance = float(config["visibility"]["numeric_tolerance"])
    visibility_audits = {}
    for record in records:
        condition_id = str(record.get("condition_id", "MISSING"))
        try:
            visibility_audits[condition_id] = check_record_visibility(
                record, tolerance=tolerance
            )
        except (KeyError, TypeError, ValueError) as error:
            visibility_audits[condition_id] = {
                "passed": False,
                "error": str(error),
            }
    checks.append(
        _check(
            "equal_weight_visibility_recomputed",
            bool(visibility_audits)
            and all(result["passed"] for result in visibility_audits.values()),
            definition=VISIBILITY_DEFINITION,
            condition_audits=visibility_audits,
        )
    )

    if missing_roles or role_error:
        return {
            "snapshot_group_id": scan.get("snapshot_group_id"),
            "task_id": scan.get("task_id"),
            "split": scan.get("split"),
            "snapshot_sha256": snapshot_sha,
            "status": "FAIL",
            "checks": checks,
            "selected_conditions": {},
        }

    canonical = roles["canonical"]
    canonical_score = float(canonical["visibility_score"])
    delta_errors = {
        str(record["condition_id"]): abs(
            float(record["delta_visibility"])
            - (float(record["visibility_score"]) - canonical_score)
        )
        for record in records
    }
    checks.append(
        _check(
            "delta_from_canonical",
            all(value <= tolerance for value in delta_errors.values()),
            absolute_errors=delta_errors,
        )
    )

    thresholds = config["thresholds"]
    strong = roles["strong_info"]
    control = roles["matched_control"]
    blind = roles["blind"]
    look_away = roles["look_away"]
    blackout = roles["all_camera_blackout"]
    strong_delta = float(strong["delta_visibility"])
    control_delta = float(control["delta_visibility"])
    blind_score = float(blind["visibility_score"])
    look_away_score = float(look_away["visibility_score"])
    blackout_score = float(blackout["visibility_score"])

    checks.extend(
        [
            _check(
                "strong_info_visibility_gain",
                strong_delta >= float(thresholds["strong_info_min_delta"])
                and bool(strong.get("operational", False)),
                observed_delta=strong_delta,
                minimum_delta=float(thresholds["strong_info_min_delta"]),
                operational=bool(strong.get("operational", False)),
            ),
            _check(
                "matched_control_low_information",
                abs(control_delta)
                <= float(thresholds["matched_control_max_abs_delta"]),
                observed_delta=control_delta,
                maximum_absolute_delta=float(
                    thresholds["matched_control_max_abs_delta"]
                ),
            ),
            _check(
                "blind_visibility",
                blind_score <= float(thresholds["blind_max_score"]),
                observed_score=blind_score,
                maximum_score=float(thresholds["blind_max_score"]),
            ),
            _check(
                "look_away_visibility",
                look_away_score <= float(thresholds["look_away_max_score"]),
                observed_score=look_away_score,
                maximum_score=float(thresholds["look_away_max_score"]),
            ),
            _check(
                "blackout_visibility",
                blackout_score <= float(thresholds["blackout_max_score"]),
                observed_score=blackout_score,
                maximum_score=float(thresholds["blackout_max_score"]),
            ),
            _check(
                "reveal_blind_separation",
                float(strong["visibility_score"]) - blind_score
                >= float(thresholds["reveal_blind_min_delta"]),
                observed_delta=float(strong["visibility_score"]) - blind_score,
                minimum_delta=float(thresholds["reveal_blind_min_delta"]),
            ),
        ]
    )

    strong_translation, strong_rotation = _geometry(strong)
    control_translation, control_rotation = _geometry(control)
    translation_difference = abs(strong_translation - control_translation)
    rotation_difference = abs(strong_rotation - control_rotation)
    checks.append(
        _check(
            "matched_control_pose_magnitude",
            translation_difference
            <= float(thresholds["matched_translation_tolerance_m"])
            and rotation_difference
            <= float(thresholds["matched_rotation_tolerance_deg"]),
            strong_info={
                "translation_m": strong_translation,
                "rotation_geodesic_deg": strong_rotation,
            },
            matched_control={
                "translation_m": control_translation,
                "rotation_geodesic_deg": control_rotation,
            },
            translation_difference_m=translation_difference,
            rotation_difference_deg=rotation_difference,
        )
    )

    eval_only_roles = set(config["safety"]["evaluation_only_roles"])
    safety_failures = []
    for record in records:
        role = str(record["condition_role"])
        must_be_eval_only = role in eval_only_roles or bool(record.get("is_extreme"))
        if must_be_eval_only and (
            not bool(record.get("evaluation_only"))
            or bool(record.get("training_eligible"))
        ):
            safety_failures.append(str(record["condition_id"]))
        if scan.get("split") != "train" and bool(record.get("training_eligible")):
            safety_failures.append(str(record["condition_id"]))
    checks.append(
        _check(
            "evaluation_only_safety",
            not safety_failures,
            failed_conditions=sorted(set(safety_failures)),
            evaluation_only_roles=sorted(eval_only_roles),
        )
    )

    manual = scan.get("manual_visual_audit", {"status": "PENDING"})
    manual_required = bool(config["manual_visual_audit"]["required"])
    montage_path = Path(str(manual.get("montage_path", "")))
    montage_exists = montage_path.is_file()
    observed_montage_sha256 = sha256_file(montage_path) if montage_exists else None
    manual_passed = not manual_required or (
        manual.get("status") == "PASS"
        and montage_exists
        and is_sha256(manual.get("montage_sha256"))
        and observed_montage_sha256 == manual.get("montage_sha256")
    )
    checks.append(
        _check(
            "manual_visual_audit",
            manual_passed,
            required=manual_required,
            observed_status=manual.get("status"),
            montage_path=str(montage_path) if manual.get("montage_path") else None,
            montage_exists=montage_exists,
            montage_sha256=manual.get("montage_sha256"),
            observed_montage_sha256=observed_montage_sha256,
        )
    )

    nonmanual_passed = all(
        check["status"] == "PASS"
        for check in checks
        if check["name"] != "manual_visual_audit"
    )
    if nonmanual_passed and manual_passed:
        status = "PASS"
    elif nonmanual_passed:
        status = "HOLD_MANUAL_AUDIT"
    else:
        status = "FAIL"
    return {
        "snapshot_group_id": scan["snapshot_group_id"],
        "task_id": scan["task_id"],
        "split": scan["split"],
        "scene_variant_id": scan["scene_variant_id"],
        "snapshot_sha256": snapshot_sha,
        "status": status,
        "checks": checks,
        "selected_conditions": {
            role: {
                "condition_id": record["condition_id"],
                "visibility_score": float(record["visibility_score"]),
                "delta_visibility": float(record["delta_visibility"]),
                "per_camera_scores": record["per_camera_scores"],
                "evaluation_only": bool(record.get("evaluation_only")),
                "training_eligible": bool(record.get("training_eligible")),
            }
            for role, record in roles.items()
        },
    }


def audit_collection(
    scans: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    gates = [audit_snapshot(scan, config) for scan in scans]
    group_ids = [str(gate.get("snapshot_group_id")) for gate in gates]
    duplicate_groups = sorted(
        group for group, count in Counter(group_ids).items() if count > 1
    )
    task_counts = Counter(str(gate.get("task_id")) for gate in gates)
    population = config["population_gate"]
    population_checks = [
        _check(
            "unique_snapshot_groups",
            not duplicate_groups,
            duplicate_snapshot_group_ids=duplicate_groups,
        ),
        _check(
            "minimum_snapshot_groups",
            len(gates) >= int(population["minimum_snapshot_groups"]),
            observed=len(gates),
            required=int(population["minimum_snapshot_groups"]),
        ),
        _check(
            "minimum_task_count",
            len(task_counts) >= int(population["minimum_task_count"]),
            observed=len(task_counts),
            required=int(population["minimum_task_count"]),
        ),
        _check(
            "minimum_states_per_task",
            bool(task_counts)
            and min(task_counts.values())
            >= int(population["minimum_states_per_task"]),
            observed=dict(sorted(task_counts.items())),
            required=int(population["minimum_states_per_task"]),
        ),
    ]
    snapshot_status_counts = Counter(gate["status"] for gate in gates)
    population_passed = all(check["status"] == "PASS" for check in population_checks)
    if population_passed and snapshot_status_counts.get("FAIL", 0) == 0:
        status = (
            "PASS"
            if snapshot_status_counts.get("HOLD_MANUAL_AUDIT", 0) == 0
            else "HOLD_MANUAL_AUDIT"
        )
    else:
        status = "FAIL"
    return {
        "schema": "dsol_constructed_blind_reveal_gate_v1",
        "status": status,
        "visibility_definition": VISIBILITY_DEFINITION,
        "snapshot_count": len(gates),
        "task_state_counts": dict(sorted(task_counts.items())),
        "snapshot_status_counts": dict(sorted(snapshot_status_counts.items())),
        "population_checks": population_checks,
        "snapshot_gates": gates,
        "m1_admission": status == "PASS",
    }


def _condition_rows(scans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for scan in scans:
        for record in scan["records"]:
            row = {
                "snapshot_group_id": scan["snapshot_group_id"],
                "snapshot_sha256": scan["snapshot"]["snapshot_sha256"],
                "task_id": scan["task_id"],
                "split": scan["split"],
                "scene_variant_id": scan["scene_variant_id"],
                "condition_id": record["condition_id"],
                "condition_role": record["condition_role"],
                "visibility_score": record["visibility_score"],
                "delta_visibility": record["delta_visibility"],
                "evaluation_only": record["evaluation_only"],
                "training_eligible": record["training_eligible"],
                "translation_m": record.get(
                    "camera_displacement_from_canonical", {}
                ).get("translation_m"),
                "rotation_geodesic_deg": record.get(
                    "camera_displacement_from_canonical", {}
                ).get("rotation_geodesic_deg"),
            }
            for camera, score in record["per_camera_scores"].items():
                row[f"camera::{camera}"] = score
            rows.append(row)
    return rows


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scans", type=Path, nargs="+")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema") != "dsol_constructed_blind_reveal_gate_config_v1":
        raise ValueError("unexpected gate config schema")
    scans = [json.loads(path.read_text(encoding="utf-8")) for path in args.scans]
    result = audit_collection(scans, config)
    result["config"] = {
        "path": str(args.config.resolve()),
        "sha256": sha256_file(args.config.resolve()),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "gate.json", result)
    with (output_dir / "snapshot_gates.jsonl").open("w", encoding="utf-8") as handle:
        for gate in result["snapshot_gates"]:
            handle.write(json.dumps(gate, sort_keys=True) + "\n")
    rows = _condition_rows(scans)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with (output_dir / "condition_records.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(
        json.dumps(
            {
                "status": result["status"],
                "snapshot_count": result["snapshot_count"],
                "m1_admission": result["m1_admission"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
