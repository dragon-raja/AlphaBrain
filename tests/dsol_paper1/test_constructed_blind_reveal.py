from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dsol_paper1.audit_constructed_blind_reveal import (
    audit_collection,
    audit_snapshot,
    main as audit_main,
)
from scripts.dsol_paper1.constructed_blind_reveal import (
    VISIBILITY_DEFINITION,
    build_snapshot_identity,
    masked_visibility,
    recompute_equal_weight_visibility,
    sha256_file,
)
from scripts.dsol_paper1.package_constructed_blind_reveal import package


CAMERAS = ("agentview", "robot0_eye_in_hand")
ENTITIES = ("target", "container")


def _visibility(agent: float, wrist: float) -> dict:
    def camera_record(fraction: float) -> dict:
        return {
            "score": fraction,
            "entities": {
                entity: {
                    "visible_pixels": int(round(fraction * 10_000)),
                    "visible_fraction": fraction,
                    "geom_source": entity,
                    "touches_border": False,
                }
                for entity in ENTITIES
            },
        }

    return {
        "definition": VISIBILITY_DEFINITION,
        "height": 100,
        "width": 100,
        "entity_names": list(ENTITIES),
        "entity_geom_sources": {entity: entity for entity in ENTITIES},
        "camera_names": list(CAMERAS),
        "score": (agent + wrist) / 2.0,
        "per_camera": {
            "agentview": camera_record(agent),
            "robot0_eye_in_hand": camera_record(wrist),
        },
    }


def _record(
    *, role: str, visibility: dict, snapshot_sha: str, translation: float = 0.0
) -> dict:
    score = recompute_equal_weight_visibility(visibility)["score"]
    canonical_score = 0.015
    evaluation_only = role in {"blind", "look_away", "all_camera_blackout"}
    return {
        "condition_id": role,
        "condition_role": role,
        "source_pose_id": role,
        "snapshot_sha256": snapshot_sha,
        "evaluation_only": evaluation_only,
        "training_eligible": role == "strong_info",
        "operational": role in {"canonical", "strong_info", "matched_control"},
        "is_extreme": role in {"blind", "look_away", "all_camera_blackout"},
        "visibility_score": score,
        "delta_visibility": score - canonical_score,
        "per_camera_scores": {
            camera: data["score"] for camera, data in visibility["per_camera"].items()
        },
        "visibility": visibility,
        "camera_displacement_from_canonical": {
            "translation_m": translation,
            "rotation_geodesic_deg": 30.0 if translation else 0.0,
        },
    }


def _config(*, manual_required: bool = True) -> dict:
    return {
        "schema": "dsol_constructed_blind_reveal_gate_config_v1",
        "visibility": {"numeric_tolerance": 1e-12},
        "thresholds": {
            "strong_info_min_delta": 0.02,
            "matched_control_max_abs_delta": 0.001,
            "blind_max_score": 0.002,
            "look_away_max_score": 0.0,
            "blackout_max_score": 0.0,
            "reveal_blind_min_delta": 0.04,
            "matched_translation_tolerance_m": 0.01,
            "matched_rotation_tolerance_deg": 1.0,
        },
        "safety": {
            "evaluation_only_roles": [
                "blind",
                "look_away",
                "all_camera_blackout",
            ]
        },
        "manual_visual_audit": {"required": manual_required},
        "population_gate": {
            "minimum_snapshot_groups": 1,
            "minimum_task_count": 1,
            "minimum_states_per_task": 1,
        },
    }


def _scan(tmp_path: Path, *, manual_status: str = "PASS") -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    components = {}
    for name, data in {
        "physics_state": b"physics",
        "model_xml": b"<mujoco/>",
        "task_definition": b"task",
    }.items():
        path = tmp_path / name
        path.write_bytes(data)
        components[name] = path
    identity = build_snapshot_identity(task_id="task_a", components=components)
    montage = tmp_path / "montage.png"
    montage.write_bytes(b"audited montage")
    snapshot_sha = identity["snapshot_sha256"]
    records = [
        _record(role="canonical", visibility=_visibility(0.02, 0.01), snapshot_sha=snapshot_sha),
        _record(
            role="strong_info",
            visibility=_visibility(0.08, 0.01),
            snapshot_sha=snapshot_sha,
            translation=0.20,
        ),
        _record(
            role="matched_control",
            visibility=_visibility(0.02, 0.01),
            snapshot_sha=snapshot_sha,
            translation=0.20,
        ),
        _record(role="blind", visibility=_visibility(0.001, 0.001), snapshot_sha=snapshot_sha),
        _record(role="look_away", visibility=_visibility(0.0, 0.0), snapshot_sha=snapshot_sha),
        _record(
            role="all_camera_blackout",
            visibility=_visibility(0.0, 0.0),
            snapshot_sha=snapshot_sha,
        ),
    ]
    for record in records:
        record["training_eligible"] = False
    return {
        "schema": "dsol_constructed_blind_reveal_scan_v1",
        "status": "PACKAGED_UNAUDITED",
        "snapshot_group_id": "task_a::scene_0::state_0",
        "task_id": "task_a",
        "split": "val",
        "scene_variant_id": "occlusion_v1",
        "snapshot": identity,
        "task_entities": list(ENTITIES),
        "camera_names": list(CAMERAS),
        "visibility_definition": VISIBILITY_DEFINITION,
        "manual_visual_audit": {
            "status": manual_status,
            "montage_path": str(montage) if manual_status == "PASS" else None,
            "montage_sha256": sha256_file(montage) if manual_status == "PASS" else None,
        },
        "records": records,
    }


def test_equal_weight_recomputation_does_not_use_camera_max() -> None:
    visibility = _visibility(0.08, 0.01)
    result = recompute_equal_weight_visibility(visibility)
    assert result["score"] == pytest.approx(0.045)
    assert result["score"] != max(result["per_camera_scores"].values())


def test_masked_visibility_zeroes_only_requested_camera() -> None:
    visibility = masked_visibility(
        _visibility(0.08, 0.01), masked_cameras=("agentview",)
    )
    result = recompute_equal_weight_visibility(visibility)
    assert result["per_camera_scores"] == {
        "agentview": 0.0,
        "robot0_eye_in_hand": pytest.approx(0.01),
    }
    assert result["score"] == pytest.approx(0.005)


def test_strict_snapshot_gate_passes_complete_constructed_state(tmp_path: Path) -> None:
    result = audit_snapshot(_scan(tmp_path), _config())
    assert result["status"] == "PASS"
    assert result["selected_conditions"]["strong_info"]["delta_visibility"] == pytest.approx(
        0.03
    )
    assert result["selected_conditions"]["strong_info"]["per_camera_scores"] == {
        "agentview": pytest.approx(0.08),
        "robot0_eye_in_hand": pytest.approx(0.01),
    }


def test_gate_holds_when_manual_visual_audit_is_pending(tmp_path: Path) -> None:
    scan = _scan(tmp_path, manual_status="PENDING")
    result = audit_collection([scan], _config())
    assert result["status"] == "HOLD_MANUAL_AUDIT"
    assert result["m1_admission"] is False


def test_gate_rejects_forged_manual_montage_hash(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    scan["manual_visual_audit"]["montage_sha256"] = "a" * 64
    result = audit_snapshot(scan, _config())
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["manual_visual_audit"]["status"] == "FAIL"


def test_gate_fails_when_one_view_uses_a_different_snapshot(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    scan["records"][1]["snapshot_sha256"] = "b" * 64
    result = audit_snapshot(scan, _config())
    assert result["status"] == "FAIL"
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["same_snapshot_across_conditions"]["status"] == "FAIL"


def test_gate_rejects_cached_max_camera_aggregation(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    strong = scan["records"][1]
    strong["visibility_score"] = 0.08
    strong["visibility"]["score"] = 0.08
    result = audit_snapshot(scan, _config())
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["equal_weight_visibility_recomputed"]["status"] == "FAIL"


def test_gate_rejects_condition_that_drops_a_task_entity(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    strong = scan["records"][1]
    strong["visibility"]["entity_names"] = ["target"]
    result = audit_snapshot(scan, _config())
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["visibility_contract_consistent"]["status"] == "FAIL"


def test_packager_binds_components_and_synthesizes_blackout(tmp_path: Path) -> None:
    source_records = []
    role_visibility = {
        "canonical": _visibility(0.02, 0.01),
        "strong": _visibility(0.08, 0.01),
        "control": _visibility(0.02, 0.01),
        "blind": _visibility(0.001, 0.001),
        "look": _visibility(0.0, 0.0),
    }
    for pose_id, visibility in role_visibility.items():
        source_records.append(
            {
                "pose_id": pose_id,
                "group": "synthetic",
                "visibility": visibility,
                "camera_displacement_from_canonical": {
                    "translation_m": 0.2 if pose_id in {"strong", "control"} else 0.0,
                    "rotation_geodesic_deg": 30.0 if pose_id in {"strong", "control"} else 0.0,
                },
            }
        )
    source_records.append(
        {"pose_id": "all_camera_blackout", "group": "sensor_controls"}
    )
    source_scan = tmp_path / "source_scan.json"
    source_scan.write_text(
        json.dumps({"status": "PASS", "records": source_records}),
        encoding="utf-8",
    )
    components = {}
    for name in ("physics", "xml", "task"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        components[name] = str(path)
    roles = {
        "canonical": "canonical",
        "strong_info": "strong",
        "matched_control": "control",
        "blind": "blind",
        "look_away": "look",
        "all_camera_blackout": "all_camera_blackout",
    }
    conditions = []
    for role, pose_id in roles.items():
        evaluation_only = role in {"blind", "look_away", "all_camera_blackout"}
        conditions.append(
            {
                "condition_role": role,
                "source_pose_id": pose_id,
                "evaluation_only": evaluation_only,
                "training_eligible": False,
                "is_extreme": evaluation_only,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "dsol_constructed_blind_reveal_package_v1",
                "task_id": "task_a",
                "snapshot_group_id": "group_a",
                "scene_variant_id": "scene_a",
                "split": "val",
                "source_scan": str(source_scan),
                "snapshot_components": components,
                "conditions": conditions,
            }
        ),
        encoding="utf-8",
    )
    packaged = package(manifest)
    indexed = {record["condition_role"]: record for record in packaged["records"]}
    assert len(packaged["snapshot"]["snapshot_sha256"]) == 64
    assert indexed["all_camera_blackout"]["visibility_score"] == 0.0
    assert indexed["all_camera_blackout"]["per_camera_scores"] == {
        "agentview": 0.0,
        "robot0_eye_in_hand": 0.0,
    }


def test_gate_rejects_training_eligible_validation_condition(tmp_path: Path) -> None:
    scan = _scan(tmp_path)
    scan["records"][1]["training_eligible"] = True
    result = audit_snapshot(scan, _config())
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["evaluation_only_safety"]["status"] == "FAIL"


def test_audit_cli_writes_gate_jsonl_and_camera_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan_path = tmp_path / "scan.json"
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "audit"
    scan_path.write_text(json.dumps(_scan(tmp_path / "snapshot")), encoding="utf-8")
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_constructed_blind_reveal.py",
            str(scan_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
    )
    audit_main()
    gate = json.loads((output_dir / "gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "PASS"
    assert gate["m1_admission"] is True
    assert (output_dir / "snapshot_gates.jsonl").read_text(encoding="utf-8").count("\n") == 1
    csv_text = (output_dir / "condition_records.csv").read_text(encoding="utf-8")
    assert "camera::agentview" in csv_text
    assert "snapshot_sha256" in csv_text
