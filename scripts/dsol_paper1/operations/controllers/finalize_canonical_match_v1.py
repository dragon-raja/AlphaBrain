#!/usr/bin/env python3
"""Certify the completed seed41 canonical artifact; never launch an evaluation.

The output is exclusive-created. This reads the final weight bytes once for a
content hash, but does not load a model, initialize CUDA or validate tensor values.
"""
from __future__ import annotations

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[4]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))


import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metrics(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_metrics(rows: list[dict], anchor: list[dict], steps: int, batch: int) -> dict:
    if len(rows) != steps or [row.get("step") for row in rows] != list(range(1, steps + 1)):
        raise ValueError("optimizer metrics are not a complete continuous sequence")
    if len(anchor) != steps or [row.get("step") for row in anchor] != list(range(1, steps + 1)):
        raise ValueError("anchor optimizer metrics are not continuous")
    if any(row.get("examples_seen") != row["step"] * batch for row in rows):
        raise ValueError("model-example accounting mismatch")
    if not all(math.isfinite(row["action_dit_loss"]) for row in rows):
        raise ValueError("non-finite training loss")
    if any(row.get("learning_rate") != other.get("learning_rate") for row, other in zip(rows, anchor)):
        raise ValueError("full learning-rate history differs from anchor")
    if rows[-1]["learning_rate"] != 5e-6:
        raise ValueError("unexpected final learning rate")
    return {"continuous_optimizer_steps": steps, "examples_seen": rows[-1]["examples_seen"],
            "all_training_losses_finite": True, "full_lr_history_matches_anchor": True,
            "final_learning_rate": rows[-1]["learning_rate"],
            "final_training_loss": rows[-1]["action_dit_loss"]}


def inspect_weight_header(path: Path) -> dict:
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors prefix")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if not 1 <= header_bytes <= min(10_000_000, size - 8):
            raise ValueError("invalid safetensors header length")
        header = json.loads(stream.read(header_bytes))
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if len(tensors) != 935:
        raise ValueError("unexpected tensor count")
    offsets = sorted(value["data_offsets"] for value in tensors.values())
    cursor = 0
    for start, stop in offsets:
        if start != cursor or stop < start:
            raise ValueError("non-contiguous tensor payload extents")
        cursor = stop
    if 8 + header_bytes + cursor != size:
        raise ValueError("payload extent does not match file size")
    return {"tensor_count": len(tensors), "header_bytes": header_bytes,
            "payload_extents_contiguous": True, "file_size_bytes": size,
            "tensor_values_loaded_or_checked": False}


def finalize(audit_dir: Path, release_path: Path) -> dict:
    destination = audit_dir / "completed_run_receipt.json"
    if destination.exists():
        raise FileExistsError("completion receipt already exists; never overwrite")
    audit_path = audit_dir / "audit.json"
    audit = json.loads(audit_path.read_text())
    release = json.loads(release_path.read_text())
    if audit.get("saved_recipe_match") is not True or audit.get("config_differences") or audit.get("manifest_differences"):
        raise ValueError("saved training recipes did not match")
    expected_launcher_diff = [{
        "path": "scripts/dsol_paper1/training/run_libero_pair_train.sh",
        "left": "b3f7978245863b3cc56211936df5a08c406504911077679093e2fc784784268f",
        "right": "650ce7b0e018ba62cc2546a4ebc074d6ba949ad19a14de1a7698ae45d4a55d37",
    }]
    if audit.get("critical_code_differences") != expected_launcher_diff:
        raise ValueError("unexpected code delta; requires separate review")
    canonical = Path(audit["right"]["run_dir"])
    anchor = Path(audit["left"]["run_dir"])
    if str(canonical) != release["output_dir"] or str(anchor) != release["anchor_run"]:
        raise ValueError("audit and training release identify different runs")
    for side, run in ((audit["left"], anchor), (audit["right"], canonical)):
        if side.get("issues"):
            raise ValueError("run audit reported unresolved issues")
        for relative, key in (("metrics.jsonl", "metrics_sha256"),
                              ("run_manifest.json", "manifest_sha256"),
                              ("final_model/framework_config.yaml", "config_sha256")):
            if digest_file(run / relative) != side[key]:
                raise ValueError("saved artifact changed after recipe audit: " + relative)
    manifest = audit["right"]["manifest"]
    if digest_file(release_path) != manifest["budget_decision_sha256"]:
        raise ValueError("historical training release changed")
    metrics = validate_metrics(read_metrics(canonical / "metrics.jsonl"),
                               read_metrics(anchor / "metrics.jsonl"),
                               release["optimizer_updates"], release["global_model_examples_per_update"])
    log_path = canonical / "logs/train_20260907_125837.log"
    log = log_path.read_text()
    if "Training complete. Self-contained final model saved at" not in log or "... and that's all, folks!" not in log:
        raise ValueError("training log lacks expected normal completion markers")
    weight_path = canonical / "final_model/model.safetensors"
    stat_before = weight_path.stat()
    header = inspect_weight_header(weight_path)
    content_sha256 = digest_file(weight_path)
    stat_after = weight_path.stat()
    if (stat_before.st_ino, stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_ino, stat_after.st_size, stat_after.st_mtime_ns):
        raise ValueError("weight artifact changed during content hashing")
    result = {
        "schema": "dsol_canonical_match_completed_run_receipt_v1",
        "status": "COMPLETED_ARTIFACT_AUDIT_PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_completed_at_utc": "2026-09-07T07:47:55+00:00",
        "scientific_evaluation_released_by_this_receipt": False,
        "active_camera_release": False,
        "canonical_run": str(canonical), "anchor_run": str(anchor),
        "training_release": {"path": str(release_path), "sha256": digest_file(release_path)},
        "recipe_audit": {"path": str(audit_path), "sha256": digest_file(audit_path), "saved_recipe_match": True},
        "completion_log": {"path": str(log_path), "sha256": digest_file(log_path), "normal_completion_markers": True},
        "metrics": metrics,
        "final_weights": {"path": str(weight_path), "content_sha256": content_sha256,
                          "content_sha256_verified": True, **header},
        "treatment": "same-record external image broad_a versus canonical; wrist/language/action supervision retained",
        "reviewed_code_delta": {"path": expected_launcher_diff[0]["path"],
                                "scope": "explicit initial-checkpoint override and provenance; see training_match_anchor_audit_20260907_zh.md"},
        "limitations": [
            "One matched seed41 pair; not a cross-seed or cross-model generality result.",
            "Recipe/identity matching does not establish bitwise RNG or sampled-gradient equality.",
            "This verifies saved-weight content identity and structural completeness, not tensor finiteness or inference capability.",
            "No task success rate is produced. Evaluation requires its own frozen protocol and budget.",
            "Historical initial-weight and dataset payload hashes were prelaunch audited; not rehashed by this finalizer.",
        ],
        "finalizer_sha256": digest_file(Path(__file__).resolve()),
    }
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return {"receipt": str(destination), "status": result["status"], "final_weight_sha256": content_sha256}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--training-release", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.audit_dir.resolve(), args.training_release.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
