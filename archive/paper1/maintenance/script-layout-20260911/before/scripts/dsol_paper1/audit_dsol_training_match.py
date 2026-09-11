#!/usr/bin/env python3
"""Audit saved DSOL training recipes and prepare a non-executable matched draft.

Never imports a model, changes an archived run, manages GPUs or starts training.
Recipe equality is not proof of identical sampled examples, RNG or dependencies.
Large model weights are deliberately not rehashed by this small-artifact audit.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


CONFIG_EXCLUSIONS = {"/run_id", "/output_root_dir", "/output_dir", "/datasets/vla_data/dsol_arm"}
MANIFEST_MATCH_FIELDS = (
    "seed", "num_gpus", "steps", "scheduler_total_steps",
    "global_model_examples_per_update", "source_data_items_per_update",
    "examples_per_data_item", "gradient_accumulation_steps", "data_root",
    "data_manifest_sha256", "pretrained_checkpoint_manifest_sha256", "python",
    "calibration", "skip_final_save",
)
VALID_ARMS = {"canonical_unique", "broad_unpaired_practical"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and value:
        result = {}
        for key, item in value.items():
            result.update(_flatten(item, prefix + "/" + str(key)))
        return result
    return {prefix: value}


def config_differences(left: dict, right: dict) -> list[dict]:
    a, b = _flatten(left), _flatten(right)
    differences = []
    for key in sorted(set(a) | set(b)):
        if key in CONFIG_EXCLUSIONS:
            continue
        if key not in a or key not in b or type(a[key]) is not type(b[key]) or a[key] != b[key]:
            differences.append({"path": key, "left": a.get(key), "right": b.get(key),
                                "left_present": key in a, "right_present": key in b})
    return differences


def inspect_run(run_dir: Path) -> dict:
    """Read actual saved config, run identity, final metric and small manifests."""
    run_dir = run_dir.resolve()
    config_path = run_dir / "final_model/framework_config.yaml"
    manifest_path = run_dir / "run_manifest.json"
    config = yaml.safe_load(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(config, dict) or not isinstance(manifest, dict):
        raise ValueError("saved config and manifest must be objects")
    missing = [key for key in MANIFEST_MATCH_FIELDS if key not in manifest]
    if missing:
        raise ValueError("missing manifest fields: " + ", ".join(missing))
    trainer, data = config["trainer"], config["datasets"]["vla_data"]
    issues = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            issues.append(message)

    for key, value in (("seed", config["seed"]), ("arm", data["dsol_arm"]),
                       ("steps", trainer["max_train_steps"]),
                       ("scheduler_total_steps", trainer["scheduler_total_steps"]),
                       ("gradient_accumulation_steps", trainer["gradient_accumulation_steps"]),
                       ("data_root", data["data_root_dir"]),
                       ("examples_per_data_item", data["examples_per_item"])):
        check(manifest.get(key) == value, "config/manifest mismatch: " + key)
    check(trainer.get("is_resume") is False, "resume training is not a same-initialization treatment")
    check(manifest["arm"] in VALID_ARMS, "this audit only covers canonical unique / broad practical")
    check(manifest["examples_per_data_item"] == 1, "paired exposure requires a different contract")
    batch = data["per_device_batch_size"] * manifest["num_gpus"] * manifest["gradient_accumulation_steps"]
    check(batch == manifest["global_model_examples_per_update"], "global model-example batch mismatch")
    check(batch == manifest["source_data_items_per_update"], "source-item batch mismatch")
    small_artifacts = {}
    for label, path, expected in (
        ("data_manifest", Path(data["data_root_dir"]) / "manifest.json", manifest["data_manifest_sha256"]),
        ("initial_source_manifest", Path(manifest["pretrained_checkpoint_manifest"]),
         manifest["pretrained_checkpoint_manifest_sha256"]),
    ):
        actual = sha256_file(path)
        small_artifacts[label] = {"path": str(path.resolve()), "sha256": actual, "recorded_sha256": expected}
        check(actual == expected, label + " content hash no longer matches archived declaration")
    initial_path = Path(trainer["pretrained_checkpoint"]).resolve()
    check(Path(manifest["pretrained_checkpoint_manifest"]).resolve().parent == initial_path,
          "initial source manifest does not belong to configured initial checkpoint")
    if "pretrained_checkpoint" in manifest:
        check(Path(manifest["pretrained_checkpoint"]).resolve() == initial_path, "initial checkpoint path mismatch")
    final = None
    metric_count = 0
    with (run_dir / "metrics.jsonl").open() as stream:
        for line in stream:
            if line.strip():
                final = json.loads(line)
                metric_count += 1
    if final is None:
        raise ValueError("empty training metrics")
    check(final.get("step") == manifest["steps"], "final metric step differs from declared completed budget")
    check(final.get("examples_seen") == manifest["steps"] * batch, "final model-example count mismatch")
    weights = run_dir / "final_model/model.safetensors"
    check(weights.is_file(), "final model weights absent")
    return {
        "run_dir": str(run_dir), "config": config, "manifest": manifest,
        "config_sha256": sha256_file(config_path), "manifest_sha256": sha256_file(manifest_path),
        "metrics_sha256": sha256_file(run_dir / "metrics.jsonl"), "metric_rows": metric_count,
        "last_step": final.get("step"), "last_learning_rate": final.get("learning_rate"),
        "last_examples_seen": final.get("examples_seen"), "small_artifacts": small_artifacts,
        "model_weights": {"path": str(weights), "present": weights.is_file(),
                          "size_bytes": weights.stat().st_size if weights.is_file() else None,
                          "content_sha256_verified": False},
        "issues": issues,
    }


def compare_runs(left: dict, right: dict) -> dict:
    configuration = config_differences(left["config"], right["config"])
    manifest = [{"field": key, "left": left["manifest"].get(key), "right": right["manifest"].get(key)}
                for key in MANIFEST_MATCH_FIELDS
                if left["manifest"].get(key) != right["manifest"].get(key)]
    code_a, code_b = (x["manifest"].get("critical_code_sha256", {}) for x in (left, right))
    code = [{"path": key, "left": code_a.get(key), "right": code_b.get(key)}
            for key in sorted(set(code_a) | set(code_b)) if code_a.get(key) != code_b.get(key)]
    treatment_pair = {left["manifest"].get("arm"), right["manifest"].get("arm")} == VALID_ARMS
    recipe_match = not configuration and not manifest and not left["issues"] and not right["issues"] and treatment_pair
    return {
        "schema": "dsol_training_match_audit_v1", "launch_enabled": False,
        "formal_release": False, "saved_recipe_match": recipe_match,
        "critical_code_identical": bool(code_a) and bool(code_b) and not code,
        "allowed_treatment_pair": treatment_pair,
        "config_differences": configuration, "manifest_differences": manifest,
        "critical_code_differences": code,
        "left": left, "right": right,
        "limitations": [
            "Recipe match is necessary, not proof of sampled-example/RNG/runtime equivalence.",
            "Code hash differences require semantic review; they do not alone prove a changed training objective.",
            "Model weight content, full dataset payload and dependency identity are not certified here.",
        ],
    }


def propose_canonical(anchor: dict, *, run_id: str, output_root: Path, repo_root: Path) -> dict:
    if anchor["issues"]:
        raise ValueError("anchor artifact inconsistencies: " + "; ".join(anchor["issues"]))
    if anchor["manifest"]["arm"] != "broad_unpaired_practical":
        raise ValueError("proposal anchor must be broad_unpaired_practical")
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- ." for c in run_id) or " " in run_id or run_id in (".", ".."):
        raise ValueError("unsafe run_id")
    if not output_root.is_absolute():
        raise ValueError("training output root must be absolute")
    destination = output_root / run_id
    if destination.exists():
        raise ValueError("proposed training destination already exists")
    config = copy.deepcopy(anchor["config"])
    config["datasets"]["vla_data"]["dsol_arm"] = "canonical_unique"
    config.update(run_id=run_id, output_root_dir=str(output_root), output_dir=str(destination))
    if config_differences(anchor["config"], config):
        raise AssertionError("proposal changed an unapproved training parameter")
    code_checks = []
    for relative, expected in anchor["manifest"].get("critical_code_sha256", {}).items():
        path = (repo_root / relative).resolve()
        if repo_root.resolve() not in path.parents:
            raise ValueError("critical code path escapes repository")
        actual = sha256_file(path) if path.is_file() else None
        code_checks.append({"path": relative, "anchor_sha256": expected,
                            "current_sha256": actual, "identical": actual == expected})
    return {
        "schema": "dsol_matched_canonical_training_proposal_v1", "status": "DRAFT_NOT_TRAINED",
        "launch_enabled": False, "formal_release": False,
        "anchor_run": anchor["run_dir"], "anchor_config_sha256": anchor["config_sha256"],
        "anchor_manifest_sha256": anchor["manifest_sha256"],
        "initialization": "same_original_pretrained_checkpoint_not_old_canonical_continuation",
        "proposed_config": config,
        "frozen_execution_identity": {key: anchor["manifest"][key] for key in MANIFEST_MATCH_FIELDS},
        "critical_code_checks": code_checks,
        "saved_recipe_differences_except_treatment_and_output": [],
        "remaining_requirements": [
            "Verify initial weights, dataset payload and runtime/dependency provenance.",
            "Review current-vs-anchor code differences and bind launcher effective config before launch.",
            "Freeze bounded training compute/output budget; this draft never invokes a launcher.",
            "Audit completed canonical training before publishing a matched causal comparison.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path)
    parser.add_argument("--propose-canonical-run-id")
    parser.add_argument("--training-output-root", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if bool(args.candidate_run) == bool(args.propose_canonical_run_id):
        parser.error("choose exactly one candidate comparison or canonical proposal")
    if args.propose_canonical_run_id and args.training_output_root is None:
        parser.error("canonical proposal requires --training-output-root")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty; never overwrite an audit")
    try:
        anchor = inspect_run(args.anchor_run)
        if args.candidate_run:
            result = compare_runs(anchor, inspect_run(args.candidate_run))
        else:
            result = propose_canonical(anchor, run_id=args.propose_canonical_run_id,
                                       output_root=args.training_output_root, repo_root=args.repo_root)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"formal_release": False, "error": str(error)}))
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "audit.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    if "proposed_config" in result:
        with (args.output_dir / "proposed_framework_config.yaml").open("x") as stream:
            yaml.safe_dump(result["proposed_config"], stream, sort_keys=False)
    print(json.dumps({"output": str(output.resolve()), "formal_release": False,
                      "saved_recipe_match": result.get("saved_recipe_match"),
                      "status": result.get("status", "AUDIT_ONLY")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
