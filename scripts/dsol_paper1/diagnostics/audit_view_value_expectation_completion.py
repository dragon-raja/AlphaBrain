#!/usr/bin/env python3
"""Fail-closed completion audit for the formal view-value experiment."""

from __future__ import annotations

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[3]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))


import argparse
import collections
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

try:
    from scripts.dsol_paper1.diagnostics.audit_view_value_expectation_heldout_run import (
        _load_shards,
        audit as audit_heldout,
    )
    from scripts.dsol_paper1.explicit_flow_noise import ExplicitFlowNoiseBank, sha256_file
except ModuleNotFoundError:
    from scripts.dsol_paper1.diagnostics.audit_view_value_expectation_heldout_run import _load_shards, audit as audit_heldout
    from scripts.dsol_paper1.explicit_flow_noise import ExplicitFlowNoiseBank, sha256_file


CALIBRATION_PROTOCOLS = {
    "A": (6208, "b5541a9fd584a684b7bc8e1e08c89f2508b216ca60b07b4c1f28cc4422dd95b7"),
    "B": (3072, "039f9f15fe89cd6bccc08e450555836865f0b875e1001395bdf5fb4af5d2cb59"),
    "C": (1536, "b6a9e16f6e666eb906f3b4f055c793e0e04b352c6d80c6c5a78e447d19b6b7b0"),
    "D": (2048, "3d99c62b0aa252dd1e1b15b17b3b7bbbda3615176120d2a81e398c2722a97eec"),
}
HELDOUT_PROTOCOLS = {
    41: (9216, "f5327dcd2860ee67dc9e6ab557e3f1aacbac311bf5872cac13f85fad35b04636"),
    42: (3072, "f3d8ac66c814d4ca3c922295aa04ecdb0fbb96fd9a5f26b405fd4759e7d02e38"),
    43: (3072, "bf6f9af3f57de3ccdea42178f69080a039e348c546475b0a251f5ccf172af76c"),
}
EXPECTED_FILE_SHA256 = {
    "population/population.json": "878ec0a0bd18322c935ba3c4c115d310bfbd9125b36a7d6a0a9d1be9229ac698",
    "calibration/analysis/analysis.json": "dea8b7465d3864960ace4d309580c8696526a452a6d3d02c084359d001b799e2",
    "calibration/analysis/state_results.csv": "14862cbabc2af7c1c7225ce53eaebe1ae7805154b510d8e9775ae74c27eb1c5a",
    "heldout/analysis-primary/primary-analysis.json": "79d58466ec2deef22675391746f608a2b26872c1d542c2fd47505028859f4e3d",
    "heldout/analysis-primary/reserve-decision.json": "9b274eb392ce5d83a042500043e065716b0f107ea26321b4788f284ccd7dab98",
    "heldout/analysis-primary/state-method-results.csv": (
        "5431d552cfa71a2d4c0b1b94dd977a60810664e848dcc863b6054d9ad59acd3d"
    ),
}
EXPECTED_CLAIM_SCOPE = (
    "fixed-state E0 counterfactual views in the frozen 97-view LIBERO bank; "
    "not official LIBERO-Plus full-benchmark performance and not physical active camera acquisition"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256_concat(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def validate_population(population: Mapping[str, Any]) -> dict[str, Any]:
    require(population.get("schema") == "dsol_view_value_expectation_population_v1", "bad population schema")
    require(population.get("status") == "PASS", "population did not pass")
    groups = population.get("population")
    require(isinstance(groups, dict), "population groups are missing")
    calibration = groups.get("calibration", {}).get("states", [])
    heldout = groups.get("heldout_test", {}).get("states", [])
    require(len(calibration) == 16, "calibration population is not 16 states")
    require(len(heldout) == 48, "held-out population is not 48 states")
    calibration_pairs = {str(row["pair_key"]) for row in calibration}
    heldout_pairs = {str(row["pair_key"]) for row in heldout}
    calibration_sources = {str(row["source_group"]) for row in calibration}
    heldout_sources = {str(row["source_group"]) for row in heldout}
    require(len(calibration_pairs) == 16, "duplicate calibration pair keys")
    require(len(heldout_pairs) == 48, "duplicate held-out pair keys")
    require(not calibration_pairs & heldout_pairs, "calibration and held-out pair keys overlap")
    require(not calibration_sources & heldout_sources, "calibration and held-out sources overlap")
    calibration_tasks = collections.Counter(str(row["task_id"]) for row in calibration)
    heldout_tasks = collections.Counter(str(row["task_id"]) for row in heldout)
    require(
        len(calibration_tasks) == 8 and set(calibration_tasks.values()) == {2},
        "bad calibration task stratification",
    )
    require(len(heldout_tasks) == 8 and set(heldout_tasks.values()) == {6}, "bad held-out task stratification")
    return {
        "calibration_state_count": len(calibration),
        "heldout_state_count": len(heldout),
        "task_count": len(calibration_tasks),
        "source_disjoint": True,
        "state_disjoint": True,
    }


def audit_calibration_stage(root: Path, stage: str) -> dict[str, Any]:
    expected_count, expected_protocol_sha256 = CALIBRATION_PROTOCOLS[stage]
    protocol_path = root / f"protocols/calibration-stage-{stage}.json"
    run_dir = root / f"calibration/stage-{stage}"
    manifest_path = run_dir / "run_manifest.json"
    noise_manifest_path = root / f"noise-banks-h10/bank_{stage}.manifest.json"
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)

    require(
        protocol.get("schema") == "dsol_view_value_expectation_calibration_stage_v1",
        f"stage {stage}: bad protocol schema",
    )
    require(protocol.get("status") == "PASS", f"stage {stage}: protocol did not pass")
    require(protocol.get("stage") == stage, f"stage {stage}: protocol stage mismatch")
    specs = protocol.get("specs")
    require(isinstance(specs, list), f"stage {stage}: protocol specs missing")
    require(len(specs) == expected_count == int(protocol["episode_count"]), f"stage {stage}: episode count mismatch")
    expected = {str(spec["episode_id"]): (index, spec) for index, spec in enumerate(specs)}
    require(len(expected) == len(specs), f"stage {stage}: duplicate protocol episode IDs")

    protocol_sha256 = sha256_file(protocol_path)
    require(protocol_sha256 == expected_protocol_sha256, f"stage {stage}: frozen protocol SHA-256 changed")
    require(
        Path(manifest["protocol"]).resolve() == protocol_path.resolve(),
        f"stage {stage}: manifest protocol path mismatch",
    )
    require(manifest.get("protocol_sha256") == protocol_sha256, f"stage {stage}: manifest protocol SHA-256 mismatch")
    noise_manifest_sha256 = sha256_file(noise_manifest_path)
    require(
        Path(manifest["noise_bank_manifest"]).resolve() == noise_manifest_path.resolve(),
        f"stage {stage}: manifest noise path mismatch",
    )
    require(
        manifest.get("noise_bank_manifest_sha256") == noise_manifest_sha256,
        f"stage {stage}: noise manifest SHA-256 mismatch",
    )
    require(manifest.get("require_explicit_noise") is True, f"stage {stage}: explicit noise not required")
    shard_count = int(manifest["eval_worker_count"])
    require(shard_count == 32, f"stage {stage}: expected 32 evaluator shards")

    bank = ExplicitFlowNoiseBank(noise_manifest_path, verify_file=True)
    require(bank.manifest.get("bank_id") == stage == protocol.get("bank_id"), f"stage {stage}: noise bank ID mismatch")
    require(
        set(bank.manifest["state_keys"])
        == {str(spec["pair_key"]) for spec in specs},
        f"stage {stage}: noise state population mismatch",
    )

    loaded, shard_counts, ignored_partial_tails = _load_shards(run_dir, shard_count)
    require(ignored_partial_tails == 0, f"stage {stage}: incomplete trailing shard write")
    seen: set[str] = set()
    candidate_counts: collections.Counter[str] = collections.Counter()
    pair_episode_ids: dict[str, set[str]] = collections.defaultdict(set)
    pair_physics: dict[str, set[str]] = collections.defaultdict(set)
    pair_environment_seeds: dict[str, set[int]] = collections.defaultdict(set)
    noise_signatures: dict[tuple[str, int, int], tuple[int, str]] = {}
    total_policy_calls = 0
    for shard, row in loaded:
        episode_id = str(row.get("episode_id", ""))
        require(episode_id in expected, f"stage {stage}: result absent from protocol: {episode_id}")
        require(episode_id not in seen, f"stage {stage}: duplicate episode: {episode_id}")
        seen.add(episode_id)
        protocol_index, spec = expected[episode_id]
        require(protocol_index % shard_count == shard, f"stage {stage}: episode in wrong shard: {episode_id}")
        for field, value in spec.items():
            require(row.get(field) == value, f"stage {stage}: protocol field {field} differs: {episode_id}")
        require(row.get("status") == "complete", f"stage {stage}: incomplete episode: {episode_id}")
        require(row.get("explicit_flow_noise") is True, f"stage {stage}: explicit noise missing: {episode_id}")
        require(row.get("noise_bank_id") == stage, f"stage {stage}: result used another noise bank: {episode_id}")
        require(
            row.get("noise_bank_manifest_sha256") == noise_manifest_sha256,
            f"stage {stage}: result noise manifest mismatch: {episode_id}",
        )

        pair_key = str(row["pair_key"])
        repeat_id = int(row["policy_repeat_id"])
        candidate_counts[str(row["selected_candidate_id"])] += 1
        pair_episode_ids[pair_key].add(episode_id)
        pair_environment_seeds[pair_key].add(int(row["environment_seed"]))
        metrics = row.get("initial_metrics", {})
        before = str(metrics.get("physics_state_sha256", ""))
        after = str(metrics.get("post_wait_physics_state_sha256_exact", ""))
        require(before and before == after, f"stage {stage}: physics changed during setup: {episode_id}")
        pair_physics[pair_key].add(before)

        calls = row.get("policy_calls")
        require(isinstance(calls, list), f"stage {stage}: policy-call ledger missing: {episode_id}")
        require(len(calls) == int(row["inference_calls"]), f"stage {stage}: policy-call count mismatch: {episode_id}")
        require(
            [int(call["replan_index"]) for call in calls] == list(range(len(calls))),
            f"stage {stage}: replan indices are not contiguous: {episode_id}",
        )
        total_policy_calls += len(calls)
        for call in calls:
            replan_index = int(call["replan_index"])
            expected_noise = bank.get(pair_key, repeat_id, replan_index)
            require(
                int(call["policy_repeat_id"]) == repeat_id,
                f"stage {stage}: policy-call repeat mismatch: {episode_id}",
            )
            require(
                call.get("noise_seed") == expected_noise["noise_seed"],
                f"stage {stage}: noise seed mismatch: {episode_id}",
            )
            require(
                call.get("noise_sha256") == expected_noise["noise_sha256"],
                f"stage {stage}: noise SHA-256 mismatch: {episode_id}",
            )
            key = (pair_key, repeat_id, replan_index)
            signature = (int(call["noise_seed"]), str(call["noise_sha256"]))
            require(
                noise_signatures.setdefault(key, signature) == signature,
                f"stage {stage}: paired candidates used different noise: {key}",
            )

    require(seen == set(expected), f"stage {stage}: result set incomplete: {len(seen)}/{len(expected)}")
    expected_shards = [0] * shard_count
    for index in range(len(specs)):
        expected_shards[index % shard_count] += 1
    require(shard_counts == expected_shards, f"stage {stage}: shard counts differ from protocol")
    require(
        all(len(values) == 1 for values in pair_physics.values()),
        f"stage {stage}: physics differs within a state",
    )
    require(
        all(len(values) == 1 for values in pair_environment_seeds.values()),
        f"stage {stage}: environment seed differs within a state",
    )
    expected_by_pair: dict[str, set[str]] = collections.defaultdict(set)
    for spec in specs:
        expected_by_pair[str(spec["pair_key"])].add(str(spec["episode_id"]))
    complete_matrices = sum(pair_episode_ids[key] == values for key, values in expected_by_pair.items())
    require(complete_matrices == int(protocol["state_count"]), f"stage {stage}: incomplete state matrices")
    return {
        "status": "PASS_COMPLETE",
        "protocol_sha256": protocol_sha256,
        "noise_bank_manifest_sha256": noise_manifest_sha256,
        "noise_file_sha256": str(bank.manifest["noise_file_sha256"]),
        "episode_count": len(seen),
        "shard_count": shard_count,
        "shard_min": min(shard_counts),
        "shard_max": max(shard_counts),
        "state_count": len(pair_episode_ids),
        "candidate_count_per_state": int(protocol["candidate_count_per_state"]),
        "unique_candidate_id_count": len(candidate_counts),
        "policy_noise_repeats": int(protocol["policy_noise_repeats"]),
        "complete_state_matrices": int(complete_matrices),
        "total_policy_calls": total_policy_calls,
        "paired_noise_keys": len(noise_signatures),
        "physics_pair_violations": 0,
        "environment_seed_pair_violations": 0,
    }


def validate_final_decision(
    calibration: Mapping[str, Any], heldout: Mapping[str, Any], decision: Mapping[str, Any]
) -> None:
    selector = heldout["selector_population_gate"]
    expected = {
        "schema": "dsol_view_value_expectation_final_decision_v1",
        "status": "VIEW_HEADROOM_NOT_CONFIRMED",
        "calibration_headroom_gate": calibration["status"],
        "heldout_selector_gate": selector["status"],
        "selector_method": selector["best_rule_frozen_on_calibration"],
        "cross_checkpoint_mean_gain_pp": selector["cross_checkpoint_mean_gain_pp"],
        "cross_checkpoint_mean_harm_probability": selector["cross_checkpoint_mean_harm_probability"],
        "direction_consistent_positive": selector["direction_consistent_positive"],
        "noise_repeats_per_condition": heldout["noise_repeats_per_condition"],
        "claim_scope": EXPECTED_CLAIM_SCOPE,
    }
    for field, value in expected.items():
        require(decision.get(field) == value, f"final decision field differs from analyses: {field}")
    require(
        isinstance(decision.get("generated_at"), str) and decision["generated_at"],
        "final decision timestamp missing",
    )


def audit_completion(root: Path, repo: Path) -> dict[str, Any]:
    root = root.resolve()
    repo = repo.resolve()
    require(root.is_dir(), f"experiment root is missing: {root}")
    require(repo.is_dir(), f"repository root is missing: {repo}")

    immutable_hashes = {}
    for relative, expected_sha256 in EXPECTED_FILE_SHA256.items():
        actual = sha256_file(root / relative)
        require(actual == expected_sha256, f"frozen artifact SHA-256 changed: {relative}")
        immutable_hashes[relative] = actual

    population = load_json(root / "population/population.json")
    population_audit = validate_population(population)
    calibration_runs = {stage: audit_calibration_stage(root, stage) for stage in CALIBRATION_PROTOCOLS}

    heldout_runs = {}
    for index, (seed, (expected_count, expected_protocol_sha256)) in enumerate(HELDOUT_PROTOCOLS.items()):
        protocol_path = root / f"protocols/heldout-primary-seed{seed}.json"
        run_dir = root / f"heldout/primary-seed{seed}"
        current = audit_heldout(
            protocol_path=protocol_path,
            run_dir=run_dir,
            run_manifest_path=run_dir / "run_manifest.json",
            noise_manifest_path=root / "noise-banks-h10/bank_E.manifest.json",
            require_complete=True,
            verify_noise_file=index == 0,
        )
        require(
            current["protocol_sha256"] == expected_protocol_sha256,
            f"seed {seed}: frozen protocol SHA-256 changed",
        )
        require(current["result_episode_count"] == expected_count, f"seed {seed}: result count changed")
        require(current["complete_state_matrices"] == 48, f"seed {seed}: incomplete state matrices")
        stored = load_json(run_dir / "heldout-run-audit.json")
        require(stored == current, f"seed {seed}: stored audit receipt differs from fresh audit")
        heldout_runs[str(seed)] = current

    freeze = load_json(root / "protocols/heldout-freeze-receipt.json")
    require(freeze.get("status") == "PASS", "held-out freeze receipt did not pass")
    require(
        freeze.get("selection_uses_heldout_policy_outcomes") is False,
        "held-out outcomes were used to freeze rules",
    )
    require(freeze.get("best_noncanonical_rule") == "calibration_global_fixed_pose", "frozen selector changed")
    require(freeze.get("calibration_global_fixed_pose") == "broad_train_053", "frozen fixed pose changed")

    calibration = load_json(root / "calibration/analysis/analysis.json")
    require(
        calibration.get("schema") == "dsol_view_value_expectation_calibration_analysis_v1",
        "bad calibration analysis schema",
    )
    require(calibration.get("status") == "VIEW_HEADROOM_NOT_CONFIRMED", "calibration gate changed")
    require(
        calibration.get("state_count") == 16
        and calibration.get("strong_state_count") == 0,
        "calibration state gate counts changed",
    )

    heldout = load_json(root / "heldout/analysis-primary/primary-analysis.json")
    require(heldout.get("schema") == "dsol_view_value_expectation_heldout_analysis_v1", "bad held-out analysis schema")
    require(heldout.get("status") == "PRIMARY_COMPLETE_PRECISION_SUFFICIENT", "held-out analysis status changed")
    selector = heldout.get("selector_population_gate", {})
    require(selector.get("status") == "SELECTOR_GAIN_NOT_CONFIRMED", "held-out selector gate changed")
    require(
        selector.get("best_rule_frozen_on_calibration")
        == "calibration_global_fixed_pose",
        "analyzed selector changed",
    )
    require(selector.get("direction_consistent_positive") is False, "directional result changed")
    require(selector.get("final_precision_halfwidth_at_most_5pp") is True, "primary precision is not sufficient")

    reserve = load_json(root / "heldout/analysis-primary/reserve-decision.json")
    require(reserve == {
        "activate_bank_F": False,
        "machine_generated_before_bank_F_opened": True,
        "reasons": [],
        "schema": "dsol_view_value_expectation_reserve_decision_v1",
        "status": "PRIMARY_PRECISION_SUFFICIENT",
    }, "reserve decision changed")
    allowed_heldout = {"analysis-primary", "primary-seed41", "primary-seed42", "primary-seed43"}
    actual_heldout = {path.name for path in (root / "heldout").iterdir()}
    require(
        actual_heldout == allowed_heldout,
        "unexpected held-out outputs exist, including a possible unapproved Bank F run",
    )

    decision = load_json(root / "final-report/final_decision.json")
    validate_final_decision(calibration, heldout, decision)
    markdown = root / "final-report/view_value_expectation_final_zh.md"
    markdown_text = markdown.read_text(encoding="utf-8")
    for required_text in (
        "VIEW_HEADROOM_NOT_CONFIRMED",
        "SELECTOR_GAIN_NOT_CONFIRMED",
        "固定物理状态下的 97 个反事实外部视角",
        "不等同于官方 LIBERO-Plus 全量成绩",
        "不证明物理主动相机获取有效",
    ):
        require(required_text in markdown_text, f"final report is missing claim boundary: {required_text}")

    report_minimum_sizes = {
        root / "final-report/final_decision.json": 500,
        root / "final-report/heldout_metrics.csv": 500,
        markdown: 500,
        root / "final-report/view_value_expectation_final_zh.pdf": 10_000,
        **{
            root / f"final-report/previews/page-{page:02d}.png": 10_000
            for page in range(1, 7)
        },
    }
    report_files = list(report_minimum_sizes)
    for path, minimum_size in report_minimum_sizes.items():
        require(
            path.is_file() and path.stat().st_size > minimum_size,
            f"final report artifact missing or too small: {path}",
        )
    report_hashes = {str(path.relative_to(root)): sha256_file(path) for path in report_files}

    visibility_summary = load_json(root / "visibility-scan/analysis/summary.json")
    require(visibility_summary.get("scan_count") == 64, "visibility scan count changed")
    require(visibility_summary.get("candidate_record_count") == 7424, "visibility candidate count changed")
    visibility_diagnostic = root / "visibility-scan/analysis/visibility_diagnostics.png"
    require(visibility_diagnostic.stat().st_size > 10_000, "visibility diagnostic is missing")

    rank_paths = sorted((root / "accel-ensemble").glob("rank-shard-*.jsonl"))
    render_paths = sorted((root / "accel-ensemble").glob("render-shard-*.jsonl"))
    require(len(rank_paths) == len(render_paths) == 8, "Accel shard count changed")
    rank_rows = sum(path.read_text(encoding="utf-8").count("\n") for path in rank_paths)
    render_rows = sum(path.read_text(encoding="utf-8").count("\n") for path in render_paths)
    require(rank_rows == render_rows == 64, "Accel result count changed")
    rank_sha256 = sha256_concat(rank_paths)
    require(
        rank_sha256
        == "80d40a71f44fab6bec91ca935697f46f30c18844696cc20f5f2b240d32c436ac",
        "ordered Accel rank ledger changed",
    )

    post_log = (root / "logs/post-calibration.log").read_text(encoding="utf-8")
    finalize_log = (root / "logs/finalize.log").read_text(encoding="utf-8")
    require(
        "view_value_expectation_post_calibration_complete=" in post_log,
        "post-calibration completion marker missing",
    )
    require("view_value_expectation_finalize_complete=" in finalize_log, "finalize completion marker missing")

    wandb = load_json(root / "wandb/wandb_receipt.json")
    require(wandb.get("status") == "PASS", "W&B logging did not pass")
    require(wandb.get("mode") == "offline", "W&B logging was not offline")
    require(wandb.get("media_or_artifact_uploads") is False, "W&B receipt reports uploaded media or artifacts")
    require(Path(wandb["run_directory"]).is_dir(), "W&B offline run directory is missing")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout
    require(not dirty, "repository worktree is dirty; commit the completion state before auditing")

    return {
        "schema": "dsol_view_value_expectation_completion_audit_v1",
        "status": "PASS_COMPLETE",
        "experiment_root": str(root),
        "repository": str(repo),
        "repository_commit": commit,
        "repository_worktree_clean": True,
        "population": population_audit,
        "calibration_runs": calibration_runs,
        "heldout_runs": heldout_runs,
        "frozen_artifact_sha256": immutable_hashes,
        "visibility": {
            "scan_count": 64,
            "candidate_record_count": 7424,
            "summary_sha256": sha256_file(root / "visibility-scan/analysis/summary.json"),
            "diagnostic_sha256": sha256_file(visibility_diagnostic),
        },
        "accel": {
            "rank_count": rank_rows,
            "render_count": render_rows,
            "ordered_rank_ledger_sha256": rank_sha256,
            "ordered_render_ledger_sha256": sha256_concat(render_paths),
        },
        "analysis": {
            "calibration_status": calibration["status"],
            "calibration_source_equal_success_gain_pp": calibration["source_equal_success_gain_pp"],
            "heldout_status": heldout["status"],
            "selector_gate": selector["status"],
            "selector_method": selector["best_rule_frozen_on_calibration"],
            "cross_checkpoint_mean_gain_pp": selector["cross_checkpoint_mean_gain_pp"],
            "cross_checkpoint_mean_harm_probability": selector["cross_checkpoint_mean_harm_probability"],
            "reserve_status": reserve["status"],
            "bank_F_activated": reserve["activate_bank_F"],
        },
        "final_decision": decision,
        "final_report_sha256": report_hashes,
        "wandb": wandb,
        "completion_markers": {
            "post_calibration": True,
            "finalize": True,
        },
        "auditor": str(Path(__file__).resolve()),
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_completion(args.root, args.repo)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
