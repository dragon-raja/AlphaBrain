from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.dsol_paper1 import run_matched_view_accel_v1 as scorer


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def ranking_fixture():
    rows = []
    for index, candidate in enumerate(sorted(scorer.EXPECTED_CANDIDATES)):
        scores = np.asarray([0.1 + index / 100 + member / 1000 for member in range(8)])
        rows.append({"candidate_id": candidate, "member_accel_3": scores.tolist(),
                     "mean_accel_3": float(scores.mean()), "std_accel_3": float(scores.std())})
    return {"ranking": rows, "selected_candidate_id": rows[0]["candidate_id"]}


def state_fixture(index=0):
    return {
        "pair_key": f"oracle-v2::development::task-{index}::demo-1",
        "asset_source_pair_key": f"legacy::heldout_test::task-{index}::demo-1",
        "task_id": f"task-{index}", "source_group": f"task-{index}::demo-1",
        "split": "development", "v2_role": "development",
        "static_assets": {"physics_state_sha256": "a" * 64, "policy_inputs_sha256": "b" * 64},
    }


def completed_fixture(state, manifest_hash):
    return {
        **ranking_fixture(), "status": "PASS", "pair_key": state["asset_source_pair_key"],
        "population_pair_key": state["pair_key"], "manifest_sha256": manifest_hash,
        "checkpoint_sha256": scorer.EXPECTED_WEIGHT_HASH, "candidate_count": 97,
        "ensemble_size": 8, "ensemble_seeds": scorer.seed_list(state["asset_source_pair_key"]),
        "render_artifact_sha256": state["static_assets"]["policy_inputs_sha256"],
        "physics_state_sha256": state["static_assets"]["physics_state_sha256"],
    }


def asset_fixture(tmp_path):
    state = state_fixture()
    artifact = tmp_path / "policy_inputs.npz"
    np.savez_compressed(
        artifact, candidate_ids=np.asarray(sorted(scorer.EXPECTED_CANDIDATES)),
        external_images=np.zeros((97, 224, 224, 3), dtype=np.uint8),
        wrist_images=np.zeros((97, 224, 224, 3), dtype=np.uint8),
        robot_states=np.zeros((97, 8), dtype=np.float32),
        camera_intrinsics=np.zeros((97, 3, 3)), camera_to_world_opencv=np.zeros((97, 4, 4)),
    )
    render_path, broad_path = tmp_path / "render.json", tmp_path / "ranking.json"
    render = {"status": "PASS", "candidate_count": 97, "pair_key": state["asset_source_pair_key"],
              "artifact": str(artifact), "artifact_sha256": scorer.sha256_file(artifact),
              "physics_state_sha256": "a" * 64, "task_id": state["task_id"],
              "source_group": state["source_group"]}
    write_json(render_path, render)
    state["static_assets"].update(
        render_receipt=str(render_path), render_receipt_sha256=scorer.sha256_file(render_path),
        policy_inputs=str(artifact), policy_inputs_sha256=scorer.sha256_file(artifact),
    )
    broad = {**ranking_fixture(), "status": "PASS", "pair_key": state["asset_source_pair_key"],
             "task_id": state["task_id"], "source_group": state["source_group"],
             "ensemble_seeds": scorer.seed_list(state["asset_source_pair_key"]), "ensemble_size": 8,
             "candidate_count": 97, "render_artifact_sha256": render["artifact_sha256"],
             "physics_state_sha256": "a" * 64, "checkpoint_action_horizon": 10}
    write_json(broad_path, broad)
    return state, broad_path


def test_code_identities_point_to_real_accel_implementation():
    identities = scorer.code_identities()
    paths = [row["path"] for row in identities]
    assert any(path.endswith("scripts/dsol_paper1/accel_inference.py") for path in paths)
    assert not any(path.endswith("scripts/cabi_vla/accel_inference.py") for path in paths)
    assert any(path.endswith("accel_core.py") for path in paths)


def test_legacy_seed_list_matches_actual_historical_broad_receipt():
    key = "expectation-v1::heldout_test::goal_cream_cheese_bowl::demo_21::frame-00092"
    assert scorer.seed_list(key) == [1356888342, 912500968, 3950629998, 2217870736,
                                      77349860, 3937220499, 2402935644, 2963028792]
    assert scorer.seed_list("oracle-v2::development::" + key) != scorer.seed_list(key)


def test_actual_cached_artifact_matrix_and_legacy_seed_validation(tmp_path):
    state, broad_path = asset_fixture(tmp_path)
    assert scorer.validate_assets(state) == scorer.identity(broad_path)
    broad = scorer.read(broad_path)
    broad["ensemble_seeds"][0] += 1
    write_json(broad_path, broad)
    with pytest.raises(ValueError, match="frozen seeds"):
        scorer.validate_assets(state)


def test_cached_artifact_hash_and_physics_mismatch_rejected(tmp_path):
    state, _ = asset_fixture(tmp_path)
    bad = deepcopy(state)
    bad["static_assets"]["policy_inputs_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash changed"):
        scorer.validate_assets(bad)
    bad = deepcopy(state)
    bad["static_assets"]["physics_state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="physical state mismatch"):
        scorer.validate_assets(bad)


@pytest.mark.parametrize("mutation", ["duplicate", "short_members", "nan", "bad_mean", "wrong_selection"])
def test_incomplete_or_invalid_97_by_8_scores_rejected(mutation):
    row = ranking_fixture()
    if mutation == "duplicate":
        row["ranking"][1]["candidate_id"] = row["ranking"][0]["candidate_id"]
    elif mutation == "short_members":
        row["ranking"][0]["member_accel_3"].pop()
    elif mutation == "nan":
        row["ranking"][0]["member_accel_3"][0] = float("nan")
    elif mutation == "bad_mean":
        row["ranking"][0]["mean_accel_3"] += 1
    else:
        row["selected_candidate_id"] = "canonical"
    with pytest.raises(ValueError):
        scorer.validate_ranking_matrix(row)


def test_existing_result_requires_full_matching_identity():
    state = state_fixture()
    row = completed_fixture(state, "c" * 64)
    scorer.validate_result(row, state, "c" * 64)
    for key in ("pair_key", "population_pair_key", "checkpoint_sha256", "manifest_sha256",
                "physics_state_sha256", "render_artifact_sha256"):
        bad = deepcopy(row)
        bad[key] = "different"
        with pytest.raises(ValueError, match="refusing to overwrite"):
            scorer.validate_result(bad, state, "c" * 64)


def prepared_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer, "ROOT", tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"small synthetic weight file, never loaded")
    weight_hash = scorer.sha256_file(weights)
    monkeypatch.setattr(scorer, "EXPECTED_WEIGHT_HASH", weight_hash)
    write_json(checkpoint / "framework_config.yaml", {"framework": {
        "name": "PaliGemmaPi05", "pi05": True, "gripper_remap": False,
        "action_model": {"action_dim": 7, "action_horizon": 10, "state_dim": 8,
                         "num_inference_steps": 10},
    }})
    write_json(checkpoint / "vlm_pretrained" / "tokenizer.json", {"fixture": True})
    states = [state_fixture(index) for index in range(8)]
    selection = tmp_path / "selection.json"
    receipt = tmp_path / "receipt.json"
    write_json(selection, {"status": "FROZEN_OUTCOME_BLIND_DEVELOPMENT_SUBSET", "states": states})
    write_json(receipt, {"status": "COMPLETED_ARTIFACT_AUDIT_PASS",
                         "final_weights": {"path": str(weights), "content_sha256": weight_hash}})
    broad_identity = {"path": "/fixture/broad.json", "sha256": "9" * 64}
    monkeypatch.setattr(scorer, "validate_assets", lambda _state: broad_identity)
    return selection, receipt, tmp_path / "canonical-accel", weights


def test_prepare_hashes_weights_once_and_never_overwrites(tmp_path, monkeypatch):
    selection, receipt, output, weights = prepared_fixture(tmp_path, monkeypatch)
    actual_sha = scorer.sha256_file
    calls = []

    def tracked_sha(path):
        calls.append(Path(path))
        return actual_sha(path)

    monkeypatch.setattr(scorer, "sha256_file", tracked_sha)
    manifest_path = scorer.prepare(selection, receipt, output)
    assert calls.count(weights) == 1
    before = manifest_path.read_bytes()
    with pytest.raises(FileExistsError):
        scorer.prepare(selection, receipt, output)
    assert calls.count(weights) == 1
    assert manifest_path.read_bytes() == before


def test_shard_resumes_without_weight_rehash_or_model_load(tmp_path, monkeypatch):
    selection, receipt, output, weights = prepared_fixture(tmp_path, monkeypatch)
    manifest_path = scorer.prepare(selection, receipt, output)
    manifest = scorer.read(manifest_path)
    state = manifest["states"][0]
    row = completed_fixture(state, scorer.sha256_file(manifest_path))
    state_dir = output / "states" / hashlib.sha256(state["asset_source_pair_key"].encode()).hexdigest()[:20]
    write_json(state_dir / "ranking.json", row)
    ledger = output / "rank-shard-00.jsonl"
    ledger.write_text(json.dumps(row) + "\n")
    original = ledger.read_bytes()
    actual_sha = scorer.sha256_file

    def prohibit_weight_hash(path):
        assert Path(path) != weights, "shards must not rehash the full checkpoint"
        return actual_sha(path)

    monkeypatch.setattr(scorer, "sha256_file", prohibit_weight_hash)
    monkeypatch.setattr(scorer, "rank_state", lambda *_a, **_kw: pytest.fail("resuming must not score again"))
    scorer.run(manifest_path, 0, 8, "cpu")
    assert ledger.read_bytes() == original
    with pytest.raises(ValueError, match="shard allocation"):
        scorer.run(manifest_path, 0, 1, "cpu")
    tokenizer = weights.parent / "vlm_pretrained" / "tokenizer.json"
    tokenizer.write_text("changed")
    with pytest.raises(ValueError, match="tokenizer/configuration"):
        scorer.run(manifest_path, 0, 8, "cpu")


def test_output_is_restricted_to_new_landscape_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer, "ROOT", tmp_path)
    scorer.validate_output_root(tmp_path / "canonical-accel")
    with pytest.raises(ValueError, match="new landscape root"):
        scorer.validate_output_root(tmp_path / "historical-broad")
    with pytest.raises(ValueError, match="new landscape root"):
        scorer.validate_output_root(tmp_path.parent / "old-experiment" / "canonical-accel")


def test_wrong_denoising_configuration_is_rejected_before_weight_hash(tmp_path, monkeypatch):
    selection, receipt, output, weights = prepared_fixture(tmp_path, monkeypatch)
    config_path = weights.parent / "framework_config.yaml"
    config = scorer.read(config_path)
    config["framework"]["action_model"]["num_inference_steps"] = 5
    write_json(config_path, config)
    with pytest.raises(ValueError, match="num_inference_steps"):
        scorer.prepare(selection, receipt, output)
    assert not output.exists()
