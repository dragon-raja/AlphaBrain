"""Synthetic protocol fixtures shared by protocol and execution regressions."""
from copy import deepcopy

from scripts.dsol_paper1.protocols.build_matched_view_landscape_protocol_v1 import EXPECTED_CANDIDATES


def population_fixture():
    states = [
        {"task_id": f"task-{task}", "source_group": f"task-{task}::demo-{demo}",
         "pair_key": f"oracle-v2::development::task-{task}::demo-{demo}",
         "asset_source_pair_key": f"legacy::{task}::{demo}",
         "split": "development", "v2_role": "development",
         "environment_seed": task * 10 + demo, "construction_spec_sha256": "a" * 64,
         "source_state_index": 17, "static_assets": {"physics_state_sha256": "b" * 64}}
        for task in range(8) for demo in range(6)
    ]
    return {"status": "PASS", "population": {"development": {"states": states},
                                                "test": {"states": ["MUST_NOT_BE_READ"]}}}


def protocol_fixture(population):
    blocks = [{
        "state": deepcopy(state), "scene_construction": {"sha256": "c" * 64},
        "candidates": [{"selected_candidate_id": candidate, "pose": {"yaw": index},
                        "candidate_features": {"visibility_score": index / 97}}
                       for index, candidate in enumerate(EXPECTED_CANDIDATES)],
    } for state in population["population"]["development"]["states"]]
    return [{
        "schema": "dsol_compact_view_matrix_protocol_v1", "status": "PASS_FROZEN",
        "role": "development", "noise_bank_id": "O",
        "policy_repeat_ids": list(range(index * 4, index * 4 + 4)),
        "state_blocks": deepcopy(blocks), "catalog": "/fake/catalog.json",
        "catalog_sha256": "d" * 64, "population": "/fake/population.json",
        "population_sha256": "e" * 64, "sensor_control": "both",
    } for index in range(8)]
