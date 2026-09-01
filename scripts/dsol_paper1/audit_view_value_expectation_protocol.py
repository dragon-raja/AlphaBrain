#!/usr/bin/env python3
"""Fail-closed static audit for the frozen view-value expectation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "dsol_view_value_expectation_protocol_v1"
STATUS = "FROZEN_PREREGISTRATION_RUNNER_HOLD"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    require(payload.get("schema") == SCHEMA, "unexpected protocol schema")
    require(payload.get("status") == STATUS, "protocol must remain runner HOLD")

    randomness = payload["randomness_contract"]
    environment = randomness["environment_seed"]
    flow = randomness["policy_flow_noise"]
    require(environment["must_not_be_derived_from_policy_noise_seed"], "environment and policy seeds must be separate")
    require(flow["distribution"] == "iid_standard_normal", "formal flow noise must match deployment distribution")
    require(flow["explicit_tensor_injection_required"], "seed-only policy noise is not releasable")
    require(flow["shared_exactly_across_views_within_state_repeat_and_replan_index"], "paired views must share noise")
    require(flow["independent_across_repeat_and_replan_index"], "noise repeats and replans must be independent")
    banks = randomness["noise_banks"]
    required_banks = {
        "screen_all97",
        "screen_top24",
        "screen_top6",
        "calibration_confirmation",
        "heldout_primary",
        "heldout_precision_reserve",
    }
    require(required_banks <= set(banks), "noise bank partition contract is incomplete")
    require(len({banks[name] for name in required_banks}) == len(required_banks), "noise bank identities must be unique")
    require(not banks["bank_overlap_allowed"], "noise banks must not overlap")

    population = payload["population"]
    require(population["source_group_disjoint_between_calibration_and_test"], "calibration/test sources must be disjoint")
    require(population["state_disjoint_between_calibration_and_test"], "calibration/test states must be disjoint")
    require(population["heldout_test"]["physical_state_count"] > population["calibration"]["physical_state_count"], "test needs more states than dense calibration")

    candidate_count = int(payload["candidate_space"]["candidate_count"])
    composition = payload["candidate_space"]["composition"]
    require(sum(int(value) for value in composition.values()) == candidate_count, "candidate composition does not sum to catalog size")

    calibration = payload["calibration_design"]
    stage_counts = [
        int(calibration["screen_all97"]["candidate_count"]),
        int(calibration["screen_top24"]["candidate_count_including_canonical"]),
        int(calibration["screen_top6"]["candidate_count_including_canonical"]),
        len(calibration["confirmation"]["conditions"]),
    ]
    require(stage_counts[0] == candidate_count, "first stage must screen the full catalog")
    require(all(left > right for left, right in zip(stage_counts, stage_counts[1:])), "candidate stages must strictly shrink")
    require(calibration["confirmation"]["candidate_must_be_frozen_before_bank_D_is_opened"], "confirmation candidate must be frozen")
    require(not calibration["posthoc_candidate_replacement_after_confirmation"], "posthoc candidate replacement is forbidden")

    heldout = payload["heldout_selector_analysis"]
    require(heldout["primary_noise_sequences_per_condition"] == heldout["reserve_noise_sequences_per_condition"], "primary and reserve precision banks must be balanced")
    require(heldout["reserve_activation_rule"]["decision_must_be_machine_generated_before_bank_F_is_opened"], "reserve activation must be auditable")

    statistics = payload["statistics"]
    require(statistics["independent_unit"] == "source_demonstration_group", "noise repeats must not become independent units")
    require(statistics["noise_repeats_are_not_independent_task_samples"], "noise pseudoreplication guard is required")
    headroom = payload["claim_gates"]["stable_candidate_space_headroom"]
    require(0 < headroom["minimum_confirmed_strong_state_fraction"] < 1, "headroom prevalence gate must be a fraction")
    require(headroom["minimum_confirmed_source_groups"] >= 4, "headroom must span multiple source groups")
    require(headroom["minimum_confirmed_tasks"] >= 3, "headroom must span multiple tasks")

    calibration_states = int(population["calibration"]["physical_state_count"])
    calibration_episodes = calibration_states * (
        stage_counts[0] * int(calibration["screen_all97"]["new_full_episode_noise_sequences_per_candidate"])
        + stage_counts[1] * int(calibration["screen_top24"]["new_full_episode_noise_sequences_per_candidate"])
        + stage_counts[2] * int(calibration["screen_top6"]["new_full_episode_noise_sequences_per_candidate"])
        + stage_counts[3] * int(calibration["confirmation"]["fixed_full_episode_noise_sequences_per_condition"])
    )
    test_states = int(population["heldout_test"]["physical_state_count"])
    seed41_conditions = len(heldout["checkpoint_seed_41_full_baselines"])
    confirm_conditions = len(heldout["checkpoint_seeds_42_43_confirmation"])
    primary = int(heldout["primary_noise_sequences_per_condition"])
    reserve = int(heldout["reserve_noise_sequences_per_condition"])
    heldout_primary_episodes = test_states * primary * (seed41_conditions + 2 * confirm_conditions)
    heldout_max_episodes = test_states * (primary + reserve) * (seed41_conditions + 2 * confirm_conditions)

    return {
        "schema": "dsol_view_value_expectation_protocol_audit_v1",
        "status": "PASS_STATIC_DESIGN_RUNNER_STILL_HOLD",
        "protocol_schema": SCHEMA,
        "protocol_status": STATUS,
        "calibration_projected_episodes": calibration_episodes,
        "heldout_primary_projected_episodes": heldout_primary_episodes,
        "heldout_max_with_reserve_projected_episodes": heldout_max_episodes,
        "projected_primary_total_episodes": calibration_episodes + heldout_primary_episodes,
        "projected_max_total_episodes": calibration_episodes + heldout_max_episodes,
        "formal_execution_authorized": False,
        "next_required_release": "explicit_noise_runner_and_manifest_audit_receipt",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.protocol.read_text())
    result = audit(payload)
    result["protocol"] = str(args.protocol.resolve())
    result["protocol_sha256"] = sha256(args.protocol)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
