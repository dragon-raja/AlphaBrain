"""Fail-closed A1 protocol validation and arithmetic planning; never launches jobs.

Artifact hashes are declarations: this module validates their shape and consistency,
not the truth of audit receipts or simulator feasibility. A release check is a
necessary preflight, not authorization to run a robot or experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = "view-acquisition-a1/v1"
SPLITS = ("development", "calibration", "confirmation")
GATES = (
    "input_contract", "training_match", "restore_roundtrip", "no_acquisition_equivalence",
    "camera_pose_sync", "hold_drift", "query_permissions",
    "noise_independence", "failure_denominator", "evidence_controls",
)
_MISSING = object()
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_INPUTS = {
    "candidate_images", "unread_candidate_images", "simulator_state",
    "hidden_branch", "z", "oracle_value", "future_observation", "q_outcome",
}


@dataclass(frozen=True)
class ProtocolIssue:
    code: str
    path: str
    message: str
    severity: str = "error"


class ProtocolValidationError(ValueError):
    def __init__(self, issues: list[ProtocolIssue]):
        self.issues = issues
        super().__init__("; ".join(f"{x.code} at {x.path}: {x.message}" for x in issues))


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Load strict JSON (duplicate keys and non-finite numbers are rejected)."""
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant: {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"Non-finite JSON number: {value}")
        return parsed

    with Path(path).open(encoding="utf-8") as stream:
        result = json.load(stream, object_pairs_hook=pairs, parse_constant=invalid_constant, parse_float=finite_float)
    if not isinstance(result, dict):
        raise ValueError("Protocol root must be an object")
    return result


def protocol_digest(protocol: dict[str, Any]) -> str:
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _at(document: Any, pointer: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return _MISSING
    value = document
    for part in pointer[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            value = value.get(part, _MISSING)
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return _MISSING
    return value


def _is_int(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _is_number(value: Any, minimum: float = 0, *, strict: bool = False) -> bool:
    return (
        type(value) in (int, float) and math.isfinite(value)
        and (value > minimum if strict else value >= minimum)
    )


def validate_protocol(protocol: Any, *, for_release: bool = False) -> list[ProtocolIssue]:
    """Return stable, auditable issues instead of silently completing a draft.

    ``unresolved`` contains exact JSON pointers to null, empty-list, or explicit
    TBD fields. Missing keys are schema errors, not automatically accepted drafts.
    All unresolved fields block ``require_release``.
    """
    issues: list[ProtocolIssue] = []

    def issue(code: str, path: str, message: str, severity: str = "error") -> None:
        item = ProtocolIssue(code, path, message, severity)
        if item not in issues:
            issues.append(item)

    if not isinstance(protocol, dict):
        return [ProtocolIssue("ROOT_TYPE", "/", "Protocol must be an object")]
    try:
        json.dumps(protocol, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return [ProtocolIssue("NON_JSON_VALUE", "/", "Protocol must contain only finite JSON values")]
    unresolved_raw = protocol.get("unresolved", _MISSING)
    unresolved = set()
    if not isinstance(unresolved_raw, list) or any(not isinstance(p, str) for p in unresolved_raw):
        issue("UNRESOLVED_INDEX", "/unresolved", "Provide a list of exact JSON pointers")
    else:
        unresolved = set(unresolved_raw)
        if len(unresolved) != len(unresolved_raw):
            issue("DUPLICATE_UNRESOLVED", "/unresolved", "Pointers must be unique")
        for pointer in sorted(unresolved):
            value = _at(protocol, pointer)
            if value is _MISSING:
                issue("BAD_UNRESOLVED_POINTER", pointer, "Pointer does not name an existing field")
            elif value is not None and value != [] and value not in ("TBD", "UNRESOLVED", ""):
                issue("STALE_UNRESOLVED_POINTER", pointer, "Field has a value; resolve or remove the pointer")

    def get(path: str, check: Any = None, message: str = "Invalid value", default: Any = None) -> Any:
        value = _at(protocol, path)
        if value is _MISSING:
            issue("MISSING_FIELD", path, "Required field is absent")
            return default
        if value is None or value in ("TBD", "UNRESOLVED", "") or (value == [] and path in unresolved):
            if path not in unresolved:
                issue("UNDECLARED_UNRESOLVED", path, "Missing value must be listed in /unresolved")
            issue("UNRESOLVED", path, "Explicitly unresolved; execution release is blocked",
                  "error" if for_release else "unresolved")
            return default
        if check is not None and not check(value):
            issue("INVALID_VALUE", path, message)
            return default
        return value

    def obj(path: str) -> dict[str, Any]:
        return get(path, lambda x: isinstance(x, dict), "Expected object", {})

    def rows(path: str) -> list[Any]:
        return get(path, lambda x: isinstance(x, list), "Expected array", [])

    def string(path: str) -> str | None:
        return get(path, lambda x: isinstance(x, str) and bool(x.strip()), "Expected nonempty string")

    def integer(path: str, minimum: int = 1) -> int | None:
        return get(path, lambda x: _is_int(x, minimum), f"Expected integer >= {minimum}")

    def number(path: str, *, positive: bool = True) -> float | None:
        return get(path, lambda x: _is_number(x, strict=positive), "Expected finite positive/nonnegative number")

    def digest(path: str) -> str | None:
        return get(path, lambda x: isinstance(x, str) and bool(_SHA256.fullmatch(x)), "Expected lowercase SHA-256")

    def exact(path: str, expected: Any) -> Any:
        return get(path, lambda x: type(x) is type(expected) and x == expected, f"Must equal {expected!r}")

    def strings(path: str, *, nonempty: bool = True) -> list[str]:
        value = rows(path)
        if (nonempty and not value and path not in unresolved) or any(not isinstance(x, str) or not x for x in value):
            issue("STRING_LIST", path, "Expected nonempty string IDs")
            return []
        if len(set(value)) != len(value):
            issue("DUPLICATE_ID", path, "IDs must be unique")
        return value

    def vector(path: str, size: int) -> list[float] | None:
        return get(path, lambda x: isinstance(x, list) and len(x) == size and
                   all(type(y) in (int, float) and math.isfinite(y) for y in x), f"Expected {size} finite numbers")

    def unique_rows(path: str, key: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, row in enumerate(rows(path)):
            prefix = f"{path}/{index}"
            if not isinstance(row, dict):
                issue("ROW_TYPE", prefix, "Expected object")
                continue
            identifier = string(f"{prefix}/{key}")
            if identifier is not None:
                if identifier in seen:
                    issue("DUPLICATE_ID", f"{prefix}/{key}", f"Duplicate ID {identifier}")
                seen.add(identifier)
            result.append(row)
        return result

    exact("/schema_version", SCHEMA_VERSION)
    string("/protocol_id")
    status = get("/status", lambda x: x in ("DRAFT", "FROZEN"), "Expected DRAFT or FROZEN")
    if status != "FROZEN":
        issue("NOT_FROZEN", "/status", "Draft cannot pass release validation", "error" if for_release else "warning")
    exact("/scope", "A1_KINEMATIC_EXTERNAL")
    exact("/launch_enabled", False)
    if for_release and unresolved:
        issue("UNRESOLVED_RELEASE", "/unresolved", "Release requires an empty unresolved index")
    obj("/release")
    approved = string("/release/approved_by")
    frozen_at = string("/release/frozen_at_utc")
    if frozen_at is not None:
        try:
            datetime.strptime(frozen_at, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            issue("FROZEN_TIMESTAMP", "/release/frozen_at_utc", "Use a real UTC date YYYY-MM-DDTHH:MM:SSZ")
    if for_release and (not approved or not frozen_at):
        issue("RELEASE_RECEIPT", "/release", "Named approval and freeze timestamp are required")
    obj("/gates")
    for gate in GATES:
        exact(f"/gates/{gate}", True)

    obj("/input_contract")
    contract_id = string("/input_contract/id")
    policy_inputs = strings("/input_contract/policy_inputs")
    selector_inputs = strings("/input_contract/selector_inputs")
    for field, values in (("policy_inputs", policy_inputs), ("selector_inputs", selector_inputs)):
        if _FORBIDDEN_INPUTS.intersection(values):
            issue("PRIVILEGED_INPUT", f"/input_contract/{field}", "Deployment inputs include unread/privileged evidence")
        allowed_inputs = {"external_rgb", "wrist_rgb", "language", "robot_state"}
        if field == "selector_inputs":
            allowed_inputs.update({"camera_geometry", "candidate_action", "action_cost"})
        if not set(values).issubset(allowed_inputs):
            issue("UNKNOWN_INPUT", f"/input_contract/{field}", "Unknown input fields require a new audited contract")
    exact("/input_contract/candidate_images", "forbidden")
    exact("/input_contract/along_path_images", False)
    exact("/input_contract/image_readout", "after_arrival_only")
    exact("/input_contract/query_unit", "observation_bundle")
    if strings("/input_contract/bundle_image_keys") != ["external_rgb", "wrist_rgb"]:
        issue("QUERY_BUNDLE", "/input_contract/bundle_image_keys", "One query reads one external+wrist observation bundle, not one image")
    exact("/input_contract/rules_share_input_permissions", True)
    exact("/input_contract/base_policy_history", "current_frame")
    if "robot_state" in selector_inputs and "robot_state" not in policy_inputs:
        exact("/input_contract/extra_selector_state_declared", True)
    else:
        get("/input_contract/extra_selector_state_declared", lambda x: type(x) is bool, "Expected boolean")
    if not {"external_rgb", "wrist_rgb", "language"}.issubset(policy_inputs):
        issue("POLICY_INTERFACE", "/input_contract/policy_inputs", "A1 contract requires external, wrist and language")

    treatments = unique_rows("/treatments", "id")
    if len(treatments) != 2 or {x.get("coverage") for x in treatments if isinstance(x.get("coverage"), str)} != {"narrow", "broad"}:
        issue("TREATMENT_PAIR", "/treatments", "Require exactly one narrow and one broad treatment")
    match_fields = ("initial_checkpoint_sha256", "training_sources_sha256", "action_labels_sha256",
                    "unique_states", "image_exposures", "optimizer_updates")
    matched: dict[str, list[Any]] = {key: [] for key in match_fields}
    optimizer_contracts: list[dict[str, Any]] = []
    checkpoint_hashes: list[str] = []
    for index, treatment in enumerate(treatments):
        prefix = f"/treatments/{index}"
        checkpoint_path = string(f"{prefix}/checkpoint_path")
        if checkpoint_path is not None and not Path(checkpoint_path).is_absolute():
            issue("CHECKPOINT_PATH", prefix, "Checkpoint path must be absolute")
        checkpoint_hash = digest(f"{prefix}/checkpoint_sha256")
        if checkpoint_hash:
            checkpoint_hashes.append(checkpoint_hash)
        digest(f"{prefix}/training_manifest_sha256")
        if string(f"{prefix}/input_contract_id") != contract_id:
            issue("INPUT_CONTRACT_MISMATCH", prefix, "Both treatments must use the declared common input contract")
        for field in match_fields:
            value = digest(f"{prefix}/{field}") if field.endswith("sha256") else integer(f"{prefix}/{field}")
            if value is not None:
                matched[field].append(value)
        strings(f"{prefix}/noise_bank_ids")
        optimizer = obj(f"{prefix}/optimizer_contract")
        string(f"{prefix}/optimizer_contract/name")
        string(f"{prefix}/optimizer_contract/schedule")
        number(f"{prefix}/optimizer_contract/peak_learning_rate")
        number(f"{prefix}/optimizer_contract/end_learning_rate", positive=False)
        number(f"{prefix}/optimizer_contract/weight_decay", positive=False)
        integer(f"{prefix}/optimizer_contract/schedule_steps")
        integer(f"{prefix}/optimizer_contract/warmup_steps", 0)
        optimizer_contracts.append(optimizer)
    for field, values in matched.items():
        if len(values) == 2 and values[0] != values[1]:
            issue("TREATMENT_MATCH", f"/treatments/*/{field}", "Narrow/Broad must match; otherwise declare a different recipe protocol")
    if len(checkpoint_hashes) == 2 and checkpoint_hashes[0] == checkpoint_hashes[1]:
        issue("IDENTICAL_CHECKPOINTS", "/treatments", "Distinct training treatments resolve to the same checkpoint hash")
    if len(optimizer_contracts) == 2 and optimizer_contracts[0] != optimizer_contracts[1]:
        issue("OPTIMIZER_MISMATCH", "/treatments/*/optimizer_contract", "Compare effective optimizer/schedule parameters, not only update counts or file hashes")

    obj("/noise")
    exact("/noise/shared_across_treatments", True)
    exact("/noise/acquisition_consumes_base_policy_noise", False)
    if strings("/noise/key_fields") != ["source_id", "bank_id", "repeat", "base_policy_call"]:
        issue("NOISE_KEY", "/noise/key_fields", "Key must isolate source/bank/repeat and base-policy call index")
    if set(strings("/noise/independent_streams")) != {"base_policy", "camera", "selector"}:
        issue("NOISE_STREAMS", "/noise/independent_streams", "Use independent base_policy, camera and selector streams")
    banks = unique_rows("/noise/banks", "id")
    if {x.get("role") for x in banks if isinstance(x.get("role"), str)} != {"discovery", "refinement", "confirmation"} or len(banks) != 3:
        issue("NOISE_ROLES", "/noise/banks", "Declare separate discovery/refinement/confirmation banks")
    seeds: list[int] = []
    hashes: list[str] = []
    bank_ids = {x.get("id") for x in banks if isinstance(x.get("id"), str)}
    for index, bank in enumerate(banks):
        prefix = f"/noise/banks/{index}"
        seed, bank_hash = integer(f"{prefix}/seed", 0), digest(f"{prefix}/sha256")
        integer(f"{prefix}/repeats")
        if seed is not None:
            seeds.append(seed)
        if bank_hash:
            hashes.append(bank_hash)
    if len(set(seeds)) != len(seeds) or len(set(hashes)) != len(hashes):
        issue("NOISE_BANK_REUSE", "/noise/banks", "Discovery/refinement/confirmation bank seeds and hashes must differ")
    for index, treatment in enumerate(treatments):
        ids = treatment.get("noise_bank_ids")
        if isinstance(ids, list) and all(isinstance(x, str) for x in ids) and set(ids) != bank_ids:
            issue("NOISE_TREATMENT_MISMATCH", f"/treatments/{index}/noise_bank_ids", "Treatments must declare the same complete bank IDs")

    sources = unique_rows("/sources", "source_id")
    if not sources and "/sources" not in unresolved:
        issue("EMPTY_SOURCE_MANIFEST", "/sources", "Resolved protocols need actual source records, not only planning counts")
    source_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    cluster_splits: dict[str, set[str]] = {}
    for index, source in enumerate(sources):
        prefix = f"/sources/{index}"
        cluster = string(f"{prefix}/cluster_id")
        string(f"{prefix}/task_id")
        split = get(f"{prefix}/split", lambda x: isinstance(x, str) and x in SPLITS, "Unknown source split")
        strings(f"{prefix}/branches")
        prior = get(f"{prefix}/seen_in_prior_research", lambda x: type(x) is bool, "Expected boolean")
        if split:
            source_by_split[split].append(source)
            if cluster:
                cluster_splits.setdefault(cluster, set()).add(split)
        if split == "confirmation" and prior is True:
            issue("HISTORICAL_CONFIRMATION", prefix, "Final confirmation sources must not be historical research data")
    for cluster, splits in sorted(cluster_splits.items()):
        if len(splits) > 1:
            issue("CLUSTER_LEAKAGE", "/sources", f"Cluster {cluster!r} occurs across {sorted(splits)}")
    digest("/source_manifest_sha256")
    counts = obj("/planning_counts")
    for split in SPLITS:
        obj(f"/planning_counts/{split}")
        expected_clusters = integer(f"/planning_counts/{split}/base_clusters", 0)
        expected_sources = integer(f"/planning_counts/{split}/sources", 0)
        expected_branches = integer(f"/planning_counts/{split}/branches_per_source")
        if expected_clusters is not None and expected_sources is not None and expected_clusters > expected_sources:
            issue("COUNT_ORDER", f"/planning_counts/{split}", "Clusters cannot exceed sources")
        actual = source_by_split[split]
        if sources and expected_sources is not None:
            actual_clusters = {x.get("cluster_id") for x in actual if isinstance(x.get("cluster_id"), str)}
            if len(actual) != expected_sources or len(actual_clusters) != expected_clusters:
                issue("SOURCE_COUNT_MISMATCH", f"/planning_counts/{split}", "Planning counts disagree with resolved source manifest")
            if any(isinstance(x.get("branches"), list) and len(x["branches"]) != expected_branches for x in actual):
                issue("BRANCH_COUNT_MISMATCH", f"/planning_counts/{split}", "Branches per source disagree with planning counts")

    obj("/environment")
    number("/environment/control_frequency_hz")
    total = integer("/environment/total_steps")
    trigger = integer("/environment/trigger_step", 0)
    exact("/environment/acquisition_consumes_horizon", True)
    exact("/environment/hold_controller", "maintain_targets")
    number("/environment/max_hold_position_error_m")
    number("/environment/max_hold_rotation_error_rad")
    obj("/acquisition")
    exact("/acquisition/camera_mode", "kinematic_external")
    exact("/acquisition/same_duration_required", True)
    obj("/acquisition/speed_limits")
    for field in ("linear_m_s", "angular_rad_s", "linear_acceleration_m_s2", "angular_acceleration_rad_s2"):
        number(f"/acquisition/speed_limits/{field}")
    obj("/acquisition/workspace_bounds")
    lower = vector("/acquisition/workspace_bounds/min_xyz", 3)
    upper = vector("/acquisition/workspace_bounds/max_xyz", 3)
    if lower and upper and any(lo >= hi for lo, hi in zip(lower, upper)):
        issue("WORKSPACE_BOUNDS", "/acquisition/workspace_bounds", "Each lower bound must be smaller than upper bound")
    actions = unique_rows("/acquisition/actions", "id")
    kinds = [x.get("kind") for x in actions]
    if len(actions) != 6 or kinds.count("continue") != 1 or kinds.count("hold") != 1 or kinds.count("move") != 4:
        issue("ACTION_SET", "/acquisition/actions", "A1 requires continue + hold + exactly four move actions")
    durations: list[int] = []
    for index, action in enumerate(actions):
        prefix = f"/acquisition/actions/{index}"
        if set(action) - {"id", "kind", "duration_steps", "path_interpolation", "target_pose"}:
            issue("UNKNOWN_ACTION_FIELD", prefix, "Unknown action parameters cannot be silently ignored")
        kind = get(f"{prefix}/kind", lambda x: x in ("continue", "hold", "move"), "Unknown action kind")
        duration = integer(f"{prefix}/duration_steps", 0 if kind == "continue" else 1)
        if kind == "continue" and duration != 0:
            issue("CONTINUE_DELAY", prefix, "Continue must not receive a forced wait")
        elif kind in ("hold", "move") and duration is not None:
            durations.append(duration)
        if kind == "move":
            exact(f"{prefix}/path_interpolation", "quintic_xyz_slerp")
            obj(f"{prefix}/target_pose")
            position = vector(f"{prefix}/target_pose/position_xyz", 3)
            quaternion = vector(f"{prefix}/target_pose/quaternion_wxyz", 4)
            if quaternion is not None and not math.isclose(sum(x*x for x in quaternion), 1.0, rel_tol=0, abs_tol=1e-6):
                issue("QUATERNION_NORM", f"{prefix}/target_pose/quaternion_wxyz", "Quaternion must have unit norm")
            if position and lower and upper and any(p < lo or p > hi for p, lo, hi in zip(position, lower, upper)):
                issue("TARGET_OUTSIDE_WORKSPACE", prefix, "Move endpoint is outside declared workspace")
    if len(set(durations)) > 1:
        issue("DURATION_MISMATCH", "/acquisition/actions", "Hold and all four moves must have identical durations")
    if total is not None and trigger is not None:
        if trigger >= total or (durations and trigger + max(durations) >= total):
            issue("HORIZON_EXHAUSTED", "/environment", "Trigger/acquisition must leave positive continuation budget")

    blocks = strings("/evidence_blocks")
    if set(blocks) != {"equivalent_projection", "reveal", "matched_negative"}:
        issue("EVIDENCE_BLOCKS", "/evidence_blocks", "Declare all three evidence blocks; do not infer them from Q outcomes")
    exact("/evidence_blocks_frozen_before_outcomes", True)
    controls = strings("/diagnostic_control_ids")
    rule_ids = strings("/rule_ids")
    groups = unique_rows("/evaluation_groups", "id")
    if not groups:
        issue("NO_EVALUATION_GROUPS", "/evaluation_groups", "Specify at least one countable evaluation group")
    action_ids = {x.get("id") for x in actions if isinstance(x.get("id"), str)}
    bank_map = {x["id"]: x for x in banks if isinstance(x.get("id"), str)}
    evaluation_keys: set[tuple[str, str, str, str]] = set()
    for index, group in enumerate(groups):
        prefix = f"/evaluation_groups/{index}"
        split = get(f"{prefix}/split", lambda x: isinstance(x, str) and x in SPLITS, "Unknown source split")
        bank_id = string(f"{prefix}/bank_id")
        kind = get(f"{prefix}/kind", lambda x: x in ("actions", "controls", "rules"), "Unknown evaluation kind")
        members = strings(f"{prefix}/member_ids")
        allowed = {"actions": action_ids, "controls": set(controls), "rules": set(rule_ids)}.get(kind, set())
        if not set(members).issubset(allowed):
            issue("UNKNOWN_EVALUATION_MEMBER", prefix, "Group names undeclared actions/controls/rules")
        if bank_id not in bank_map:
            issue("UNKNOWN_NOISE_BANK", prefix, "Group refers to an undeclared bank")
        if split and bank_id and kind:
            for member in members:
                key = (split, bank_id, kind, member)
                if key in evaluation_keys:
                    issue("DUPLICATE_EVALUATION", prefix, "Same split/bank/condition is listed twice")
                evaluation_keys.add(key)
        if split == "confirmation":
            if bank_id in bank_map and bank_map[bank_id].get("role") != "confirmation":
                issue("CONFIRMATION_BANK", prefix, "Final confirmation must use the independent confirmation bank")
            digest(f"{prefix}/frozen_rule_manifest_sha256")
            exact(f"{prefix}/rules_frozen_before_confirmation", True)
        if split and isinstance(counts.get(split), dict) and counts[split].get("sources") == 0:
            issue("EMPTY_EVALUATION_SPLIT", prefix, "Cannot evaluate an empty planned split")

    contrasts = unique_rows("/primary_contrasts", "id")
    if not contrasts:
        issue("NO_PRIMARY_CONTRASTS", "/primary_contrasts", "Specify primary estimands before release")
    for index, contrast in enumerate(contrasts):
        prefix = f"/primary_contrasts/{index}"
        for side in ("left_rule", "right_rule"):
            if string(f"{prefix}/{side}") not in rule_ids:
                issue("CONTRAST_RULE", f"{prefix}/{side}", "Contrast refers to an unknown rule")
        if contrast.get("left_rule") == contrast.get("right_rule"):
            issue("SELF_CONTRAST", prefix, "A rule cannot be compared with itself")
        get(f"{prefix}/estimand", lambda x: x in ("success_difference", "net_value_difference", "training_interaction"), "Unknown estimand")
        get(f"{prefix}/minimum_effect", lambda x: _is_number(x) and x <= 1, "Effect threshold must be in [0,1]")
    obj("/analysis")
    exact("/analysis/cluster_unit", "cluster_id")
    string("/analysis/interval_method")
    string("/analysis/multiplicity")
    exact("/analysis/fit_and_evaluate_same_samples", False)
    exact("/analysis/grouped_development_validation", True)
    get("/analysis/cost_weights", lambda x: isinstance(x, dict) and set(x) == {"elapsed_seconds", "observation_queries"}
        and all(_is_number(v) for v in x.values()), "Specify finite nonnegative seconds/query weights")
    obj("/budget")
    cap = integer("/budget/max_episodes")
    exact("/budget/planning_failures_in_denominator", True)
    estimate = _episode_counts(protocol)
    if cap is not None and estimate["planned_episodes"] is not None and estimate["planned_episodes"] > cap:
        issue("EPISODE_CAP", "/budget/max_episodes", "Planned episodes exceed frozen maximum")
    if for_release and estimate["planned_episodes"] is None:
        issue("UNCOUNTABLE_PLAN", "/evaluation_groups", "Release requires fully computable episode counts")
    return issues


def _episode_counts(protocol: dict[str, Any]) -> dict[str, Any]:
    """Only arithmetic. Counts remain estimates until source manifests resolve."""
    counts = protocol.get("planning_counts", {})
    noise = protocol.get("noise", {})
    banks = noise.get("banks", []) if isinstance(noise, dict) else []
    bank_map = {b["id"]: b for b in banks if isinstance(b, dict) and isinstance(b.get("id"), str)} if isinstance(banks, list) else {}
    treatments = protocol.get("treatments", [])
    treatment_count = len(treatments) if isinstance(treatments, list) else 0
    blocks = protocol.get("evidence_blocks", [])
    block_count = len(blocks) if isinstance(blocks, list) else 0
    result: list[dict[str, Any]] = []
    groups = protocol.get("evaluation_groups", [])
    if not isinstance(groups, list):
        groups = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        split = group.get("split")
        sample = counts.get(split, {}) if isinstance(counts, dict) and isinstance(split, str) else {}
        sample = sample if isinstance(sample, dict) else {}
        bank_id = group.get("bank_id")
        bank = bank_map.get(bank_id, {}) if isinstance(bank_id, str) else {}
        members = group.get("member_ids", [])
        member_count = len(members) if isinstance(members, list) else 0
        factors = [sample.get("sources"), sample.get("branches_per_source"), block_count,
                   treatment_count, member_count, bank.get("repeats")]
        valid = all(_is_int(value, 1) for value in factors)
        episodes = math.prod(factors) if valid else None
        result.append({"id": group.get("id"), "split": split, "kind": group.get("kind"),
                       "base_clusters": sample.get("base_clusters"), "source_records": sample.get("sources"),
                       "branch_instances": sample.get("sources", 0) * sample.get("branches_per_source", 0)
                       if all(_is_int(sample.get(k)) for k in ("sources", "branches_per_source")) else None,
                       "evidence_blocks": block_count, "treatments": treatment_count,
                       "conditions": member_count, "noise_repeats": bank.get("repeats"), "episodes": episodes})
    total = sum(item["episodes"] for item in result) if result and all(x["episodes"] is not None for x in result) else None
    sources = protocol.get("sources", [])
    source_list = sources if isinstance(sources, list) else []
    return {"groups": result, "planned_episodes": total,
            "resolved_unique_clusters": len({s["cluster_id"] for s in source_list if isinstance(s, dict) and isinstance(s.get("cluster_id"), str)}),
            "resolved_source_records": len(source_list),
            "count_basis": "resolved_manifest" if source_list else "draft_planning_counts_only",
            "independence_note": "Branches, evidence blocks, treatments and noise repeats are NOT independent source clusters; group counts must not be summed as independent clusters."}


def build_plan(protocol: dict[str, Any]) -> dict[str, Any]:
    issues = validate_protocol(protocol)
    release_issues = validate_protocol(protocol, for_release=True)
    try:
        digest = protocol_digest(protocol)
    except (TypeError, ValueError):
        digest = None
    return {"schema_version": SCHEMA_VERSION, "protocol_id": protocol.get("protocol_id"),
            "protocol_sha256": digest, "launch_enabled": False,
            "draft_schema_valid": not any(x.severity == "error" for x in issues),
            "release_valid": not any(x.severity != "warning" for x in release_issues),
            "issues": [asdict(x) for x in issues], "release_issues": [asdict(x) for x in release_issues],
            **_episode_counts(protocol),
            "limits": ["No jobs or simulators are launched.",
                       "Hashes and gate receipts are declarations, not independently verified artifacts.",
                       "Runtime path feasibility, controller behavior and observation permissions require separate runner audits."]}


def require_release(protocol: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on any unresolved/error issue. Never starts an experiment."""
    issues = validate_protocol(protocol, for_release=True)
    blockers = [issue for issue in issues if issue.severity != "warning"]
    if blockers:
        raise ProtocolValidationError(blockers)
    return build_plan(protocol)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--require-release", action="store_true", help="Validate release conditions; still never launch")
    args = parser.parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        plan = build_plan(protocol)
    except (OSError, TypeError, ValueError) as error:
        print(json.dumps({"launch_enabled": False, "error": str(error)}))
        return 2
    print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if (plan["release_valid"] if args.require_release else plan["draft_schema_valid"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
