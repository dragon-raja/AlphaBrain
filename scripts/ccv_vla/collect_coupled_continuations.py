from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ccv import (
    assert_deployable_arrays,
    best_candidate_index,
    coupled_policy_continuations,
    frozen_source_split,
    milestone_violation_count,
    profile_from_signatures,
    stable_seed,
    summary_signature,
)
from evaluate_libero_closed_loop import (
    Pi05Policy,
    RemotePi05Policy,
    _atomic_write_json,
    _load_reference_arrays,
    _policy_observation,
    _restore_recorded_state,
    is_failure_continuation,
    is_recovery_action,
)
from evaluate_physical_process_oracle import _physical_state, capture_runtime_snapshot
from evaluate_sequential_oracle import ParallelBranchPool, physical_candidate_rows
from libero_full_episode_collector import object_grasped
from libero_snapshot_collector import DEFAULT_BDDL, _step
from onpolicy_support import STAGES, classify_boundary_stage


EXPECTED_PREREGISTRATION_SHA256 = "060512b914d01a3a4092ccf3f7e955396733f2befbf83db79d16ec82c70b3bd0"
ENGINEERING_EXCLUDED_SOURCES = (36,)
TIMELINE_REPLANS = (0, 8, 16, 24, 32, 40, 48, 56)
FORMAL_CONFIG = {
    "candidate_count": 16,
    "execution_horizon": 2,
    "lookahead_actions": 8,
    "continuation_repeats": 6,
    "selection_repeats": 2,
    "max_actions": 320,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def save_audit_snapshot(path: Path, snapshot: Mapping[str, Any]) -> None:
    arrays = {"sim_state": np.asarray(snapshot["sim_state"], dtype=np.float64)}
    arrays.update(
        {
            f"controller__{key}": np.asarray(value)
            for key, value in snapshot["controller_state"].items()
        }
    )
    atomic_npz(path, arrays)


def raw_milestones(summary: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            bool(summary.get("regrasp_reached", summary.get("stable_grasp_at_end", False))),
            bool(summary["lift_reached"]),
            bool(summary["transport_reached"]),
            bool(summary["success"]),
        ],
        dtype=np.uint8,
    )


def collect_state(
    policy: Pi05Policy | RemotePi05Policy,
    output_root: Path,
    *,
    pair_id: str,
    source_initial_state_index: int,
    source_partition: str,
    state_label: str,
    observed_stage: str,
    capture_reasons: Sequence[str],
    replan_index: int,
    observation: Mapping[str, Any],
    candidates: np.ndarray,
    candidate_batch_seed: int,
    candidate_inference_seconds: float,
    immediate_rows: Sequence[Mapping[str, Any]],
    continuation_rows: Sequence[Sequence[Mapping[str, Any]]],
    continuation_cost: Mapping[str, int],
    selected_index: int,
    selection_repeats: int,
    audit_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    state_id = f"{state_label}-r{replan_index:03d}"
    state_root = output_root / "groups" / pair_id / "states" / state_id
    deployable_path = state_root / "deployable.npz"
    labels_path = state_root / "labels.npz"
    audit_path = state_root / "audit_snapshot.npz"

    policy_input = _policy_observation(observation)
    feature, feature_seconds = policy.extract_feature(observation)
    candidate_keys = np.stack(
        [
            np.full(len(candidates), int(candidate_batch_seed), dtype=np.int64),
            np.arange(len(candidates), dtype=np.int64),
        ],
        axis=1,
    )
    deployable = {
        "agentview_image": np.asarray(policy_input["image"][0], dtype=np.uint8),
        "wrist_image": np.asarray(policy_input["image"][1], dtype=np.uint8),
        "robot_state": np.asarray(policy_input["state"], dtype=np.float32),
        "vla_feature": np.asarray(feature, dtype=np.float16),
        "candidates": np.asarray(candidates, dtype=np.float32),
        "candidate_seeds": candidate_keys,
    }
    assert_deployable_arrays(deployable)

    signatures = np.asarray(
        [
            [summary_signature(summary) for summary in candidate_rows]
            for candidate_rows in continuation_rows
        ],
        dtype=np.float32,
    )
    raw = np.asarray(
        [
            [raw_milestones(summary) for summary in candidate_rows]
            for candidate_rows in continuation_rows
        ],
        dtype=np.uint8,
    )
    profiles = np.stack([profile_from_signatures(rows) for rows in signatures])
    direct_signatures = np.stack(
        [summary_signature(row["direct"]) for row in immediate_rows]
    ).astype(np.float32)
    labels = {
        "continuation_signatures": signatures,
        "continuation_profiles": profiles,
        "raw_milestones": raw,
        "direct_signatures": direct_signatures,
        "immediate_correct": np.asarray(
            [row["immediate_correct"] for row in immediate_rows], dtype=np.uint8
        ),
        "failure_continuation": np.asarray(
            [row["failure_continuation"] for row in immediate_rows], dtype=np.uint8
        ),
        "premature_commitment": np.asarray(
            [row["premature_commitment"] for row in immediate_rows], dtype=np.uint8
        ),
    }

    atomic_npz(deployable_path, deployable)
    atomic_npz(labels_path, labels)
    save_audit_snapshot(audit_path, audit_snapshot)
    return {
        "state_id": state_id,
        "pair_id": pair_id,
        "source_initial_state_index": int(source_initial_state_index),
        "source_partition": source_partition,
        "stage": observed_stage,
        "capture_reasons": list(capture_reasons),
        "replan_index": int(replan_index),
        "deployable_file": str(deployable_path.relative_to(output_root)),
        "labels_file": str(labels_path.relative_to(output_root)),
        "audit_file": str(audit_path.relative_to(output_root)),
        "candidate_count": int(len(candidates)),
        "continuation_repeats": int(signatures.shape[1]),
        "trajectory_selected_index": int(selected_index),
        "trajectory_selection_repeats": int(selection_repeats),
        "raw_milestone_violations": milestone_violation_count(raw),
        "candidate_inference_wall_seconds": float(candidate_inference_seconds),
        "feature_wall_seconds": float(feature_seconds),
        "continuation_cost": continuation_cost,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CCV depth-coupled continuation labels")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-socket", type=Path)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--group-offset", type=int, default=0)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--execution-horizon", type=int, default=2)
    parser.add_argument("--lookahead-actions", type=int, default=8)
    parser.add_argument("--continuation-repeats", type=int, default=6)
    parser.add_argument("--selection-repeats", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=320)
    parser.add_argument("--run-kind", choices=("smoke", "formal"), required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_formal(args: argparse.Namespace, train_groups: Sequence[Mapping[str, Any]]) -> None:
    if "confirmation" in str(args.episode_root).lower():
        raise ValueError("confirmation groups are sealed")
    if args.seed != 41:
        raise ValueError("Gate 0 collection is frozen to seed 41")
    if args.run_kind == "formal":
        for field, expected in FORMAL_CONFIG.items():
            if getattr(args, field) != expected:
                raise ValueError(f"formal collection requires {field}={expected}")
        if len(train_groups) != 102:
            raise ValueError(f"formal collection requires 102 train groups, found {len(train_groups)}")
        source_ids = {int(group["source_initial_state_index"]) for group in train_groups}
        if len(source_ids) != 30:
            raise ValueError(f"formal collection requires 30 train source IDs, found {len(source_ids)}")
        actual_hash = file_sha256(args.preregistration)
        if actual_hash != EXPECTED_PREREGISTRATION_SHA256:
            raise ValueError(
                f"preregistration digest changed: {actual_hash} != {EXPECTED_PREREGISTRATION_SHA256}"
            )


def write_root_metadata(
    args: argparse.Namespace,
    train_groups: Sequence[Mapping[str, Any]],
    fit_sources: Sequence[int],
    holdout_sources: Sequence[int],
) -> None:
    metadata = {
        "experiment": "ccv_vla_gate0_coupled_continuations",
        "status": "collecting",
        "seed": args.seed,
        "split": "train",
        "run_kind": args.run_kind,
        "episode_root": str(args.episode_root.resolve()),
        "preregistration": str(args.preregistration.resolve()),
        "preregistration_sha256": file_sha256(args.preregistration),
        "candidate_count": args.candidate_count,
        "execution_horizon": args.execution_horizon,
        "lookahead_actions": args.lookahead_actions,
        "continuation_repeats": args.continuation_repeats,
        "selection_repeats": args.selection_repeats,
        "trajectory_mode": "coupled_continuation_teacher",
        "max_actions": args.max_actions,
        "all_train_group_count": len(train_groups),
        "source_split_salt": "ccv-vla-gate0-v1",
        "fit_source_ids": list(fit_sources),
        "holdout_source_ids": list(holdout_sources),
        "engineering_excluded_source_ids": list(ENGINEERING_EXCLUDED_SOURCES),
        "timeline_replans": list(TIMELINE_REPLANS),
    }
    path = args.output_root / "metadata.json"
    if path.exists():
        existing = json.loads(path.read_text())
        ignored = {"status", "completed_groups"}
        comparable = {key: value for key, value in existing.items() if key not in ignored}
        expected = {key: value for key, value in metadata.items() if key not in ignored}
        if comparable != expected:
            raise ValueError("output root metadata does not match this collection configuration")
    _atomic_write_json(path, metadata)


def rebuild_manifest(output_root: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    group_rows = []
    groups_root = output_root / "groups"
    if groups_root.exists():
        for complete in sorted(groups_root.glob("*/complete.json")):
            group_rows.append(json.loads(complete.read_text()))
    payload = {
        **dict(metadata),
        "status": "partial",
        "completed_groups": len(group_rows),
        "groups": group_rows,
    }
    _atomic_write_json(output_root / "manifest.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    from libero.libero.envs import OffScreenRenderEnv

    if (args.checkpoint is None) == (args.policy_socket is None):
        raise ValueError("provide exactly one of --checkpoint or --policy-socket")
    source_manifest = json.loads((args.episode_root / "manifest.json").read_text())
    all_train_groups = sorted(
        [group for group in source_manifest["groups"] if group["split"] == "train"],
        key=lambda group: group["pair_id"],
    )
    validate_formal(args, all_train_groups)
    split_fit_sources, holdout_sources = frozen_source_split(
        [int(group["source_initial_state_index"]) for group in all_train_groups]
    )
    fit_sources = tuple(
        source_id
        for source_id in split_fit_sources
        if source_id not in ENGINEERING_EXCLUDED_SOURCES
    )
    write_root_metadata(args, all_train_groups, fit_sources, holdout_sources)
    metadata = json.loads((args.output_root / "metadata.json").read_text())

    groups = all_train_groups[args.group_offset :]
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    policy = (
        RemotePi05Policy(args.policy_socket)
        if args.policy_socket is not None
        else Pi05Policy(args.checkpoint, args.device)
    )
    env_kwargs = {
        "bddl_file_name": str(Path(source_manifest.get("bddl", DEFAULT_BDDL))),
        "camera_heights": 224,
        "camera_widths": 224,
    }
    live_env = OffScreenRenderEnv(**env_kwargs)
    live_env.seed(args.seed)
    branch_pool = ParallelBranchPool(args.candidate_count, env_kwargs, args.seed)
    try:
        for group in groups:
            pair_id = str(group["pair_id"])
            complete_path = args.output_root / "groups" / pair_id / "complete.json"
            if complete_path.exists():
                print(json.dumps({"pair_id": pair_id, "status": "already_complete"}), flush=True)
                continue
            source_id = int(group["source_initial_state_index"])
            if source_id in ENGINEERING_EXCLUDED_SOURCES:
                partition = "engineering_excluded"
            elif source_id in holdout_sources:
                partition = "holdout"
            else:
                partition = "fit"
            reference = _load_reference_arrays(args.episode_root, group, "slipped")
            observation = _restore_recorded_state(
                live_env,
                reference,
                int(group["feedback_reveal_time"]),
            )
            initial_distance = float(
                np.linalg.norm(
                    np.asarray(observation["cream_cheese_1_pos"])
                    - np.asarray(observation["robot0_eef_pos"])
                )
            )
            captured: set[str] = set()
            state_rows = []
            recovery_started = False
            previous_failure = False
            actions_executed = 0
            replan_index = 0
            last_state = None
            success = bool(live_env.check_success())
            while actions_executed < args.max_actions and not success:
                pool_seed = stable_seed("ccv-candidate-pool", args.seed, pair_id, replan_index)
                candidates, inference_seconds = policy.predict_sample_batch(
                    observation,
                    count=args.candidate_count,
                    seed=pool_seed,
                )
                distance = float(
                    np.linalg.norm(
                        np.asarray(observation["cream_cheese_1_pos"])
                        - np.asarray(observation["robot0_eef_pos"])
                    )
                )
                stage = classify_boundary_stage(
                    replan_index=replan_index,
                    grasped=object_grasped(live_env),
                    previous_failure_continuation=previous_failure,
                    recovery_started=recovery_started,
                    eef_object_distance=distance,
                    initial_eef_object_distance=initial_distance,
                    candidate0_closes=bool(
                        np.any(candidates[0, : args.execution_horizon, -1] > 0.2)
                    ),
                )
                snapshot = dict(capture_runtime_snapshot(live_env))
                snapshot["physical_state"] = _physical_state(
                    live_env,
                    observation,
                    bool(live_env.check_success()),
                )
                capture_reasons = []
                if stage not in captured:
                    capture_reasons.append(f"semantic:{stage}")
                if replan_index in TIMELINE_REPLANS:
                    capture_reasons.append(f"timeline:r{replan_index}")
                current_trace = [dict(snapshot["physical_state"])]
                immediate_rows, endpoints = physical_candidate_rows(
                    branch_pool,
                    snapshot,
                    observation,
                    current_trace,
                    candidates,
                    execution_horizon=args.execution_horizon,
                )
                label_repeats = (
                    args.continuation_repeats
                    if capture_reasons
                    else args.selection_repeats
                )
                continuation_rows, continuation_cost = coupled_policy_continuations(
                    branch_pool,
                    policy,
                    endpoint_count=len(endpoints),
                    seed=args.seed,
                    pair_id=pair_id,
                    state_id=f"live-r{replan_index:03d}",
                    execution_horizon=args.execution_horizon,
                    lookahead_actions=args.lookahead_actions,
                    repeats=label_repeats,
                )
                selection_profiles = np.stack(
                    [
                        profile_from_signatures(
                            np.stack(
                                [
                                    summary_signature(summary)
                                    for summary in rows[: args.selection_repeats]
                                ]
                            )
                        )
                        for rows in continuation_rows
                    ]
                )
                selected_index = best_candidate_index(selection_profiles)
                last_state = (
                    snapshot,
                    observation,
                    candidates,
                    pool_seed,
                    inference_seconds,
                    replan_index,
                    immediate_rows,
                    continuation_rows,
                    continuation_cost,
                    selected_index,
                    label_repeats,
                )
                if capture_reasons:
                    state_label = (
                        stage
                        if stage not in captured
                        else f"timeline_{replan_index:03d}"
                    )
                    row = collect_state(
                        policy,
                        args.output_root,
                        pair_id=pair_id,
                        source_initial_state_index=source_id,
                        source_partition=partition,
                        state_label=state_label,
                        observed_stage=stage,
                        capture_reasons=capture_reasons,
                        replan_index=replan_index,
                        observation=observation,
                        candidates=candidates,
                        candidate_batch_seed=pool_seed,
                        candidate_inference_seconds=inference_seconds,
                        immediate_rows=immediate_rows,
                        continuation_rows=continuation_rows,
                        continuation_cost=continuation_cost,
                        selected_index=selected_index,
                        selection_repeats=args.selection_repeats,
                        audit_snapshot=snapshot,
                    )
                    state_rows.append(row)
                    if stage not in captured:
                        captured.add(stage)
                    print(
                        json.dumps(
                            {
                                "pair_id": pair_id,
                                "stage": stage,
                                "capture_reasons": capture_reasons,
                                "replan_index": replan_index,
                                "states_complete": len(state_rows),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

                previous_failure = False
                for action in candidates[selected_index, : args.execution_horizon]:
                    grasped = object_grasped(live_env)
                    previous_failure = previous_failure or is_failure_continuation(
                        action, grasped=grasped
                    )
                    recovery_started = recovery_started or is_recovery_action(
                        action,
                        grasped=grasped,
                        eef_position=observation["robot0_eef_pos"],
                        object_position=observation["cream_cheese_1_pos"],
                    )
                    observation = _step(live_env, action)
                    actions_executed += 1
                    success = bool(live_env.check_success())
                    if success or actions_executed >= args.max_actions:
                        break
                replan_index += 1

            if not success and "final_failure" not in captured and last_state is not None:
                (
                    snapshot,
                    final_observation,
                    candidates,
                    pool_seed,
                    inference_seconds,
                    final_replan,
                    immediate_rows,
                    continuation_rows,
                    continuation_cost,
                    selected_index,
                    label_repeats,
                ) = last_state
                duplicate = next(
                    (row for row in state_rows if row["replan_index"] == final_replan),
                    None,
                )
                if duplicate is not None:
                    duplicate["capture_reasons"] = sorted(
                        set(duplicate["capture_reasons"]) | {"semantic:final_failure"}
                    )
                else:
                    if label_repeats != args.continuation_repeats:
                        continuation_rows, continuation_cost = coupled_policy_continuations(
                            branch_pool,
                            policy,
                            endpoint_count=len(candidates),
                            seed=args.seed,
                            pair_id=pair_id,
                            state_id=f"live-r{final_replan:03d}",
                            execution_horizon=args.execution_horizon,
                            lookahead_actions=args.lookahead_actions,
                            repeats=args.continuation_repeats,
                        )
                    row = collect_state(
                        policy,
                        args.output_root,
                        pair_id=pair_id,
                        source_initial_state_index=source_id,
                        source_partition=partition,
                        state_label="final_failure",
                        observed_stage="final_failure",
                        capture_reasons=("semantic:final_failure",),
                        replan_index=final_replan,
                        observation=final_observation,
                        candidates=candidates,
                        candidate_batch_seed=pool_seed,
                        candidate_inference_seconds=inference_seconds,
                        immediate_rows=immediate_rows,
                        continuation_rows=continuation_rows,
                        continuation_cost=continuation_cost,
                        selected_index=selected_index,
                        selection_repeats=args.selection_repeats,
                        audit_snapshot=snapshot,
                    )
                    state_rows.append(row)
                captured.add("final_failure")

            group_payload = {
                "pair_id": pair_id,
                "source_initial_state_index": source_id,
                "source_partition": partition,
                "success": success,
                "actions": actions_executed,
                "replans": replan_index,
                "captured_stages": sorted(captured),
                "missing_stages": sorted(set(STAGES) - captured),
                "states": state_rows,
            }
            _atomic_write_json(complete_path, group_payload)
            rebuild_manifest(args.output_root, metadata)
            print(
                json.dumps(
                    {
                        "pair_id": pair_id,
                        "status": "complete",
                        "states": len(state_rows),
                        "success": success,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        branch_pool.close()
        live_env.close()
        policy.close()

    manifest = rebuild_manifest(args.output_root, metadata)
    manifest["status"] = "complete" if len(manifest["groups"]) == len(all_train_groups) else "partial"
    _atomic_write_json(args.output_root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
