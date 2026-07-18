from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for source_dir in (REPO_ROOT / "scripts" / "fresh_vla", REPO_ROOT / "scripts" / "cora_vla"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from scripts.basin_vla.policy_relativity import STAGES, TARGET_POLICIES, parse_cache_name, select_cache_files
from scripts.cora_vla.evaluate_sequential_oracle import (
    ParallelBranchPool,
    batched_policy_continuations,
    continuation_selection_key,
)
from scripts.fresh_vla.evaluate_libero_closed_loop import RemotePi05Policy
from scripts.fresh_vla.evaluate_physical_process_oracle import _physical_state, restore_runtime_snapshot
from scripts.fresh_vla.libero_snapshot_collector import DEFAULT_BDDL


DEFAULT_CACHE_ROOT = Path("/share/longjunyu/fresh-vla/cora-vla/onpolicy-support-v1/cache")
DEFAULT_EPISODE_ROOT = Path("/share/longjunyu/fresh-vla/libero-full-episode-v2-128")
DEFAULT_OUTPUT_ROOT = Path("/share/longjunyu/basin-vla/policy-relativity-gate0-v1")
COMMON_ROLLOUT_SEED = 260718


def _assert_unsealed_path(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    forbidden = {"test", "tests", "confirmation", "confirm", "sealed"}
    if lowered & forbidden or any("confirmation" in part for part in lowered):
        raise ValueError(f"refusing sealed path: {path}")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_cached_state(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        snapshot = {
            "sim_state": np.asarray(arrays["sim_state"], dtype=np.float64),
            "controller_state": {
                key.split("controller__", 1)[1]: np.asarray(arrays[key])
                for key in arrays.files
                if key.startswith("controller__")
            },
        }
        candidates = np.asarray(arrays["candidates"], dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[0] != 16 or candidates.shape[2] != 7:
        raise ValueError(f"unexpected cached candidate shape in {path}: {candidates.shape}")
    return snapshot, candidates


def selected_formal_caches(cache_root: Path) -> list[Path]:
    paths = []
    for tag in ("seed41-a", "seed41-b"):
        directory = cache_root / tag
        _assert_unsealed_path(directory)
        paths.extend(sorted(directory.glob("*.npz")))
    return select_cache_files(paths, per_stage=3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect same-state cross-policy continuation rankings")
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--target-policy-seed", type=int, choices=TARGET_POLICIES, required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--episode-root", type=Path, default=DEFAULT_EPISODE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    for path in (args.cache_root, args.episode_root, args.output_root):
        _assert_unsealed_path(path)
    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    source_by_pair = {
        str(group["pair_id"]): int(group["source_initial_state_index"])
        for group in manifest["groups"]
        if group["split"] == "val"
    }
    caches = selected_formal_caches(args.cache_root)
    if args.max_states is not None:
        caches = caches[: args.max_states]
    if not caches:
        raise ValueError("no preregistered caches selected")

    env_kwargs = {
        "bddl_file_name": str(Path(manifest.get("bddl", DEFAULT_BDDL))),
        "camera_heights": 224,
        "camera_widths": 224,
    }
    main_env = OffScreenRenderEnv(**env_kwargs)
    main_env.seed(COMMON_ROLLOUT_SEED)
    pool = ParallelBranchPool(16, env_kwargs, COMMON_ROLLOUT_SEED)
    policy = RemotePi05Policy(args.policy_socket)
    completed = 0
    invalid = 0
    target_dir = args.output_root / f"policy-{args.target_policy_seed}" / "records"
    try:
        for cache_path in caches:
            pair_id, stage, replan_index = parse_cache_name(cache_path)
            state_id = cache_path.stem
            output = target_dir / f"{state_id}.json"
            if output.exists() and not args.overwrite:
                completed += 1
                continue
            try:
                snapshot, candidates = load_cached_state(cache_path)
                observation = restore_runtime_snapshot(main_env, snapshot)
                prefix_trace = [
                    _physical_state(main_env, observation, bool(main_env.check_success()))
                ]
                endpoints = pool.prepare(snapshot, prefix_trace, candidates, execution_horizon=2)
                continuation_rows, batch_calls, simulator_actions = batched_policy_continuations(
                    pool,
                    policy,
                    endpoints,
                    seed=COMMON_ROLLOUT_SEED,
                    pair_id=pair_id,
                    outcome="slipped",
                    replan_index=replan_index,
                    execution_horizon=2,
                    lookahead_actions=8,
                    repeats=2,
                )
                payload = {
                    "schema_version": 1,
                    "status": "valid",
                    "state_id": state_id,
                    "pair_id": pair_id,
                    "source_initial_state_index": source_by_pair[pair_id],
                    "stage": stage,
                    "replan_index": replan_index,
                    "source_candidate_policy_seed": 41,
                    "target_continuation_policy_seed": args.target_policy_seed,
                    "candidate_cache": str(cache_path),
                    "candidate_count": len(candidates),
                    "execution_horizon": 2,
                    "lookahead_actions": 8,
                    "continuation_repeats": 2,
                    "common_rollout_seed": COMMON_ROLLOUT_SEED,
                    "continuation_keys": [
                        [round(value, 7) for value in continuation_selection_key(rows)]
                        for rows in continuation_rows
                    ],
                    "continuation_repeat_summaries": continuation_rows,
                    "direct_endpoint_summaries": endpoints,
                    "direct_endpoint_sha256": hashlib.sha256(
                        json.dumps(endpoints, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "policy_batch_calls": batch_calls,
                    "search_simulator_actions": int(len(candidates) * 2 + simulator_actions),
                    "policy_checkpoint_realpath": policy.checkpoint_realpath,
                    "policy_runtime": policy.runtime_identity,
                }
            except Exception as exception:
                invalid += 1
                payload = {
                    "schema_version": 1,
                    "status": "invalid",
                    "state_id": state_id,
                    "pair_id": pair_id,
                    "stage": stage,
                    "target_continuation_policy_seed": args.target_policy_seed,
                    "error": f"{type(exception).__name__}: {exception}",
                    "traceback": traceback.format_exc(),
                }
                atomic_json(output, payload)
                print(json.dumps({"state_id": state_id, "status": "invalid", "error": payload["error"]}), flush=True)
                if args.fail_fast:
                    raise
            else:
                atomic_json(output, payload)
                print(json.dumps({"state_id": state_id, "status": "valid"}), flush=True)
            completed += 1
    finally:
        policy.close()
        pool.close()
        main_env.close()

    stages = [parse_cache_name(path)[1] for path in caches]
    run = {
        "schema_version": 1,
        "status": "complete",
        "target_continuation_policy_seed": args.target_policy_seed,
        "selected_states": len(caches),
        "completed_states": completed,
        "invalid_states_in_this_invocation": invalid,
        "stage_counts": {stage: stages.count(stage) for stage in STAGES},
        "data_policy": "existing validation cache only; test and confirmation fail closed",
    }
    atomic_json(args.output_root / f"policy-{args.target_policy_seed}" / "run.json", run)
    print(json.dumps(run, indent=2), flush=True)


if __name__ == "__main__":
    main()
