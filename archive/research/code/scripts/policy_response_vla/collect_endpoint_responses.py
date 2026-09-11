from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for source_dir in (
    REPO_ROOT / "scripts" / "fresh_vla",
    REPO_ROOT / "scripts" / "cora_vla",
    REPO_ROOT / "scripts" / "ccv_vla",
):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from ccv import stable_seed
from evaluate_libero_closed_loop import RemotePi05Policy
from evaluate_physical_process_oracle import _physical_state, restore_runtime_snapshot
from evaluate_sequential_oracle import ParallelBranchPool
from libero_snapshot_collector import DEFAULT_BDDL


FORBIDDEN_PARTS = {"test", "tests", "confirmation", "confirm", "sealed", "holdout"}
EXPECTED_RESPONSE_SHAPE = (16, 2, 10, 7)


def assert_unsealed(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    if lowered & FORBIDDEN_PARTS or any("confirmation" in part for part in lowered):
        raise ValueError(f"refusing sealed path: {path}")


def load_snapshot(path: Path) -> dict[str, Any]:
    assert_unsealed(path)
    with np.load(path, allow_pickle=False) as arrays:
        return {
            "sim_state": np.asarray(arrays["sim_state"], dtype=np.float64),
            "controller_state": {
                key.split("controller__", 1)[1]: np.asarray(arrays[key])
                for key in arrays.files
                if key.startswith("controller__")
            },
        }


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_fit_states(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group in manifest["groups"]:
        if group["source_partition"] != "fit":
            continue
        for state in group["states"]:
            if state["source_partition"] != "fit":
                raise RuntimeError("group/state partition mismatch")
            rows.append({"group": group, "state": state})
    return sorted(rows, key=lambda row: (row["group"]["pair_id"], row["state"]["state_id"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect label-disjoint endpoint policy responses")
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--ccv-root", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--state-offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    for path in (args.ccv_root, args.episode_root, args.output_root, args.preregistration):
        assert_unsealed(path)
    if args.output_root.exists() and not args.resume:
        raise FileExistsError(f"output exists; inspect it and pass --resume: {args.output_root}")

    ccv_manifest = json.loads((args.ccv_root / "manifest.json").read_text())
    if ccv_manifest.get("status") != "complete" or ccv_manifest.get("split") != "train":
        raise ValueError("requires the complete train-only CCV collection")
    states = selected_fit_states(ccv_manifest)[args.state_offset :]
    if args.max_states is not None:
        states = states[: args.max_states]
    if not states:
        raise ValueError("no fit states selected")

    source_manifest = json.loads((args.episode_root / "manifest.json").read_text())
    env_kwargs = {
        "bddl_file_name": str(Path(source_manifest.get("bddl", DEFAULT_BDDL))),
        "camera_heights": 224,
        "camera_widths": 224,
    }
    run_config = {
        "schema_version": 1,
        "experiment": "policy_response_surrogate_gate_minus1",
        "ccv_root": str(args.ccv_root.resolve()),
        "episode_root": str(args.episode_root.resolve()),
        "preregistration": str(args.preregistration.resolve()),
        "preregistration_sha256": file_sha256(args.preregistration),
        "source_partition": "fit",
        "response_repeats": 2,
        "response_horizon": 8,
        "execution_horizon": 2,
        "ccv_holdout_states_opened": 0,
        "test_or_confirmation_states_opened": 0,
    }
    config_path = args.output_root / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != run_config:
            raise RuntimeError("resume run_config mismatch")
    else:
        atomic_json(config_path, run_config)

    env = OffScreenRenderEnv(**env_kwargs)
    env.seed(260719)
    pool = ParallelBranchPool(16, env_kwargs, 260719)
    policy = RemotePi05Policy(args.policy_socket)
    policy_identity = {
        "checkpoint_realpath": policy.checkpoint_realpath,
        "model_size_bytes": policy.model_size_bytes,
        **policy.runtime_identity,
    }
    identity_path = args.output_root / "policy_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != policy_identity:
            raise RuntimeError("resume policy identity mismatch")
    else:
        atomic_json(identity_path, policy_identity)
    records = []
    try:
        for ordinal, item in enumerate(states, start=1):
            group, state = item["group"], item["state"]
            pair_id, state_id = str(group["pair_id"]), str(state["state_id"])
            relative = Path("records") / pair_id / f"{state_id}.npz"
            output = args.output_root / relative
            if output.exists():
                if not args.resume:
                    raise FileExistsError(output)
                with np.load(output, allow_pickle=False) as existing:
                    if tuple(existing["response_actions"].shape) != EXPECTED_RESPONSE_SHAPE:
                        raise RuntimeError(f"invalid resume record: {output}")
                records.append(str(relative))
                print(json.dumps({"state": state_id, "status": "resumed"}), flush=True)
                continue

            deployable_path = args.ccv_root / state["deployable_file"]
            labels_path = args.ccv_root / state["labels_file"]
            audit_path = args.ccv_root / state["audit_file"]
            for path in (deployable_path, labels_path, audit_path):
                assert_unsealed(path)
            with np.load(deployable_path, allow_pickle=False) as deployable:
                candidates = np.asarray(deployable["candidates"], dtype=np.float32)
            with np.load(labels_path, allow_pickle=False) as labels:
                profiles = np.asarray(labels["continuation_profiles"], dtype=np.float32)
                direct = np.asarray(labels["direct_signatures"], dtype=np.float32)
            if candidates.shape != (16, 10, 7) or profiles.shape != (16, 6) or direct.shape != (16, 6):
                raise RuntimeError(f"unexpected source shapes for {pair_id}/{state_id}")

            snapshot = load_snapshot(audit_path)
            observation = restore_runtime_snapshot(env, snapshot)
            prefix_trace = [_physical_state(env, observation, bool(env.check_success()))]
            pool.prepare(snapshot, prefix_trace, candidates, execution_horizon=2)
            endpoint_observations = pool.reset_continuation(16)
            responses = []
            inference_seconds = 0.0
            for repeat in range(2):
                seed = stable_seed("policy-response-label-disjoint-v1", pair_id, state_id, repeat)
                actions, seconds = policy.predict_observation_batch_coupled(
                    endpoint_observations, seed=seed
                )
                responses.append(actions)
                inference_seconds += seconds
            response_actions = np.stack(responses, axis=1).astype(np.float32)
            if response_actions.shape != EXPECTED_RESPONSE_SHAPE:
                raise RuntimeError(f"unexpected response shape: {response_actions.shape}")
            atomic_npz(
                output,
                {
                    "candidate_prefixes": candidates[:, :2],
                    "response_actions": response_actions,
                    "continuation_profiles": profiles,
                    "direct_signatures": direct,
                    "source_id": np.asarray(int(group["source_initial_state_index"]), dtype=np.int64),
                },
            )
            records.append(str(relative))
            print(
                json.dumps(
                    {
                        "completed": ordinal,
                        "selected": len(states),
                        "pair_id": pair_id,
                        "state_id": state_id,
                        "inference_seconds": round(inference_seconds, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        policy.close()
        pool.close()
        env.close()

    all_records = sorted(
        str(path.relative_to(args.output_root))
        for path in (args.output_root / "records").glob("*/*.npz")
    )
    atomic_json(
        args.output_root / "manifest.json",
        {
            **run_config,
            "status": "complete",
            "state_offset": args.state_offset,
            "selected_state_count": len(states),
            "collected_state_count": len(all_records),
            "records": all_records,
        },
    )


if __name__ == "__main__":
    main()
