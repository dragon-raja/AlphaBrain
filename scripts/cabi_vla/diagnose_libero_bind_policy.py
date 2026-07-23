from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def edge_factors(edge_id: str) -> tuple[str, str]:
    try:
        source, target = edge_id.rsplit("-", 1)
    except ValueError as error:
        raise ValueError(f"invalid edge id: {edge_id!r}") from error
    if not source or not target:
        raise ValueError(f"invalid edge id: {edge_id!r}")
    return source, target


def mean_squared_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.square(np.asarray(left) - np.asarray(right)).mean())


def factor_sensitivity(predictions: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Measure instruction interventions under one observation and shared flow noise."""

    factors = {edge: edge_factors(edge) for edge in predictions}
    source_pairs = []
    target_pairs = []
    edges = sorted(predictions)
    for index, left in enumerate(edges):
        left_source, left_target = factors[left]
        for right in edges[index + 1 :]:
            right_source, right_target = factors[right]
            row = {
                "left_edge": left,
                "right_edge": right,
                "chunk_mse": mean_squared_difference(predictions[left], predictions[right]),
                "first_step_mse": mean_squared_difference(
                    np.asarray(predictions[left])[0], np.asarray(predictions[right])[0]
                ),
            }
            if left_target == right_target and left_source != right_source:
                source_pairs.append(row)
            if left_source == right_source and left_target != right_target:
                target_pairs.append(row)

    def summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        if not rows:
            raise ValueError("factor sensitivity requires both source and target pairs")
        values = np.asarray([row["chunk_mse"] for row in rows], dtype=np.float64)
        first = np.asarray([row["first_step_mse"] for row in rows], dtype=np.float64)
        return {
            "pair_count": len(rows),
            "chunk_mse_mean": float(values.mean()),
            "chunk_mse_min": float(values.min()),
            "chunk_mse_max": float(values.max()),
            "first_step_mse_mean": float(first.mean()),
        }

    source_summary = summary(source_pairs)
    target_summary = summary(target_pairs)
    return {
        "source_interventions": source_summary,
        "target_interventions": target_summary,
        "source_pairs": source_pairs,
        "target_pairs": target_pairs,
        "source_to_target_sensitivity_ratio": float(
            source_summary["chunk_mse_mean"]
            / max(target_summary["chunk_mse_mean"], 1e-12)
        ),
    }


def action_chunk(actions: np.ndarray, start: int, horizon: int) -> np.ndarray:
    chunk = np.asarray(actions[start : start + horizon], dtype=np.float32)
    if len(chunk) < horizon:
        chunk = np.concatenate(
            [chunk, np.zeros((horizon - len(chunk), actions.shape[1]), np.float32)]
        )
    return chunk


def progress_bin(frame_index: int, action_count: int) -> str:
    fraction = frame_index / max(action_count, 1)
    index = min(int(fraction * 4), 3)
    return ("q1", "q2", "q3", "q4")[index]


def summarize_teacher_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "chunk_mse",
        "first_step_mse",
        "translation_mse",
        "rotation_mse",
        "gripper_mse",
    )

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        result: dict[str, float | int] = {"row_count": len(selected)}
        for name in metric_names:
            result[name] = float(np.mean([float(row[name]) for row in selected]))
        return result

    by_edge: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_phase: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_progress: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_edge[str(row["edge_id"])].append(row)
        by_phase[str(row["phase"])].append(row)
        by_progress[str(row["progress_bin"])].append(row)
    return {
        "overall": aggregate(rows),
        "by_edge": {key: aggregate(value) for key, value in sorted(by_edge.items())},
        "by_phase": {key: aggregate(value) for key, value in sorted(by_phase.items())},
        "by_progress": {
            key: aggregate(by_progress[key])
            for key in ("q1", "q2", "q3", "q4")
            if key in by_progress
        },
    }


def policy_example(
    agentview: np.ndarray,
    wrist: np.ndarray,
    state: np.ndarray,
    language: str,
) -> dict[str, Any]:
    return {
        "image": [np.asarray(agentview), np.asarray(wrist)],
        "lang": language,
        "language": language,
        "state": np.asarray(state, dtype=np.float32),
    }


def anchor_key(
    edge_id: str,
    state_index: int,
    field: str,
    decision_point: str | None = None,
) -> str:
    decision = "" if decision_point is None else f"{decision_point}__"
    return f"{edge_id}__state_{state_index:02d}__{decision}{field}"


def resolve_decision_point(
    manifest: Mapping[str, Any], requested: str
) -> str | None:
    decision_points = manifest.get("decision_points", {})
    if requested == "auto":
        return "source_select" if "source_select" in decision_points else None
    if requested not in decision_points:
        raise ValueError(
            f"decision point {requested!r} is absent from the training view"
        )
    return requested


def request_actions(
    connection: Client,
    examples: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    coupled: bool,
) -> np.ndarray:
    connection.send(
        {
            "op": (
                "predict_observation_batch_coupled"
                if coupled
                else "predict_observation_batch"
            ),
            "seed": int(seed),
            "examples": list(examples),
        }
    )
    response = connection.recv()
    if "error" in response:
        raise RuntimeError(response["error"])
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim != 3 or actions.shape[0] != len(examples) or not np.all(np.isfinite(actions)):
        raise ValueError(f"invalid policy response: {actions.shape}")
    return actions


def load_records(
    training_view: Path,
    *,
    state_indices: set[int],
    frame_stride: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((training_view / "manifest.json").read_text())
    withheld = set(manifest["leakage_guard"]["withheld_action_edges"])
    records = []
    for line in (training_view / "records.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["edge_id"] in withheld:
            raise ValueError("training records unexpectedly contain a held-out action edge")
        if int(row["canonical_state_index"]) not in state_indices:
            continue
        if int(row["frame_index"]) % frame_stride != 0:
            continue
        records.append(row)
    if not records:
        raise ValueError("teacher-forcing diagnosis selected no records")
    return manifest, records


def run_instruction_sensitivity(
    connection: Client,
    training_view: Path,
    manifest: Mapping[str, Any],
    *,
    state_index: int,
    physical_edge: str,
    seeds: Sequence[int],
    decision_point: str | None,
) -> dict[str, Any]:
    edges = sorted(manifest["edge_instructions"])
    with np.load(training_view / manifest["anchors_file"], allow_pickle=False) as anchors:
        agent = np.asarray(
            anchors[anchor_key(physical_edge, state_index, "agentview", decision_point)]
        )
        wrist = np.asarray(
            anchors[anchor_key(physical_edge, state_index, "wrist", decision_point)]
        )
        state = np.asarray(
            anchors[anchor_key(physical_edge, state_index, "state", decision_point)]
        )
        examples = [
            policy_example(agent, wrist, state, manifest["edge_instructions"][edge])
            for edge in edges
        ]
    rows = []
    for seed in seeds:
        actions = request_actions(connection, examples, seed=seed, coupled=True)
        rows.append({"seed": seed, **factor_sensitivity(dict(zip(edges, actions)))})
    return {
        "physical_edge": physical_edge,
        "canonical_state_index": state_index,
        "decision_point": decision_point,
        "shared_observation": True,
        "shared_flow_noise_within_seed": True,
        "action_labels_loaded": False,
        "seeds": list(seeds),
        "rows": rows,
        "summary": {
            "source_intervention_chunk_mse": float(
                np.mean([row["source_interventions"]["chunk_mse_mean"] for row in rows])
            ),
            "target_intervention_chunk_mse": float(
                np.mean([row["target_interventions"]["chunk_mse_mean"] for row in rows])
            ),
            "source_to_target_sensitivity_ratio": float(
                np.mean([row["source_to_target_sensitivity_ratio"] for row in rows])
            ),
        },
    }


def run_teacher_forcing(
    connection: Client,
    identity: Mapping[str, Any],
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    batch_size: int,
) -> dict[str, Any]:
    horizon = int(identity["horizon"])
    collection = Path(manifest["source_collection"])
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["episode_file"])].append(row)

    output_rows = []
    for episode_file, episode_records in sorted(grouped.items()):
        with np.load(collection / episode_file, allow_pickle=False) as episode:
            actions = np.asarray(episode["actions"], dtype=np.float32)
            for start in range(0, len(episode_records), batch_size):
                selected = episode_records[start : start + batch_size]
                examples = [
                    policy_example(
                        episode["agentview"][int(row["frame_index"])],
                        episode["wrist"][int(row["frame_index"])],
                        episode["robot_state"][int(row["frame_index"])],
                        str(row["language_instruction"]),
                    )
                    for row in selected
                ]
                teachers = np.stack(
                    [
                        action_chunk(actions, int(row["frame_index"]), horizon)
                        for row in selected
                    ]
                )
                for seed in seeds:
                    predictions = request_actions(
                        connection,
                        examples,
                        seed=stable_seed(seed, episode_file, start),
                        coupled=False,
                    )
                    squared = np.square(predictions - teachers)
                    for row, error in zip(selected, squared):
                        frame = int(row["frame_index"])
                        output_rows.append(
                            {
                                "sample_id": row["sample_id"],
                                "edge_id": row["edge_id"],
                                "canonical_state_index": int(row["canonical_state_index"]),
                                "frame_index": frame,
                                "phase": str(episode["phase"][frame]),
                                "progress_bin": progress_bin(frame, len(actions)),
                                "seed": seed,
                                "chunk_mse": float(error.mean()),
                                "first_step_mse": float(error[0].mean()),
                                "translation_mse": float(error[:, :3].mean()),
                                "rotation_mse": float(error[:, 3:6].mean()),
                                "gripper_mse": float(error[:, 6].mean()),
                            }
                        )
    return {
        "heldout_action_labels_loaded": False,
        "supervised_edges": sorted({str(row["edge_id"]) for row in records}),
        "selected_record_count": len(records),
        "prediction_row_count": len(output_rows),
        "seeds": list(seeds),
        "summary": summarize_teacher_rows(output_rows),
        "rows": output_rows,
    }


def atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose LIBERO-Bind policy grounding and BC fit")
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-indices", type=int, nargs="+", default=[0])
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260722, 20260723, 20260724])
    parser.add_argument("--sensitivity-physical-edge", default="red-left")
    parser.add_argument(
        "--decision-point",
        choices=("auto", "source_select", "target_select"),
        default="auto",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if args.frame_stride <= 0 or not 1 <= args.batch_size <= 16:
        raise ValueError("frame stride must be positive and batch size must be in [1, 16]")
    manifest, records = load_records(
        args.training_view,
        state_indices=set(args.state_indices),
        frame_stride=args.frame_stride,
    )
    decision_point = resolve_decision_point(manifest, args.decision_point)
    connection = Client(str(args.policy_socket), family="AF_UNIX", authkey=b"fresh-vla-local")
    identity = dict(connection.recv())
    try:
        sensitivity = run_instruction_sensitivity(
            connection,
            args.training_view,
            manifest,
            state_index=args.state_indices[0],
            physical_edge=args.sensitivity_physical_edge,
            seeds=args.seeds,
            decision_point=decision_point,
        )
        teacher_forcing = run_teacher_forcing(
            connection,
            identity,
            manifest,
            records,
            seeds=args.seeds,
            batch_size=args.batch_size,
        )
    finally:
        try:
            connection.send({"op": "close"})
        except (BrokenPipeError, EOFError, OSError):
            pass
        connection.close()
    payload = {
        "schema_version": 1,
        "status": "complete",
        "training_view": str(args.training_view),
        "policy_identity": identity,
        "diagnostic_only": True,
        "instruction_sensitivity": sensitivity,
        "teacher_forcing": teacher_forcing,
    }
    atomic_write(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_sensitivity": sensitivity["summary"]["source_intervention_chunk_mse"],
                "target_sensitivity": sensitivity["summary"]["target_intervention_chunk_mse"],
                "teacher_chunk_mse": teacher_forcing["summary"]["overall"]["chunk_mse"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
