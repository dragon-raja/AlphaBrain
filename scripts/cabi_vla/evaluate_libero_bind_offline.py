from __future__ import annotations

import argparse
import json
import os
from multiprocessing.connection import Client
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


CORNER_ORDER = ("base", "source_anchor", "target_anchor", "fourth_anchor")


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def action_transport_metrics(
    predictions: Mapping[str, np.ndarray],
    teachers: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Score a fourth corner without consuming its expert action."""

    base_p = np.asarray(predictions["base"], dtype=np.float32)
    source_p = np.asarray(predictions["source_anchor"], dtype=np.float32)
    target_p = np.asarray(predictions["target_anchor"], dtype=np.float32)
    fourth_p = np.asarray(predictions["fourth_anchor"], dtype=np.float32)
    base_t = np.asarray(teachers["base"], dtype=np.float32)
    source_t = np.asarray(teachers["source_anchor"], dtype=np.float32)
    target_t = np.asarray(teachers["target_anchor"], dtype=np.float32)
    if not all(value.shape == base_t.shape for value in (*predictions.values(), *teachers.values())):
        raise ValueError("all prediction and teacher chunks must share one shape")

    pseudo_fourth = source_t + target_t - base_t
    clipped_pseudo = np.clip(pseudo_fourth, -1.0, 1.0)
    teacher_target_effect = target_t - base_t
    teacher_source_effect = source_t - base_t
    predicted_target_transfer = fourth_p - source_p
    predicted_source_transfer = fourth_p - target_p
    observed_mses = [
        np.square(predictions[name] - teachers[name]).mean()
        for name in ("base", "source_anchor", "target_anchor")
    ]
    return {
        "observed_corner_mse": float(np.mean(observed_mses)),
        "action_free_pseudo_mse": float(np.square(fourth_p - clipped_pseudo).mean()),
        "model_self_closure_mse": float(
            np.square(fourth_p - (source_p + target_p - base_p)).mean()
        ),
        "target_effect_transfer_mse": float(
            np.square(predicted_target_transfer - teacher_target_effect).mean()
        ),
        "source_effect_transfer_mse": float(
            np.square(predicted_source_transfer - teacher_source_effect).mean()
        ),
        "target_effect_cosine": cosine(
            predicted_target_transfer, teacher_target_effect
        ),
        "source_effect_cosine": cosine(
            predicted_source_transfer, teacher_source_effect
        ),
        "pseudo_action_clipped_fraction": float(
            np.mean(np.abs(pseudo_fourth) > 1.0)
        ),
    }


def anchor_key(edge_id: str, state_index: int, field: str) -> str:
    return f"{edge_id}__state_{state_index:02d}__{field}"


def corner_example(
    anchors: Mapping[str, np.ndarray],
    instructions: Mapping[str, str],
    corner: Mapping[str, object],
    state_index: int,
) -> dict:
    physical = str(corner["physical_edge"])
    instruction = str(corner["instruction_edge"])
    return {
        "image": [
            np.asarray(anchors[anchor_key(physical, state_index, "agentview")]),
            np.asarray(anchors[anchor_key(physical, state_index, "wrist")]),
        ],
        "lang": instructions[instruction],
        "language": instructions[instruction],
        "state": np.asarray(
            anchors[anchor_key(physical, state_index, "state")], dtype=np.float32
        ),
    }


def atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Action-free CABI fourth-corner gate")
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260722])
    parser.add_argument("--max-tetrads", type=int)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    manifest = json.loads((args.training_view / "manifest.json").read_text())
    tetrads = list(manifest["tetrads"])
    if args.max_tetrads is not None:
        tetrads = tetrads[: args.max_tetrads]
    if not tetrads:
        raise ValueError("no tetrads selected")

    connection = Client(
        str(args.policy_socket), family="AF_UNIX", authkey=b"fresh-vla-local"
    )
    identity = dict(connection.recv())
    rows = []
    try:
        with np.load(args.training_view / "anchors.npz", allow_pickle=False) as anchors:
            for seed in args.seeds:
                for tetrad in tetrads:
                    state_index = int(tetrad["canonical_state_index"])
                    corners = tetrad["corners"]
                    examples = [
                        corner_example(
                            anchors,
                            manifest["edge_instructions"],
                            corners[name],
                            state_index,
                        )
                        for name in CORNER_ORDER
                    ]
                    connection.send(
                        {
                            "op": "predict_observation_batch_coupled",
                            "seed": int(seed),
                            "examples": examples,
                        }
                    )
                    response = connection.recv()
                    if "error" in response:
                        raise RuntimeError(response["error"])
                    actions = np.asarray(response["actions"], dtype=np.float32)
                    expected = (4, int(identity["horizon"]), 7)
                    if actions.shape != expected or not np.all(np.isfinite(actions)):
                        raise ValueError(f"invalid policy output: {actions.shape}")
                    predictions = dict(zip(CORNER_ORDER, actions))
                    teachers = {
                        name: np.asarray(
                            anchors[
                                anchor_key(
                                    str(corners[name]["physical_edge"]),
                                    state_index,
                                    "action",
                                )
                            ],
                            dtype=np.float32,
                        )
                        for name in ("base", "source_anchor", "target_anchor")
                    }
                    metrics = action_transport_metrics(predictions, teachers)
                    rows.append(
                        {
                            "tetrad_id": tetrad["tetrad_id"],
                            "canonical_state_index": state_index,
                            "withheld_edge": str(
                                corners["fourth_anchor"]["instruction_edge"]
                            ),
                            "seed": int(seed),
                            **metrics,
                        }
                    )
    finally:
        try:
            connection.send({"op": "close"})
        except (BrokenPipeError, EOFError, OSError):
            pass
        connection.close()

    metric_names = list(action_transport_metrics(
        {name: np.zeros((1, 1), np.float32) for name in CORNER_ORDER},
        {name: np.zeros((1, 1), np.float32) for name in CORNER_ORDER[:3]},
    ))
    summary = {
        name: float(np.mean([row[name] for row in rows])) for name in metric_names
    }
    payload = {
        "schema_version": 1,
        "status": "complete",
        "training_view": str(args.training_view),
        "policy_identity": identity,
        "tetrad_count": len(tetrads),
        "seeds": list(args.seeds),
        "note": "Fourth-corner expert actions are not loaded; pseudo closure is diagnostic only.",
        "summary": summary,
        "rows": rows,
    }
    atomic_write(args.output, payload)
    print(json.dumps({"output": str(args.output), "row_count": len(rows), **summary}))


if __name__ == "__main__":
    main()
