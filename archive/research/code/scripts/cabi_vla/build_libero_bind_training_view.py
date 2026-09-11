from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


TETRAD_SPECS = (
    {
        "name": "white-right",
        "base": "red-left",
        "source_anchor": "white-left",
        "target_anchor": "red-right",
        "fourth_instruction": "white-right",
        "fourth_physical": "white-left",
    },
    {
        "name": "yellow_white-left",
        "base": "red-right",
        "source_anchor": "yellow_white-right",
        "target_anchor": "red-left",
        "fourth_instruction": "yellow_white-left",
        "fourth_physical": "yellow_white-right",
    },
)


def padded_action_chunk(
    actions: np.ndarray,
    start: int,
    horizon: int,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError("actions must be [T, D]")
    if start < 0 or start >= len(actions):
        raise IndexError(f"action start {start} outside episode length {len(actions)}")
    if horizon <= 0:
        raise ValueError("action horizon must be positive")
    chunk = actions[start : start + horizon]
    if len(chunk) < horizon:
        chunk = np.concatenate(
            [chunk, np.zeros((horizon - len(chunk), actions.shape[1]), np.float32)]
        )
    return chunk


def transport_anchor_index(phases: np.ndarray) -> int:
    values = np.asarray(phases).astype(str)
    transport = np.flatnonzero(values == "transport")
    if len(transport) == 0:
        raise ValueError("episode has no transport phase")
    return max(0, int(transport[0]) - 1)


def _episode_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "agentview",
        "wrist",
        "robot_state",
        "actions",
        "phase",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise KeyError(f"episode {path} is missing {missing}")
    observation_count = len(arrays["agentview"])
    if len(arrays["wrist"]) != observation_count:
        raise ValueError(f"camera length mismatch in {path}")
    if len(arrays["robot_state"]) != observation_count:
        raise ValueError(f"state length mismatch in {path}")
    if len(arrays["actions"]) + 1 != observation_count:
        raise ValueError(f"action/observation length mismatch in {path}")
    return arrays


def _anchor_key(edge_id: str, state_index: int, field: str) -> str:
    return f"{edge_id}__state_{state_index:02d}__{field}"


def anchor_conditioning_is_identical(
    anchors: Mapping[str, np.ndarray],
    left_edge: str,
    right_edge: str,
    state_index: int,
) -> bool:
    """Require exact common conditioning before treating target as the intervention."""

    return all(
        np.array_equal(
            anchors[_anchor_key(left_edge, state_index, field)],
            anchors[_anchor_key(right_edge, state_index, field)],
        )
        for field in ("agentview", "wrist", "state")
    )


def teacher_edge_quality(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, dict[str, Any]]:
    grouped: dict[str, list[bool]] = {}
    for row in rows:
        grouped.setdefault(str(row["edge_id"]), []).append(bool(row.get("success")))
    return {
        edge_id: {
            "successful": sum(outcomes),
            "total": len(outcomes),
            "success_rate": sum(outcomes) / len(outcomes),
        }
        for edge_id, outcomes in sorted(grouped.items())
    }


def supervised_collection_rows(
    rows: Iterable[Mapping[str, Any]],
    edge_by_id: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Select supervised rows using the suite contract, including failed rows."""

    supervised = []
    for row in rows:
        edge_id = str(row["edge_id"])
        if edge_id not in edge_by_id:
            raise KeyError(f"collection row references unknown edge: {edge_id}")
        expected = bool(edge_by_id[edge_id]["action_supervised"])
        if "action_supervised" in row and bool(row["action_supervised"]) != expected:
            raise ValueError(
                f"collection row supervision disagrees with suite for edge {edge_id}"
            )
        if expected:
            supervised.append(row)
    return supervised


def build_training_view(
    collection_root: Path,
    output_dir: Path,
    *,
    action_horizon: int,
    stride: int,
) -> Mapping[str, Any]:
    collection_manifest = json.loads((collection_root / "manifest.json").read_text())
    suite_root = Path(collection_manifest["suite"])
    suite_manifest = json.loads((suite_root / "manifest.json").read_text())
    edge_by_id = {edge["edge_id"]: edge for edge in suite_manifest["edges"]}
    if stride <= 0:
        raise ValueError("stride must be positive")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    all_supervised_rows = supervised_collection_rows(
        collection_manifest["rows"], edge_by_id
    )
    quality = teacher_edge_quality(all_supervised_rows)
    below_gate = {
        edge_id: row["success_rate"]
        for edge_id, row in quality.items()
        if row["success_rate"] < 0.90
    }
    if below_gate:
        raise ValueError(f"teacher edges below 90% success gate: {below_gate}")
    failed = [
        row["sample_id"] for row in all_supervised_rows if not bool(row.get("success"))
    ]
    supervised_rows = [row for row in all_supervised_rows if bool(row.get("success"))]

    records: list[dict[str, Any]] = []
    anchors: dict[str, np.ndarray] = {}
    anchor_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in supervised_rows:
        episode_path = collection_root / row["episode_file"]
        arrays = _episode_arrays(episode_path)
        edge_id = str(row["edge_id"])
        state_index = int(row["canonical_state_index"])
        for frame_index in range(0, len(arrays["actions"]), stride):
            records.append(
                {
                    "sample_id": f"{row['sample_id']}--frame-{frame_index:04d}",
                    "episode_file": row["episode_file"],
                    "edge_id": edge_id,
                    "canonical_state_index": state_index,
                    "split": row["split"],
                    "frame_index": frame_index,
                    "language_instruction": row["language_instruction"],
                }
            )

        anchor_index = transport_anchor_index(arrays["phase"])
        anchor_fields = {
            "agentview": arrays["agentview"][anchor_index],
            "wrist": arrays["wrist"][anchor_index],
            "state": arrays["robot_state"][anchor_index],
            "action": padded_action_chunk(
                arrays["actions"], anchor_index, action_horizon
            ),
        }
        for field, value in anchor_fields.items():
            anchors[_anchor_key(edge_id, state_index, field)] = np.asarray(value)
        anchor_rows[(edge_id, state_index)] = {
            "edge_id": edge_id,
            "canonical_state_index": state_index,
            "split": row["split"],
            "frame_index": anchor_index,
            "language_instruction": row["language_instruction"],
        }

    tetrads = []
    excluded_tetrads = []
    available_states = sorted({state for _, state in anchor_rows})
    for state_index in available_states:
        for spec in TETRAD_SPECS:
            observed_edges = (spec["base"], spec["source_anchor"], spec["target_anchor"])
            if any((edge_id, state_index) not in anchor_rows for edge_id in observed_edges):
                continue
            if not anchor_conditioning_is_identical(
                anchors, spec["base"], spec["target_anchor"], state_index
            ):
                excluded_tetrads.append(
                    {
                        "tetrad_id": f"state-{state_index:02d}--{spec['name']}",
                        "reason": "base_target_conditioning_mismatch",
                    }
                )
                continue
            base_action = anchors[
                _anchor_key(spec["base"], state_index, "action")
            ]
            target_action = anchors[
                _anchor_key(spec["target_anchor"], state_index, "action")
            ]
            if np.array_equal(base_action, target_action):
                excluded_tetrads.append(
                    {
                        "tetrad_id": f"state-{state_index:02d}--{spec['name']}",
                        "reason": "target_intervention_has_no_action_divergence",
                    }
                )
                continue
            fourth_edge = edge_by_id[spec["fourth_instruction"]]
            if bool(fourth_edge["action_supervised"]):
                raise ValueError("fourth-corner action edge must remain withheld")
            physical_row = anchor_rows[(spec["fourth_physical"], state_index)]
            split = anchor_rows[(spec["base"], state_index)]["split"]
            if any(anchor_rows[(edge, state_index)]["split"] != split for edge in observed_edges):
                raise ValueError("tetrad corners crossed canonical-state splits")
            tetrads.append(
                {
                    "tetrad_id": f"state-{state_index:02d}--{spec['name']}",
                    "canonical_state_index": state_index,
                    "split": split,
                    "corners": {
                        "base": {
                            "physical_edge": spec["base"],
                            "instruction_edge": spec["base"],
                            "action_supervised": True,
                        },
                        "source_anchor": {
                            "physical_edge": spec["source_anchor"],
                            "instruction_edge": spec["source_anchor"],
                            "action_supervised": True,
                        },
                        "target_anchor": {
                            "physical_edge": spec["target_anchor"],
                            "instruction_edge": spec["target_anchor"],
                            "action_supervised": True,
                        },
                        "fourth_anchor": {
                            "physical_edge": physical_row["edge_id"],
                            "instruction_edge": spec["fourth_instruction"],
                            "action_supervised": False,
                        },
                    },
                }
            )

    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / "records.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
        )
        np.savez_compressed(staging / "anchors.npz", **anchors)
        report = {
            "schema_version": 1,
            "source_collection": str(collection_root),
            "source_suite": str(suite_root),
            "action_horizon": action_horizon,
            "action_dim": int(
                next(value.shape[-1] for key, value in anchors.items() if key.endswith("__action"))
            ),
            "stride": stride,
            "record_count": len(records),
            "anchor_count": len(anchor_rows),
            "tetrad_count": len(tetrads),
            "excluded_tetrads": excluded_tetrads,
            "teacher_edge_quality": quality,
            "excluded_failed_episodes": failed,
            "records_file": "records.jsonl",
            "anchors_file": "anchors.npz",
            "edge_instructions": {
                edge_id: edge["language_instruction"] for edge_id, edge in edge_by_id.items()
            },
            "tetrads": tetrads,
            "leakage_guard": {
                "withheld_action_edges": [
                    edge_id
                    for edge_id, edge in edge_by_id.items()
                    if not bool(edge["action_supervised"])
                ],
                "fourth_corner_fields": ["image", "state", "language"],
                "fourth_corner_actions_loaded": False,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(output_dir)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a leakage-safe LIBERO-Bind view")
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    report = build_training_view(
        args.collection_root,
        args.output_dir,
        action_horizon=args.action_horizon,
        stride=args.stride,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "record_count": report["record_count"],
                "tetrad_count": report["tetrad_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
