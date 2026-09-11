from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from build_libero_bind_training_view import padded_action_chunk, transport_anchor_index


DECISION_POINTS = {
    "source_select": "source",
    "target_select": "target",
}


def decision_anchor_index(phases: np.ndarray, decision_point: str) -> int:
    values = np.asarray(phases).astype(str)
    if decision_point == "source_select":
        starts = np.flatnonzero(values == "episode_start")
        if len(starts) != 1 or int(starts[0]) != 0:
            raise ValueError("source selection requires one episode_start at frame zero")
        return 0
    if decision_point == "target_select":
        return transport_anchor_index(values)
    raise ValueError(f"unknown decision point: {decision_point!r}")


def decision_anchor_key(
    edge_id: str,
    state_index: int,
    decision_point: str,
    field: str,
) -> str:
    if decision_point not in DECISION_POINTS:
        raise ValueError(f"unknown decision point: {decision_point!r}")
    return f"{edge_id}__state_{state_index:02d}__{decision_point}__{field}"


def _conditioning_equal(
    anchors: Mapping[str, np.ndarray],
    left_edge: str,
    right_edge: str,
    state_index: int,
    decision_point: str,
) -> bool:
    return all(
        np.array_equal(
            anchors[decision_anchor_key(left_edge, state_index, decision_point, field)],
            anchors[decision_anchor_key(right_edge, state_index, decision_point, field)],
        )
        for field in ("agentview", "wrist", "state")
    )


def _action_mse(
    anchors: Mapping[str, np.ndarray],
    left_edge: str,
    right_edge: str,
    state_index: int,
    decision_point: str,
) -> float:
    left = anchors[
        decision_anchor_key(left_edge, state_index, decision_point, "action")
    ]
    right = anchors[
        decision_anchor_key(right_edge, state_index, decision_point, "action")
    ]
    return float(np.square(np.asarray(left) - np.asarray(right)).mean())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_decision_view(source_view: Path, output_dir: Path) -> Mapping[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    source_manifest = json.loads((source_view / "manifest.json").read_text())
    collection_root = Path(source_manifest["source_collection"])
    collection_manifest = json.loads((collection_root / "manifest.json").read_text())
    action_horizon = int(source_manifest["action_horizon"])

    required_edges = {
        str(corner["physical_edge"])
        for tetrad in source_manifest["tetrads"]
        for corner in tetrad["corners"].values()
    }
    rows = {
        (str(row["edge_id"]), int(row["canonical_state_index"])): row
        for row in collection_manifest["rows"]
        if bool(row.get("success"))
        and bool(row.get("action_supervised"))
        and str(row["edge_id"]) in required_edges
    }

    anchors: dict[str, np.ndarray] = {}
    for (edge_id, state_index), row in sorted(rows.items()):
        episode_path = collection_root / str(row["episode_file"])
        with np.load(episode_path, allow_pickle=False) as episode:
            phases = np.asarray(episode["phase"])
            for decision_point in DECISION_POINTS:
                frame = decision_anchor_index(phases, decision_point)
                values = {
                    "agentview": np.asarray(episode["agentview"][frame]),
                    "wrist": np.asarray(episode["wrist"][frame]),
                    "state": np.asarray(episode["robot_state"][frame]),
                    "action": padded_action_chunk(
                        np.asarray(episode["actions"]), frame, action_horizon
                    ),
                }
                for field, value in values.items():
                    anchors[
                        decision_anchor_key(
                            edge_id, state_index, decision_point, field
                        )
                    ] = value

    tetrads = []
    decision_audit = []
    for source_tetrad in source_manifest["tetrads"]:
        state_index = int(source_tetrad["canonical_state_index"])
        corners = source_tetrad["corners"]
        base_edge = str(corners["base"]["physical_edge"])
        source_edge = str(corners["source_anchor"]["physical_edge"])
        target_edge = str(corners["target_anchor"]["physical_edge"])
        for decision_point, role in DECISION_POINTS.items():
            intervention_edge = source_edge if role == "source" else target_edge
            if not _conditioning_equal(
                anchors,
                base_edge,
                intervention_edge,
                state_index,
                decision_point,
            ):
                raise ValueError(
                    f"{source_tetrad['tetrad_id']} has mismatched {decision_point} conditioning"
                )
            action_mse = _action_mse(
                anchors,
                base_edge,
                intervention_edge,
                state_index,
                decision_point,
            )
            if action_mse <= 1e-8:
                raise ValueError(
                    f"{source_tetrad['tetrad_id']} has no {role} action effect at {decision_point}"
                )
            tetrad = copy.deepcopy(source_tetrad)
            tetrad["tetrad_id"] = f"{source_tetrad['tetrad_id']}--{decision_point}"
            tetrad["decision_point"] = decision_point
            tetrad["transport_roles"] = [role]
            tetrads.append(tetrad)
            decision_audit.append(
                {
                    "tetrad_id": tetrad["tetrad_id"],
                    "decision_point": decision_point,
                    "transport_role": role,
                    "intervention_action_mse": action_mse,
                    "conditioning_identical": True,
                }
            )

    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        records_source = source_view / str(source_manifest["records_file"])
        records_target = staging / "records.jsonl"
        shutil.copy2(records_source, records_target)
        np.savez_compressed(staging / "anchors.npz", **anchors)
        manifest = dict(source_manifest)
        manifest.update(
            {
                "schema_version": 2,
                "source_view": str(source_view),
                "records_file": "records.jsonl",
                "anchors_file": "anchors.npz",
                "anchor_count": len(anchors) // 4,
                "tetrad_count": len(tetrads),
                "tetrads": tetrads,
                "decision_points": {
                    "source_select": {
                        "teacher_phase": "episode_start",
                        "transport_roles": ["source"],
                    },
                    "target_select": {
                        "teacher_phase": "last_pre_transport",
                        "transport_roles": ["target"],
                    },
                },
                "decision_audit": decision_audit,
                "records_sha256": _sha256(records_source),
                "leakage_guard": {
                    **dict(source_manifest["leakage_guard"]),
                    "fourth_corner_actions_loaded": False,
                    "decision_point_actions": "observed corners only",
                },
            }
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if _sha256(records_target) != manifest["records_sha256"]:
            raise ValueError("decision view changed the balanced action records")
        staging.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add phase-correct causal anchors to a balanced LIBERO-Bind view"
    )
    parser.add_argument("--source-view", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    report = build_decision_view(args.source_view, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "record_count": int(report["record_count"]),
                "tetrad_count": int(report["tetrad_count"]),
                "records_sha256": str(report["records_sha256"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
