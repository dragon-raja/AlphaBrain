from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from build_libero_bind_decision_view import decision_anchor_index
from build_libero_bind_training_view import padded_action_chunk


ACTION_KEY = re.compile(
    r"^(?P<edge>.+)__state_(?P<state>[0-9]+)__"
    r"(?P<decision>source_select|target_select)__action$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_action_key(key: str) -> tuple[str, int, str]:
    match = ACTION_KEY.fullmatch(key)
    if match is None:
        raise ValueError(f"unsupported decision action key: {key!r}")
    return (
        match.group("edge"),
        int(match.group("state")),
        match.group("decision"),
    )


def build_horizon_view(
    source_view: Path,
    output_dir: Path,
    *,
    action_horizon: int,
) -> Mapping[str, object]:
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    source_manifest = json.loads((source_view / "manifest.json").read_text())
    old_horizon = int(source_manifest["action_horizon"])
    if action_horizon == old_horizon:
        raise ValueError("new horizon must differ from the source horizon")
    collection_root = Path(source_manifest["source_collection"])
    collection_manifest = json.loads((collection_root / "manifest.json").read_text())
    withheld_edges = set(
        map(str, source_manifest["leakage_guard"]["withheld_action_edges"])
    )
    rows = {
        (str(row["edge_id"]), int(row["canonical_state_index"])): row
        for row in collection_manifest["rows"]
        if bool(row.get("success")) and bool(row.get("action_supervised"))
    }

    source_anchors_path = source_view / str(source_manifest["anchors_file"])
    anchors: dict[str, np.ndarray] = {}
    action_audit = []
    with np.load(source_anchors_path, allow_pickle=False) as source_anchors:
        for key in source_anchors.files:
            value = np.asarray(source_anchors[key])
            if not key.endswith("__action"):
                anchors[key] = value
                continue
            edge_id, state_index, decision_point = parse_action_key(key)
            if edge_id in withheld_edges:
                raise ValueError(f"withheld action appeared in source anchors: {edge_id}")
            row = rows.get((edge_id, state_index))
            if row is None:
                raise ValueError(
                    f"missing successful supervised episode for {edge_id} state {state_index}"
                )
            with np.load(collection_root / str(row["episode_file"]), allow_pickle=False) as episode:
                frame = decision_anchor_index(episode["phase"], decision_point)
                chunk = padded_action_chunk(
                    np.asarray(episode["actions"]), frame, action_horizon
                )
            anchors[key] = chunk
            action_audit.append(
                {
                    "anchor_key": key,
                    "edge_id": edge_id,
                    "canonical_state_index": state_index,
                    "decision_point": decision_point,
                    "frame_index": frame,
                    "episode_file": str(row["episode_file"]),
                }
            )

    expected_action_keys = [key for key in anchors if key.endswith("__action")]
    if not expected_action_keys:
        raise ValueError("source view contains no decision action anchors")
    if any(anchors[key].shape != (action_horizon, 7) for key in expected_action_keys):
        raise ValueError("re-sliced action anchors have inconsistent shapes")

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
                "schema_version": max(3, int(source_manifest.get("schema_version", 1))),
                "source_view": str(source_view),
                "records_file": "records.jsonl",
                "anchors_file": "anchors.npz",
                "action_horizon": action_horizon,
                "records_sha256": _sha256(records_source),
                "horizon_view_audit": {
                    "source_action_horizon": old_horizon,
                    "target_action_horizon": action_horizon,
                    "action_anchor_count": len(action_audit),
                    "non_action_anchors_copied_unchanged": True,
                    "records_copied_unchanged": True,
                    "action_sources": "successful supervised physical edges only",
                    "actions": action_audit,
                },
                "leakage_guard": {
                    **dict(source_manifest["leakage_guard"]),
                    "fourth_corner_actions_loaded": False,
                    "horizon_reslice_uses_teacher_qa": False,
                    "horizon_reslice_action_edges": sorted(
                        {row["edge_id"] for row in action_audit}
                    ),
                },
            }
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if _sha256(records_target) != manifest["records_sha256"]:
            raise ValueError("horizon view changed records.jsonl")
        staging.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-slice observed LIBERO-Bind decision actions at a fixed horizon"
    )
    parser.add_argument("--source-view", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, required=True)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    report = build_horizon_view(
        args.source_view,
        args.output_dir,
        action_horizon=args.action_horizon,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "record_count": int(report["record_count"]),
                "tetrad_count": int(report["tetrad_count"]),
                "action_horizon": int(report["action_horizon"]),
                "records_sha256": str(report["records_sha256"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

