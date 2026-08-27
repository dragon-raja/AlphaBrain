#!/usr/bin/env python3
"""Freeze outcome-blind selectors on dense test-state visibility scans."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROTOCOL_SCHEMA = "dsol_constructed_dense_view_oracle_protocol_v1"
SELECTOR_METHODS = (
    "canonical",
    "visibility_mean",
    "visibility_min_entity",
    "visibility_hmean_entity",
    "visibility_gain_gated",
    "validation_global_fixed",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def visibility_features(record: Mapping[str, Any]) -> dict[str, float]:
    visibility = record["visibility"]
    cameras = list(visibility["camera_names"])
    entities = list(visibility["entity_names"])
    entity_means = []
    for entity in entities:
        entity_means.append(
            float(
                np.mean(
                    [
                        visibility["per_camera"][camera]["entities"][entity][
                            "visible_fraction"
                        ]
                        for camera in cameras
                    ]
                )
            )
        )
    minimum = min(entity_means) if entity_means else 0.0
    harmonic = (
        0.0
        if not entity_means or any(value <= 0 for value in entity_means)
        else float(len(entity_means) / sum(1.0 / value for value in entity_means))
    )
    return {
        "visibility_mean": float(visibility["score"]),
        "visibility_min_entity": float(minimum),
        "visibility_hmean_entity": float(harmonic),
    }


def build(
    protocol: Mapping[str, Any],
    scans: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    dense_protocol_path: Path,
    global_fixed_candidate: str,
    visibility_gain_threshold: float,
) -> dict[str, Any]:
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("status") != "PASS":
        raise ValueError("dense test protocol must use the expected schema and PASS")
    if protocol.get("split") != "test":
        raise ValueError("selector protocol may only consume the test split")
    base_specs = {
        (str(row["pair_key"]), str(row["selected_candidate_id"])): dict(row)
        for row in protocol["specs"]
    }
    pair_keys = sorted({pair_key for pair_key, _candidate in base_specs})
    if set(pair_keys) != set(scans):
        raise ValueError("test scan states differ from the dense test protocol")
    specs = []
    selected_states = []
    for pair_key in pair_keys:
        operational = {
            str(record["pose_id"]): dict(record)
            for record in scans[pair_key]
            if (pair_key, str(record["pose_id"])) in base_specs
        }
        if len(operational) != int(protocol["candidate_count"]):
            raise ValueError(f"{pair_key} does not contain the complete operational bank")
        if "canonical" not in operational or global_fixed_candidate not in operational:
            raise ValueError("canonical or validation-global candidate is absent")
        featured = {
            candidate_id: {**record, **visibility_features(record)}
            for candidate_id, record in operational.items()
        }
        selections = {"canonical": "canonical"}
        for method in (
            "visibility_mean",
            "visibility_min_entity",
            "visibility_hmean_entity",
        ):
            selections[method] = max(
                featured,
                key=lambda candidate_id, method=method: (
                    float(featured[candidate_id][method]),
                    candidate_id,
                ),
            )
        best_mean = selections["visibility_mean"]
        visibility_gain = (
            featured[best_mean]["visibility_mean"]
            - featured["canonical"]["visibility_mean"]
        )
        selections["visibility_gain_gated"] = (
            best_mean if visibility_gain >= visibility_gain_threshold else "canonical"
        )
        selections["validation_global_fixed"] = global_fixed_candidate
        if set(selections) != set(SELECTOR_METHODS):
            raise AssertionError("selector methods changed unexpectedly")
        selected_states.append(
            {
                "pair_key": pair_key,
                "task_id": str(base_specs[(pair_key, "canonical")]["task_id"]),
                "source_episode_id": str(
                    base_specs[(pair_key, "canonical")]["episode_id_source"]
                ),
                "selections": selections,
                "visibility_gain": float(visibility_gain),
            }
        )
        for method in SELECTOR_METHODS:
            candidate_id = selections[method]
            selected = dict(base_specs[(pair_key, candidate_id)])
            identity = f"{pair_key}::{method}::{candidate_id}"
            selected.update(
                {
                    "condition": f"selector__{method}",
                    "diagnostic_role": "dense_test_frozen_selector",
                    "episode_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
                    "selection_metadata": {
                        **selected.get("selection_metadata", {}),
                        "selector_method": method,
                        "selector_frozen_before_test_outcomes": True,
                        "visibility_features": {
                            key: featured[candidate_id][key]
                            for key in (
                                "visibility_mean",
                                "visibility_min_entity",
                                "visibility_hmean_entity",
                            )
                        },
                        "visibility_gain_threshold": visibility_gain_threshold,
                        "validation_global_fixed_candidate": global_fixed_candidate,
                    },
                }
            )
            specs.append(selected)
    return {
        "schema": "dsol_dense_test_frozen_selector_protocol_v1",
        "status": "PASS",
        "analysis_role": "independent_test_selector_comparison",
        "split": "test",
        "selection_uses_test_policy_outcomes": False,
        "selector_methods": list(SELECTOR_METHODS),
        "selector_method_count": len(SELECTOR_METHODS),
        "selected_state_count": len(pair_keys),
        "source_episode_count": len(
            {row["source_episode_id"] for row in selected_states}
        ),
        "episode_count": len(specs),
        "candidate_count": int(protocol["candidate_count"]),
        "visibility_gain_threshold": visibility_gain_threshold,
        "validation_global_fixed_candidate": global_fixed_candidate,
        "dense_test_protocol": str(dense_protocol_path.resolve()),
        "dense_test_protocol_sha256": sha256(dense_protocol_path),
        "selected_states": selected_states,
        "specs": specs,
    }


def load_scans(patterns: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for pattern in patterns:
        for ledger in glob.glob(pattern):
            with open(ledger, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("status") != "PASS":
                        raise ValueError(f"test feature scan did not PASS: {row['scan_id']}")
                    scan = json.loads(
                        (Path(row["output_dir"]) / "scan.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    if scan.get("status") != "PASS":
                        raise ValueError(f"test state scan did not PASS: {row['scan_id']}")
                    result[str(row["scan_id"])] = list(scan["records"])
    if not result:
        raise FileNotFoundError("no test feature scan rows matched")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_ledgers", nargs="+")
    parser.add_argument("--dense-test-protocol", type=Path, required=True)
    parser.add_argument("--global-fixed-candidate", required=True)
    parser.add_argument("--visibility-gain-threshold", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.dense_test_protocol.read_text(encoding="utf-8"))
    payload = build(
        protocol,
        load_scans(args.scan_ledgers),
        dense_protocol_path=args.dense_test_protocol,
        global_fixed_candidate=args.global_fixed_candidate,
        visibility_gain_threshold=args.visibility_gain_threshold,
    )
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "states": payload["selected_state_count"],
                "episodes": payload["episode_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
