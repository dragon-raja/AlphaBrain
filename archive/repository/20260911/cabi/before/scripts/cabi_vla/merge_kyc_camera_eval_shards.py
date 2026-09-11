from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


EpisodeKey = tuple[str, str, int, int]


def episode_key(row: Mapping[str, Any]) -> EpisodeKey:
    return (
        str(row["edge_id"]),
        str(row["camera_pose"]),
        int(row["execution_horizon"]),
        int(row["canonical_state_index"]),
    )


def _require_equal(payloads: Sequence[Mapping[str, Any]], key: str) -> Any:
    values = [payload.get(key) for payload in payloads]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"fragment {key} values do not match")
    return values[0]


def merge_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    expected_edges: Sequence[str],
    expected_state_indices: Sequence[int],
    expected_execution_horizons: Sequence[int],
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("at least one fragment is required")
    if not expected_edges or not expected_state_indices or not expected_execution_horizons:
        raise ValueError("expected axes cannot be empty")

    camera_config = _require_equal(payloads, "camera_config")
    policy_identity = _require_equal(payloads, "policy_identity")
    if not isinstance(camera_config, Mapping):
        raise ValueError("fragments must contain a camera_config object")
    if not isinstance(policy_identity, Mapping):
        raise ValueError("fragments must contain a policy_identity object")

    pose_names = [str(pose["name"]) for pose in camera_config.get("poses", [])]
    if not pose_names or len(pose_names) != len(set(pose_names)):
        raise ValueError("camera_config poses must be non-empty and unique")

    rows_by_key: dict[EpisodeKey, dict[str, Any]] = {}
    for payload in payloads:
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("every fragment must contain a rows list")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("fragment rows must be objects")
            key = episode_key(row)
            if key in rows_by_key:
                raise ValueError(f"duplicate episode key across fragments: {key}")
            rows_by_key[key] = dict(row)

    expected_keys = set(
        itertools.product(
            [str(value) for value in expected_edges],
            pose_names,
            [int(value) for value in expected_execution_horizons],
            [int(value) for value in expected_state_indices],
        )
    )
    actual_keys = set(rows_by_key)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "fragment episode grid is incomplete: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_sample={missing[:3]} unexpected_sample={unexpected[:3]}"
        )

    complete_payloads = [payload for payload in payloads if payload.get("status") == "complete"]
    if not complete_payloads:
        raise ValueError("at least one complete fragment is required as metadata template")
    result = copy.deepcopy(dict(complete_payloads[0]))
    edge_order = {value: index for index, value in enumerate(expected_edges)}
    pose_order = {value: index for index, value in enumerate(pose_names)}
    horizon_order = {
        int(value): index for index, value in enumerate(expected_execution_horizons)
    }
    state_order = {
        int(value): index for index, value in enumerate(expected_state_indices)
    }
    result["status"] = "complete"
    result["edges"] = [str(value) for value in expected_edges]
    result["poses"] = pose_names
    result["state_indices"] = [int(value) for value in expected_state_indices]
    result["execution_horizons"] = [
        int(value) for value in expected_execution_horizons
    ]
    result["expected_episode_count"] = len(expected_keys)
    result["camera_config"] = copy.deepcopy(camera_config)
    result["policy_identity"] = copy.deepcopy(policy_identity)
    result["rows"] = sorted(
        rows_by_key.values(),
        key=lambda row: (
            edge_order[str(row["edge_id"])],
            pose_order[str(row["camera_pose"])],
            horizon_order[int(row["execution_horizon"])],
            state_order[int(row["canonical_state_index"])],
        ),
    )
    return result


def _parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("comma-separated value cannot be empty")
    return values


def _parse_int_csv(value: str) -> list[int]:
    try:
        return [int(item) for item in _parse_csv(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite merged evaluation: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly merge disjoint KYC camera-evaluation fragments"
    )
    parser.add_argument("--fragment", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-edges", type=_parse_csv, required=True)
    parser.add_argument("--expected-state-indices", type=_parse_int_csv, required=True)
    parser.add_argument(
        "--expected-execution-horizons",
        type=_parse_int_csv,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = [json.loads(path.read_text()) for path in args.fragment]
    merged = merge_payloads(
        payloads,
        expected_edges=args.expected_edges,
        expected_state_indices=args.expected_state_indices,
        expected_execution_horizons=args.expected_execution_horizons,
    )
    _atomic_write(args.output, merged)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episode_count": len(merged["rows"]),
                "fragment_count": len(payloads),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
