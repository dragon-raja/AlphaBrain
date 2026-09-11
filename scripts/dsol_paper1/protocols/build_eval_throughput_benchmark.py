#!/usr/bin/env python3
"""Freeze a non-primary subset of a compact protocol for topology benchmarking."""

from __future__ import annotations

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[3]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))


import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.dsol_paper1.protocols.build_statewise_view_oracle_v2 import freeze_json
from AlphaBrain.research.dsol.data.flow_noise import sha256_file


def build_benchmark_protocol(
    source: Mapping[str, Any], *, source_path: Path, state_count: int
) -> dict[str, Any]:
    if source.get("schema") != "dsol_compact_view_matrix_protocol_v1":
        raise ValueError("source must use the compact view-matrix schema")
    blocks = list(source.get("state_blocks", []))
    if not 1 <= state_count <= len(blocks):
        raise ValueError("state_count is outside the source protocol")
    selected = blocks[:state_count]
    repeats = list(source.get("policy_repeat_ids", []))
    candidate_counts = {len(block.get("candidates", [])) for block in selected}
    if not repeats or len(candidate_counts) != 1 or min(candidate_counts) <= 0:
        raise ValueError("source protocol has an invalid matrix")
    payload = dict(source)
    payload.update(
        {
            "status": "PASS_FROZEN_NON_PRIMARY_BENCHMARK",
            "benchmark_only": True,
            "excluded_from_scientific_estimands": True,
            "source_protocol": str(source_path.resolve()),
            "source_protocol_sha256": sha256_file(source_path),
            "state_blocks": selected,
            "state_count": state_count,
            "candidate_count_per_state": next(iter(candidate_counts)),
            "episode_count": state_count * next(iter(candidate_counts)) * len(repeats),
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--state-count", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.source_protocol.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    payload = build_benchmark_protocol(
        source, source_path=source_path, state_count=args.state_count
    )
    freeze_json(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "states": payload["state_count"],
                "episodes": payload["episode_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
