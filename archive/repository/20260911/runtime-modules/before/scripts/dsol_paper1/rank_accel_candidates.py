from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from accel_core import rank_accel_candidates


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank fixed-state candidates from a Pi0.5 velocity-trace NPZ."
    )
    parser.add_argument("--trace-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    with np.load(args.trace_npz, allow_pickle=False) as payload:
        required = {"candidate_ids", "velocity_trace", "initial_noise"}
        missing = required - set(payload.files)
        if missing:
            raise KeyError(f"trace NPZ is missing {sorted(missing)}")
        ranking = rank_accel_candidates(
            [str(value) for value in payload["candidate_ids"].tolist()],
            payload["velocity_trace"],
            initial_noise=payload["initial_noise"],
        )
        if "flow_times" in payload.files:
            ranking["flow_times"] = [float(value) for value in payload["flow_times"]]
    _atomic_json(args.output, ranking)


if __name__ == "__main__":
    main()
