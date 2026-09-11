from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from accel_core import analyze_selected_relations


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _candidate_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("candidates"), list):
        rows = payload["candidates"]
    else:
        raise ValueError("candidate metadata must be a list or contain a candidates list")
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each candidate metadata row must be an object")
        candidate_id = row.get("candidate_id", row.get("pose_id"))
        if candidate_id is None:
            raise ValueError("candidate metadata row is missing candidate_id/pose_id")
        result.append({**dict(row), "candidate_id": str(candidate_id)})
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relate an Accel-selected view to frozen reference view sets."
    )
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--candidate-metadata", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
    candidates = _candidate_rows(
        json.loads(args.candidate_metadata.read_text(encoding="utf-8"))
    )
    references = json.loads(args.references.read_text(encoding="utf-8"))
    if not isinstance(references, Mapping):
        raise ValueError("references must be a JSON object")
    result = analyze_selected_relations(ranking, candidates, references)
    _atomic_json(args.output, result)


if __name__ == "__main__":
    main()
