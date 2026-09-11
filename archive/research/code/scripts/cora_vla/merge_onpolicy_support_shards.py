from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge CORA on-policy support shards")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text()) for path in args.inputs]
    rows = [row for payload in payloads for row in payload["rows"]]
    episodes = [row for payload in payloads for row in payload["episodes"]]
    row_ids = [(row["pair_id"], row["stage"]) for row in rows]
    episode_ids = [row["pair_id"] for row in episodes]
    if len(row_ids) != len(set(row_ids)) or len(episode_ids) != len(set(episode_ids)):
        raise ValueError("on-policy shards overlap")
    if len(episodes) != 13:
        raise ValueError(f"formal seed merge requires 13 episodes, got {len(episodes)}")
    result = dict(payloads[0])
    result.update(
        {
            "status": "complete",
            "group_offset": 0,
            "group_count": 13,
            "completed_groups": 13,
            "rows": sorted(rows, key=lambda row: (row["pair_id"], row["stage"])),
            "episodes": sorted(episodes, key=lambda row: row["pair_id"]),
        }
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
