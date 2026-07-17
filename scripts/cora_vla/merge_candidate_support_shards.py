from __future__ import annotations

import argparse
import json
from pathlib import Path

from candidate_support import summarize_group_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge disjoint CORA Gate 1 group shards")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text()) for path in args.inputs]
    rows = [row for payload in payloads for row in payload["rows"]]
    pre_feedback = [row for payload in payloads for row in payload["pre_feedback"]]
    identities = [(row["pair_id"], row["outcome"]) for row in rows]
    pre_identities = [row["pair_id"] for row in pre_feedback]
    if len(identities) != len(set(identities)):
        raise ValueError("candidate shards overlap")
    if len(pre_identities) != len(set(pre_identities)):
        raise ValueError("pre-feedback shards overlap")
    if len(rows) != 26 or len(pre_feedback) != 13:
        raise ValueError(f"formal merge requires 26 outcome rows and 13 pre-feedback rows, got {len(rows)} and {len(pre_feedback)}")
    result = dict(payloads[-1])
    result.update(
        {
            "status": "complete",
            "group_count": 13,
            "completed_rows": 26,
            "pre_feedback": sorted(pre_feedback, key=lambda row: row["pair_id"]),
            "pre_feedback_leakage_passed": all(row["passed"] for row in pre_feedback),
            "rows": sorted(rows, key=lambda row: (row["pair_id"], row["outcome"])),
        }
    )
    result["summary"] = summarize_group_rows(result["rows"], result["prefix_sizes"])
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
