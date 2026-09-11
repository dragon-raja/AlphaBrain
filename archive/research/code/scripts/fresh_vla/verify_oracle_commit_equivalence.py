from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

TIMING_KEYS = {
    "commit_method",
    "inference_wall_seconds",
    "client_policy_wall_seconds",
    "server_predict_action_wall_seconds",
    "commit_selector_wall_seconds",
    "episode_wall_seconds",
    "mean_inference_seconds_per_call",
}


def semantic_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in row.items() if key not in TIMING_KEYS}
    result["commit_trace"] = [
        {key: value for key, value in trace.items() if key not in {"source", "boundary_step"}}
        for trace in row.get("commit_trace", [])
    ]
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_run(
    branch_run: Path,
    feedback_run: Path,
    *,
    verify_videos: bool,
    require_byte_identical_videos: bool = False,
) -> dict[str, Any]:
    result = {"evaluations": {}, "videos": None}
    for name in ("closed_loop_isolated.json", "closed_loop_end_to_end.json", "deterministic_reach.json"):
        branch = json.loads((branch_run / name).read_text())
        feedback = json.loads((feedback_run / name).read_text())
        branch_rows = [semantic_row(row) for row in branch["rows"]]
        feedback_rows = [semantic_row(row) for row in feedback["rows"]]
        equal = branch_rows == feedback_rows
        result["evaluations"][name] = {"row_count": len(branch_rows), "semantic_equal": equal}
        if not equal:
            raise ValueError(f"Oracle semantic mismatch: {branch_run} vs {feedback_run} in {name}")

    if verify_videos:
        branch_videos = {
            str(path.relative_to(branch_run / "videos")): path for path in (branch_run / "videos").rglob("*.mp4")
        }
        feedback_videos = {
            str(path.relative_to(feedback_run / "videos")): path for path in (feedback_run / "videos").rglob("*.mp4")
        }
        if branch_videos.keys() != feedback_videos.keys():
            raise ValueError("Oracle video sets do not match")
        mismatches = [
            name
            for name in sorted(branch_videos)
            if file_sha256(branch_videos[name]) != file_sha256(feedback_videos[name])
        ]
        result["videos"] = {"count": len(branch_videos), "byte_identical": not mismatches, "mismatches": mismatches}
        if mismatches and require_byte_identical_videos:
            raise ValueError(f"Oracle videos differ: {mismatches[:3]}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify equality of the two collapsed Oracle commit schedules")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/libero-oracle-commit-final-v1"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument(
        "--require-byte-identical-videos",
        action="store_true",
        help="Fail on encoded MP4 byte drift after semantic row equivalence has passed",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = {}
    for seed in args.seeds:
        results[str(seed)] = verify_run(
            args.runs_root / f"oracle_commit_oracle_branch_safe_commit_seed{seed}",
            args.runs_root / f"oracle_commit_oracle_feedback_reveal_commit_seed{seed}",
            verify_videos=not args.skip_videos,
            require_byte_identical_videos=args.require_byte_identical_videos,
        )
    video_results = [result["videos"] for result in results.values() if result["videos"] is not None]
    video_mismatch_count = sum(len(result["mismatches"]) for result in video_results)
    payload = {
        "reason": (
            "the current manifest has feedback_reveal_time == action_divergence_time == event_time "
            "for every test group"
        ),
        "seeds": list(args.seeds),
        "equivalent": True,
        "equivalence_definition": (
            "all scheduler traces and behavior/result fields are identical after excluding method labels and "
            "wall-clock timing; encoded video bytes are audited separately"
        ),
        "video_byte_identical": all(result["byte_identical"] for result in video_results),
        "video_mismatch_count": video_mismatch_count,
        "results": results,
    }
    output = args.output or args.runs_root / "oracle_equivalence.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "equivalent": True,
                "output": str(output),
                "video_byte_identical": payload["video_byte_identical"],
                "video_mismatch_count": video_mismatch_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
