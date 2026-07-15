from __future__ import annotations

import argparse
import json
from pathlib import Path

from transcode_videos import inspect_video, is_compatible

METHODS = (
    "oracle_branch_safe_commit",
    "oracle_feedback_reveal_commit",
    "gripper_commit",
    "random_matched_commit",
    "self_consistency_commit",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify completeness and codec integrity of Oracle commit artifacts")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/libero-oracle-commit-final-v1"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--skip-frame-decode", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = {}
    git_shas = set()
    total_videos = 0
    scheduler_boundaries = {}
    for method in METHODS:
        for seed in args.seeds:
            run = args.runs_root / f"oracle_commit_{method}_seed{seed}"
            identity = None
            outputs = {}
            payloads = {}
            for name, expected in (
                ("closed_loop_isolated.json", 26),
                ("closed_loop_end_to_end.json", 26),
                ("deterministic_reach.json", 13),
            ):
                payload = json.loads((run / name).read_text())
                if payload.get("status") != "complete" or len(payload.get("rows", ())) != expected:
                    raise ValueError(f"incomplete artifact: {run / name}")
                if payload.get("policy_checkpoint_sha256_source") != "sha256sum_preflight_verified":
                    raise ValueError(f"checkpoint hash was not measured by the runner: {run / name}")
                current_identity = (
                    payload.get("git_sha"),
                    payload.get("policy_checkpoint_realpath"),
                    payload.get("policy_checkpoint_sha256"),
                    payload.get("policy_model_size_bytes"),
                    json.dumps(payload.get("policy_runtime"), sort_keys=True),
                )
                if identity is not None and current_identity != identity:
                    raise ValueError(f"runtime identity changed within {run}")
                identity = current_identity
                outputs[name] = expected
                payloads[name] = payload
                git_shas.add(payload.get("git_sha"))

            if method == "oracle_branch_safe_commit":
                scheduler_boundaries[(method, seed)] = sorted(
                    int(trace["commit_length"])
                    for row in payloads["closed_loop_end_to_end.json"]["rows"]
                    for trace in row["commit_trace"]
                    if trace.get("interrupted_by_oracle_event")
                )
            elif method == "random_matched_commit":
                scheduler_boundaries[(method, seed)] = sorted(
                    int(trace["planned_commit_length"])
                    for row in payloads["closed_loop_end_to_end.json"]["rows"]
                    for trace in row["commit_trace"]
                    if trace.get("boundary_step") is not None
                    and int(trace["global_step"]) < int(trace["boundary_step"])
                    and int(trace["global_step"]) + int(trace["planned_commit_length"]) == int(trace["boundary_step"])
                )

            expected_videos = {}
            for filename, protocol in (
                ("closed_loop_isolated.json", "isolated"),
                ("closed_loop_end_to_end.json", "end_to_end"),
            ):
                paired_steps = {}
                for row in payloads[filename]["rows"]:
                    pair_id = str(row["pair_id"])
                    paired_steps.setdefault(pair_id, []).append(int(row["completion_steps"]))
                if any(len(steps) != 2 for steps in paired_steps.values()):
                    raise ValueError(f"unpaired video rows in {run / filename}")
                for pair_id, steps in paired_steps.items():
                    expected_videos[f"{protocol}/{protocol}-{pair_id}.mp4"] = max(steps) + 1
            for row in payloads["deterministic_reach.json"]["rows"]:
                pair_id = str(row["pair_id"])
                expected_videos[f"reach/reach-{pair_id}.mp4"] = int(row["completion_steps"]) + 1

            videos = {str(path.relative_to(run / "videos")): path for path in (run / "videos").rglob("*.mp4")}
            if videos.keys() != expected_videos.keys():
                missing = sorted(expected_videos.keys() - videos.keys())
                extra = sorted(videos.keys() - expected_videos.keys())
                raise ValueError(f"video set mismatch in {run}: missing={missing[:3]} extra={extra[:3]}")
            for relative, video in videos.items():
                info = inspect_video(video, decode=not args.skip_frame_decode)
                if not is_compatible(info) or info.frame_count != expected_videos[relative]:
                    raise ValueError(f"invalid H.264 video: {video}: {info}")
            total_videos += len(videos)
            runs[f"{method}:seed{seed}"] = {
                "outputs": outputs,
                "videos": len(videos),
                "identity": identity,
            }
    if None in git_shas or len(git_shas) != 1:
        raise ValueError(f"runs do not share one frozen Git SHA: {git_shas}")
    for seed in args.seeds:
        oracle = scheduler_boundaries[("oracle_branch_safe_commit", seed)]
        random_matched = scheduler_boundaries[("random_matched_commit", seed)]
        if oracle != random_matched:
            raise ValueError(f"random commit schedule is not distribution-matched for seed {seed}")
    partials = sorted(str(path) for path in args.runs_root.rglob("*.partial.json"))
    if partials:
        raise ValueError(f"partial outputs remain: {partials[:3]}")
    payload = {
        "complete": True,
        "run_count": len(runs),
        "expected_run_count": len(METHODS) * len(args.seeds),
        "video_count": total_videos,
        "expected_video_count": len(METHODS) * len(args.seeds) * 39,
        "random_schedule_distribution_match": True,
        "git_sha": next(iter(git_shas)),
        "runs": runs,
    }
    output = args.output or args.runs_root / "artifact_verification.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"complete": True, "output": str(output), "video_count": total_videos}, sort_keys=True))


if __name__ == "__main__":
    main()
