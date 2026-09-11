#!/usr/bin/env python3
"""Run the full failure-bank search and frozen-candidate confirmation sequentially."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import time
from pathlib import Path

import failure_view_search as study


REPO = Path(__file__).resolve().parents[2]
CHECKPOINT = Path(
    "/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000/final_model"
)
LAUNCHER = REPO / "scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh"


class ScalarTracker:
    """Metrics-only W&B with bounded online initialization and offline fallback."""

    def __init__(self, root, config):
        self.run = None
        self.root = root
        self.step = 0
        self.mode = "local_json_only"
        try:
            import wandb

            settings = wandb.Settings(
                console="off",
                disable_code=True,
                disable_git=True,
                x_disable_stats=True,
                x_disable_meta=True,
                init_timeout=10,
            )
            for mode in ["online", "offline"]:
                try:
                    self.run = wandb.init(
                        project="ai2r-view-revalidation",
                        name="failure-view-search-20260831",
                        dir=str(root),
                        mode=mode,
                        config=config,
                        settings=settings,
                    )
                    self.mode = mode
                    break
                except Exception:
                    self.run = None
        except Exception:
            self.run = None
        study.atomic_json(
            root / "tracking_status.json", {"mode": self.mode, "metrics_only": True, "uploads_media_or_weights": False}
        )

    def log(self, values):
        self.step += 1
        if self.run is not None:
            try:
                self.run.log(values, step=self.step)
            except Exception:
                self.run = None
                self.mode = "local_json_only"

    def finish(self, values):
        if self.run is not None:
            try:
                self.run.summary.update(values)
                self.run.finish()
            except Exception:
                pass


def check_ports(base_port):
    for port in range(base_port, base_port + 8):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"evaluation port already in use: {port}")


def execute(root, phase, protocol_path, *, port, tracker, attempt_limit=2):
    protocol = study.read(protocol_path)
    output = root / phase
    output.mkdir(parents=True, exist_ok=True)
    if len(study.load_run(protocol, output, require_complete=False)) == protocol["episode_count"]:
        return
    for attempt in range(attempt_limit):
        check_ports(port)
        env = dict(os.environ)
        env.update(
            CHECKPOINT=str(CHECKPOINT),
            OUTPUT_DIR=str(output),
            PROTOCOL=str(protocol_path),
            POLICY_BACKEND="alphabrain",
            GPU_COUNT="8",
            EVAL_WORKER_COUNT="32",
            DSOL_GPU_DEVICES="0,1,2,3,4,5,6,7",
            BASE_PORT=str(port),
            REPLAN_STEPS="5",
            WAIT_STEPS="0",
            EVAL_SEED="20260861",
            VIDEO_EPISODES="1" if phase == "discovery" else "1000000",
            RUN_ANALYSIS="0",
            KEEPALIVE_MODE="preserve",
        )
        started = time.monotonic()
        with (output / f"launcher-attempt-{attempt + 1}.log").open("a") as log:
            process = subprocess.Popen(["bash", str(LAUNCHER)], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    time.sleep(30)
                    try:
                        rows = study.load_run(protocol, output, require_complete=False)
                    except (ValueError, KeyError) as exc:
                        # A line may be in flight; durable validation is repeated after worker exit.
                        if not isinstance(exc, ValueError) or "JSON" not in type(exc).__name__:
                            raise
                        continue
                    status = {
                        "phase": phase,
                        "completed": len(rows),
                        "expected": protocol["episode_count"],
                        "elapsed_seconds_this_attempt": round(time.monotonic() - started),
                        "attempt": attempt + 1,
                        "keepalive_mode": "preserve",
                        "status": "RUNNING",
                    }
                    study.atomic_json(root / "progress.json", status)
                    tracker.log(
                        {
                            f"{phase}/completed_episodes": len(rows),
                            f"{phase}/expected_episodes": protocol["episode_count"],
                        }
                    )
                    print(f"{phase} {len(rows)}/{protocol['episode_count']}", flush=True)
            except BaseException:
                process.terminate()
                process.wait(timeout=60)
                raise
        if process.returncode == 0:
            study.load_run(protocol, output)
            return
        study.atomic_json(
            root / "last_attempt_failure.json",
            {"phase": phase, "attempt": attempt + 1, "returncode": process.returncode},
        )
        time.sleep(15)
    raise RuntimeError(f"{phase} launcher failed after {attempt_limit} resumable attempts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=study.DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root
    study.prepare(root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    config = {
        "model": "Pi0.5 Broad64 practical",
        "training_seed": 41,
        "discovery_noise_count": 3,
        "confirmation_noise_count": 5,
        "states": 20,
        "tasks": 6,
        "source_groups": 17,
        "candidate_count": 97,
        "replan_k": 5,
        "gpu_count": 8,
        "git_commit": commit,
        "hostname": socket.gethostname(),
        "data_version": "failure-pool-v1-20260831",
    }
    study.frozen_write(
        root / "execution_identity.json",
        {
            "git_commit": commit,
            "checkpoint": str(CHECKPOINT),
            "source_sha256": {
                str(p): study.sha(p)
                for p in [
                    Path(__file__),
                    Path(study.__file__),
                    LAUNCHER,
                    REPO / "scripts/dsol_paper1/evaluate_dsol_libero_hdf5_views.py",
                    REPO / "scripts/cabi_vla/serve_alphabrain_pi05_websocket.py",
                ]
            },
        },
    )
    tracker = ScalarTracker(root, config)
    try:
        execute(root, "discovery", root / "protocols/discovery-three-noise.json", port=21300, tracker=tracker)
        study.freeze_confirmation(root)
        execute(root, "confirmation", root / "protocols/confirmation-five-noise.json", port=21400, tracker=tracker)
        analysis = study.summarize(root)
        subprocess.run(
            [
                os.sys.executable,
                str(REPO / "scripts/dsol_paper1/plot_failure_view_search.py"),
                "analysis",
                "--root",
                str(root),
            ],
            cwd=REPO,
            check=True,
        )
        study.atomic_json(
            root / "progress.json",
            {
                "status": "COMPLETE",
                "discovery_episodes": 5820,
                "confirmation_episodes": 800,
                "analysis": str(root / "analysis/final_analysis.json"),
            },
        )
        tracker.finish(
            {
                "canonical_success": analysis["canonical_success"],
                "frozen_top1_success": analysis["frozen_top1_success"],
                "frozen_top1_advantage_pp": analysis["primary_paired_comparison"]["source_equal_advantage_pp"],
            }
        )
        print("failure_view_search_complete", flush=True)
    except BaseException as exc:
        study.atomic_json(root / "controller_error.json", {"status": "ERROR", "exception_type": type(exc).__name__})
        tracker.finish({"status": "ERROR"})
        raise


if __name__ == "__main__":
    main()
