#!/usr/bin/env bash
# New bounded static diagnostic controller. Existing rollout runner is unchanged.
set -euo pipefail
DSOL_LANDSCAPE_REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export DSOL_LANDSCAPE_REPO_ROOT PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$DSOL_LANDSCAPE_REPO_ROOT"
exec timeout --signal=TERM --kill-after=120s 129600s /alphabrain/.venv/bin/python - "$@" <<'PY_CONTROLLER'
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
import yaml


REPO = Path(os.environ["DSOL_LANDSCAPE_REPO_ROOT"])
RELEASE = REPO / "configs/dsol_paper1/matched_view_landscape_release_v1.json"
ROOT = Path("/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908")
RUNNER = REPO / "scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh"
AUDITOR = REPO / "scripts/dsol_paper1/audit_statewise_view_oracle_v2_run.py"
ANALYZER = REPO / "scripts/dsol_paper1/summarize_dsol_libero_hdf5_closed_loop.py"
ACTIVE = None
STOPPED = False


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, payload, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    else:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            temporary = Path(stream.name)
        temporary.replace(path)


def runner_code_hash():
    from scripts.dsol_paper1.shared_runtime_paths import shared_code_paths
    paths = [REPO / "scripts/cabi_vla/serve_alphabrain_pi05_websocket.py",
             REPO / "scripts/cabi_vla/serve_openpi_deterministic.py",
             REPO / "scripts/dsol_paper1/evaluate_dsol_libero_hdf5_views.py", ANALYZER, RUNNER]
    paths.extend(shared_code_paths(REPO))
    return hashlib.sha256(subprocess.check_output(["sha256sum", *map(str, paths)])).hexdigest()


def route_specs(manifest):
    table = {row["name"]: row for row in manifest["protocols"]}
    route = [("smoke", "smoke-O-development.json"),
             ("wave-00-remainder-a", "dense-O-development-wave-00-remainder-a.json"),
             ("wave-00-remainder-b", "dense-O-development-wave-00-remainder-b.json")]
    route += [(f"wave-{index:02d}", f"dense-O-development-wave-{index:02d}.json") for index in range(1, 8)]
    return [{"label": label, **table[name]} for label, name in route]


def preflight():
    release = read(RELEASE)
    require(release["root"] == str(ROOT), "root differs from frozen controller root")
    require(release["status"] == "STATIC_DIAGNOSTIC_ONLY", "not a static-only release")
    require(release["authorized_stages"] == [1, 2, 3], "unexpected authorized stages")
    require(not any(release[key] for key in ("training_release", "active_camera_release", "robocasa_release")), "out-of-scope release")
    execution = release["execution"]
    for key, value in {"gpu_count": 8, "eval_worker_count": 32, "policy_server_copies_per_gpu": 2,
                       "base_port": 26200, "replan_steps": 5, "wait_steps": 0, "flow_denoising_steps": 10,
                       "max_wall_seconds": 129600, "termination_grace_seconds": 120,
                       "automatic_retry": False, "require_explicit_noise": True, "keepalive_mode": "managed"}.items():
        require(execution[key] == value, "unreviewed execution setting: " + key)
    for key in ("protocol_manifest", "checkpoint_receipt", "catalog"):
        require(sha(release[key]["path"]) == release[key]["sha256"], key + " hash mismatch")
    for relative, expected in release["critical_files"].items():
        require(sha(REPO / relative) == expected, "critical source changed: " + relative)
    require(runner_code_hash() == release["runner_code_sha256"], "rollout code changed")
    checkpoint = Path(release["canonical_checkpoint"]["path"])
    require(sha(checkpoint / "framework_config.yaml") == release["canonical_checkpoint"]["framework_config_sha256"], "checkpoint config changed")
    config = yaml.safe_load((checkpoint / "framework_config.yaml").read_text())
    require(config["framework"]["action_model"]["num_inference_steps"] == 10, "flow denoising grid differs from ten-step protocol")
    require((checkpoint / "model.safetensors").stat().st_size == release["canonical_checkpoint"]["weights_size_bytes"], "checkpoint size changed")
    receipt = read(release["checkpoint_receipt"]["path"])
    require(receipt["status"] == "COMPLETED_ARTIFACT_AUDIT_PASS", "checkpoint audit is not PASS")
    require(receipt["final_weights"]["content_sha256"] == release["canonical_checkpoint"]["weights_sha256"], "checkpoint receipt hash identity mismatch")
    bank = read(release["noise_bank"]["path"])
    require(sha(release["noise_bank"]["path"]) == release["noise_bank"]["manifest_sha256"], "noise manifest changed")
    require(bank["noise_file_sha256"] == release["noise_bank"]["file_sha256"] and bank["bank_id"] == "O", "noise file declaration changed")
    manifest = read(release["protocol_manifest"]["path"])
    require(manifest["wave_00_reuse"]["status"] == "PASS_EXACT_DISJOINT_SPEC_PARTITION", "smoke partition not verified")
    require(manifest["wave_00_reuse"]["all_expanded_fields_identical"] and not manifest["wave_00_reuse"]["episode_id_overlap_between_parts"], "invalid smoke reuse")
    route = route_specs(manifest)
    require(sum(part["episode_count"] for part in route) == 24832, "episode budget mismatch")
    require([part["episode_count"] for part in route] == [48, 1504, 1552] + [3104] * 7, "unexpected partition sizes")
    require([part["label"] for part in route] == release["episode_budget"]["route"], "execution route differs from release")
    for reference in [manifest["selection"], manifest["source_provenance"], *manifest["protocols"]]:
        require(sha(reference["path"]) == reference["sha256"], "frozen protocol source changed: " + reference["path"])
    # Verify expanded IDs/specs without reading outcomes or launching MuJoCo.
    from scripts.dsol_paper1.evaluate_dsol_libero_hdf5_views import protocol_spec_count, protocol_spec_at
    seen = set()
    for part in route:
        protocol = read(part["path"])
        require(protocol_spec_count(protocol) == part["episode_count"], "expanded count mismatch")
        for index in range(part["episode_count"]):
            episode_id = protocol_spec_at(protocol, index)["episode_id"]
            require(episode_id not in seen, "duplicate episode across smoke/dense route")
            seen.add(episode_id)
    identity = {"release_sha256": sha(RELEASE), "protocol_manifest_sha256": release["protocol_manifest"]["sha256"],
                "controller_sha256": sha(REPO / "scripts/dsol_paper1/run_matched_view_landscape_v1.sh"),
                "checkpoint_sha256": release["canonical_checkpoint"]["weights_sha256"],
                "framework_config_sha256": release["canonical_checkpoint"]["framework_config_sha256"],
                "noise_bank_manifest_sha256": release["noise_bank"]["manifest_sha256"],
                "noise_bank_file_sha256": release["noise_bank"]["file_sha256"],
                "runner_code_sha256": release["runner_code_sha256"]}
    return release, route, identity


def validate_manifest(manifest, release, part):
    expected = {"checkpoint": release["canonical_checkpoint"]["path"],
                "checkpoint_sha256": release["canonical_checkpoint"]["weights_sha256"],
                "protocol_sha256": part["sha256"], "policy_backend": "alphabrain",
                "noise_bank_manifest_sha256": release["noise_bank"]["manifest_sha256"],
                "code_sha256": release["runner_code_sha256"], "require_explicit_noise": True,
                "gpu_count": 8, "eval_worker_count": 32, "policy_server_copies_per_gpu": 2,
                "replan_steps": 5, "wait_steps": 0, "eval_seed": 20260818,
                "video_episodes_per_worker": 0, "max_episodes_per_shard": None,
                "keepalive_mode": "managed", "run_analysis": False}
    for key, value in expected.items():
        require(manifest.get(key) == value, "run manifest mismatch: " + key)


def reusable(output, identity, release, part):
    receipt_path = output / "segment-receipt.json"
    if not receipt_path.exists():
        require(not output.exists() or not any(output.iterdir()), "partial/unreceipted segment exists; no automatic retry: " + str(output))
        return False
    receipt = read(receipt_path)
    require(receipt["status"] == "PASS_COMPLETE" and receipt["identity"] == identity, "resume identity or PASS status mismatch")
    require(receipt["protocol_sha256"] == part["sha256"] and receipt["episode_count"] == part["episode_count"], "resume protocol mismatch")
    require({str(path) for path in output.glob("episodes-shard-*.jsonl")} == set(receipt["ledger_sha256"]), "resume ledger set changed")
    for path, expected in receipt["ledger_sha256"].items():
        require(sha(path) == expected, "resume ledger content changed")
    require(sha(output / "run_manifest.json") == receipt["run_manifest_sha256"], "resume run manifest changed")
    require(sha(output / "audit.json") == receipt["audit_sha256"], "resume audit changed")
    require(read(output / "audit.json")["status"] == "PASS_COMPLETE", "resume audit not PASS")
    validate_manifest(read(output / "run_manifest.json"), release, part)
    return True


def resources_available(base_port):
    # Do not kill or borrow another experiment's allocations. Only established
    # keepalive jobs may be temporarily managed by the unchanged scoped runner.
    for attempt in range(20):
        occupied = []
        rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
        for value in set(rows.split()):
            if not value.isdigit():
                continue
            try:
                command = (Path("/proc") / value / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except FileNotFoundError:
                continue
            if "/workspace/ai2r/gpu_compute_keepalive/gpu_compute_keepalive.py" not in command:
                occupied.append(value)
        ports_busy = []
        for port in range(base_port, base_port + 16):
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    ports_busy.append(port)
        if not occupied and not ports_busy:
            return
        if attempt < 19:
            time.sleep(0.5)
    raise RuntimeError(f"GPU/port ownership conflict; no processes changed: pids={occupied}, ports={ports_busy}")


def stop(signum, frame):
    global STOPPED
    STOPPED = True
    if ACTIVE is not None and ACTIVE.poll() is None:
        # This PID is our own GNU timeout supervisor, which forwards TERM only
        # to its launched runner group. Never pkill, killall, or match user jobs.
        ACTIVE.terminate()


def run_command(command, environment, log_path, deadline):
    global ACTIVE
    require(not STOPPED, "controller interrupted")
    remaining = int(deadline - time.time())
    require(remaining > 0, "36-hour controller budget exhausted")
    with log_path.open("a") as log:
        ACTIVE = subprocess.Popen(["timeout", "--signal=TERM", "--kill-after=120s", str(remaining) + "s", *command],
                                  env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        code = ACTIVE.wait()
        ACTIVE = None
    require(not STOPPED and code == 0, f"segment command failed/interrupted (exit {code}); automatic retry disabled")


def execute(release, route, identity):
    accel_root = ROOT / "canonical-accel"
    accel_completion = read(accel_root / "completion.json")
    require(accel_completion.get("status") == "PASS_COMPLETE", "canonical Accel prerequisite not PASS")
    require(accel_completion.get("manifest_sha256") == sha(accel_root / "manifest.json"), "canonical Accel manifest changed")
    require(accel_completion.get("checkpoint_sha256") == identity["checkpoint_sha256"], "canonical Accel uses another checkpoint")
    require([accel_completion.get(key) for key in ("state_count", "candidate_count", "ensemble_size")] == [8, 97, 8], "canonical Accel prerequisite has wrong scope")
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (ROOT / "matched-view-landscape-controller.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget_path = ROOT / "controller-budget.json"
        if budget_path.exists():
            budget = read(budget_path)
            require(budget["identity"] == identity, "controller budget identity changed")
        else:
            budget = {"identity": identity, "first_started_epoch": time.time(), "first_started_utc": now(), "max_wall_seconds": 129600}
            write_json(budget_path, budget, exclusive=True)
        deadline = budget["first_started_epoch"] + 129600
        require(time.time() < deadline, "36-hour frozen wall-clock budget already exhausted")
        status_path = ROOT / "controller-status.json"
        state = {"schema": "dsol_matched_view_landscape_controller_status_v1", "identity": identity,
                 "pid": os.getpid(), "status": "PREFLIGHT", "completed_unique_episodes": 0,
                 "episode_budget": 24832, "current_segment": None, "updated_at_utc": now()}
        write_json(status_path, state)
        try:
            # One full content verification before any GPU use. The unchanged
            # old runner additionally records its own full checkpoint hash.
            checkpoint = Path(release["canonical_checkpoint"]["path"])
            require(sha(checkpoint / "model.safetensors") == identity["checkpoint_sha256"], "final weight bytes changed")
            bank = read(release["noise_bank"]["path"])
            require(sha(bank["noise_file"]) == identity["noise_bank_file_sha256"], "noise bytes changed")
            for part in route:
                output = ROOT / "closed-loop/canonical/O" / part["label"]
                state.update(current_segment=part["label"], updated_at_utc=now())
                if reusable(output, identity, release, part):
                    state["completed_unique_episodes"] += part["episode_count"]
                    state.update(status="REUSED_VERIFIED_PASS", updated_at_utc=now())
                    write_json(status_path, state)
                    print(json.dumps({"event": "reuse_verified_pass", "segment": part["label"]}), flush=True)
                    continue
                require(time.time() < deadline and not STOPPED, "controller stopped or budget exhausted")
                # Recheck small frozen sources before each new segment.
                require(preflight()[2] == identity, "frozen identity changed between segments")
                resources_available(release["execution"]["base_port"])
                output.mkdir(parents=True, exist_ok=True)
                state.update(status="RUNNING", updated_at_utc=now())
                write_json(status_path, state)
                write_json(output / "controller-status.json", state)
                print(json.dumps({"event": "segment_start", "segment": part["label"], "episodes": part["episode_count"], "utc": now()}), flush=True)
                environment = os.environ.copy()
                environment.update({"CHECKPOINT": str(checkpoint), "OUTPUT_DIR": str(output), "PROTOCOL": part["path"],
                    "NOISE_BANK_MANIFEST": release["noise_bank"]["path"], "REQUIRE_EXPLICIT_NOISE": "1",
                    "GPU_COUNT": "8", "EVAL_WORKER_COUNT": "32", "POLICY_SERVER_COPIES_PER_GPU": "2",
                    "POLICY_CPU_THREADS": "2", "SIM_CPU_THREADS": "1", "DSOL_GPU_DEVICES": "0,1,2,3,4,5,6,7",
                    "BASE_PORT": "26200", "REPLAN_STEPS": "5", "WAIT_STEPS": "0", "EVAL_SEED": "20260818",
                    "VIDEO_EPISODES": "0", "RUN_ANALYSIS": "0", "KEEPALIVE_MODE": "managed",
                    "POLICY_BACKEND": "alphabrain", "ANALYZER": str(ANALYZER), "MAX_EPISODES_PER_SHARD": "",
                    "POLICY_PYTHON": "/alphabrain/.venv/bin/python", "SIM_PYTHON": "/workspace/envs/fresh-libero/bin/python",
                    "RUNTIME": "/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus",
                    "SIM_CONFIG": "/share/longjunyu/alphabrain/envs/libero-plus-runtime-config-v1", "FFMPEG_EXE": "/usr/bin/ffmpeg"})
                segment_log = logs / (part["label"] + ".log")
                run_command(["bash", str(RUNNER)], environment, segment_log, deadline)
                validate_manifest(read(output / "run_manifest.json"), release, part)
                require(preflight()[2] == identity, "frozen identity changed during segment")
                state.update(status="AUDITING", updated_at_utc=now())
                write_json(status_path, state)
                run_command([sys.executable, str(AUDITOR), "--protocol", part["path"], "--noise-bank-manifest", release["noise_bank"]["path"],
                             "--run-manifest", str(output / "run_manifest.json"), "--episode-ledgers", str(output / "episodes-shard-*.jsonl"),
                             "--output", str(output / "audit.json")], environment, segment_log, deadline)
                audit = read(output / "audit.json")
                require(audit["status"] == "PASS_COMPLETE" and audit["episode_count"] == part["episode_count"], "segment audit not complete")
                require(audit["every_policy_call_matches_frozen_noise_bank"], "noise identity audit failed")
                receipt = {"schema": "dsol_matched_view_landscape_segment_receipt_v1", "status": "PASS_COMPLETE",
                    "created_at_utc": now(), "identity": identity, "protocol_sha256": part["sha256"], "episode_count": part["episode_count"],
                    "run_manifest_sha256": sha(output / "run_manifest.json"), "audit_sha256": sha(output / "audit.json"),
                    "ledger_sha256": {str(path): sha(path) for path in sorted(output.glob("episodes-shard-*.jsonl"))}}
                write_json(output / "segment-receipt.json", receipt, exclusive=True)
                state["completed_unique_episodes"] += part["episode_count"]
                state.update(status="SEGMENT_PASS", updated_at_utc=now())
                write_json(output / "controller-status.json", state)
                write_json(status_path, state)
                print(json.dumps({"event": "segment_pass", "segment": part["label"], "completed_unique_episodes": state["completed_unique_episodes"]}), flush=True)
            require(state["completed_unique_episodes"] == 24832, "final unique episode budget mismatch")
            state.update(status="PASS_COMPLETE_STATIC_DIAGNOSTIC", current_segment=None, updated_at_utc=now())
            write_json(status_path, state)
            print(json.dumps(state), flush=True)
        except BaseException as error:
            state.update(status="STOPPED_NO_AUTOMATIC_RETRY", error=str(error), updated_at_utc=now())
            write_json(status_path, state)
            if state["current_segment"]:
                output = ROOT / "closed-loop/canonical/O" / state["current_segment"]
                if output.exists():
                    write_json(output / "controller-status.json", state)
            raise


def main():
    parser = argparse.ArgumentParser(description="Bounded 24,832-episode static canonical landscape; never trains or moves cameras during rollout.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true", help="small-artifact/protocol validation only; no writes or GPU use")
    modes.add_argument("--run", action="store_true", help="default: execute frozen smoke, residuals, and waves; no automatic retry")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    release, route, identity = preflight()
    if args.check_only:
        print(json.dumps({"status": "PASS_READY_FOR_BOUNDED_STATIC_RUN", "identity": identity, "route": route,
                          "unique_episode_count": 24832, "full_weight_or_noise_bytes_rehashed": False,
                          "gpu_use": False, "writes": False}, indent=2))
    else:
        execute(release, route, identity)


if __name__ == "__main__":
    main()
PY_CONTROLLER
