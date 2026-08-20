#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RETENTION_SESSION=${DSOL_RETENTION_SESSION:-dsol-original-retention-s41-v3}
RETENTION_ROOT=${DSOL_RETENTION_ROOT:-/share/longjunyu/alphabrain/experiments/libero-original-full-retention-v3}
DATA_ROOT=${DSOL_BROAD64_DATA_ROOT:-/share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad64_stride2}
OUTPUT_ROOT=${DSOL_BROAD64_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs}
CONTROL_ROOT=${DSOL_BROAD64_CONTROL_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad64-parallel-m-a-v1}
POLL_SECONDS=${DSOL_WAIT_POLL_SECONDS:-60}
SMOKE_STEPS=${DSOL_PARALLEL_SMOKE_STEPS:-20}
FORMAL_STEPS=${DSOL_PARALLEL_FORMAL_STEPS:-2000}
SEED=${DSOL_PARALLEL_SEED:-41}

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid DSOL_WAIT_POLL_SECONDS" >&2; exit 2; }
[[ -s "$DATA_ROOT/manifest.json" ]] || { echo "missing Broad64 manifest: $DATA_ROOT" >&2; exit 1; }
mkdir -p "$CONTROL_ROOT"

exec > >(tee -a "$CONTROL_ROOT/controller.log") 2>&1
printf 'controller_start=%s git_commit=%s retention_session=%s\n' \
  "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)" "$RETENTION_SESSION"

while tmux has-session -t "$RETENTION_SESSION" 2>/dev/null; do
  printf 'waiting_for_retention=%s at=%s\n' "$RETENTION_SESSION" "$(date -u +%FT%TZ)"
  sleep "$POLL_SECONDS"
done

[[ -s "$RETENTION_ROOT/official-pi05-frozen-original2000/metrics.json" ]] || {
  echo "retention session ended without Official metrics" >&2
  exit 1
}
[[ -s "$RETENTION_ROOT/broad64-seed41-original2000/metrics.json" ]] || {
  echo "retention session ended without Broad64 metrics" >&2
  exit 1
}
[[ -s "$RETENTION_ROOT/broad64-seed41-vs-official.json" ]] || {
  echo "retention session ended without paired comparison" >&2
  exit 1
}

arms=(broad_unpaired_state_matched broad_paired_fm broad_paired_consistency)
devices=(0,1 2,3 4,5)
ports=(31941 31942 31943)

run_wave() {
  local steps=$1
  local tag=$2
  local skip_final_save=$3
  local pids=()
  local names=()

  printf 'wave_start=%s steps=%s tag=%s\n' "$(date -u +%FT%TZ)" "$steps" "$tag"
  for index in "${!arms[@]}"; do
    local arm=${arms[$index]}
    local gpu_list=${devices[$index]}
    local port=${ports[$index]}
    local job_log="$CONTROL_ROOT/${tag}-${arm}.log"
    (
      export DSOL_PAIR_DATA_ROOT="$DATA_ROOT"
      export DSOL_PAIR_OUTPUT_ROOT="$OUTPUT_ROOT"
      export DSOL_GPU_DEVICES="$gpu_list"
      export DSOL_MAIN_PROCESS_PORT="$port"
      export DSOL_GLOBAL_EXAMPLES=32
      export DSOL_SKIP_FINAL_SAVE="$skip_final_save"
      export WANDB_MODE=${WANDB_MODE:-offline}
      "$REPO_ROOT/scripts/dsol_paper1/run_libero_pair_train.sh" \
        "$arm" "$SEED" 2 "$steps" "$tag"
    ) >"$job_log" 2>&1 &
    pids+=("$!")
    names+=("$arm")
    printf 'launched arm=%s pid=%s devices=%s port=%s log=%s\n' \
      "$arm" "${pids[-1]}" "$gpu_list" "$port" "$job_log"
  done

  local failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      printf 'completed arm=%s tag=%s\n' "${names[$index]}" "$tag"
    else
      printf 'failed arm=%s tag=%s\n' "${names[$index]}" "$tag" >&2
      failed=1
    fi
  done
  (( failed == 0 )) || return 1
  printf 'wave_complete=%s steps=%s tag=%s\n' "$(date -u +%FT%TZ)" "$steps" "$tag"
}

run_wave "$SMOKE_STEPS" broad64-parallel-smoke-v1 1
run_wave "$FORMAL_STEPS" broad64-parallel-formal-v1 0

python - "$CONTROL_ROOT/completion.json" "$DATA_ROOT" "$SEED" "$REPO_ROOT" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

output, data_root, seed, repo_root = sys.argv[1:]
payload = {
    "schema": "dsol_broad64_parallel_pairing_completion_v1",
    "status": "COMPLETE",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "data_root": data_root,
    "seed": int(seed),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip(),
    "arms": [
        "broad_unpaired_state_matched",
        "broad_paired_fm",
        "broad_paired_consistency",
    ],
    "global_batch_size": 32,
    "gpus_per_job": 2,
}
path = Path(output)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
printf 'controller_complete=%s\n' "$(date -u +%FT%TZ)"
