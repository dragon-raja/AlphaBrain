#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${DSOL_M_B_DATA_ROOT:-/share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad64_stride2}
OUTPUT_ROOT=${DSOL_M_B_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs}
CONTROL_ROOT=${DSOL_M_B_CONTROL_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-view-revalidation-m-b-v1/training}
FORMAL_STEPS=${DSOL_M_B_STEPS:-2000}
RUN_TAG=${DSOL_M_B_RUN_TAG:-m-b-formal-v1}

[[ -s "$DATA_ROOT/manifest.json" ]] || { echo "missing Broad64 manifest: $DATA_ROOT" >&2; exit 1; }
mkdir -p "$CONTROL_ROOT/logs"
exec > >(tee -a "$CONTROL_ROOT/controller.log") 2>&1

run_job() {
  local arm=$1 seed=$2 devices=$3 port=$4
  local run_id="dsol_${arm}_${RUN_TAG}_seed${seed}_g2_gb32_steps${FORMAL_STEPS}"
  local final_model="$OUTPUT_ROOT/$run_id/final_model/model.safetensors"
  if [[ -s "$final_model" ]]; then
    printf 'reuse_complete_training=%s\n' "$run_id"
    return
  fi
  DSOL_PAIR_DATA_ROOT="$DATA_ROOT" \
  DSOL_PAIR_OUTPUT_ROOT="$OUTPUT_ROOT" \
  DSOL_GPU_DEVICES="$devices" \
  DSOL_MAIN_PROCESS_PORT="$port" \
  DSOL_GLOBAL_EXAMPLES=32 \
  DSOL_SKIP_FINAL_SAVE=0 \
  WANDB_MODE=${WANDB_MODE:-offline} \
    "$REPO_ROOT/scripts/dsol_paper1/run_libero_pair_train.sh" \
      "$arm" "$seed" 2 "$FORMAL_STEPS" "$RUN_TAG"
  [[ -s "$final_model" ]] || { echo "training ended without checkpoint: $run_id" >&2; exit 1; }
}

run_wave() {
  local wave=$1
  shift
  local pids=() names=() spec arm seed devices port
  printf 'wave_start=%s wave=%s\n' "$(date -u +%FT%TZ)" "$wave"
  for spec in "$@"; do
    IFS=: read -r arm seed devices port <<< "$spec"
    (
      run_job "$arm" "$seed" "$devices" "$port"
    ) > "$CONTROL_ROOT/logs/${wave}-${arm}-seed${seed}.log" 2>&1 &
    pids+=("$!")
    names+=("${arm}-seed${seed}")
    printf 'launched=%s pid=%s devices=%s\n' "${names[-1]}" "${pids[-1]}" "$devices"
  done
  local failed=0 index
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      printf 'completed=%s\n' "${names[$index]}"
    else
      printf 'failed=%s\n' "${names[$index]}" >&2
      failed=1
    fi
  done
  (( failed == 0 )) || return 1
  printf 'wave_complete=%s wave=%s\n' "$(date -u +%FT%TZ)" "$wave"
}

printf 'controller_start=%s git_commit=%s\n' \
  "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)"

run_wave wave1 \
  broad_unpaired_practical:41:0,1:32141 \
  broad_unpaired_practical:42:2,3:32142 \
  broad_unpaired_practical:43:4,5:32143 \
  broad_paired_consistency:42:6,7:32144

run_wave wave2 \
  broad_paired_consistency:43:0,1:32145

python - "$CONTROL_ROOT/completion.json" "$DATA_ROOT" "$OUTPUT_ROOT" "$REPO_ROOT" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

output, data_root, output_root, repo_root = sys.argv[1:]
payload = {
    "schema": "dsol_view_revalidation_m_b_training_completion_v1",
    "status": "COMPLETE",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip(),
    "data_root": data_root,
    "output_root": output_root,
    "global_batch_size": 32,
    "gpus_per_job": 2,
    "formal_steps": 2000,
    "trained": {
        "broad_unpaired_practical": [41, 42, 43],
        "broad_paired_consistency": [42, 43],
    },
    "reused": {
        "broad_paired_consistency": [41],
    },
}
path = Path(output)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY

printf 'controller_complete=%s output=%s\n' \
  "$(date -u +%FT%TZ)" "$CONTROL_ROOT/completion.json"
