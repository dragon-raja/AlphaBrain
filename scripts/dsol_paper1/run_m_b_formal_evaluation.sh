#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TRAIN_COMPLETION=${DSOL_M_B_TRAIN_COMPLETION:-/share/longjunyu/alphabrain/experiments/dsol-view-revalidation-m-b-v1/training/completion.json}
OUTPUT_ROOT=${DSOL_M_B_EVAL_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-view-revalidation-m-b-v1}
RUN_ROOT=${DSOL_PAIR_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs}
CAMERA_PROTOCOL=${DSOL_CAMERA_PROTOCOL:-/share/longjunyu/alphabrain/experiments/libero-plus-camera-full-v1/protocol.json}
CAMERA_BASELINE=${DSOL_CAMERA_BASELINE:-/share/longjunyu/alphabrain/experiments/libero-plus-camera-full-v1/official-pi05-frozen-camera1599}
OFFICIAL_CHECKPOINT=${OFFICIAL_CHECKPOINT:-/share/longjunyu/alphabrain/pretrained_models/openpi/pi05_libero_pytorch}
WAIT_SECONDS=${DSOL_WAIT_POLL_SECONDS:-30}
LEASE_DIR=${AI2R_KEEPALIVE_LEASE_DIR:-/run/ai2r/gpu-keepalive-leases}
LEASE_OWNER=$BASHPID

PRACTICAL_41=$RUN_ROOT/dsol_broad_unpaired_practical_m-b-formal-v1_seed41_g2_gb32_steps2000/final_model
PRACTICAL_42=$RUN_ROOT/dsol_broad_unpaired_practical_m-b-formal-v1_seed42_g2_gb32_steps2000/final_model
PRACTICAL_43=$RUN_ROOT/dsol_broad_unpaired_practical_m-b-formal-v1_seed43_g2_gb32_steps2000/final_model
CONSISTENCY_41=$RUN_ROOT/dsol_broad_paired_consistency_broad64-parallel-formal-v1_seed41_g2_gb32_steps2000/final_model
CONSISTENCY_42=$RUN_ROOT/dsol_broad_paired_consistency_m-b-formal-v1_seed42_g2_gb32_steps2000/final_model
CONSISTENCY_43=$RUN_ROOT/dsol_broad_paired_consistency_m-b-formal-v1_seed43_g2_gb32_steps2000/final_model

SPECS=(
  "broad64-practical|41|$PRACTICAL_41"
  "broad64-practical|42|$PRACTICAL_42"
  "broad64-practical|43|$PRACTICAL_43"
  "broad64-paired-consistency|41|$CONSISTENCY_41"
  "broad64-paired-consistency|42|$CONSISTENCY_42"
  "broad64-paired-consistency|43|$CONSISTENCY_43"
)

mkdir -p "$OUTPUT_ROOT/pipeline_logs" "$LEASE_DIR"
exec > >(tee -a "$OUTPUT_ROOT/formal_evaluation_controller.log") 2>&1

reset_low_keepalive() {
  bash /workspace/ai2r/gpu_compute_keepalive/stop_all.sh gpu-keepalive >/dev/null || true
  bash /workspace/ai2r/gpu_compute_keepalive/start_all.sh 0 4096 gpu-keepalive >/dev/null || true
}

cleanup() {
  local gpu lease owner
  reset_low_keepalive
  for gpu in 0 1 2 3 4 5 6 7; do
    lease="$LEASE_DIR/gpu-$gpu.lease"
    owner=""
    [[ -s "$lease" ]] && read -r owner < "$lease" || true
    if [[ "$owner" == "$LEASE_OWNER" ]]; then
      unlink "$lease"
    fi
  done
}
trap cleanup EXIT INT TERM

for gpu in 0 1 2 3 4 5 6 7; do
  temporary="$LEASE_DIR/gpu-$gpu.lease.$LEASE_OWNER.tmp"
  printf '%s\n' "$LEASE_OWNER" > "$temporary"
  mv "$temporary" "$LEASE_DIR/gpu-$gpu.lease"
done

while [[ ! -s "$TRAIN_COMPLETION" ]]; do
  printf 'waiting_for_training=%s at=%s\n' "$TRAIN_COMPLETION" "$(date -u +%FT%TZ)"
  sleep "$WAIT_SECONDS"
done
jq -e '.status == "COMPLETE"' "$TRAIN_COMPLETION" >/dev/null
for spec in "${SPECS[@]}"; do
  IFS='|' read -r method seed checkpoint <<< "$spec"
  [[ -s "$checkpoint/model.safetensors" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 1; }
done

run_candidate() {
  local method=$1 seed=$2 checkpoint=$3 mode=$4 init_count=$5 output=$6 port=$7
  if [[ -s "$output/metrics.json" ]]; then
    printf 'reuse_complete_eval=%s seed=%s mode=%s\n' "$method" "$seed" "$mode"
    return
  fi
  printf 'eval_start=%s method=%s seed=%s mode=%s\n' "$(date -u +%FT%TZ)" "$method" "$seed" "$mode"
  CHECKPOINT="$checkpoint" \
  OUTPUT_DIR="$output" \
  PROTOCOL="$CAMERA_PROTOCOL" \
  GPU_COUNT=8 \
  BASE_PORT="$port" \
  INIT_STATE_COUNT="$init_count" \
  EVAL_MODES="$mode" \
  PROBE_SAMPLES=0 \
  VIDEO_EPISODES=2 \
  POLICY_PYTHON=/alphabrain/.venv/bin/python \
    bash "$REPO_ROOT/scripts/cabi_vla/run_alphabrain_pi05_libero_plus_view_eval.sh"
  printf 'eval_complete=%s method=%s seed=%s mode=%s\n' "$(date -u +%FT%TZ)" "$method" "$seed" "$mode"
}

CAMERA_ROOT=$OUTPUT_ROOT/camera_full
mkdir -p "$CAMERA_ROOT"
for spec in "${SPECS[@]}"; do
  IFS='|' read -r method seed checkpoint <<< "$spec"
  run_candidate "$method" "$seed" "$checkpoint" camera_full 1 \
    "$CAMERA_ROOT/${method}-seed${seed}" 19500
done

camera_args=()
for spec in "${SPECS[@]}"; do
  IFS='|' read -r method seed checkpoint <<< "$spec"
  normalized_method=${method//-/_}
  camera_args+=(--run "$normalized_method" "$seed" "$CAMERA_ROOT/${method}-seed${seed}")
done
/alphabrain/.venv/bin/python "$REPO_ROOT/scripts/dsol_paper1/analyze_m_b_multiseed.py" \
  --benchmark camera_full \
  --baseline-dir "$CAMERA_BASELINE" \
  "${camera_args[@]}" \
  --output-json "$CAMERA_ROOT/multiseed_metrics.json" \
  --output-report "$CAMERA_ROOT/multiseed_report.md"

ORIGINAL_ROOT=$OUTPUT_ROOT/original_full
ORIGINAL_BASELINE=$ORIGINAL_ROOT/official-pi05-frozen-original2000
mkdir -p "$ORIGINAL_ROOT"
if [[ ! -s "$ORIGINAL_BASELINE/metrics.json" ]]; then
  printf 'eval_start=%s method=official mode=original_full\n' "$(date -u +%FT%TZ)"
  CHECKPOINT="$OFFICIAL_CHECKPOINT" \
  PROTOCOL="$CAMERA_PROTOCOL" \
  OUTPUT_DIR="$ORIGINAL_BASELINE" \
  GPU_COUNT=8 \
  BASE_PORT=19600 \
  INIT_STATE_COUNT=50 \
  EVAL_MODES=original_full \
  PROBE_SAMPLES=0 \
  VIDEO_EPISODES=2 \
    bash "$REPO_ROOT/scripts/cabi_vla/run_pi05_libero_plus_view_study.sh"
  reset_low_keepalive
fi

for spec in "${SPECS[@]}"; do
  IFS='|' read -r method seed checkpoint <<< "$spec"
  run_candidate "$method" "$seed" "$checkpoint" original_full 50 \
    "$ORIGINAL_ROOT/${method}-seed${seed}" 19700
done

original_args=()
for spec in "${SPECS[@]}"; do
  IFS='|' read -r method seed checkpoint <<< "$spec"
  normalized_method=${method//-/_}
  original_args+=(--run "$normalized_method" "$seed" "$ORIGINAL_ROOT/${method}-seed${seed}")
done
/alphabrain/.venv/bin/python "$REPO_ROOT/scripts/dsol_paper1/analyze_m_b_multiseed.py" \
  --benchmark original_full \
  --baseline-dir "$ORIGINAL_BASELINE" \
  "${original_args[@]}" \
  --output-json "$ORIGINAL_ROOT/multiseed_metrics.json" \
  --output-report "$ORIGINAL_ROOT/multiseed_report.md"

python - "$OUTPUT_ROOT/formal_evaluation_completion.json" "$REPO_ROOT" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

output, repo_root = sys.argv[1:]
payload = {
    "schema": "dsol_view_revalidation_m_b_formal_evaluation_completion_v1",
    "status": "COMPLETE",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip(),
    "camera_full_models": 6,
    "original_full_models": 6,
    "training_seeds": [41, 42, 43],
}
path = Path(output)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
printf 'formal_evaluation_complete=%s\n' "$(date -u +%FT%TZ)"
