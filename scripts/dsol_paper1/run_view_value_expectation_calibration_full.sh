#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${DSOL_VIEW_EXPECTATION_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-view-value-expectation-v1}
CHECKPOINT=${CHECKPOINT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000/final_model}
PROTOCOL=$ROOT/protocols/calibration-stage-A.json
OUTPUT=$ROOT/calibration/stage-A
NOISE=$ROOT/noise-banks-h10/bank_A.manifest.json

mkdir -p "$ROOT/logs" "$OUTPUT"
exec > >(tee -a "$ROOT/logs/calibration-full.log") 2>&1

count=$(find "$OUTPUT" -maxdepth 1 -name 'episodes-shard-*.jsonl' -type f \
  -exec awk 'NF{n++} END{print n+0}' {} \; 2>/dev/null | \
  awk '{sum += $1} END {print sum + 0}')
if [[ "$count" != 6208 ]]; then
  CHECKPOINT="$CHECKPOINT" \
  OUTPUT_DIR="$OUTPUT" \
  PROTOCOL="$PROTOCOL" \
  NOISE_BANK_MANIFEST="$NOISE" \
  REQUIRE_EXPLICIT_NOISE=1 \
  GPU_COUNT=8 \
  EVAL_WORKER_COUNT=32 \
  DSOL_GPU_DEVICES=0,1,2,3,4,5,6,7 \
  BASE_PORT=22400 \
  REPLAN_STEPS=5 \
  WAIT_STEPS=0 \
  VIDEO_EPISODES=0 \
  RUN_ANALYSIS=0 \
  KEEPALIVE_MODE=managed \
    "$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh"
else
  printf 'stage_A_skip_complete episodes=%s\n' "$count"
fi

exec "$REPO_ROOT/scripts/dsol_paper1/run_view_value_expectation_calibration_tail.sh"
