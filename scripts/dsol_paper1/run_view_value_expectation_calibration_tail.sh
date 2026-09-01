#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${DSOL_VIEW_EXPECTATION_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-view-value-expectation-v1}
CHECKPOINT=${CHECKPOINT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000/final_model}
POPULATION=$ROOT/population/population.json
CATALOG=$REPO_ROOT/configs/dsol_paper1/libero_view_catalog_v2_m1.json
SCAN_GLOB=$ROOT/visibility-scan/shard-*.jsonl
PROTOCOL_ROOT=$ROOT/protocols
CALIBRATION_ROOT=$ROOT/calibration
NOISE_ROOT=$ROOT/noise-banks-h10
GPU_COUNT=${GPU_COUNT:-8}
EVAL_WORKER_COUNT=${EVAL_WORKER_COUNT:-32}
GPU_DEVICES=${DSOL_GPU_DEVICES:-0,1,2,3,4,5,6,7}

mkdir -p "$ROOT/logs" "$PROTOCOL_ROOT" "$CALIBRATION_ROOT"
exec > >(tee -a "$ROOT/logs/calibration-tail.log") 2>&1

count_rows() {
  local directory=$1
  find "$directory" -maxdepth 1 -name 'episodes-shard-*.jsonl' -type f \
    -exec awk 'NF{n++} END{print n+0}' {} \; 2>/dev/null | \
    awk '{sum += $1} END {print sum + 0}'
}

run_stage() {
  local stage=$1
  local expected=$2
  local port=$3
  local protocol=$PROTOCOL_ROOT/calibration-stage-$stage.json
  local output=$CALIBRATION_ROOT/stage-$stage
  local actual
  actual=$(count_rows "$output")
  if [[ "$actual" == "$expected" ]]; then
    printf 'stage_%s_skip_complete episodes=%s\n' "$stage" "$actual"
    return
  fi
  CHECKPOINT="$CHECKPOINT" \
  OUTPUT_DIR="$output" \
  PROTOCOL="$protocol" \
  NOISE_BANK_MANIFEST="$NOISE_ROOT/bank_${stage}.manifest.json" \
  REQUIRE_EXPLICIT_NOISE=1 \
  GPU_COUNT="$GPU_COUNT" \
  EVAL_WORKER_COUNT="$EVAL_WORKER_COUNT" \
  DSOL_GPU_DEVICES="$GPU_DEVICES" \
  BASE_PORT="$port" \
  REPLAN_STEPS=5 \
  WAIT_STEPS=0 \
  VIDEO_EPISODES=0 \
  RUN_ANALYSIS=0 \
  KEEPALIVE_MODE=managed \
    "$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh"
}

stage_a_count=$(count_rows "$CALIBRATION_ROOT/stage-A")
[[ "$stage_a_count" == 6208 ]] || {
  echo "stage A must complete before calibration tail: $stage_a_count/6208" >&2
  exit 2
}

python "$REPO_ROOT/scripts/dsol_paper1/build_view_value_expectation_calibration_stage.py" \
  --stage B --population "$POPULATION" --catalog "$CATALOG" \
  --scan-ledgers "$SCAN_GLOB" \
  --previous-results "$CALIBRATION_ROOT/stage-A/episodes-shard-*.jsonl" \
  --previous-protocol "$PROTOCOL_ROOT/calibration-stage-A.json" \
  --output "$PROTOCOL_ROOT/calibration-stage-B.json"
run_stage B 3072 22500

python "$REPO_ROOT/scripts/dsol_paper1/build_view_value_expectation_calibration_stage.py" \
  --stage C --population "$POPULATION" --catalog "$CATALOG" \
  --scan-ledgers "$SCAN_GLOB" \
  --previous-results "$CALIBRATION_ROOT/stage-B/episodes-shard-*.jsonl" \
  --previous-protocol "$PROTOCOL_ROOT/calibration-stage-B.json" \
  --output "$PROTOCOL_ROOT/calibration-stage-C.json"
run_stage C 1536 22600

python "$REPO_ROOT/scripts/dsol_paper1/build_view_value_expectation_calibration_stage.py" \
  --stage D --population "$POPULATION" --catalog "$CATALOG" \
  --scan-ledgers "$SCAN_GLOB" \
  --previous-results "$CALIBRATION_ROOT/stage-C/episodes-shard-*.jsonl" \
  --previous-protocol "$PROTOCOL_ROOT/calibration-stage-C.json" \
  --output "$PROTOCOL_ROOT/calibration-stage-D.json"
run_stage D 2048 22700

python "$REPO_ROOT/scripts/dsol_paper1/analyze_view_value_expectation_calibration.py" \
  "$CALIBRATION_ROOT/stage-D/episodes-shard-*.jsonl" \
  --protocol "$PROTOCOL_ROOT/calibration-stage-D.json" \
  --output-dir "$CALIBRATION_ROOT/analysis"

printf 'view_value_expectation_calibration_complete=%s\n' "$(date -u +%FT%TZ)"
