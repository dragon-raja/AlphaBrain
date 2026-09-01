#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${DSOL_VIEW_EXPECTATION_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-view-value-expectation-v1}
CALIBRATION_SESSION=${CALIBRATION_SESSION:-dsol-view-expectation-calibration-v1}
POPULATION=$ROOT/population/population.json
CATALOG=$REPO_ROOT/configs/dsol_paper1/libero_view_catalog_v2_m1.json
SCAN_ROOT=$ROOT/visibility-scan
PROTOCOL_ROOT=$ROOT/protocols
CALIBRATION_ROOT=$ROOT/calibration
ACCEL_ROOT=$ROOT/accel-ensemble
HELDOUT_ROOT=$ROOT/heldout
NOISE_ROOT=$ROOT/noise-banks-h10
RUNTIME=${RUNTIME:-/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus}
SIM_CONFIG=${SIM_CONFIG:-/share/longjunyu/alphabrain/envs/libero-plus-runtime-config-v1}
SIM_PYTHON=${SIM_PYTHON:-/workspace/envs/fresh-libero/bin/python}
POLICY_PYTHON=${POLICY_PYTHON:-/alphabrain/.venv/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs}
GPU_COUNT=${GPU_COUNT:-8}
EVAL_WORKER_COUNT=${EVAL_WORKER_COUNT:-32}
GPU_DEVICES=${DSOL_GPU_DEVICES:-0,1,2,3,4,5,6,7}

checkpoint_for_seed() {
  local seed=$1
  printf '%s/dsol_broad_unpaired_practical_m-b-formal-v1_seed%s_g2_gb32_steps2000/final_model\n' \
    "$CHECKPOINT_ROOT" "$seed"
}

count_rows() {
  local directory=$1
  local pattern=$2
  if [[ ! -d "$directory" ]]; then
    echo 0
    return
  fi
  find "$directory" -maxdepth 1 -name "$pattern" -type f \
    -exec awk 'NF{n++} END{print n+0}' {} \; 2>/dev/null | \
    awk '{sum += $1} END {print sum + 0}'
}

wait_for_calibration() {
  while tmux has-session -t "$CALIBRATION_SESSION" 2>/dev/null; do
    printf 'waiting_for_calibration stage_A_rows=%s time=%s\n' \
      "$(count_rows "$CALIBRATION_ROOT/stage-A" 'episodes-shard-*.jsonl')" \
      "$(date -u +%FT%TZ)"
    sleep 60
  done
  [[ -s "$CALIBRATION_ROOT/analysis/analysis.json" ]] || {
    echo "calibration session ended without analysis.json" >&2
    exit 2
  }
}

stopped_keepalives=()
active_pids=()

stop_keepalives() {
  stopped_keepalives=()
  local gpu
  for gpu in $(seq 0 7); do
    if tmux has-session -t "gpu-keepalive-$gpu" 2>/dev/null; then
      tmux kill-session -t "gpu-keepalive-$gpu"
      stopped_keepalives+=("$gpu")
    fi
  done
}

restore_keepalives() {
  local gpu
  for gpu in "${stopped_keepalives[@]:-}"; do
    [[ -n "$gpu" ]] || continue
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      1 8192 "gpu-keepalive-$gpu" "$gpu" >/dev/null || true
  done
  stopped_keepalives=()
}

cleanup_accel() {
  local pid
  for pid in "${active_pids[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  restore_keepalives
}

run_accel_stage() {
  local stage=$1
  local expected=64
  local actual
  actual=$(count_rows "$ACCEL_ROOT" "$stage-shard-*.jsonl")
  if [[ "$actual" == "$expected" ]]; then
    printf 'accel_%s_skip_complete=%s\n' "$stage" "$actual"
    return
  fi
  active_pids=()
  local shard gpu checkpoint
  checkpoint=$(checkpoint_for_seed 41)
  for shard in $(seq 0 7); do
    gpu=$shard
    if [[ "$stage" == render ]]; then
      LIBERO_CONFIG_PATH="$SIM_CONFIG" \
      PYTHONPATH="$REPO_ROOT:$REPO_ROOT/scripts/cabi_vla:$REPO_ROOT/scripts/dsol_paper1" \
        "$SIM_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/run_view_value_expectation_accel_ensemble.py" \
          --stage render --population "$POPULATION" --scan-root "$SCAN_ROOT" \
          --output-root "$ACCEL_ROOT" --runtime "$RUNTIME" --config-root "$SIM_CONFIG" \
          --catalog "$CATALOG" --render-gpu "$gpu" --num-shards 8 --shard-index "$shard" \
          > "$ACCEL_ROOT/logs/render-shard-$shard.log" 2>&1 &
    else
      CUDA_VISIBLE_DEVICES=$gpu \
      PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
      ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1 \
      PYTHONPATH="$REPO_ROOT:/projects/openpi/src:/projects/openpi/packages/openpi-client/src" \
        "$POLICY_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/run_view_value_expectation_accel_ensemble.py" \
          --stage rank --population "$POPULATION" --scan-root "$SCAN_ROOT" \
          --output-root "$ACCEL_ROOT" --checkpoint "$checkpoint" --catalog "$CATALOG" \
          --device cuda:0 --ensemble-size 8 --batch-size 16 \
          --num-shards 8 --shard-index "$shard" \
          > "$ACCEL_ROOT/logs/rank-shard-$shard.log" 2>&1 &
    fi
    active_pids+=("$!")
  done
  local failed=0 pid
  for pid in "${active_pids[@]}"; do
    wait "$pid" || failed=1
  done
  active_pids=()
  [[ "$failed" == 0 ]] || { echo "Accel $stage stage failed" >&2; exit 1; }
  actual=$(count_rows "$ACCEL_ROOT" "$stage-shard-*.jsonl")
  [[ "$actual" == "$expected" ]] || {
    echo "Accel $stage expected $expected records, found $actual" >&2
    exit 1
  }
}

run_heldout_seed() {
  local seed=$1
  local bank=$2
  local protocol=$3
  local output=$4
  local expected actual checkpoint
  expected=$(jq -r '.episode_count' "$protocol")
  actual=$(count_rows "$output" 'episodes-shard-*.jsonl')
  if [[ "$actual" == "$expected" ]]; then
    printf 'heldout_seed%s_bank%s_skip_complete=%s\n' "$seed" "$bank" "$actual"
    return
  fi
  checkpoint=$(checkpoint_for_seed "$seed")
  CHECKPOINT="$checkpoint" \
  OUTPUT_DIR="$output" \
  PROTOCOL="$protocol" \
  NOISE_BANK_MANIFEST="$NOISE_ROOT/bank_${bank}.manifest.json" \
  REQUIRE_EXPLICIT_NOISE=1 \
  GPU_COUNT="$GPU_COUNT" \
  EVAL_WORKER_COUNT="$EVAL_WORKER_COUNT" \
  DSOL_GPU_DEVICES="$GPU_DEVICES" \
  BASE_PORT=22900 \
  REPLAN_STEPS=5 \
  WAIT_STEPS=0 \
  VIDEO_EPISODES=0 \
  RUN_ANALYSIS=0 \
  KEEPALIVE_MODE=managed \
    "$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh"
}

run_primary_analysis() {
  "$POLICY_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/analyze_view_value_expectation_heldout.py" \
    --seed41-protocol "$PROTOCOL_ROOT/heldout-primary-seed41.json" \
    --seed41-results "$HELDOUT_ROOT/primary-seed41/episodes-shard-*.jsonl" \
    --seed42-protocol "$PROTOCOL_ROOT/heldout-primary-seed42.json" \
    --seed42-results "$HELDOUT_ROOT/primary-seed42/episodes-shard-*.jsonl" \
    --seed43-protocol "$PROTOCOL_ROOT/heldout-primary-seed43.json" \
    --seed43-results "$HELDOUT_ROOT/primary-seed43/episodes-shard-*.jsonl" \
    --output-dir "$HELDOUT_ROOT/analysis-primary"
}

run_final_analysis() {
  "$POLICY_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/analyze_view_value_expectation_heldout.py" \
    --seed41-protocol "$PROTOCOL_ROOT/heldout-primary-seed41.json" \
    --seed41-results "$HELDOUT_ROOT/primary-seed41/episodes-shard-*.jsonl" \
    --seed41-reserve-protocol "$PROTOCOL_ROOT/heldout-reserve-seed41.json" \
    --seed41-reserve-results "$HELDOUT_ROOT/reserve-seed41/episodes-shard-*.jsonl" \
    --seed42-protocol "$PROTOCOL_ROOT/heldout-primary-seed42.json" \
    --seed42-results "$HELDOUT_ROOT/primary-seed42/episodes-shard-*.jsonl" \
    --seed42-reserve-protocol "$PROTOCOL_ROOT/heldout-reserve-seed42.json" \
    --seed42-reserve-results "$HELDOUT_ROOT/reserve-seed42/episodes-shard-*.jsonl" \
    --seed43-protocol "$PROTOCOL_ROOT/heldout-primary-seed43.json" \
    --seed43-results "$HELDOUT_ROOT/primary-seed43/episodes-shard-*.jsonl" \
    --seed43-reserve-protocol "$PROTOCOL_ROOT/heldout-reserve-seed43.json" \
    --seed43-reserve-results "$HELDOUT_ROOT/reserve-seed43/episodes-shard-*.jsonl" \
    --output-dir "$HELDOUT_ROOT/analysis-final"
}

mkdir -p "$ROOT/logs" "$ACCEL_ROOT/logs" "$HELDOUT_ROOT" "$PROTOCOL_ROOT"
exec > >(tee -a "$ROOT/logs/post-calibration.log") 2>&1

wait_for_calibration

trap cleanup_accel EXIT INT TERM
stop_keepalives
run_accel_stage render
run_accel_stage rank
restore_keepalives
trap - EXIT INT TERM

"$POLICY_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/build_view_value_expectation_heldout_protocols.py" \
  --population "$POPULATION" --catalog "$CATALOG" \
  --scan-ledgers "$SCAN_ROOT/shard-*.jsonl" \
  --calibration-stage-a-results "$CALIBRATION_ROOT/stage-A/episodes-shard-*.jsonl" \
  --accel-results "$ACCEL_ROOT/rank-shard-*.jsonl" \
  --output-dir "$PROTOCOL_ROOT"

for seed in 41 42 43; do
  run_heldout_seed "$seed" E "$PROTOCOL_ROOT/heldout-primary-seed$seed.json" \
    "$HELDOUT_ROOT/primary-seed$seed"
done
run_primary_analysis

if jq -e '.activate_bank_F == true' "$HELDOUT_ROOT/analysis-primary/reserve-decision.json" >/dev/null; then
  for seed in 41 42 43; do
    "$POLICY_PYTHON" "$REPO_ROOT/scripts/dsol_paper1/build_view_value_expectation_reserve_protocol.py" \
      --primary-protocol "$PROTOCOL_ROOT/heldout-primary-seed$seed.json" \
      --reserve-decision "$HELDOUT_ROOT/analysis-primary/reserve-decision.json" \
      --output "$PROTOCOL_ROOT/heldout-reserve-seed$seed.json"
    run_heldout_seed "$seed" F "$PROTOCOL_ROOT/heldout-reserve-seed$seed.json" \
      "$HELDOUT_ROOT/reserve-seed$seed"
  done
  run_final_analysis
else
  printf 'heldout_precision_reserve=not_required\n'
fi

printf 'view_value_expectation_post_calibration_complete=%s\n' "$(date -u +%FT%TZ)"
