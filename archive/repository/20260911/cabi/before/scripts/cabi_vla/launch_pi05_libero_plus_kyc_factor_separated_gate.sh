#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SCRIPT=$(realpath "$0")
COMMAND=${1:-status}
SESSION=${PLUS_FS_GATE_SESSION:-plus-kyc-factor-separated-gate}
VIEW_ROOT=${PLUS_FS_VIEW_ROOT:-/share/longjunyu/alphabrain/datasets/libero-plus/views/pi05-goal-factor-separated-v1}
RUN_ROOT=${PLUS_FS_RUN_ROOT:-/share/longjunyu/alphabrain/experiments/libero-plus-kyc-factor-separated-v1/runs}
OUTPUT_ROOT=${PLUS_FS_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/libero-plus-kyc-factor-separated-v1}
PROTOCOL=${PLUS_FS_PROTOCOL:-/share/longjunyu/alphabrain/experiments/libero-plus-camera-background-v1/protocol-v1.json}
DEPENDENCY=${PLUS_FS_DEPENDENCY:-/share/longjunyu/alphabrain/experiments/libero-plus-kyc-matched-v1/final/metrics.json}
TAG=${PLUS_FS_RUN_TAG:-factor-separated-goal-v1}
STEPS=${PLUS_FS_STEPS:-33000}
WAIT_SECONDS=${PLUS_FS_WAIT_SECONDS:-60}
LOG=$OUTPUT_ROOT/orchestrator.log
SEEDS=(41 42 43)
METHODS=(control kyc)

training_session() {
  local method=$1 seed=$2
  echo "plus-fs-${method}-s${seed}"
}

training_arm() {
  case "$1" in
    control) echo visual_lora_control ;;
    kyc) echo visual_lora_kyc ;;
    *) return 2 ;;
  esac
}

run_id() {
  local method=$1 seed=$2
  echo "pi05_plus_mv_$(training_arm "$method")_${TAG}_seed${seed}_steps${STEPS}"
}

checkpoint_dir() {
  local method=$1 seed=$2
  echo "$RUN_ROOT/$(run_id "$method" "$seed")/final_model"
}

evaluation_dir() {
  local method=$1 seed=$2
  echo "$OUTPUT_ROOT/${method}_seed${seed}"
}

episode_count() {
  local output=$1
  if [[ ! -d "$output" ]]; then
    echo 0
    return
  fi
  find "$output" -maxdepth 1 -name 'episodes-shard-*.jsonl' -type f \
    -exec awk 'NF {count += 1} END {print count + 0}' {} + 2>/dev/null \
    | awk '{total += $1} END {print total + 0}'
}

expected_count() {
  PYTHONPATH="$REPO_ROOT/scripts/cabi_vla" PROTOCOL="$PROTOCOL" \
    "$REPO_ROOT/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path
from evaluate_pi05_libero_plus_views import build_episode_specs

protocol = json.loads(Path(os.environ["PROTOCOL"]).read_text())
print(len(build_episode_specs(
    protocol,
    modes=["composition"],
    suites=["libero_goal"],
    init_state_count=2,
)))
PY
}

last_step() {
  local method=$1 seed=$2
  local metrics="$RUN_ROOT/$(run_id "$method" "$seed")/metrics.jsonl"
  if [[ ! -s "$metrics" ]]; then
    echo 0
    return
  fi
  METRICS="$metrics" "$REPO_ROOT/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

lines = [line for line in Path(os.environ["METRICS"]).read_text().splitlines() if line]
row = json.loads(lines[-1]) if lines else {}
print(int(row.get("step", row.get("global_step", 0))))
PY
}

print_status() {
  mkdir -p "$OUTPUT_ROOT"
  local expected
  expected=$(expected_count)
  if [[ -s "$VIEW_ROOT/manifest.json" ]]; then
    echo "factor_view=READY"
  else
    echo "factor_view=WAITING"
  fi
  if [[ -s "$DEPENDENCY" ]]; then
    echo "matched_gate_dependency=COMPLETE"
  else
    echo "matched_gate_dependency=WAITING"
  fi
  for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      local checkpoint evaluation
      checkpoint=$(checkpoint_dir "$method" "$seed")
      evaluation=$(evaluation_dir "$method" "$seed")
      if [[ -s "$checkpoint/model.safetensors" ]]; then
        echo "$method seed=$seed training=COMPLETE evaluation=$(episode_count "$evaluation")/$expected"
      else
        echo "$method seed=$seed training=$(last_step "$method" "$seed")/$STEPS evaluation=$(episode_count "$evaluation")/$expected"
      fi
    done
  done
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "orchestrator=RUNNING session=$SESSION"
  elif [[ -s "$OUTPUT_ROOT/final/metrics.json" ]]; then
    echo "orchestrator=COMPLETE output=$OUTPUT_ROOT/final"
  else
    echo "orchestrator=NOT_RUNNING"
  fi
}

wait_for_inputs() {
  while [[ ! -s "$VIEW_ROOT/manifest.json" ]]; do
    if ! tmux has-session -t plus-goal-factor-view 2>/dev/null; then
      echo "factor view builder stopped without a manifest" >&2
      exit 1
    fi
    echo "waiting_for_factor_view=$(date -u +%FT%TZ)"
    sleep "$WAIT_SECONDS"
  done
  while [[ ! -s "$DEPENDENCY" ]]; do
    if ! tmux has-session -t plus-kyc-matched-gate 2>/dev/null; then
      echo "matched KYC gate stopped without final metrics" >&2
      exit 1
    fi
    echo "waiting_for_matched_gate=$(date -u +%FT%TZ)"
    sleep "$WAIT_SECONDS"
  done
}

start_training() {
  local gpu=0
  for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      local checkpoint session arm
      checkpoint=$(checkpoint_dir "$method" "$seed")
      session=$(training_session "$method" "$seed")
      arm=$(training_arm "$method")
      if [[ -s "$checkpoint/model.safetensors" ]]; then
        gpu=$((gpu + 1))
        continue
      fi
      if tmux has-session -t "$session" 2>/dev/null; then
        gpu=$((gpu + 1))
        continue
      fi
      tmux new-session -d -s "$session" env \
        PLUS_MV_DATA_ROOT="$VIEW_ROOT" \
        PLUS_MV_OUTPUT_ROOT="$RUN_ROOT" \
        PLUS_MV_RUN_TAG="$TAG" \
        "$REPO_ROOT/scripts/cabi_vla/run_pi05_libero_plus_multiview_train.sh" \
        "$arm" "$seed" "$gpu" "$STEPS"
      echo "training_started=$(date -u +%FT%TZ) method=$method seed=$seed gpu=$gpu"
      gpu=$((gpu + 1))
    done
  done
}

wait_for_training() {
  while true; do
    local pending=0
    for seed in "${SEEDS[@]}"; do
      for method in "${METHODS[@]}"; do
        local checkpoint session
        checkpoint=$(checkpoint_dir "$method" "$seed")
        if [[ ! -s "$checkpoint/model.safetensors" ]]; then
          session=$(training_session "$method" "$seed")
          if ! tmux has-session -t "$session" 2>/dev/null; then
            echo "training ended without checkpoint: method=$method seed=$seed" >&2
            exit 1
          fi
          pending=$((pending + 1))
        fi
      done
    done
    if [[ "$pending" == 0 ]]; then
      return
    fi
    echo "waiting_for_factor_training=$(date -u +%FT%TZ) pending=$pending"
    sleep "$WAIT_SECONDS"
  done
}

run_worker() {
  mkdir -p "$OUTPUT_ROOT"
  exec > >(tee -a "$LOG") 2>&1
  echo "factor_separated_gate_started=$(date -u +%FT%TZ) commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  wait_for_inputs
  start_training
  wait_for_training
  local expected
  expected=$(expected_count)
  echo "expected_goal_episodes_per_run=$expected"

  local run_index=0
  for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      local checkpoint output count
      checkpoint=$(checkpoint_dir "$method" "$seed")
      output=$(evaluation_dir "$method" "$seed")
      count=$(episode_count "$output")
      if [[ "$count" -ne "$expected" ]]; then
        CHECKPOINT="$checkpoint" \
        OUTPUT_DIR="$output" \
        PROTOCOL="$PROTOCOL" \
        EVAL_MODES=composition \
        EVAL_SUITES=libero_goal \
        PROBE_SAMPLES=0 \
        VIDEO_EPISODES=72 \
        SKIP_ANALYSIS=1 \
        BASE_PORT=$((19400 + run_index * 10)) \
          "$REPO_ROOT/scripts/cabi_vla/run_alphabrain_pi05_libero_plus_view_eval.sh"
      fi
      run_index=$((run_index + 1))
    done
  done

  local args=()
  for seed in "${SEEDS[@]}"; do
    args+=(--control "$seed=$(evaluation_dir control "$seed")")
    args+=(--kyc "$seed=$(evaluation_dir kyc "$seed")")
  done
  mkdir -p "$OUTPUT_ROOT/final"
  PYTHONPATH="$REPO_ROOT/scripts/cabi_vla" "$REPO_ROOT/.venv/bin/python" \
    "$REPO_ROOT/scripts/cabi_vla/analyze_pi05_libero_plus_kyc_matched.py" \
    "${args[@]}" \
    --interpretation factor_separated_category_composition \
    --output-json "$OUTPUT_ROOT/final/metrics.json" \
    --output-report "$OUTPUT_ROOT/final/report_zh.md" \
    --output-figure "$OUTPUT_ROOT/final/summary.png"
  cp "$VIEW_ROOT/manifest.json" "$OUTPUT_ROOT/final/training_view_manifest.json"
  echo "factor_separated_gate_complete=$(date -u +%FT%TZ) output=$OUTPUT_ROOT/final"
}

case "$COMMAND" in
  start)
    mkdir -p "$OUTPUT_ROOT"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "orchestrator already running: $SESSION"
      exit 0
    fi
    if [[ -s "$OUTPUT_ROOT/final/metrics.json" ]]; then
      echo "factor-separated KYC gate already complete: $OUTPUT_ROOT/final"
      exit 0
    fi
    tmux new-session -d -s "$SESSION" "$SCRIPT" worker
    echo "orchestrator started: $SESSION log=$LOG"
    ;;
  status)
    print_status
    ;;
  worker)
    run_worker
    ;;
  *)
    echo "usage: $0 {start|status}" >&2
    exit 2
    ;;
esac
