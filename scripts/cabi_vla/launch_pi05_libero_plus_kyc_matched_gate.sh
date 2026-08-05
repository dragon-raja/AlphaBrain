#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SCRIPT=$(realpath "$0")
COMMAND=${1:-status}
SESSION=${PLUS_KYC_GATE_SESSION:-plus-kyc-matched-gate}
RUN_ROOT=${PLUS_MV_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/libero-plus-mv-rgb-v1/runs}
OUTPUT_ROOT=${PLUS_KYC_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/libero-plus-kyc-matched-v1}
PROTOCOL=${PLUS_KYC_PROTOCOL:-/share/longjunyu/alphabrain/experiments/libero-plus-camera-background-v1/protocol-v1.json}
TAG=${PLUS_KYC_RUN_TAG:-matched-v1}
STEPS=${PLUS_KYC_STEPS:-33000}
WAIT_SECONDS=${PLUS_KYC_WAIT_SECONDS:-60}
LOG=$OUTPUT_ROOT/orchestrator.log
SEEDS=(41 42 43)
METHODS=(control kyc)

training_session() {
  local method=$1 seed=$2
  echo "plus-${method}-s${seed}"
}

run_id() {
  local method=$1 seed=$2
  echo "pi05_plus_mv_visual_lora_${method}_${TAG}_seed${seed}_steps${STEPS}"
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
print(len(build_episode_specs(protocol, modes=["composition"], init_state_count=2)))
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
  local expected=unknown
  if [[ -s "$PROTOCOL" ]]; then
    expected=$(expected_count)
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

wait_for_training() {
  while true; do
    local pending=0
    for seed in "${SEEDS[@]}"; do
      for method in "${METHODS[@]}"; do
        local checkpoint
        checkpoint=$(checkpoint_dir "$method" "$seed")
        if [[ ! -s "$checkpoint/model.safetensors" ]]; then
          local session
          session=$(training_session "$method" "$seed")
          if ! tmux has-session -t "$session" 2>/dev/null; then
            echo "training ended without checkpoint: method=$method seed=$seed session=$session" >&2
            exit 1
          fi
          pending=$((pending + 1))
        fi
      done
    done
    if [[ "$pending" == 0 ]]; then
      echo "all_training_checkpoints_ready=$(date -u +%FT%TZ)"
      return
    fi
    echo "waiting_for_training=$(date -u +%FT%TZ) pending=$pending"
    sleep "$WAIT_SECONDS"
  done
}

run_worker() {
  mkdir -p "$OUTPUT_ROOT"
  exec > >(tee -a "$LOG") 2>&1
  echo "matched_kyc_gate_started=$(date -u +%FT%TZ) commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  wait_for_training
  local expected
  expected=$(expected_count)
  echo "expected_episodes_per_run=$expected"

  local run_index=0
  for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      local checkpoint output count
      checkpoint=$(checkpoint_dir "$method" "$seed")
      output=$(evaluation_dir "$method" "$seed")
      count=$(episode_count "$output")
      if [[ "$count" -ne "$expected" ]]; then
        echo "evaluation_started=$(date -u +%FT%TZ) method=$method seed=$seed existing=$count/$expected"
        CHECKPOINT="$checkpoint" \
        OUTPUT_DIR="$output" \
        PROTOCOL="$PROTOCOL" \
        EVAL_MODES=composition \
        PROBE_SAMPLES=0 \
        VIDEO_EPISODES=16 \
        SKIP_ANALYSIS=1 \
        BASE_PORT=$((18800 + run_index * 10)) \
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
    --output-json "$OUTPUT_ROOT/final/metrics.json" \
    --output-report "$OUTPUT_ROOT/final/report_zh.md" \
    --output-figure "$OUTPUT_ROOT/final/summary.png"
  echo "matched_kyc_gate_complete=$(date -u +%FT%TZ) output=$OUTPUT_ROOT/final"
}

case "$COMMAND" in
  start)
    mkdir -p "$OUTPUT_ROOT"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "orchestrator already running: $SESSION"
      exit 0
    fi
    if [[ -s "$OUTPUT_ROOT/final/metrics.json" ]]; then
      echo "matched KYC gate already complete: $OUTPUT_ROOT/final"
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
