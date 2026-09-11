#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SCRIPT=$(realpath "$0")
COMMAND=${1:-status}
SESSION=${PLUS_MV_GATE_SESSION:-plus-mv-gate-eval}
RUN_ROOT=${PLUS_MV_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/libero-plus-mv-rgb-v1/runs}
EVAL_ROOT=${PLUS_MV_GATE_EVAL_ROOT:-/share/longjunyu/alphabrain/experiments/libero-plus-mv-rgb-v1/gate-v1}
OFFICIAL_BASELINE=${PLUS_MV_OFFICIAL_BASELINE:-/share/longjunyu/alphabrain/experiments/libero-plus-view-gap-v1/pi05-libero-official-v1}
TRAIN_STEPS=${PLUS_MV_WAVE_STEPS:-33000}
SEED=${PLUS_MV_WAVE_SEED:-41}
LOG="$EVAL_ROOT/orchestrator.log"

NAMES=(action_b100 action_b025 visual_b100 visual_b025)
TRAIN_SESSIONS=(plus-mv-a100-s41 plus-mv-a025-s41 plus-mv-v100-s41 plus-mv-v025-s41)
RUN_IDS=(
  "pi05_plus_mv_action_only_gate-v1-b100_seed${SEED}_steps${TRAIN_STEPS}"
  "pi05_plus_mv_action_only_gate-v1-b025_seed${SEED}_steps${TRAIN_STEPS}"
  "pi05_plus_mv_visual_lora_gate-v1-b100_seed${SEED}_steps${TRAIN_STEPS}"
  "pi05_plus_mv_visual_lora_gate-v1-b025_seed${SEED}_steps${TRAIN_STEPS}"
)

training_ready() {
  local index=$1
  [[ -s "$RUN_ROOT/${RUN_IDS[$index]}/final_model/model.safetensors" ]] \
    && ! tmux has-session -t "${TRAIN_SESSIONS[$index]}" 2>/dev/null
}

print_status() {
  for index in "${!NAMES[@]}"; do
    state=WAITING
    if training_ready "$index"; then
      state=TRAINED
    elif ! tmux has-session -t "${TRAIN_SESSIONS[$index]}" 2>/dev/null; then
      state=TRAINING_STOPPED
    fi
    if [[ -s "$EVAL_ROOT/${NAMES[$index]}-gap/metrics.json" ]]; then
      state=GAP_EVALUATED
    fi
    printf '%-14s %-18s %s\n' "${NAMES[$index]}" "$state" "$RUN_ROOT/${RUN_IDS[$index]}"
  done
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "orchestrator=RUNNING session=$SESSION"
  elif [[ -s "$EVAL_ROOT/final-best/metrics.json" ]]; then
    echo "orchestrator=COMPLETE output=$EVAL_ROOT/final-best"
  else
    echo "orchestrator=NOT_RUNNING"
  fi
}

run_worker() {
  mkdir -p "$EVAL_ROOT"
  exec > >(tee -a "$LOG") 2>&1
  echo "gate_worker_started=$(date -u +%FT%TZ)"

  while true; do
    pending=0
    for index in "${!NAMES[@]}"; do
      if training_ready "$index"; then
        continue
      fi
      pending=1
      if ! tmux has-session -t "${TRAIN_SESSIONS[$index]}" 2>/dev/null; then
        echo "training stopped without a complete checkpoint: ${NAMES[$index]}" >&2
        exit 1
      fi
    done
    if [[ "$pending" == 0 ]]; then
      break
    fi
    sleep 60
  done
  echo "all_training_checkpoints_ready=$(date -u +%FT%TZ)"

  for index in "${!NAMES[@]}"; do
    name=${NAMES[$index]}
    checkpoint="$RUN_ROOT/${RUN_IDS[$index]}/final_model"
    output="$EVAL_ROOT/${name}-gap"
    if [[ -s "$output/metrics.json" ]]; then
      echo "skip_completed_gap_eval=$name"
      continue
    fi
    CHECKPOINT="$checkpoint" OUTPUT_DIR="$output" \
      EVAL_MODES=gap PROBE_SAMPLES=0 VIDEO_EPISODES=8 \
      BASE_PORT=$((18300 + index * 20)) \
      "$REPO_ROOT/scripts/cabi_vla/run_alphabrain_pi05_libero_plus_view_eval.sh"
  done

  compare_dir="$EVAL_ROOT/comparison"
  mkdir -p "$compare_dir"
  "$REPO_ROOT/.venv/bin/python" \
    "$REPO_ROOT/scripts/cabi_vla/compare_pi05_libero_plus_multiview_gate.py" \
    --baseline "official_pi05=$OFFICIAL_BASELINE" \
    --run "action_b100=$EVAL_ROOT/action_b100-gap" \
    --run "action_b025=$EVAL_ROOT/action_b025-gap" \
    --run "visual_b100=$EVAL_ROOT/visual_b100-gap" \
    --run "visual_b025=$EVAL_ROOT/visual_b025-gap" \
    --output-json "$compare_dir/metrics.json" \
    --output-report "$compare_dir/report_zh.md" \
    --output-figure "$compare_dir/summary.png"

  best_name=$(METRICS="$compare_dir/metrics.json" "$REPO_ROOT/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path
print(json.loads(Path(os.environ["METRICS"]).read_text())["best_run"])
PY
)
  best_index=-1
  for index in "${!NAMES[@]}"; do
    if [[ "${NAMES[$index]}" == "$best_name" ]]; then
      best_index=$index
      break
    fi
  done
  if [[ "$best_index" -lt 0 ]]; then
    echo "unknown best run: $best_name" >&2
    exit 1
  fi

  candidate_output="$EVAL_ROOT/${best_name}-candidates"
  if [[ ! -s "$candidate_output/metrics.json" ]]; then
    CHECKPOINT="$RUN_ROOT/${RUN_IDS[$best_index]}/final_model" \
      OUTPUT_DIR="$candidate_output" \
      EVAL_MODES=candidates PROBE_SAMPLES=3 VIDEO_EPISODES=16 \
      BASE_PORT=18400 \
      "$REPO_ROOT/scripts/cabi_vla/run_alphabrain_pi05_libero_plus_view_eval.sh"
  fi

  mapfile -t combined_episode_files < <(
    find "$EVAL_ROOT/${best_name}-gap" "$candidate_output" \
      -maxdepth 1 -name 'episodes-shard-*.jsonl' -type f | sort
  )
  final_dir="$EVAL_ROOT/final-best"
  mkdir -p "$final_dir"
  "$REPO_ROOT/.venv/bin/python" \
    "$REPO_ROOT/scripts/cabi_vla/analyze_pi05_libero_plus_views.py" \
    --episodes "${combined_episode_files[@]}" \
    --output-json "$final_dir/metrics.json" \
    --output-figure "$final_dir/summary.png" \
    --output-report "$final_dir/report_zh.md"
  BEST_RUN="$best_name" RUN_ID="${RUN_IDS[$best_index]}" \
    "$REPO_ROOT/.venv/bin/python" - <<'PY' >"$final_dir/selection.json"
import json
import os
print(json.dumps({"best_run": os.environ["BEST_RUN"], "run_id": os.environ["RUN_ID"]}, indent=2))
PY
  echo "gate_worker_complete=$(date -u +%FT%TZ) best_run=$best_name output=$final_dir"
}

case "$COMMAND" in
  start)
    mkdir -p "$EVAL_ROOT"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "orchestrator already running: $SESSION"
      exit 0
    fi
    if [[ -s "$EVAL_ROOT/final-best/metrics.json" ]]; then
      echo "gate already complete: $EVAL_ROOT/final-best"
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
