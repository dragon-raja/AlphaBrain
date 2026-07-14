#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
OUTPUT_ROOT=${FRESH_CLOSED_LOOP_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2}
EPISODE_ROOT=${FRESH_EPISODE_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-v2-128}
WINDOW_ROOT=${FRESH_WINDOW_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-windows-v2-128}
STEPS=${FRESH_FINAL_TRAIN_STEPS:-27607}
VAL_TAG="val_budget${STEPS}"
SEEDS=(41 42 43)

mkdir -p "$OUTPUT_ROOT/final_gate_logs"
exec > >(tee -a "$OUTPUT_ROOT/final_gate_logs/pipeline.log") 2>&1

log_stage() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"
}

export FRESH_CLOSED_LOOP_OUTPUT_ROOT="$OUTPUT_ROOT"
export FRESH_EPISODE_ROOT="$EPISODE_ROOT"
export FRESH_WINDOW_ROOT="$WINDOW_ROOT"

log_stage "wait_full_h_calibration"
while true; do
  complete=0
  for seed in "${SEEDS[@]}"; do
    checkpoint="$OUTPUT_ROOT/fresh_closed_loop_full_h_seed${seed}/final_model/model.safetensors"
    if tmux has-session -t "fresh-final-v2-full-${seed}" 2>/dev/null; then
      continue
    fi
    if [ -f "$checkpoint" ]; then
      complete=$((complete + 1))
    else
      echo "Full-H seed $seed stopped without a final checkpoint" >&2
      exit 1
    fi
  done
  if [ "$complete" = "${#SEEDS[@]}" ]; then
    break
  fi
  log_stage "full_h_checkpoints=${complete}/${#SEEDS[@]}"
  sleep 60
done

log_stage "validation_end_to_end_start"
pids=()
names=()
for index in "${!SEEDS[@]}"; do
  seed=${SEEDS[$index]}
  output="$OUTPUT_ROOT/fresh_closed_loop_full_h_seed${seed}/closed_loop_end_to_end_${VAL_TAG}.json"
  if [ -f "$output" ]; then
    log_stage "skip_completed_validation_seed=${seed}"
    continue
  fi
  FRESH_EVAL_ONLY=end_to_end \
  FRESH_EVAL_SPLIT=val \
  FRESH_EVAL_OUTPUT_TAG="$VAL_TAG" \
  FRESH_EVAL_MAX_STEPS=320 \
  FRESH_SAVE_EVAL_VIDEOS=1 \
  FRESH_EVAL_VIDEO_GROUPS=13 \
  bash "$REPO_ROOT/scripts/fresh_vla/run_libero_closed_loop_eval.sh" full_h "$seed" "$index" \
    >"$OUTPUT_ROOT/final_gate_logs/validation_seed${seed}.log" 2>&1 &
  pids+=("$!")
  names+=("$seed")
done
failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "Full-H validation failed for seed ${names[$index]}" >&2
    failed=1
  fi
done
if [ "$failed" = 1 ]; then
  exit 1
fi

log_stage "validation_summary"
PYTHONPATH="$REPO_ROOT/scripts/fresh_vla${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
  "$REPO_ROOT/scripts/fresh_vla/summarize_libero_baseline_validation.py" \
  --runs-root "$OUTPUT_ROOT" \
  --tag "$VAL_TAG" \
  --training-steps "$STEPS"
baseline_result="$OUTPUT_ROOT/baseline_validation_${VAL_TAG}/results.json"
baseline_valid=$(
  "$PYTHON" -c 'import json,sys; print("true" if json.load(open(sys.argv[1]))["baseline_valid"] else "false")' \
    "$baseline_result"
)
if [ "$baseline_valid" != true ]; then
  log_stage "baseline_invalid_stop_before_test"
  exit 3
fi

log_stage "common_budget_frozen_steps=${STEPS}"
bash "$REPO_ROOT/scripts/fresh_vla/run_libero_closed_loop_train_matrix.sh" "$STEPS"

log_stage "test_closed_loop_matrix_start"
FRESH_EVAL_ONLY=all \
FRESH_EVAL_SPLIT=test \
FRESH_EVAL_OUTPUT_TAG= \
FRESH_EVAL_MAX_STEPS=320 \
FRESH_SAVE_EVAL_VIDEOS=1 \
FRESH_EVAL_VIDEO_GROUPS=13 \
bash "$REPO_ROOT/scripts/fresh_vla/run_libero_closed_loop_eval_matrix.sh"

log_stage "offline_matrix_start"
bash "$REPO_ROOT/scripts/fresh_vla/run_libero_closed_loop_offline_eval_matrix.sh"

log_stage "group_level_summaries"
"$PYTHON" "$REPO_ROOT/scripts/fresh_vla/summarize_libero_closed_loop.py" --runs-root "$OUTPUT_ROOT"
"$PYTHON" "$REPO_ROOT/scripts/fresh_vla/summarize_libero_episode_offline.py" --runs-root "$OUTPUT_ROOT"
"$PYTHON" "$REPO_ROOT/scripts/fresh_vla/summarize_libero_reach.py" \
  --runs-root "$OUTPUT_ROOT" \
  --input-name deterministic_reach.json

log_stage "final_decision"
"$PYTHON" "$REPO_ROOT/scripts/fresh_vla/finalize_libero_decision.py"
log_stage "pipeline_complete"
