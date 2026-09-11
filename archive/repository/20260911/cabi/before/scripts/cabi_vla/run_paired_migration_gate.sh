#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
POLL_SECONDS=${CABI_MIGRATION_POLL_SECONDS:-30}
EVALUATION_ROOT=${CABI_EVAL_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/evaluations}
COMPARISON_ROOT=${CABI_COMPARISON_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/comparisons}

BASELINE_CHECKPOINT=${1:?usage: run_paired_migration_gate.sh BASELINE_CHECKPOINT BASELINE_PREFIX BASELINE_GATE_SESSION BASELINE_GPU METHOD_CHECKPOINT METHOD_PREFIX METHOD_GATE_SESSION METHOD_GPU COMPARISON_NAME}
BASELINE_PREFIX=${2:?missing baseline prefix}
BASELINE_GATE_SESSION=${3:?missing baseline gate session}
BASELINE_GPU=${4:?missing baseline GPU}
METHOD_CHECKPOINT=${5:?missing method checkpoint}
METHOD_PREFIX=${6:?missing method prefix}
METHOD_GATE_SESSION=${7:?missing method gate session}
METHOD_GPU=${8:?missing method GPU}
COMPARISON_NAME=${9:?missing comparison name}

for value in "$BASELINE_PREFIX" "$BASELINE_GATE_SESSION" "$METHOD_PREFIX" "$METHOD_GATE_SESSION" "$COMPARISON_NAME"; do
  if [[ ! "$value" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "run identifiers contain unsupported characters" >&2
    exit 2
  fi
done
if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CABI_MIGRATION_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

baseline_train_output="$EVALUATION_ROOT/${BASELINE_PREFIX}_train0_k3/closed_loop_train.json"
method_train_output="$EVALUATION_ROOT/${METHOD_PREFIX}_train0_k3/closed_loop_train.json"

wait_for_checkpoint_gate() {
  local session=$1
  local output=$2
  echo "waiting checkpoint gate: $session"
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
  if [[ ! -s "$output" ]]; then
    echo "checkpoint gate ended without closed-loop calibration: $output" >&2
    exit 1
  fi
}

wait_for_checkpoint_gate "$BASELINE_GATE_SESSION" "$baseline_train_output"
wait_for_checkpoint_gate "$METHOD_GATE_SESSION" "$method_train_output"

render_evaluation() {
  local run_name=$1
  local output=$2
  local run_dir="$EVALUATION_ROOT/$run_name"
  if [[ -d "$run_dir/videos" ]]; then
    echo "video render skip complete: $run_dir/videos"
    return 0
  fi
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/cabi_vla/render_libero_bind_eval_frames.py" \
    --evaluation "$output" \
    --frame-dir "$run_dir/frames" \
    --output-dir "$run_dir/videos" \
    --codecs h264,av1 \
    --fps 20 >"$run_dir/render.log" 2>&1
}

render_paired() {
  local label=$1
  local baseline_run=$2
  local baseline_output=$3
  local method_run=$4
  local method_output=$5
  local output_dir="$COMPARISON_ROOT/${COMPARISON_NAME}_${label}_paired_videos"
  if [[ -d "$output_dir" ]]; then
    echo "paired render skip complete: $output_dir"
    return 0
  fi
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/cabi_vla/render_libero_bind_paired_videos.py" \
    --baseline-evaluation "$baseline_output" \
    --baseline-frame-dir "$EVALUATION_ROOT/$baseline_run/frames" \
    --method-evaluation "$method_output" \
    --method-frame-dir "$EVALUATION_ROOT/$method_run/frames" \
    --output-dir "$output_dir" \
    --baseline-name BC \
    --method-name CABI \
    --codecs h264,av1 \
    --fps 20 >"$COMPARISON_ROOT/${COMPARISON_NAME}_${label}_paired_render.log" 2>&1
}

# Preserve visual failure evidence even when the baseline calibration blocks
# the larger validation run.
render_evaluation "${BASELINE_PREFIX}_train0_k3" "$baseline_train_output"
render_evaluation "${METHOD_PREFIX}_train0_k3" "$method_train_output"
render_paired \
  train0 \
  "${BASELINE_PREFIX}_train0_k3" "$baseline_train_output" \
  "${METHOD_PREFIX}_train0_k3" "$method_train_output"

baseline_supervised_success=$(jq '[.rows[] | select(.action_supervised) | if .success then 1 else 0 end] | add // 0' "$baseline_train_output")
baseline_supervised_total=$(jq '[.rows[] | select(.action_supervised)] | length' "$baseline_train_output")
baseline_rate=$(jq -n --argjson success "$baseline_supervised_success" --argjson total "$baseline_supervised_total" '$success / $total')

mkdir -p "$COMPARISON_ROOT"
decision_output="$COMPARISON_ROOT/${COMPARISON_NAME}_orchestration.json"
if [[ -e "$decision_output" ]]; then
  echo "refusing to overwrite orchestration decision: $decision_output" >&2
  exit 1
fi
if ! jq -e -n --argjson rate "$baseline_rate" '$rate >= 0.70' >/dev/null; then
  temporary="${decision_output}.tmp-$$"
  jq -n \
    --arg decision BASELINE_INVALID \
    --arg baseline_output "$baseline_train_output" \
    --arg method_output "$method_train_output" \
    --argjson baseline_success "$baseline_supervised_success" \
    --argjson baseline_total "$baseline_supervised_total" \
    --argjson baseline_rate "$baseline_rate" \
    '{schema_version:1,decision:$decision,baseline_train_output:$baseline_output,method_train_output:$method_output,baseline_supervised_success:$baseline_success,baseline_supervised_total:$baseline_total,baseline_supervised_rate:$baseline_rate,full_val_started:false}' \
    >"$temporary"
  mv "$temporary" "$decision_output"
  echo "migration_gate_decision=BASELINE_INVALID output=$decision_output"
  exit 0
fi

baseline_val_run="${BASELINE_PREFIX}_val_k3"
method_val_run="${METHOD_PREFIX}_val_k3"
baseline_val_output="$EVALUATION_ROOT/$baseline_val_run/closed_loop_val.json"
method_val_output="$EVALUATION_ROOT/$method_val_run/closed_loop_val.json"

run_val() {
  local checkpoint=$1
  local run_name=$2
  local gpu=$3
  local output=$4
  local run_dir="$EVALUATION_ROOT/$run_name"
  if [[ -s "$output" ]]; then
    echo "full-val skip complete: $output"
    return 0
  fi
  if [[ -e "$run_dir/closed_loop_val.partial.json" ]]; then
    echo "partial full-val requires manual audit: $run_dir/closed_loop_val.partial.json" >&2
    return 1
  fi
  mkdir -p "$run_dir"
  CABI_EVAL_SPLIT=val \
  CABI_EVAL_HORIZONS=3 \
  CABI_EVAL_FRAME_EPISODES=5 \
    "$REPO_ROOT/scripts/cabi_vla/run_libero_bind_eval.sh" \
      "$checkpoint" "$run_name" "$gpu" >"$run_dir/evaluator.log" 2>&1
}

echo "stage=paired_full_val status=run baseline=$baseline_val_run method=$method_val_run"
run_val "$BASELINE_CHECKPOINT" "$baseline_val_run" "$BASELINE_GPU" "$baseline_val_output" &
baseline_pid=$!
run_val "$METHOD_CHECKPOINT" "$method_val_run" "$METHOD_GPU" "$method_val_output" &
method_pid=$!
baseline_status=0
method_status=0
wait "$baseline_pid" || baseline_status=$?
wait "$method_pid" || method_status=$?
if [[ "$baseline_status" != 0 || "$method_status" != 0 ]]; then
  echo "paired full-val failed: baseline=$baseline_status method=$method_status" >&2
  exit 1
fi

render_evaluation "$baseline_val_run" "$baseline_val_output"
render_evaluation "$method_val_run" "$method_val_output"
render_paired \
  val \
  "$baseline_val_run" "$baseline_val_output" \
  "$method_val_run" "$method_val_output"

comparison_output="$COMPARISON_ROOT/${COMPARISON_NAME}.json"
if [[ ! -s "$comparison_output" ]]; then
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/cabi_vla/compare_libero_bind_policies.py" \
    --baseline "$baseline_val_output" \
    --method "$method_val_output" \
    --output "$comparison_output" \
    --decision-horizon 3
fi
pilot_decision=$(jq -r .pilot_decision "$comparison_output")
temporary="${decision_output}.tmp-$$"
jq -n \
  --arg decision "$pilot_decision" \
  --arg baseline_train_output "$baseline_train_output" \
  --arg method_train_output "$method_train_output" \
  --arg baseline_val_output "$baseline_val_output" \
  --arg method_val_output "$method_val_output" \
  --arg comparison_output "$comparison_output" \
  --argjson baseline_rate "$baseline_rate" \
  '{schema_version:1,decision:$decision,baseline_supervised_rate:$baseline_rate,baseline_train_output:$baseline_train_output,method_train_output:$method_train_output,baseline_val_output:$baseline_val_output,method_val_output:$method_val_output,comparison_output:$comparison_output,full_val_started:true}' \
  >"$temporary"
mv "$temporary" "$decision_output"
echo "migration_gate_decision=$pilot_decision comparison=$comparison_output"
