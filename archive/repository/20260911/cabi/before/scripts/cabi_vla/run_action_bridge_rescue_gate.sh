#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
POLL_SECONDS=${CABI_ACTION_BRIDGE_POLL_SECONDS:-30}
EVALUATION_ROOT=${CABI_EVAL_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/evaluations}
COMPARISON_ROOT=${CABI_COMPARISON_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/comparisons}

BC_CHECKPOINT=${1:?usage: run_action_bridge_rescue_gate.sh BC_CHECKPOINT BC_PREFIX BC_GATE_SESSION BC_GPU BRIDGE_CHECKPOINT BRIDGE_PREFIX BRIDGE_GATE_SESSION BRIDGE_GPU METHOD_CHECKPOINT METHOD_PREFIX METHOD_GATE_SESSION METHOD_GPU COMPARISON_NAME}
BC_PREFIX=${2:?missing BC prefix}
BC_GATE_SESSION=${3:?missing BC gate session}
BC_GPU=${4:?missing BC GPU}
BRIDGE_CHECKPOINT=${5:?missing bridge checkpoint}
BRIDGE_PREFIX=${6:?missing bridge prefix}
BRIDGE_GATE_SESSION=${7:?missing bridge gate session}
BRIDGE_GPU=${8:?missing bridge GPU}
METHOD_CHECKPOINT=${9:?missing method checkpoint}
METHOD_PREFIX=${10:?missing method prefix}
METHOD_GATE_SESSION=${11:?missing method gate session}
METHOD_GPU=${12:?missing method GPU}
COMPARISON_NAME=${13:?missing comparison name}

for value in \
  "$BC_PREFIX" "$BC_GATE_SESSION" \
  "$BRIDGE_PREFIX" "$BRIDGE_GATE_SESSION" \
  "$METHOD_PREFIX" "$METHOD_GATE_SESSION" "$COMPARISON_NAME"; do
  if [[ ! "$value" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "run identifiers contain unsupported characters" >&2
    exit 2
  fi
done
if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CABI_ACTION_BRIDGE_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

bc_train_run="${BC_PREFIX}_train0_k3"
bridge_train_run="${BRIDGE_PREFIX}_train0_k3"
method_train_run="${METHOD_PREFIX}_train0_k3"
bc_train_output="$EVALUATION_ROOT/$bc_train_run/closed_loop_train.json"
bridge_train_output="$EVALUATION_ROOT/$bridge_train_run/closed_loop_train.json"
method_train_output="$EVALUATION_ROOT/$method_train_run/closed_loop_train.json"

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

wait_for_checkpoint_gate "$BC_GATE_SESSION" "$bc_train_output"
wait_for_checkpoint_gate "$BRIDGE_GATE_SESSION" "$bridge_train_output"
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
  local baseline_name=$2
  local baseline_run=$3
  local baseline_output=$4
  local method_run=$5
  local method_output=$6
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
    --baseline-name "$baseline_name" \
    --method-name CABI-Bridge-Closure \
    --codecs h264,av1 \
    --fps 20 >"$COMPARISON_ROOT/${COMPARISON_NAME}_${label}_paired_render.log" 2>&1
}

for pair in \
  "$bc_train_run $bc_train_output" \
  "$bridge_train_run $bridge_train_output" \
  "$method_train_run $method_train_output"; do
  read -r run_name output <<<"$pair"
  render_evaluation "$run_name" "$output"
done
render_paired train0_bc BC "$bc_train_run" "$bc_train_output" "$method_train_run" "$method_train_output"
render_paired train0_bridge CABI-Bridge "$bridge_train_run" "$bridge_train_output" "$method_train_run" "$method_train_output"

success_count() {
  local output=$1
  local supervised=$2
  jq --argjson supervised "$supervised" \
    '[.rows[] | select(.action_supervised == $supervised and .success)] | length' \
    "$output"
}

bc_id=$(success_count "$bc_train_output" true)
bc_ood=$(success_count "$bc_train_output" false)
bridge_id=$(success_count "$bridge_train_output" true)
bridge_ood=$(success_count "$bridge_train_output" false)
method_id=$(success_count "$method_train_output" true)
method_ood=$(success_count "$method_train_output" false)

comparator_name=none
comparator_prefix=
comparator_checkpoint=
comparator_gpu=
comparator_train_output=
comparator_id=0
comparator_ood=0
if (( bridge_id >= 3 )); then
  comparator_name=action_bridge
  comparator_prefix=$BRIDGE_PREFIX
  comparator_checkpoint=$BRIDGE_CHECKPOINT
  comparator_gpu=$BRIDGE_GPU
  comparator_train_output=$bridge_train_output
  comparator_id=$bridge_id
  comparator_ood=$bridge_ood
elif (( bc_id >= 3 )); then
  comparator_name=plain_bc
  comparator_prefix=$BC_PREFIX
  comparator_checkpoint=$BC_CHECKPOINT
  comparator_gpu=$BC_GPU
  comparator_train_output=$bc_train_output
  comparator_id=$bc_id
  comparator_ood=$bc_ood
fi

mkdir -p "$COMPARISON_ROOT"
decision_output="$COMPARISON_ROOT/${COMPARISON_NAME}_orchestration.json"
if [[ -e "$decision_output" ]]; then
  echo "refusing to overwrite orchestration decision: $decision_output" >&2
  exit 1
fi

write_early_decision() {
  local decision=$1
  local reason=$2
  local temporary="${decision_output}.tmp-$$"
  jq -n \
    --arg decision "$decision" \
    --arg reason "$reason" \
    --arg comparator "$comparator_name" \
    --arg bc_output "$bc_train_output" \
    --arg bridge_output "$bridge_train_output" \
    --arg method_output "$method_train_output" \
    --argjson bc_id "$bc_id" --argjson bc_ood "$bc_ood" \
    --argjson bridge_id "$bridge_id" --argjson bridge_ood "$bridge_ood" \
    --argjson method_id "$method_id" --argjson method_ood "$method_ood" \
    '{schema_version:1,decision:$decision,reason:$reason,comparator:$comparator,bc_train_output:$bc_output,bridge_train_output:$bridge_output,method_train_output:$method_output,state0:{bc:{id_success:$bc_id,ood_success:$bc_ood},bridge:{id_success:$bridge_id,ood_success:$bridge_ood},method:{id_success:$method_id,ood_success:$method_ood}},full_val_started:false}' \
    >"$temporary"
  mv "$temporary" "$decision_output"
  echo "action_bridge_gate_decision=$decision reason=$reason output=$decision_output"
}

if [[ "$comparator_name" == none ]]; then
  write_early_decision BASELINE_INVALID NO_OBSERVED_BASELINE_REACHED_3_OF_4
  exit 0
fi
if (( method_id < 3 )); then
  write_early_decision ACTION_BRIDGE_INVALID METHOD_DID_NOT_REACH_3_OF_4_OBSERVED
  exit 0
fi
if (( method_id + 1 < comparator_id )); then
  write_early_decision PILOT_DOES_NOT_CLEAR_ACTION_BRIDGE_GATE OBSERVED_DEGRADATION_EXCEEDS_ONE_EDGE
  exit 0
fi
if (( method_ood < 1 || method_ood <= comparator_ood )); then
  write_early_decision PILOT_DOES_NOT_CLEAR_ACTION_BRIDGE_GATE NO_STATE0_ACTION_FREE_GAIN
  exit 0
fi

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

bc_val_run="${BC_PREFIX}_val_k3"
bridge_val_run="${BRIDGE_PREFIX}_val_k3"
method_val_run="${METHOD_PREFIX}_val_k3"
bc_val_output="$EVALUATION_ROOT/$bc_val_run/closed_loop_val.json"
bridge_val_output="$EVALUATION_ROOT/$bridge_val_run/closed_loop_val.json"
method_val_output="$EVALUATION_ROOT/$method_val_run/closed_loop_val.json"

echo "stage=three_way_full_val status=run"
run_val "$BC_CHECKPOINT" "$bc_val_run" "$BC_GPU" "$bc_val_output" &
bc_pid=$!
run_val "$BRIDGE_CHECKPOINT" "$bridge_val_run" "$BRIDGE_GPU" "$bridge_val_output" &
bridge_pid=$!
run_val "$METHOD_CHECKPOINT" "$method_val_run" "$METHOD_GPU" "$method_val_output" &
method_pid=$!
bc_status=0
bridge_status=0
method_status=0
wait "$bc_pid" || bc_status=$?
wait "$bridge_pid" || bridge_status=$?
wait "$method_pid" || method_status=$?
if [[ "$bc_status" != 0 || "$bridge_status" != 0 || "$method_status" != 0 ]]; then
  echo "three-way full-val failed: bc=$bc_status bridge=$bridge_status method=$method_status" >&2
  exit 1
fi

render_evaluation "$bc_val_run" "$bc_val_output"
render_evaluation "$bridge_val_run" "$bridge_val_output"
render_evaluation "$method_val_run" "$method_val_output"
render_paired val_bc BC "$bc_val_run" "$bc_val_output" "$method_val_run" "$method_val_output"
render_paired val_bridge CABI-Bridge "$bridge_val_run" "$bridge_val_output" "$method_val_run" "$method_val_output"

bc_comparison="$COMPARISON_ROOT/${COMPARISON_NAME}_vs_bc.json"
bridge_comparison="$COMPARISON_ROOT/${COMPARISON_NAME}_vs_bridge.json"
if [[ ! -s "$bc_comparison" ]]; then
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/cabi_vla/compare_libero_bind_policies.py" \
    --baseline "$bc_val_output" --method "$method_val_output" \
    --output "$bc_comparison" --decision-horizon 3
fi
if [[ ! -s "$bridge_comparison" ]]; then
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/cabi_vla/compare_libero_bind_policies.py" \
    --baseline "$bridge_val_output" --method "$method_val_output" \
    --output "$bridge_comparison" --decision-horizon 3
fi

selected_comparison=$bc_comparison
if [[ "$comparator_name" == action_bridge ]]; then
  selected_comparison=$bridge_comparison
fi
decision=$(jq -r .pilot_decision "$selected_comparison")
temporary="${decision_output}.tmp-$$"
jq -n \
  --arg decision "$decision" --arg comparator "$comparator_name" \
  --arg bc_train_output "$bc_train_output" --arg bridge_train_output "$bridge_train_output" \
  --arg method_train_output "$method_train_output" --arg bc_val_output "$bc_val_output" \
  --arg bridge_val_output "$bridge_val_output" --arg method_val_output "$method_val_output" \
  --arg bc_comparison "$bc_comparison" --arg bridge_comparison "$bridge_comparison" \
  --arg selected_comparison "$selected_comparison" \
  --argjson bc_id "$bc_id" --argjson bc_ood "$bc_ood" \
  --argjson bridge_id "$bridge_id" --argjson bridge_ood "$bridge_ood" \
  --argjson method_id "$method_id" --argjson method_ood "$method_ood" \
  '{schema_version:1,decision:$decision,comparator:$comparator,state0:{bc:{id_success:$bc_id,ood_success:$bc_ood},bridge:{id_success:$bridge_id,ood_success:$bridge_ood},method:{id_success:$method_id,ood_success:$method_ood}},bc_train_output:$bc_train_output,bridge_train_output:$bridge_train_output,method_train_output:$method_train_output,bc_val_output:$bc_val_output,bridge_val_output:$bridge_val_output,method_val_output:$method_val_output,bc_comparison:$bc_comparison,bridge_comparison:$bridge_comparison,selected_comparison:$selected_comparison,full_val_started:true}' \
  >"$temporary"
mv "$temporary" "$decision_output"
echo "action_bridge_gate_decision=$decision comparator=$comparator_name output=$decision_output"
