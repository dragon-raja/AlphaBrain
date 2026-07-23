#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
POLL_SECONDS=${CAFC_GATE_POLL_SECONDS:-30}
EVALUATION_ROOT=${CABI_EVAL_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/evaluations}
COMPARISON_ROOT=${CABI_COMPARISON_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/comparisons}

PLAIN_CHECKPOINT=${1:?usage: run_cafc_migration_gate.sh PLAIN_CHECKPOINT PLAIN_PREFIX PLAIN_GATE_SESSION PLAIN_GPU GROUNDED_CHECKPOINT GROUNDED_PREFIX GROUNDED_GATE_SESSION GROUNDED_GPU COMPARISON_NAME}
PLAIN_PREFIX=${2:?missing plain CAFC prefix}
PLAIN_GATE_SESSION=${3:?missing plain CAFC gate session}
PLAIN_GPU=${4:?missing plain CAFC GPU}
GROUNDED_CHECKPOINT=${5:?missing Bridge+CAFC checkpoint}
GROUNDED_PREFIX=${6:?missing Bridge+CAFC prefix}
GROUNDED_GATE_SESSION=${7:?missing Bridge+CAFC gate session}
GROUNDED_GPU=${8:?missing Bridge+CAFC GPU}
COMPARISON_NAME=${9:?missing comparison name}

BC_CHECKPOINT=${CAFC_BC_CHECKPOINT:-/share/longjunyu/cabi-vla/runs/cabi_bind_pi05_bc_smoke_seed41_steps33000_edge-balanced-3epoch-v12/final_model}
BRIDGE_CHECKPOINT=${CAFC_BRIDGE_CHECKPOINT:-/share/longjunyu/cabi-vla/runs/cabi_bind_pi05_action_bridge_smoke_seed41_steps33000_edge-balanced-3epoch-v12/final_model}
CLOSURE_CHECKPOINT=${CAFC_CLOSURE_CHECKPOINT:-/share/longjunyu/cabi-vla/runs/cabi_bind_pi05_action_bridge_closure_smoke_seed41_steps33000_edge-balanced-3epoch-v12/final_model}
BC_GPU=${CAFC_BC_GPU:-0}
BRIDGE_GPU=${CAFC_BRIDGE_GPU:-1}
CLOSURE_GPU=${CAFC_CLOSURE_GPU:-2}
BC_PREFIX=${CAFC_BC_PREFIX:-bc33000_edge_balanced_seed41_v12}
BRIDGE_PREFIX=${CAFC_BRIDGE_PREFIX:-cabi_bridge33000_edge_balanced_seed41_v12}
CLOSURE_PREFIX=${CAFC_CLOSURE_PREFIX:-cabi_bridge_closure33000_edge_balanced_seed41_v12}

for value in \
  "$PLAIN_PREFIX" "$PLAIN_GATE_SESSION" \
  "$GROUNDED_PREFIX" "$GROUNDED_GATE_SESSION" "$COMPARISON_NAME"; do
  if [[ ! "$value" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "run identifiers contain unsupported characters" >&2
    exit 2
  fi
done
for value in "$POLL_SECONDS" "$PLAIN_GPU" "$GROUNDED_GPU" "$BC_GPU" "$BRIDGE_GPU" "$CLOSURE_GPU"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "poll interval and GPU ids must be non-negative integers" >&2
    exit 2
  fi
done
if (( POLL_SECONDS < 1 )); then
  echo "CAFC_GATE_POLL_SECONDS must be positive" >&2
  exit 2
fi
for checkpoint in \
  "$PLAIN_CHECKPOINT" "$GROUNDED_CHECKPOINT" \
  "$BC_CHECKPOINT" "$BRIDGE_CHECKPOINT" "$CLOSURE_CHECKPOINT"; do
  parent=${checkpoint%/final_model}
  if [[ "$checkpoint" == "$PLAIN_CHECKPOINT" || "$checkpoint" == "$GROUNDED_CHECKPOINT" ]]; then
    continue
  fi
  if [[ ! -s "$checkpoint/model.safetensors" || ! -s "$checkpoint/framework_config.yaml" ]]; then
    echo "incomplete fixed comparator checkpoint: $checkpoint (run root: $parent)" >&2
    exit 1
  fi
done

plain_train_run=${PLAIN_PREFIX}_train0_k3
grounded_train_run=${GROUNDED_PREFIX}_train0_k3
plain_train_output=$EVALUATION_ROOT/$plain_train_run/closed_loop_train.json
grounded_train_output=$EVALUATION_ROOT/$grounded_train_run/closed_loop_train.json
bc_train_run=${BC_PREFIX}_train0_k3
bridge_train_run=${BRIDGE_PREFIX}_train0_k3
closure_train_run=${CLOSURE_PREFIX}_train0_k3
bc_train_output=$EVALUATION_ROOT/$bc_train_run/closed_loop_train.json
bridge_train_output=$EVALUATION_ROOT/$bridge_train_run/closed_loop_train.json
closure_train_output=$EVALUATION_ROOT/$closure_train_run/closed_loop_train.json

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

wait_for_checkpoint_gate "$PLAIN_GATE_SESSION" "$plain_train_output"
wait_for_checkpoint_gate "$GROUNDED_GATE_SESSION" "$grounded_train_output"
for output in "$bc_train_output" "$bridge_train_output" "$closure_train_output"; do
  if [[ ! -s "$output" ]]; then
    echo "missing fixed-comparator state-0 evaluation: $output" >&2
    exit 1
  fi
done

render_evaluation() {
  local run_name=$1
  local output=$2
  local run_dir=$EVALUATION_ROOT/$run_name
  local video_dir=$run_dir/videos_h264_av1
  if [[ -s "$video_dir/manifest.json" ]] && \
     jq -e '.codecs | (index("h264") != null and index("av1") != null)' "$video_dir/manifest.json" >/dev/null; then
    echo "video render skip complete: $video_dir"
    return 0
  fi
  if [[ -e "$video_dir" ]]; then
    echo "partial or incompatible video output requires audit: $video_dir" >&2
    return 1
  fi
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/cabi_vla/render_libero_bind_eval_frames.py" \
    --evaluation "$output" \
    --frame-dir "$run_dir/frames" \
    --output-dir "$video_dir" \
    --codecs h264,av1 \
    --fps 20 >"$run_dir/render_h264_av1.log" 2>&1
}

render_paired() {
  local label=$1
  local baseline_name=$2
  local method_name=$3
  local baseline_run=$4
  local baseline_output=$5
  local method_run=$6
  local method_output=$7
  local output_dir=$COMPARISON_ROOT/${COMPARISON_NAME}_${label}_paired_h264_av1
  if [[ -s "$output_dir/manifest.json" ]] && \
     jq -e '.codecs | (index("h264") != null and index("av1") != null)' "$output_dir/manifest.json" >/dev/null; then
    echo "paired render skip complete: $output_dir"
    return 0
  fi
  if [[ -e "$output_dir" ]]; then
    echo "partial paired video output requires audit: $output_dir" >&2
    return 1
  fi
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/cabi_vla/render_libero_bind_paired_videos.py" \
    --baseline-evaluation "$baseline_output" \
    --baseline-frame-dir "$EVALUATION_ROOT/$baseline_run/frames" \
    --method-evaluation "$method_output" \
    --method-frame-dir "$EVALUATION_ROOT/$method_run/frames" \
    --output-dir "$output_dir" \
    --baseline-name "$baseline_name" \
    --method-name "$method_name" \
    --codecs h264,av1 \
    --fps 20 >"$COMPARISON_ROOT/${COMPARISON_NAME}_${label}_paired_render.log" 2>&1
}

mkdir -p "$COMPARISON_ROOT"
render_evaluation "$plain_train_run" "$plain_train_output"
render_evaluation "$grounded_train_run" "$grounded_train_output"
render_paired train0_plain BC CAFC \
  "$bc_train_run" "$bc_train_output" "$plain_train_run" "$plain_train_output"
render_paired train0_grounded CABI-Bridge Bridge+CAFC \
  "$bridge_train_run" "$bridge_train_output" "$grounded_train_run" "$grounded_train_output"

success_count() {
  local output=$1
  local supervised=$2
  jq --argjson supervised "$supervised" \
    '[.rows[] | select(.action_supervised == $supervised and .success)] | length' \
    "$output"
}

plain_id=$(success_count "$plain_train_output" true)
plain_ood=$(success_count "$plain_train_output" false)
grounded_id=$(success_count "$grounded_train_output" true)
grounded_ood=$(success_count "$grounded_train_output" false)
decision_output=$COMPARISON_ROOT/${COMPARISON_NAME}_orchestration.json
if [[ -e "$decision_output" ]]; then
  echo "refusing to overwrite CAFC decision: $decision_output" >&2
  exit 1
fi

write_early_decision() {
  local decision=$1
  local reason=$2
  local temporary=${decision_output}.tmp-$$
  jq -n \
    --arg decision "$decision" --arg reason "$reason" \
    --arg plain_output "$plain_train_output" --arg grounded_output "$grounded_train_output" \
    --argjson plain_id "$plain_id" --argjson plain_ood "$plain_ood" \
    --argjson grounded_id "$grounded_id" --argjson grounded_ood "$grounded_ood" \
    '{schema_version:1,decision:$decision,reason:$reason,decision_horizon:3,full_val_started:false,state0:{plain_cafc:{id_success:$plain_id,ood_success:$plain_ood,output:$plain_output},bridge_cafc:{id_success:$grounded_id,ood_success:$grounded_ood,output:$grounded_output}}}' \
    >"$temporary"
  mv "$temporary" "$decision_output"
  echo "cafc_gate_decision=$decision reason=$reason output=$decision_output"
}

if (( plain_id < 3 && grounded_id < 3 )); then
  write_early_decision BASELINE_INVALID NEITHER_CAFC_ARM_REACHED_3_OF_4_OBSERVED_STATE0_SUCCESSES
  exit 0
fi
if (( (plain_id < 3 || plain_ood < 1) && (grounded_id < 3 || grounded_ood < 1) )); then
  write_early_decision STOP_CAFC NO_CAFC_ARM_REACHED_THE_STATE0_MIGRATION_GATE
  exit 0
fi

run_val() {
  local checkpoint=$1
  local run_name=$2
  local gpu=$3
  local output=$4
  local run_dir=$EVALUATION_ROOT/$run_name
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

# Keep evaluator run names short enough for the Unix-domain policy socket.
bc_val_run=bc_cafc_gate_s41_v14_val_k3
bridge_val_run=bridge_cafc_gate_s41_v14_val_k3
closure_val_run=closure_cafc_gate_s41_v14_val_k3
plain_val_run=${PLAIN_PREFIX}_val_k3
grounded_val_run=${GROUNDED_PREFIX}_val_k3
bc_val_output=$EVALUATION_ROOT/$bc_val_run/closed_loop_val.json
bridge_val_output=$EVALUATION_ROOT/$bridge_val_run/closed_loop_val.json
closure_val_output=$EVALUATION_ROOT/$closure_val_run/closed_loop_val.json
plain_val_output=$EVALUATION_ROOT/$plain_val_run/closed_loop_val.json
grounded_val_output=$EVALUATION_ROOT/$grounded_val_run/closed_loop_val.json

echo "stage=five_way_full_val status=run"
run_val "$BC_CHECKPOINT" "$bc_val_run" "$BC_GPU" "$bc_val_output" & bc_pid=$!
run_val "$BRIDGE_CHECKPOINT" "$bridge_val_run" "$BRIDGE_GPU" "$bridge_val_output" & bridge_pid=$!
run_val "$CLOSURE_CHECKPOINT" "$closure_val_run" "$CLOSURE_GPU" "$closure_val_output" & closure_pid=$!
run_val "$PLAIN_CHECKPOINT" "$plain_val_run" "$PLAIN_GPU" "$plain_val_output" & plain_pid=$!
run_val "$GROUNDED_CHECKPOINT" "$grounded_val_run" "$GROUNDED_GPU" "$grounded_val_output" & grounded_pid=$!

statuses=()
failed=0
for item in \
  "$bc_pid:bc" "$bridge_pid:bridge" "$closure_pid:closure" \
  "$plain_pid:plain" "$grounded_pid:grounded"; do
  pid=${item%%:*}
  name=${item#*:}
  status=0
  wait "$pid" || status=$?
  statuses+=("$name=$status")
  if (( status != 0 )); then
    failed=1
  fi
done
if (( failed != 0 )); then
  printf 'five-way full-val failed: %s\n' "${statuses[*]}" >&2
  exit 1
fi

for pair in \
  "$bc_val_run $bc_val_output" \
  "$bridge_val_run $bridge_val_output" \
  "$closure_val_run $closure_val_output" \
  "$plain_val_run $plain_val_output" \
  "$grounded_val_run $grounded_val_output"; do
  read -r run_name output <<<"$pair"
  render_evaluation "$run_name" "$output"
done
render_paired val_plain BC CAFC \
  "$bc_val_run" "$bc_val_output" "$plain_val_run" "$plain_val_output"
render_paired val_grounded CABI-Bridge Bridge+CAFC \
  "$bridge_val_run" "$bridge_val_output" "$grounded_val_run" "$grounded_val_output"

compare() {
  local baseline=$1
  local method=$2
  local output=$3
  if [[ -s "$output" ]]; then
    echo "comparison skip complete: $output"
    return 0
  fi
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/cabi_vla/compare_libero_bind_policies.py" \
    --baseline "$baseline" --method "$method" --output "$output" --decision-horizon 3
}

plain_exact=$COMPARISON_ROOT/${COMPARISON_NAME}_plain_vs_bc.json
grounded_exact=$COMPARISON_ROOT/${COMPARISON_NAME}_grounded_vs_bridge.json
plain_strong=$COMPARISON_ROOT/${COMPARISON_NAME}_plain_vs_closure.json
grounded_strong=$COMPARISON_ROOT/${COMPARISON_NAME}_grounded_vs_closure.json
compare "$bc_val_output" "$plain_val_output" "$plain_exact"
compare "$bridge_val_output" "$grounded_val_output" "$grounded_exact"
compare "$closure_val_output" "$plain_val_output" "$plain_strong"
compare "$closure_val_output" "$grounded_val_output" "$grounded_strong"

"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/cabi_vla/decide_cafc_gate.py" \
  --plain-state0 "$plain_train_output" \
  --grounded-state0 "$grounded_train_output" \
  --plain-exact "$plain_exact" \
  --grounded-exact "$grounded_exact" \
  --plain-strong "$plain_strong" \
  --grounded-strong "$grounded_strong" \
  --output "$decision_output"
echo "cafc_gate_complete decision=$(jq -r .decision "$decision_output") output=$decision_output"
