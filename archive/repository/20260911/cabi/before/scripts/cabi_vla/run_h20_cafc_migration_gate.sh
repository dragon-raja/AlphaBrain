#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
POLL_SECONDS=${H20_GATE_POLL_SECONDS:-30}
EVALUATION_ROOT=${CABI_EVAL_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/evaluations}
COMPARISON_ROOT=${CABI_COMPARISON_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/comparisons}

BC_CHECKPOINT=${1:?missing BC-H20 checkpoint}
BC_PREFIX=${2:?missing BC-H20 prefix}
BC_GATE_SESSION=${3:?missing BC-H20 gate session}
BC_GPU=${4:?missing BC-H20 GPU}
BRIDGE_CHECKPOINT=${5:?missing Bridge-H20 checkpoint}
BRIDGE_PREFIX=${6:?missing Bridge-H20 prefix}
BRIDGE_GATE_SESSION=${7:?missing Bridge-H20 gate session}
BRIDGE_GPU=${8:?missing Bridge-H20 GPU}
PLAIN_CHECKPOINT=${9:?missing CAFC-H20 checkpoint}
PLAIN_PREFIX=${10:?missing CAFC-H20 prefix}
PLAIN_GATE_SESSION=${11:?missing CAFC-H20 gate session}
PLAIN_GPU=${12:?missing CAFC-H20 GPU}
GROUNDED_CHECKPOINT=${13:?missing Bridge+CAFC-H20 checkpoint}
GROUNDED_PREFIX=${14:?missing Bridge+CAFC-H20 prefix}
GROUNDED_GATE_SESSION=${15:?missing Bridge+CAFC-H20 gate session}
GROUNDED_GPU=${16:?missing Bridge+CAFC-H20 GPU}
COMPARISON_NAME=${17:?missing comparison name}

CLOSURE_CHECKPOINT=${H20_CLOSURE_CHECKPOINT:-/share/longjunyu/cabi-vla/runs/cabi_bind_pi05_action_bridge_closure_smoke_seed41_steps33000_edge-balanced-3epoch-v12/final_model}
CLOSURE_GPU=${H20_CLOSURE_GPU:-2}
CLOSURE_STATE0=${H20_CLOSURE_STATE0:-/share/longjunyu/cabi-vla/evaluations/cabi_bridge_closure33000_edge_balanced_seed41_v12_train0_k3/closed_loop_train.json}

for value in \
  "$BC_PREFIX" "$BC_GATE_SESSION" "$BRIDGE_PREFIX" "$BRIDGE_GATE_SESSION" \
  "$PLAIN_PREFIX" "$PLAIN_GATE_SESSION" \
  "$GROUNDED_PREFIX" "$GROUNDED_GATE_SESSION" "$COMPARISON_NAME"; do
  if [[ ! "$value" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "run identifiers contain unsupported characters" >&2
    exit 2
  fi
done
for value in \
  "$POLL_SECONDS" "$BC_GPU" "$BRIDGE_GPU" "$PLAIN_GPU" \
  "$GROUNDED_GPU" "$CLOSURE_GPU"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "poll interval and GPU ids must be non-negative integers" >&2
    exit 2
  fi
done
if (( POLL_SECONDS < 1 )); then
  echo "H20_GATE_POLL_SECONDS must be positive" >&2
  exit 2
fi
if [[ ! -s "$CLOSURE_CHECKPOINT/model.safetensors" || ! -s "$CLOSURE_STATE0" ]]; then
  echo "frozen H10 closure comparator is incomplete" >&2
  exit 1
fi

bc_state0=$EVALUATION_ROOT/${BC_PREFIX}_train0_k3/closed_loop_train.json
bridge_state0=$EVALUATION_ROOT/${BRIDGE_PREFIX}_train0_k3/closed_loop_train.json
plain_state0=$EVALUATION_ROOT/${PLAIN_PREFIX}_train0_k3/closed_loop_train.json
grounded_state0=$EVALUATION_ROOT/${GROUNDED_PREFIX}_train0_k3/closed_loop_train.json

wait_for_gate() {
  local session=$1
  local output=$2
  echo "waiting checkpoint gate: $session"
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
  if [[ ! -s "$output" ]]; then
    echo "checkpoint gate ended without state-0 evaluation: $output" >&2
    exit 1
  fi
}

wait_for_gate "$BC_GATE_SESSION" "$bc_state0"
wait_for_gate "$BRIDGE_GATE_SESSION" "$bridge_state0"
wait_for_gate "$PLAIN_GATE_SESSION" "$plain_state0"
wait_for_gate "$GROUNDED_GATE_SESSION" "$grounded_state0"

for checkpoint in \
  "$BC_CHECKPOINT" "$BRIDGE_CHECKPOINT" "$PLAIN_CHECKPOINT" "$GROUNDED_CHECKPOINT"; do
  if [[ ! -s "$checkpoint/model.safetensors" || ! -s "$checkpoint/framework_config.yaml" ]]; then
    echo "incomplete H20 checkpoint: $checkpoint" >&2
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
    return 0
  fi
  if [[ -e "$video_dir" ]]; then
    echo "partial video output requires audit: $video_dir" >&2
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
    return 0
  fi
  if [[ -e "$output_dir" ]]; then
    echo "partial paired output requires audit: $output_dir" >&2
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

bc_state0_run=${BC_PREFIX}_train0_k3
bridge_state0_run=${BRIDGE_PREFIX}_train0_k3
plain_state0_run=${PLAIN_PREFIX}_train0_k3
grounded_state0_run=${GROUNDED_PREFIX}_train0_k3
mkdir -p "$COMPARISON_ROOT"
render_evaluation "$bc_state0_run" "$bc_state0"
render_evaluation "$bridge_state0_run" "$bridge_state0"
render_evaluation "$plain_state0_run" "$plain_state0"
render_evaluation "$grounded_state0_run" "$grounded_state0"
render_paired train0_plain BC-H20 CAFC-H20 \
  "$bc_state0_run" "$bc_state0" "$plain_state0_run" "$plain_state0"
render_paired train0_grounded Bridge-H20 Bridge+CAFC-H20 \
  "$bridge_state0_run" "$bridge_state0" "$grounded_state0_run" "$grounded_state0"

success_count() {
  local output=$1
  local supervised=$2
  jq --argjson supervised "$supervised" \
    '[.rows[] | select(.action_supervised == $supervised and .success)] | length' \
    "$output"
}

bc_id=$(success_count "$bc_state0" true)
bc_ood=$(success_count "$bc_state0" false)
bridge_id=$(success_count "$bridge_state0" true)
bridge_ood=$(success_count "$bridge_state0" false)
plain_id=$(success_count "$plain_state0" true)
plain_ood=$(success_count "$plain_state0" false)
grounded_id=$(success_count "$grounded_state0" true)
grounded_ood=$(success_count "$grounded_state0" false)
decision_output=$COMPARISON_ROOT/${COMPARISON_NAME}_orchestration.json
if [[ -e "$decision_output" ]]; then
  echo "refusing to overwrite H20 decision: $decision_output" >&2
  exit 1
fi

write_early_decision() {
  local decision=$1
  local reason=$2
  local temporary=${decision_output}.tmp-$$
  jq -n \
    --arg decision "$decision" --arg reason "$reason" \
    --arg bc "$bc_state0" --arg bridge "$bridge_state0" \
    --arg plain "$plain_state0" --arg grounded "$grounded_state0" \
    --argjson bc_id "$bc_id" --argjson bc_ood "$bc_ood" \
    --argjson bridge_id "$bridge_id" --argjson bridge_ood "$bridge_ood" \
    --argjson plain_id "$plain_id" --argjson plain_ood "$plain_ood" \
    --argjson grounded_id "$grounded_id" --argjson grounded_ood "$grounded_ood" \
    '{schema_version:1,decision:$decision,reason:$reason,training_action_horizon:20,decision_horizon:3,full_val_started:false,state0:{bc_h20:{id_success:$bc_id,ood_success:$bc_ood,output:$bc},bridge_h20:{id_success:$bridge_id,ood_success:$bridge_ood,output:$bridge},cafc_h20:{id_success:$plain_id,ood_success:$plain_ood,output:$plain},bridge_cafc_h20:{id_success:$grounded_id,ood_success:$grounded_ood,output:$grounded}}}' \
    >"$temporary"
  mv "$temporary" "$decision_output"
  echo "h20_gate_decision=$decision reason=$reason output=$decision_output"
}

plain_exact_valid=0
grounded_exact_valid=0
if (( plain_id >= 3 && bc_id >= 3 )); then plain_exact_valid=1; fi
if (( grounded_id >= 3 && bridge_id >= 3 )); then grounded_exact_valid=1; fi
plain_eligible=0
grounded_eligible=0
if (( plain_exact_valid == 1 && plain_ood >= 1 )); then plain_eligible=1; fi
if (( grounded_exact_valid == 1 && grounded_ood >= 1 )); then grounded_eligible=1; fi

if (( plain_eligible == 0 && grounded_eligible == 0 )); then
  if (( plain_exact_valid == 0 && grounded_exact_valid == 0 )); then
    write_early_decision BASELINE_INVALID NO_H20_CAFC_ARM_HAS_A_VALID_OBSERVED_EXACT_COMPARISON
  else
    write_early_decision STOP_HORIZON_EXTENSION NO_H20_CAFC_ARM_REACHED_THE_STATE0_MIGRATION_GATE
  fi
  exit 0
fi

run_val() {
  local checkpoint=$1
  local run_name=$2
  local gpu=$3
  local output=$4
  local run_dir=$EVALUATION_ROOT/$run_name
  if [[ -s "$output" ]]; then
    return 0
  fi
  if [[ -e "$run_dir/closed_loop_val.partial.json" ]]; then
    echo "partial full-val requires audit: $run_dir" >&2
    return 1
  fi
  mkdir -p "$run_dir"
  CABI_EVAL_SPLIT=val \
  CABI_EVAL_HORIZONS=3 \
  CABI_EVAL_FRAME_EPISODES=5 \
    "$REPO_ROOT/scripts/cabi_vla/run_libero_bind_eval.sh" \
      "$checkpoint" "$run_name" "$gpu" >"$run_dir/evaluator.log" 2>&1
}

bc_val_run=bc_h20_s41_v15_val_k3
bridge_val_run=bridge_h20_s41_v15_val_k3
closure_val_run=closure_h10_h20gate_s41_v15_val_k3
plain_val_run=cafc_h20_s41_v15_val_k3
grounded_val_run=bridge_cafc_h20_s41_v15_val_k3
bc_val=$EVALUATION_ROOT/$bc_val_run/closed_loop_val.json
bridge_val=$EVALUATION_ROOT/$bridge_val_run/closed_loop_val.json
closure_val=$EVALUATION_ROOT/$closure_val_run/closed_loop_val.json
plain_val=$EVALUATION_ROOT/$plain_val_run/closed_loop_val.json
grounded_val=$EVALUATION_ROOT/$grounded_val_run/closed_loop_val.json

echo "stage=five_way_full_val status=run"
run_val "$BC_CHECKPOINT" "$bc_val_run" "$BC_GPU" "$bc_val" & bc_pid=$!
run_val "$BRIDGE_CHECKPOINT" "$bridge_val_run" "$BRIDGE_GPU" "$bridge_val" & bridge_pid=$!
run_val "$CLOSURE_CHECKPOINT" "$closure_val_run" "$CLOSURE_GPU" "$closure_val" & closure_pid=$!
run_val "$PLAIN_CHECKPOINT" "$plain_val_run" "$PLAIN_GPU" "$plain_val" & plain_pid=$!
run_val "$GROUNDED_CHECKPOINT" "$grounded_val_run" "$GROUNDED_GPU" "$grounded_val" & grounded_pid=$!

failed=0
for item in \
  "$bc_pid:bc" "$bridge_pid:bridge" "$closure_pid:closure" \
  "$plain_pid:plain" "$grounded_pid:grounded"; do
  pid=${item%%:*}
  status=0
  wait "$pid" || status=$?
  if (( status != 0 )); then
    echo "full-val arm failed: ${item#*:} status=$status" >&2
    failed=1
  fi
done
if (( failed != 0 )); then exit 1; fi

for pair in \
  "$bc_val_run $bc_val" "$bridge_val_run $bridge_val" \
  "$closure_val_run $closure_val" "$plain_val_run $plain_val" \
  "$grounded_val_run $grounded_val"; do
  read -r run_name output <<<"$pair"
  render_evaluation "$run_name" "$output"
done
render_paired val_plain BC-H20 CAFC-H20 \
  "$bc_val_run" "$bc_val" "$plain_val_run" "$plain_val"
render_paired val_grounded Bridge-H20 Bridge+CAFC-H20 \
  "$bridge_val_run" "$bridge_val" "$grounded_val_run" "$grounded_val"

compare() {
  local baseline=$1
  local method=$2
  local output=$3
  if [[ ! -s "$output" ]]; then
    "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/cabi_vla/compare_libero_bind_policies.py" \
      --baseline "$baseline" --method "$method" --output "$output" --decision-horizon 3
  fi
}

plain_exact=$COMPARISON_ROOT/${COMPARISON_NAME}_plain_vs_bc_h20.json
grounded_exact=$COMPARISON_ROOT/${COMPARISON_NAME}_grounded_vs_bridge_h20.json
plain_strong=$COMPARISON_ROOT/${COMPARISON_NAME}_plain_vs_closure_h10.json
grounded_strong=$COMPARISON_ROOT/${COMPARISON_NAME}_grounded_vs_closure_h10.json
compare "$bc_val" "$plain_val" "$plain_exact"
compare "$bridge_val" "$grounded_val" "$grounded_exact"
compare "$closure_val" "$plain_val" "$plain_strong"
compare "$closure_val" "$grounded_val" "$grounded_strong"

"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/cabi_vla/decide_h20_cafc_gate.py" \
  --bc-state0 "$bc_state0" \
  --bridge-state0 "$bridge_state0" \
  --plain-state0 "$plain_state0" \
  --grounded-state0 "$grounded_state0" \
  --plain-exact "$plain_exact" \
  --grounded-exact "$grounded_exact" \
  --plain-strong "$plain_strong" \
  --grounded-strong "$grounded_strong" \
  --output "$decision_output"
echo "h20_gate_complete decision=$(jq -r .decision "$decision_output") output=$decision_output"

