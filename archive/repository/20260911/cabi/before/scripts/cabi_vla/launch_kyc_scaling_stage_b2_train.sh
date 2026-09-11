#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${KYC_SCALING_DATA_ROOT:-/share/longjunyu/cabi-vla/kyc-scaling-v3}
OUTPUT_ROOT=${KYC_OUTPUT_ROOT:-$DATA_ROOT/runs}
SELECTION=${1:?usage: launch_kyc_scaling_stage_b2_train.sh SELECTION_JSON}
STEPS=${KYC_SCALING_STEPS:-33000}

if [[ ! -s "$SELECTION" ]]; then
  echo "missing Stage B2 selection: $SELECTION" >&2
  exit 1
fi

run_id() {
  local catalog_size=$1
  local arm=$2
  local seed=$3
  printf 'kyc_%s_scale-n%s-fixed-wrist-on_h20_seed%s_steps%s' \
    "$arm" "$catalog_size" "$seed" "$STEPS"
}

start_job() {
  local catalog_size=$1
  local arm=$2
  local seed=$3
  local gpu_id=$4
  local id
  local session="kyc-b2-n${catalog_size}-${arm}-s${seed}"
  local data_view="$DATA_ROOT/views/libero-bind-kyc-n${catalog_size}-fixed-h20"
  id=$(run_id "$catalog_size" "$arm" "$seed")
  final_model="$OUTPUT_ROOT/$id/final_model/model.safetensors"
  if [[ ! -s "$data_view/manifest.json" ]]; then
    echo "missing scaling data view: $data_view" >&2
    exit 1
  fi
  if [[ -s "$final_model" ]]; then
    echo "already complete: $id"
    return
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already running: $session"
    return
  fi
  if [[ -e "$OUTPUT_ROOT/$id" ]]; then
    echo "incomplete Stage B2 run requires inspection: $OUTPUT_ROOT/$id" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "export KYC_OUTPUT_ROOT='$OUTPUT_ROOT'; \
     export KYC_DATA_ROOT_OVERRIDE='$data_view'; \
     export KYC_RUN_TAG='scale-n${catalog_size}-fixed-wrist-on'; \
     export KYC_WRIST_MODE=on; \
     export KYC_CABI_ANCHOR_PERIOD=1000000; \
     exec bash scripts/cabi_vla/run_kyc_train.sh \
       '$arm' '$seed' '$gpu_id' '$STEPS'"
  echo "started: $session gpu=$gpu_id"
}

wait_job() {
  local catalog_size=$1
  local arm=$2
  local seed=$3
  local id
  local session="kyc-b2-n${catalog_size}-${arm}-s${seed}"
  id=$(run_id "$catalog_size" "$arm" "$seed")
  final_model="$OUTPUT_ROOT/$id/final_model/model.safetensors"
  while [[ ! -s "$final_model" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "Stage B2 training stopped without checkpoint: $session" >&2
      exit 1
    fi
    sleep 60
  done
}

mapfile -t budgets < <(jq -r '.training_budgets[]' "$SELECTION")
mapfile -t seeds < <(jq -r '.confirmation_seeds[]' "$SELECTION")
jobs=()
for catalog_size in "${budgets[@]}"; do
  for seed in "${seeds[@]}"; do
    for arm in poseaug_control kyc; do
      jobs+=("$catalog_size $arm $seed")
    done
  done
done

for ((wave_start = 0; wave_start < ${#jobs[@]}; wave_start += 8)); do
  wave=("${jobs[@]:wave_start:8}")
  for index in "${!wave[@]}"; do
    read -r catalog_size arm seed <<<"${wave[$index]}"
    start_job "$catalog_size" "$arm" "$seed" "$index"
  done
  for specification in "${wave[@]}"; do
    read -r catalog_size arm seed <<<"$specification"
    wait_job "$catalog_size" "$arm" "$seed"
  done
done

echo "KYC scaling Stage B2 training complete"
