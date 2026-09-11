#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${KYC_SCALING_DATA_ROOT:-/share/longjunyu/cabi-vla/kyc-scaling-v3}
OUTPUT_ROOT=${KYC_OUTPUT_ROOT:-$DATA_ROOT/runs}
SEED=${KYC_SCALING_SEED:-41}
STEPS=${KYC_SCALING_STEPS:-33000}

start_job() {
  local catalog_size=$1
  local arm=$2
  local gpu_id=$3
  local cell="n${catalog_size}-fixed"
  local tag="scale-${cell}-wrist-on"
  local data_view="$DATA_ROOT/views/libero-bind-kyc-${cell}-h20"
  local run_id="kyc_${arm}_${tag}_h20_seed${SEED}_steps${STEPS}"
  local final_model="$OUTPUT_ROOT/$run_id/final_model/model.safetensors"
  local session="kyc-b1-n${catalog_size}-${arm}-s${SEED}"

  if [[ ! -s "$data_view/manifest.json" ]]; then
    echo "missing completed scaling data view: $data_view" >&2
    exit 1
  fi
  if [[ -s "$final_model" ]]; then
    echo "already complete: $run_id"
    return
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already running: $session"
    return
  fi
  if [[ -e "$OUTPUT_ROOT/$run_id" ]]; then
    echo "incomplete run requires inspection: $OUTPUT_ROOT/$run_id" >&2
    exit 1
  fi

  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "export KYC_OUTPUT_ROOT='$OUTPUT_ROOT'; \
     export KYC_DATA_ROOT_OVERRIDE='$data_view'; \
     export KYC_RUN_TAG='$tag'; \
     export KYC_WRIST_MODE=on; \
     export KYC_CABI_ANCHOR_PERIOD=1000000; \
     exec bash scripts/cabi_vla/run_kyc_train.sh '$arm' '$SEED' '$gpu_id' '$STEPS'"
  echo "started: $session gpu=$gpu_id"
}

wait_job() {
  local catalog_size=$1
  local arm=$2
  local cell="n${catalog_size}-fixed"
  local tag="scale-${cell}-wrist-on"
  local run_id="kyc_${arm}_${tag}_h20_seed${SEED}_steps${STEPS}"
  local final_model="$OUTPUT_ROOT/$run_id/final_model/model.safetensors"
  local session="kyc-b1-n${catalog_size}-${arm}-s${SEED}"
  while [[ ! -s "$final_model" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "training stopped without a final checkpoint: $session" >&2
      exit 1
    fi
    sleep 60
  done
}

wave_one=(
  "10 poseaug_control 0"
  "10 kyc 1"
  "10 poseaug_rgb 2"
  "45 poseaug_control 3"
  "45 kyc 4"
  "45 poseaug_rgb 5"
  "215 poseaug_control 6"
  "215 kyc 7"
)

for specification in "${wave_one[@]}"; do
  read -r catalog_size arm gpu_id <<<"$specification"
  start_job "$catalog_size" "$arm" "$gpu_id"
done
for specification in "${wave_one[@]}"; do
  read -r catalog_size arm _gpu_id <<<"$specification"
  wait_job "$catalog_size" "$arm"
done

wave_two=(
  "1000 poseaug_control 0"
  "1000 kyc 1"
)
for specification in "${wave_two[@]}"; do
  read -r catalog_size arm gpu_id <<<"$specification"
  start_job "$catalog_size" "$arm" "$gpu_id"
done
for specification in "${wave_two[@]}"; do
  read -r catalog_size arm _gpu_id <<<"$specification"
  wait_job "$catalog_size" "$arm"
done

echo "KYC scaling Stage B1 complete"

