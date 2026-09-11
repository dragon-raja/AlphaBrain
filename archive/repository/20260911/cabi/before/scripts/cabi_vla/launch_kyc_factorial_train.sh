#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${KYC_SCALING_DATA_ROOT:-/share/longjunyu/cabi-vla/kyc-scaling-v3}
OUTPUT_ROOT=${KYC_OUTPUT_ROOT:-$DATA_ROOT/runs}
CATALOG_SIZE=${1:?usage: launch_kyc_factorial_train.sh CATALOG_SIZE [SEED]}
SEED=${2:-41}
STEPS=${KYC_SCALING_STEPS:-33000}

if [[ ! "$CATALOG_SIZE" =~ ^[1-9][0-9]*$ || ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "CATALOG_SIZE must be positive and SEED must be non-negative" >&2
  exit 2
fi

run_id() {
  local scene=$1
  local wrist=$2
  local arm=$3
  printf 'kyc_%s_factorial-n%s-%s-wrist-%s_h20_seed%s_steps%s' \
    "$arm" "$CATALOG_SIZE" "$scene" "$wrist" "$SEED" "$STEPS"
}

start_job() {
  local scene=$1
  local wrist=$2
  local arm=$3
  local gpu_id=$4
  local id
  local session="kyc-factorial-n${CATALOG_SIZE}-${scene}-w${wrist}-${arm}-s${SEED}"
  id=$(run_id "$scene" "$wrist" "$arm")
  final_model="$OUTPUT_ROOT/$id/final_model/model.safetensors"
  if [[ -s "$final_model" ]]; then
    echo "already complete: $id"
    return
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already running: $session"
    return
  fi
  if [[ -e "$OUTPUT_ROOT/$id" ]]; then
    echo "incomplete factorial run requires inspection: $OUTPUT_ROOT/$id" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "exec bash scripts/cabi_vla/run_kyc_factorial_train.sh \
      '$CATALOG_SIZE' '$scene' '$wrist' '$arm' '$SEED' '$gpu_id' '$STEPS'"
  echo "started: $session gpu=$gpu_id"
}

wait_job() {
  local scene=$1
  local wrist=$2
  local arm=$3
  local id
  local session="kyc-factorial-n${CATALOG_SIZE}-${scene}-w${wrist}-${arm}-s${SEED}"
  id=$(run_id "$scene" "$wrist" "$arm")
  final_model="$OUTPUT_ROOT/$id/final_model/model.safetensors"
  while [[ ! -s "$final_model" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "factorial training stopped without checkpoint: $session" >&2
      exit 1
    fi
    sleep 60
  done
}

jobs=(
  "fixed off poseaug_control 0"
  "fixed off kyc 1"
  "cue_randomized on poseaug_control 2"
  "cue_randomized on kyc 3"
  "cue_randomized off poseaug_control 4"
  "cue_randomized off kyc 5"
)

for specification in "${jobs[@]}"; do
  read -r scene wrist arm gpu_id <<<"$specification"
  start_job "$scene" "$wrist" "$arm" "$gpu_id"
done
for specification in "${jobs[@]}"; do
  read -r scene wrist arm _gpu_id <<<"$specification"
  wait_job "$scene" "$wrist" "$arm"
done

echo "KYC scene-cue by wrist factorial training complete"
