#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${KYC_SCALING_DATA_ROOT:-/share/longjunyu/cabi-vla/kyc-scaling-v3}
OUTPUT_ROOT=${KYC_OUTPUT_ROOT:-$DATA_ROOT/runs}
SUMMARY=${1:?usage: launch_kyc_factorial_confirmation_train.sh SEED41_SUMMARY}
STEPS=${KYC_SCALING_STEPS:-33000}

if [[ ! -s "$SUMMARY" ]]; then
  echo "missing seed-41 factorial summary: $SUMMARY" >&2
  exit 1
fi
budget=$(jq -r '.budget' "$SUMMARY")
scope=$(jq -r '.confirmation_rule.scope' "$SUMMARY")
mapfile -t seeds < <(jq -r '.confirmation_rule.seeds[]' "$SUMMARY")

case "$scope" in
  complete_factorial)
    cells=(
      "fixed off"
      "cue_randomized on"
      "cue_randomized off"
    )
    ;;
  wrist_off_cells)
    cells=(
      "fixed off"
      "cue_randomized off"
    )
    ;;
  *)
    echo "unknown factorial confirmation scope: $scope" >&2
    exit 2
    ;;
esac

run_id() {
  local scene=$1
  local wrist=$2
  local arm=$3
  local seed=$4
  printf 'kyc_%s_factorial-n%s-%s-wrist-%s_h20_seed%s_steps%s' \
    "$arm" "$budget" "$scene" "$wrist" "$seed" "$STEPS"
}

start_job() {
  local scene=$1
  local wrist=$2
  local arm=$3
  local seed=$4
  local gpu_id=$5
  local id
  local session="kyc-factorial-confirm-n${budget}-${scene}-w${wrist}-${arm}-s${seed}"
  id=$(run_id "$scene" "$wrist" "$arm" "$seed")
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
    echo "incomplete factorial confirmation requires inspection: $OUTPUT_ROOT/$id" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "exec bash scripts/cabi_vla/run_kyc_factorial_train.sh \
      '$budget' '$scene' '$wrist' '$arm' '$seed' '$gpu_id' '$STEPS'"
  echo "started: $session gpu=$gpu_id"
}

wait_job() {
  local scene=$1
  local wrist=$2
  local arm=$3
  local seed=$4
  local id
  local session="kyc-factorial-confirm-n${budget}-${scene}-w${wrist}-${arm}-s${seed}"
  id=$(run_id "$scene" "$wrist" "$arm" "$seed")
  final_model="$OUTPUT_ROOT/$id/final_model/model.safetensors"
  while [[ ! -s "$final_model" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "factorial confirmation stopped without checkpoint: $session" >&2
      exit 1
    fi
    sleep 60
  done
}

jobs=()
for seed in "${seeds[@]}"; do
  for cell in "${cells[@]}"; do
    read -r scene wrist <<<"$cell"
    for arm in poseaug_control kyc; do
      jobs+=("$scene $wrist $arm $seed")
    done
  done
done

for ((wave_start = 0; wave_start < ${#jobs[@]}; wave_start += 8)); do
  wave=("${jobs[@]:wave_start:8}")
  for index in "${!wave[@]}"; do
    read -r scene wrist arm seed <<<"${wave[$index]}"
    start_job "$scene" "$wrist" "$arm" "$seed" "$index"
  done
  for specification in "${wave[@]}"; do
    read -r scene wrist arm seed <<<"$specification"
    wait_job "$scene" "$wrist" "$arm" "$seed"
  done
done

echo "KYC factorial confirmation training complete: scope=$scope"
