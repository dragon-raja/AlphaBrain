#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${KYC_SCALING_DATA_ROOT:-/share/longjunyu/cabi-vla/kyc-scaling-v3}
EVAL_ROOT=${KYC_FACTORIAL_EVAL_ROOT:-$DATA_ROOT/eval/factorial}
CATALOG_SIZE=${1:?usage: launch_kyc_factorial_eval.sh CATALOG_SIZE [SEED]}
SEED=${2:-41}
EXPECTED_EPISODES=520

if [[ ! "$CATALOG_SIZE" =~ ^[1-9][0-9]*$ || ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "CATALOG_SIZE must be positive and SEED must be non-negative" >&2
  exit 2
fi

evaluation_path() {
  local train_scene=$1
  local wrist=$2
  local arm=$3
  local eval_scene=$4
  local train_scene_tag
  local eval_scene_tag
  local arm_tag
  case "$train_scene" in
    fixed) train_scene_tag=fx ;;
    cue_randomized) train_scene_tag=cue ;;
  esac
  case "$eval_scene" in
    fixed) eval_scene_tag=fx ;;
    cue_randomized) eval_scene_tag=cue ;;
  esac
  case "$arm" in
    poseaug_control) arm_tag=ctrl ;;
    kyc) arm_tag=kyc ;;
  esac
  local name="n${CATALOG_SIZE}-tr${train_scene_tag}-w${wrist}-m${arm_tag}-s${SEED}-ev${eval_scene_tag}"
  printf '%s/n%s/%s/camera_sweep_test.json' \
    "$EVAL_ROOT" "$CATALOG_SIZE" "$name"
}

start_job() {
  local train_scene=$1
  local wrist=$2
  local arm=$3
  local eval_scene=$4
  local gpu_id=$5
  local output
  local session="kyc-factorial-eval-n${CATALOG_SIZE}-${train_scene}-w${wrist}-${arm}-s${SEED}-${eval_scene}"
  output=$(evaluation_path "$train_scene" "$wrist" "$arm" "$eval_scene")
  if [[ -s "$output" ]]; then
    echo "already complete: $session"
    return
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already running: $session"
    return
  fi
  if [[ -e "${output%/camera_sweep_test.json}" ]]; then
    echo "incomplete factorial evaluation requires inspection: ${output%/camera_sweep_test.json}" >&2
    exit 1
  fi
  frame_env=()
  if [[ "$train_scene" == "$eval_scene" && "$wrist" == off ]]; then
    frame_env=(
      CABI_EVAL_FRAME_EPISODES=1
      CABI_CAMERA_FRAME_EDGES=red-left,yellow_white-right
      CABI_CAMERA_FRAME_POSES=baseline,az_m60,el_m25,rad_0900
    )
  fi
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "exec env ${frame_env[*]} bash scripts/cabi_vla/run_kyc_factorial_eval.sh \
      '$CATALOG_SIZE' '$train_scene' '$wrist' '$arm' '$SEED' '$eval_scene' '$gpu_id'"
  echo "started: $session gpu=$gpu_id"
}

wait_job() {
  local train_scene=$1
  local wrist=$2
  local arm=$3
  local eval_scene=$4
  local output
  local session="kyc-factorial-eval-n${CATALOG_SIZE}-${train_scene}-w${wrist}-${arm}-s${SEED}-${eval_scene}"
  output=$(evaluation_path "$train_scene" "$wrist" "$arm" "$eval_scene")
  while true; do
    if [[ -s "$output" ]] && jq -e \
      --argjson expected "$EXPECTED_EPISODES" \
      '.status == "complete" and (.rows | length) == $expected' \
      "$output" >/dev/null; then
      return
    fi
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "factorial evaluation stopped without result: $session" >&2
      exit 1
    fi
    sleep 30
  done
}

matched=(
  "fixed on poseaug_control fixed 0"
  "fixed on kyc fixed 1"
  "fixed off poseaug_control fixed 2"
  "fixed off kyc fixed 3"
  "cue_randomized on poseaug_control cue_randomized 4"
  "cue_randomized on kyc cue_randomized 5"
  "cue_randomized off poseaug_control cue_randomized 6"
  "cue_randomized off kyc cue_randomized 7"
)
cross_scene=(
  "fixed on poseaug_control cue_randomized 0"
  "fixed on kyc cue_randomized 1"
  "fixed off poseaug_control cue_randomized 2"
  "fixed off kyc cue_randomized 3"
  "cue_randomized on poseaug_control fixed 4"
  "cue_randomized on kyc fixed 5"
  "cue_randomized off poseaug_control fixed 6"
  "cue_randomized off kyc fixed 7"
)

run_wave() {
  local wave_name=$1
  local -n wave=$wave_name
  for specification in "${wave[@]}"; do
    read -r train_scene wrist arm eval_scene gpu_id <<<"$specification"
    start_job "$train_scene" "$wrist" "$arm" "$eval_scene" "$gpu_id"
  done
  for specification in "${wave[@]}"; do
    read -r train_scene wrist arm eval_scene _gpu_id <<<"$specification"
    wait_job "$train_scene" "$wrist" "$arm" "$eval_scene"
  done
}

run_wave matched
run_wave cross_scene

echo "KYC scene-cue by wrist factorial evaluation complete"
