#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${KYC_SCALING_DATA_ROOT:-/share/longjunyu/cabi-vla/kyc-scaling-v3}
EVAL_ROOT=${KYC_FACTORIAL_EVAL_ROOT:-$DATA_ROOT/eval/factorial}
SUMMARY=${1:?usage: launch_kyc_factorial_confirmation_eval.sh SEED41_SUMMARY}
EXPECTED_EPISODES=520

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

scene_tag() {
  case "$1" in
    fixed) printf fx ;;
    cue_randomized) printf cue ;;
  esac
}

arm_tag() {
  case "$1" in
    poseaug_control) printf ctrl ;;
    kyc) printf kyc ;;
  esac
}

evaluation_path() {
  local scene=$1
  local wrist=$2
  local arm=$3
  local seed=$4
  local name="n${budget}-tr$(scene_tag "$scene")-w${wrist}-m$(arm_tag "$arm")-s${seed}-ev$(scene_tag "$scene")"
  printf '%s/n%s/%s/camera_sweep_test.json' "$EVAL_ROOT" "$budget" "$name"
}

start_job() {
  local scene=$1
  local wrist=$2
  local arm=$3
  local seed=$4
  local gpu_id=$5
  local output
  local session="kyc-factorial-confirm-eval-n${budget}-${scene}-w${wrist}-${arm}-s${seed}"
  output=$(evaluation_path "$scene" "$wrist" "$arm" "$seed")
  if [[ -s "$output" ]]; then
    echo "already evaluated: $session"
    return
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already running: $session"
    return
  fi
  if [[ -e "${output%/camera_sweep_test.json}" ]]; then
    echo "incomplete factorial confirmation evaluation: ${output%/camera_sweep_test.json}" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "exec bash scripts/cabi_vla/run_kyc_factorial_eval.sh \
      '$budget' '$scene' '$wrist' '$arm' '$seed' '$scene' '$gpu_id'"
  echo "started: $session gpu=$gpu_id"
}

wait_job() {
  local scene=$1
  local wrist=$2
  local arm=$3
  local seed=$4
  local output
  local session="kyc-factorial-confirm-eval-n${budget}-${scene}-w${wrist}-${arm}-s${seed}"
  output=$(evaluation_path "$scene" "$wrist" "$arm" "$seed")
  while true; do
    if [[ -s "$output" ]] && jq -e \
      --argjson expected "$EXPECTED_EPISODES" \
      '.status == "complete" and (.rows | length) == $expected' \
      "$output" >/dev/null; then
      return
    fi
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "factorial confirmation evaluation stopped: $session" >&2
      exit 1
    fi
    sleep 30
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

echo "KYC factorial confirmation evaluation complete: scope=$scope"
