#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${KYC_SCALING_DATA_ROOT:-/share/longjunyu/cabi-vla/kyc-scaling-v3}
EVAL_ROOT=${KYC_SCALING_EVAL_ROOT:-$DATA_ROOT/eval/stage-b1}
OFFICIAL_ROOT=${KYC_OFFICIAL_OUTPUT_ROOT:-/share/longjunyu/kyc-official-data/runs}
SEED=${KYC_SCALING_SEED:-41}
EXPECTED_EPISODES=520

official_complete() {
  for arm in image kyc; do
    for seed in 0 1 2; do
      if [[ ! -s "$OFFICIAL_ROOT/official_act_lift_randomized_${arm}_seed${seed}/epoch_20000.pth" ]]; then
        return 1
      fi
    done
  done
}

while ! official_complete; do
  for arm in image kyc; do
    for seed in 0 1 2; do
      checkpoint="$OFFICIAL_ROOT/official_act_lift_randomized_${arm}_seed${seed}/epoch_20000.pth"
      session="kyc-official-${arm}-s${seed}"
      if [[ ! -s "$checkpoint" ]] && ! tmux has-session -t "$session" 2>/dev/null; then
        echo "official KYC task stopped before completion: $session" >&2
        exit 1
      fi
    done
  done
  sleep 60
done

evaluation_path() {
  local catalog_size=$1
  local arm=$2
  printf '%s/n%s/n%s-%s-s%s-fixed-wrist-on/camera_sweep_test.json' \
    "$EVAL_ROOT" "$catalog_size" "$catalog_size" "$arm" "$SEED"
}

start_evaluation() {
  local catalog_size=$1
  local arm=$2
  local gpu_id=$3
  local output
  local session="kyc-b1-eval-n${catalog_size}-${arm}-s${SEED}"
  output=$(evaluation_path "$catalog_size" "$arm")
  if [[ -s "$output" ]] || tmux has-session -t "$session" 2>/dev/null; then
    return
  fi
  if [[ -e "${output%/camera_sweep_test.json}" ]]; then
    echo "incomplete evaluation requires inspection: ${output%/camera_sweep_test.json}" >&2
    exit 1
  fi
  frame_env=()
  if [[ "$catalog_size" == 10 ]] \
    && [[ "$arm" == poseaug_control || "$arm" == kyc ]]; then
    frame_env=(
      CABI_EVAL_FRAME_EPISODES=1
      CABI_CAMERA_FRAME_EDGES=red-left,yellow_white-right
      CABI_CAMERA_FRAME_POSES=baseline,az_m60,el_m25,rad_0900
    )
  fi
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "exec env ${frame_env[*]} bash scripts/cabi_vla/run_kyc_scaling_eval.sh \
      '$catalog_size' '$arm' '$SEED' '$gpu_id'"
}

wait_evaluation() {
  local catalog_size=$1
  local arm=$2
  local output
  local session="kyc-b1-eval-n${catalog_size}-${arm}-s${SEED}"
  output=$(evaluation_path "$catalog_size" "$arm")
  while true; do
    if [[ -s "$output" ]] && jq -e \
      --argjson expected "$EXPECTED_EPISODES" \
      '.status == "complete" and (.rows | length) == $expected' \
      "$output" >/dev/null; then
      return
    fi
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "prefetched evaluation stopped without a complete result: $session" >&2
      exit 1
    fi
    sleep 30
  done
}

primary_wave=(
  "10 poseaug_control 2"
  "10 kyc 3"
  "45 poseaug_control 4"
  "45 kyc 5"
  "215 poseaug_control 6"
  "215 kyc 7"
)
rgb_wave=(
  "10 poseaug_rgb 2"
  "45 poseaug_rgb 3"
)

for wave_name in primary_wave rgb_wave; do
  declare -n wave="$wave_name"
  for specification in "${wave[@]}"; do
    read -r catalog_size arm gpu_id <<<"$specification"
    start_evaluation "$catalog_size" "$arm" "$gpu_id"
  done
  for specification in "${wave[@]}"; do
    read -r catalog_size arm _gpu_id <<<"$specification"
    wait_evaluation "$catalog_size" "$arm"
  done
  unset -n wave
done

echo "KYC Stage B1 prefetched evaluations complete"
