#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${KYC_SCALING_DATA_ROOT:-/share/longjunyu/cabi-vla/kyc-scaling-v3}
EVAL_ROOT=${KYC_SCALING_EVAL_ROOT:-$DATA_ROOT/eval/stage-b1}
FOV_ROOT=${KYC_BOUNDARY_FOV_ROOT:-/share/longjunyu/cabi-vla/camera-viewpoint-study-v2/fov_guard_test40-49_v5}
PYTHON=${KYC_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
SELECTION=${1:?usage: launch_kyc_scaling_stage_b2_eval.sh SELECTION_JSON}
EXPECTED_EPISODES=520

if [[ ! -s "$SELECTION" ]]; then
  echo "missing Stage B2 selection: $SELECTION" >&2
  exit 1
fi

evaluation_path() {
  local catalog_size=$1
  local arm=$2
  local seed=$3
  printf '%s/n%s/n%s-%s-s%s-fixed-wrist-on/camera_sweep_test.json' \
    "$EVAL_ROOT" "$catalog_size" "$catalog_size" "$arm" "$seed"
}

start_job() {
  local catalog_size=$1
  local arm=$2
  local seed=$3
  local gpu_id=$4
  local output
  local session="kyc-b2-eval-n${catalog_size}-${arm}-s${seed}"
  output=$(evaluation_path "$catalog_size" "$arm" "$seed")
  if [[ -s "$output" ]]; then
    echo "already evaluated: n=$catalog_size arm=$arm seed=$seed"
    return
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already running: $session"
    return
  fi
  if [[ -e "${output%/camera_sweep_test.json}" ]]; then
    echo "incomplete Stage B2 evaluation requires inspection: ${output%/camera_sweep_test.json}" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "exec bash scripts/cabi_vla/run_kyc_scaling_eval.sh \
      '$catalog_size' '$arm' '$seed' '$gpu_id'"
  echo "started: $session gpu=$gpu_id"
}

wait_job() {
  local catalog_size=$1
  local arm=$2
  local seed=$3
  local output
  local session="kyc-b2-eval-n${catalog_size}-${arm}-s${seed}"
  output=$(evaluation_path "$catalog_size" "$arm" "$seed")
  while true; do
    if [[ -s "$output" ]] && jq -e \
      --argjson expected "$EXPECTED_EPISODES" \
      '.status == "complete" and (.rows | length) == $expected' \
      "$output" >/dev/null; then
      return
    fi
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "Stage B2 evaluation stopped without result: $session" >&2
      exit 1
    fi
    sleep 30
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

fov_json=("$FOV_ROOT"/*.json)
for catalog_size in "${budgets[@]}"; do
  for seed in "${seeds[@]}"; do
    analysis_dir="$EVAL_ROOT/analysis/n${catalog_size}/seed${seed}"
    if [[ -e "$analysis_dir" ]]; then
      continue
    fi
    "$PYTHON" scripts/cabi_vla/compare_kyc_camera_evaluations.py \
      --evaluation "poseaug_control=$(evaluation_path "$catalog_size" poseaug_control "$seed")" \
      --evaluation "kyc=$(evaluation_path "$catalog_size" kyc "$seed")" \
      --fov-json "${fov_json[@]}" \
      --output-dir "$analysis_dir" \
      --reference poseaug_control
  done
done

echo "KYC scaling Stage B2 evaluation complete"
