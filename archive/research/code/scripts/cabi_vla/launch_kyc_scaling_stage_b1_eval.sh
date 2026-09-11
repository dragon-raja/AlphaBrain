#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${KYC_SCALING_DATA_ROOT:-/share/longjunyu/cabi-vla/kyc-scaling-v3}
RUN_ROOT=${KYC_OUTPUT_ROOT:-$DATA_ROOT/runs}
EVAL_ROOT=${KYC_SCALING_EVAL_ROOT:-$DATA_ROOT/eval/stage-b1}
FOV_ROOT=${KYC_BOUNDARY_FOV_ROOT:-/share/longjunyu/cabi-vla/camera-viewpoint-study-v2/fov_guard_test40-49_v5}
PYTHON=${KYC_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
SEED=${KYC_SCALING_SEED:-41}
STEPS=${KYC_SCALING_STEPS:-33000}
EXPECTED_EPISODES=520

checkpoint_path() {
  local catalog_size=$1
  local arm=$2
  local tag="scale-n${catalog_size}-fixed-wrist-on"
  printf '%s/kyc_%s_%s_h20_seed%s_steps%s/final_model/model.safetensors' \
    "$RUN_ROOT" "$arm" "$tag" "$SEED" "$STEPS"
}

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
  local checkpoint
  local output
  local session="kyc-b1-eval-n${catalog_size}-${arm}-s${SEED}"
  checkpoint=$(checkpoint_path "$catalog_size" "$arm")
  output=$(evaluation_path "$catalog_size" "$arm")

  if [[ ! -s "$checkpoint" ]]; then
    echo "missing completed checkpoint: $checkpoint" >&2
    exit 1
  fi
  if [[ -s "$output" ]]; then
    echo "already evaluated: n=$catalog_size arm=$arm"
    return
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already running: $session"
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
  echo "started: $session gpu=$gpu_id"
}

wait_for_evaluation() {
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
      echo "evaluation stopped without a complete result: $session" >&2
      exit 1
    fi
    sleep 30
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
wave_two=(
  "1000 poseaug_control 0"
  "1000 kyc 1"
)

for specification in "${wave_one[@]}"; do
  read -r catalog_size arm gpu_id <<<"$specification"
  start_evaluation "$catalog_size" "$arm" "$gpu_id"
done
for specification in "${wave_one[@]}"; do
  read -r catalog_size arm _gpu_id <<<"$specification"
  wait_for_evaluation "$catalog_size" "$arm"
done
for specification in "${wave_two[@]}"; do
  read -r catalog_size arm gpu_id <<<"$specification"
  start_evaluation "$catalog_size" "$arm" "$gpu_id"
done
for specification in "${wave_two[@]}"; do
  read -r catalog_size arm _gpu_id <<<"$specification"
  wait_for_evaluation "$catalog_size" "$arm"
done

fov_json=("$FOV_ROOT"/*.json)
for catalog_size in 10 45 215 1000; do
  analysis_dir="$EVAL_ROOT/analysis/n${catalog_size}"
  if [[ -e "$analysis_dir" ]]; then
    continue
  fi
  comparison_args=(
    --evaluation "poseaug_control=$(evaluation_path "$catalog_size" poseaug_control)"
    --evaluation "kyc=$(evaluation_path "$catalog_size" kyc)"
  )
  if [[ "$catalog_size" == 10 || "$catalog_size" == 45 ]]; then
    comparison_args+=(
      --evaluation "poseaug_rgb=$(evaluation_path "$catalog_size" poseaug_rgb)"
    )
  fi
  "$PYTHON" scripts/cabi_vla/compare_kyc_camera_evaluations.py \
    "${comparison_args[@]}" \
    --fov-json "${fov_json[@]}" \
    --output-dir "$analysis_dir" \
    --reference poseaug_control
done

echo "KYC scaling Stage B1 evaluation complete"
