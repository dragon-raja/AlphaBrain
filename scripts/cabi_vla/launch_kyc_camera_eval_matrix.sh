#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN_ROOT=${KYC_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/kyc-runs}
EVAL_ROOT=${KYC_CAMERA_EVAL_ROOT:-/share/longjunyu/cabi-vla/kyc-camera-eval-v1}
BOUNDARY_FOV_ROOT=${KYC_BOUNDARY_FOV_ROOT:-/share/longjunyu/cabi-vla/camera-viewpoint-study-v2/fov_guard_test40-49_v5}
DENSE_FOV_ROOT=${KYC_DENSE_FOV_ROOT:-/share/longjunyu/cabi-vla/camera-viewpoint-study-v2/fov_guard_policy_dense_state40_v6}
PYTHON=${KYC_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}

wait_for_file() {
  local path=$1
  while [[ ! -s "$path" ]]; do
    sleep 30
  done
}

wait_for_session_exit() {
  local session=$1
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 15
  done
}

evaluation_path() {
  local scope=$1
  local method=$2
  local seed=$3
  printf '%s/%s/%s_s%s_%s/camera_sweep_test.json' \
    "$EVAL_ROOT" "$scope" "$method" "$seed" "$scope"
}

start_evaluation() {
  local scope=$1
  local method=$2
  local seed=$3
  local gpu=$4
  local session="kyc-eval-${scope}-${method}-s${seed}"
  local result
  result=$(evaluation_path "$scope" "$method" "$seed")
  local run_dir=${result%/camera_sweep_test.json}
  if [[ -s "$result" ]]; then
    return
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    return
  fi
  if [[ -e "$run_dir" ]]; then
    echo "incomplete evaluation requires inspection: $run_dir" >&2
    exit 1
  fi

  frame_env=()
  if [[ "$scope" == gate && "$seed" == 41 ]] \
    && [[ "$method" == poseaug_control || "$method" == kyc ]]; then
    frame_env=(
      CABI_EVAL_FRAME_EPISODES=1
      CABI_CAMERA_FRAME_EDGES=red-left,yellow_white-right
      CABI_CAMERA_FRAME_POSES=baseline,az_m60,az_m90,el_m25,el_m32,rad_0900,rad_0750
    )
  fi
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "exec env ${frame_env[*]} bash scripts/cabi_vla/run_kyc_camera_eval.sh $scope $method $seed $gpu"
}

wait_for_evaluation() {
  local scope=$1
  local method=$2
  local seed=$3
  local expected=$4
  local result
  result=$(evaluation_path "$scope" "$method" "$seed")
  while [[ ! -s "$result" ]] || ! jq -e \
    --argjson expected "$expected" \
    '.status == "complete" and (.rows | length) == $expected' \
    "$result" >/dev/null; do
    sleep 30
  done
}

control41="$RUN_ROOT/kyc_poseaug_control_h20_seed41_steps33000/final_model/model.safetensors"
kyc41="$RUN_ROOT/kyc_kyc_h20_seed41_steps33000/final_model/model.safetensors"
wait_for_file "$control41"
wait_for_file "$kyc41"
wait_for_session_exit kyc-full-s41-poseaug_control
wait_for_session_exit kyc-full-s41-kyc
wait_for_file "$DENSE_FOV_ROOT/red-left.json"

start_evaluation dense poseaug_control 41 6
start_evaluation dense kyc 41 7

required_checkpoints=(
  "$RUN_ROOT/kyc_poseaug_rgb_h20_seed41_steps33000/final_model/model.safetensors"
  "$RUN_ROOT/kyc_poseaug_control_h20_seed42_steps33000/final_model/model.safetensors"
  "$RUN_ROOT/kyc_kyc_h20_seed42_steps33000/final_model/model.safetensors"
  "$RUN_ROOT/kyc_poseaug_control_h20_seed43_steps33000/final_model/model.safetensors"
  "$RUN_ROOT/kyc_kyc_h20_seed43_steps33000/final_model/model.safetensors"
)
for checkpoint in "${required_checkpoints[@]}"; do
  wait_for_file "$checkpoint"
done

wait_for_evaluation dense poseaug_control 41 140
wait_for_evaluation dense kyc 41 140

gate_jobs=(
  "base 41 0"
  "poseaug_rgb 41 1"
  "poseaug_control 41 2"
  "kyc 41 3"
  "poseaug_control 42 4"
  "kyc 42 5"
  "poseaug_control 43 6"
  "kyc 43 7"
)
for specification in "${gate_jobs[@]}"; do
  read -r method seed gpu <<<"$specification"
  start_evaluation gate "$method" "$seed" "$gpu"
done

# Reuse GPU 0 for the fixed-pose module control as soon as the Base rollout
# finishes. This keeps the primary eight-way gate fully parallel while still
# evaluating every preregistered seed-41 context arm.
wait_for_evaluation gate base 41 520
wait_for_session_exit kyc-eval-gate-base-s41
start_evaluation gate pm_fixed 41 0

for specification in "${gate_jobs[@]}"; do
  read -r method seed _gpu <<<"$specification"
  wait_for_evaluation gate "$method" "$seed" 520
done
wait_for_evaluation gate pm_fixed 41 520

fov_boundary=("$BOUNDARY_FOV_ROOT"/*.json)
fov_dense=("$DENSE_FOV_ROOT"/*.json)
analysis_root="$EVAL_ROOT/analysis"
mkdir -p "$analysis_root"

if [[ ! -e "$analysis_root/dense_s41" ]]; then
  "$PYTHON" scripts/cabi_vla/compare_kyc_camera_evaluations.py \
    --evaluation "poseaug_control=$(evaluation_path dense poseaug_control 41)" \
    --evaluation "kyc=$(evaluation_path dense kyc 41)" \
    --fov-json "${fov_dense[@]}" \
    --output-dir "$analysis_root/dense_s41" \
    --reference poseaug_control
fi

if [[ ! -e "$analysis_root/gate_s41" ]]; then
  "$PYTHON" scripts/cabi_vla/compare_kyc_camera_evaluations.py \
    --evaluation "base=$(evaluation_path gate base 41)" \
    --evaluation "poseaug_rgb=$(evaluation_path gate poseaug_rgb 41)" \
    --evaluation "pm_fixed=$(evaluation_path gate pm_fixed 41)" \
    --evaluation "poseaug_control=$(evaluation_path gate poseaug_control 41)" \
    --evaluation "kyc=$(evaluation_path gate kyc 41)" \
    --fov-json "${fov_boundary[@]}" \
    --output-dir "$analysis_root/gate_s41" \
    --reference poseaug_control
fi

for seed in 42 43; do
  if [[ ! -e "$analysis_root/gate_s${seed}" ]]; then
    "$PYTHON" scripts/cabi_vla/compare_kyc_camera_evaluations.py \
      --evaluation "poseaug_control=$(evaluation_path gate poseaug_control "$seed")" \
      --evaluation "kyc=$(evaluation_path gate kyc "$seed")" \
      --fov-json "${fov_boundary[@]}" \
      --output-dir "$analysis_root/gate_s${seed}" \
      --reference poseaug_control
  fi
done

if [[ ! -e "$analysis_root/gate_seed_summary" ]]; then
  "$PYTHON" scripts/cabi_vla/summarize_kyc_camera_seeds.py \
    --control "41=$(evaluation_path gate poseaug_control 41)" \
    --control "42=$(evaluation_path gate poseaug_control 42)" \
    --control "43=$(evaluation_path gate poseaug_control 43)" \
    --kyc "41=$(evaluation_path gate kyc 41)" \
    --kyc "42=$(evaluation_path gate kyc 42)" \
    --kyc "43=$(evaluation_path gate kyc 43)" \
    --poseaug-rgb "$(evaluation_path gate poseaug_rgb 41)" \
    --fov-json "${fov_boundary[@]}" \
    --output-dir "$analysis_root/gate_seed_summary"
fi

for method in poseaug_control kyc; do
  run_dir="$EVAL_ROOT/gate/${method}_s41_gate"
  video_dir="$run_dir/videos_av1"
  if [[ ! -e "$video_dir" ]]; then
    PYTHONPATH="$REPO_ROOT/scripts/cabi_vla:$REPO_ROOT/scripts/fresh_vla" \
      "$PYTHON" scripts/cabi_vla/render_libero_bind_eval_frames.py \
      --evaluation "$run_dir/camera_sweep_test.json" \
      --frame-dir "$run_dir/frames" \
      --output-dir "$video_dir" \
      --codecs av1 \
      --fps 20
  fi
done
