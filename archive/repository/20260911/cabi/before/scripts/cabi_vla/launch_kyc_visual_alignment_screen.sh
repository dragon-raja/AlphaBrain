#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${KYC_VISUAL_ALIGNMENT_ROOT:-/share/longjunyu/cabi-vla/kyc-visual-alignment-v1}
SOURCE_DATA=${KYC_VISUAL_ALIGNMENT_DATA:-/share/longjunyu/cabi-vla/kyc-scaling-v3/views/libero-bind-kyc-n10-cue-h20}
SEED=${1:-41}
STEPS=${2:-2000}
RUN_TAG=visual-align-screen-v1
RUN_ROOT="$DATA_ROOT/runs/screen-steps${STEPS}"
EVAL_ROOT="$DATA_ROOT/eval/screen-s${SEED}-steps${STEPS}"

if [[ ! "$SEED" =~ ^[0-9]+$ || ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "seed must be non-negative and steps must be positive" >&2
  exit 2
fi
if [[ ! -s "$SOURCE_DATA/manifest.json" ]]; then
  echo "missing visual-alignment data view: $SOURCE_DATA" >&2
  exit 1
fi

run_id() {
  local arm=$1
  printf 'kyc_%s_%s_h20_seed%s_steps%s' "$arm" "$RUN_TAG" "$SEED" "$STEPS"
}

start_train() {
  local arm=$1
  local gpu=$2
  local id
  local session="kyc-va-${arm}-s${SEED}-u${STEPS}"
  id=$(run_id "$arm")
  if [[ -s "$RUN_ROOT/$id/final_model/model.safetensors" ]]; then
    echo "training already complete: $id"
    return
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "training already running: $session"
    return
  fi
  if [[ -e "$RUN_ROOT/$id" ]]; then
    echo "incomplete training requires inspection: $RUN_ROOT/$id" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "exec env KYC_DATA_ROOT_OVERRIDE='$SOURCE_DATA' KYC_WRIST_MODE=on \
      KYC_RUN_TAG='$RUN_TAG' KYC_OUTPUT_ROOT='$RUN_ROOT' \
      KYC_CABI_ANCHOR_PERIOD=1000000 \
      bash scripts/cabi_vla/run_kyc_train.sh '$arm' '$SEED' '$gpu' '$STEPS'"
  echo "started training: $session gpu=$gpu"
}

wait_train() {
  local arm=$1
  local id
  local session="kyc-va-${arm}-s${SEED}-u${STEPS}"
  id=$(run_id "$arm")
  while [[ ! -s "$RUN_ROOT/$id/final_model/model.safetensors" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "training stopped without checkpoint: $session" >&2
      exit 1
    fi
    sleep 30
  done
}

start_eval() {
  local arm=$1
  local label=$2
  local gpu=$3
  local id
  local session="kyc-va-eval-${label}-s${SEED}-u${STEPS}"
  local run_name="${label}-s${SEED}-u${STEPS}"
  id=$(run_id "$arm")
  if [[ -s "$EVAL_ROOT/$run_name/camera_sweep_test.json" ]]; then
    echo "evaluation already complete: $run_name"
    return
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "evaluation already running: $session"
    return
  fi
  if [[ -e "$EVAL_ROOT/$run_name" ]]; then
    echo "incomplete evaluation requires inspection: $EVAL_ROOT/$run_name" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "exec env CABI_CAMERA_OUTPUT_ROOT='$EVAL_ROOT' CABI_EVAL_SPLIT=test \
      CABI_CAMERA_CONFIG='$REPO_ROOT/docs/cabi_vla/configs/camera_pose_policy_gate_v6.json' \
      CABI_EVAL_STATE_INDICES=40,42,44,47,49 \
      CABI_EVAL_EDGES=red-left,red-right,white-left,yellow_white-right \
      CABI_CAMERA_POSES=baseline,az_m60,az_p60,el_m25,el_p25,rad_0900,rad_1250 \
      CABI_EVAL_HORIZONS=3 CABI_EVAL_MAX_STEPS=320 \
      CABI_SCENE_CUE_MODE=cue_randomized CABI_EVAL_FRAME_EPISODES=0 \
      bash scripts/cabi_vla/run_libero_bind_camera_sweep.sh \
      '$RUN_ROOT/$id/final_model' '$run_name' '$gpu'"
  echo "started evaluation: $session gpu=$gpu"
}

wait_eval() {
  local label=$1
  local session="kyc-va-eval-${label}-s${SEED}-u${STEPS}"
  local output="$EVAL_ROOT/${label}-s${SEED}-u${STEPS}/camera_sweep_test.json"
  while [[ ! -s "$output" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "evaluation stopped without result: $session" >&2
      exit 1
    fi
    sleep 30
  done
}

jobs=(
  "poseaug_rgb_fla rgb-fla 2"
  "poseaug_control_fla ctrl-fla 3"
  "kyc_fla kyc-fla 4"
)

for job in "${jobs[@]}"; do
  read -r arm _label gpu <<<"$job"
  start_train "$arm" "$gpu"
done
for job in "${jobs[@]}"; do
  read -r arm _label _gpu <<<"$job"
  wait_train "$arm"
done

for job in "${jobs[@]}"; do
  read -r arm label gpu <<<"$job"
  start_eval "$arm" "$label" "$gpu"
done
for job in "${jobs[@]}"; do
  read -r _arm label _gpu <<<"$job"
  wait_eval "$label"
done

kyc_id=$(run_id kyc_fla)
ray_name="kyc-fla-s${SEED}-u${STEPS}"
ray_output="$EVAL_ROOT/ray/$ray_name/ray_use.json"
if [[ ! -s "$ray_output" ]]; then
  env CABI_RAY_DIAGNOSTIC_ROOT="$EVAL_ROOT/ray" \
    CABI_EVAL_SPLIT=test CABI_EVAL_STATE_INDICES=40,42,44,47,49 \
    CABI_EVAL_EDGES=red-left,red-right,white-left,yellow_white-right \
    CABI_CAMERA_POSES=baseline,az_m60,az_p60,el_m25,el_p25,rad_0900,rad_1250 \
    CABI_SCENE_CUE_MODE=cue_randomized \
    bash scripts/cabi_vla/run_kyc_ray_diagnostic.sh \
    "$RUN_ROOT/$kyc_id/final_model" "$ray_name" 4
fi

summary="$EVAL_ROOT/summary.json"
if [[ ! -s "$summary" ]]; then
  PYTHONPATH="$REPO_ROOT/scripts/cabi_vla" "$REPO_ROOT/.venv/bin/python" \
    scripts/cabi_vla/summarize_kyc_visual_alignment_screen.py \
    --evaluation "poseaug_rgb_fla=$EVAL_ROOT/rgb-fla-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --evaluation "poseaug_control_fla=$EVAL_ROOT/ctrl-fla-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --evaluation "kyc_fla=$EVAL_ROOT/kyc-fla-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --ray-diagnostic "$ray_output" \
    --output "$summary"
fi

echo "visual alignment screen complete: $summary"
