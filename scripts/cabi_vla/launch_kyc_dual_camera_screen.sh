#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SEED=${1:-41}
STEPS=${2:-2000}
RUN_TAG=${KYC_DUAL_RUN_TAG:-dualcam-screen-v1}
RUN_ROOT=${KYC_DUAL_RUN_ROOT:-/share/longjunyu/cabi-vla/kyc-runs}
EVAL_ROOT=${KYC_DUAL_EVAL_ROOT:-/share/longjunyu/cabi-vla/dual-camera-kyc-screen-v1}
CAMERA_CONFIG=${KYC_DUAL_CAMERA_CONFIG:-$REPO_ROOT/docs/cabi_vla/configs/camera_pose_policy_gate_v6.json}
SUMMARY="$EVAL_ROOT/summary.json"
FIGURE="$EVAL_ROOT/dual_camera_screen.png"

if [[ ! "$SEED" =~ ^[0-9]+$ || ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "seed must be non-negative and steps must be positive" >&2
  exit 2
fi

run_id() {
  local arm=$1
  printf 'kyc_%s_%s_h20_seed%s_steps%s' "$arm" "$RUN_TAG" "$SEED" "$STEPS"
}

checkpoint() {
  printf '%s/%s/final_model' "$RUN_ROOT" "$(run_id "$1")"
}

wait_for_training() {
  local arm=$1
  local label=$2
  local session="dualcam-screen-${label}-s${SEED}"
  local model
  model=$(checkpoint "$arm")
  while [[ ! -s "$model/model.safetensors" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "training stopped without a final checkpoint: $session" >&2
      exit 1
    fi
    sleep 30
  done
  echo "checkpoint ready: $model"
}

start_eval() {
  local arm=$1
  local label=$2
  local gpu=$3
  local wrist_mode=$4
  local record_frames=$5
  local run_name="${label}-s${SEED}-u${STEPS}"
  local session="dualcam-eval-${label}-s${SEED}"
  local output="$EVAL_ROOT/$run_name/camera_sweep_test.json"
  local partial="$EVAL_ROOT/$run_name/camera_sweep_test.partial.json"
  local model
  model=$(checkpoint "$arm")
  if [[ -s "$output" ]]; then
    echo "evaluation already complete: $run_name"
    return
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "evaluation already running: $session"
    return
  fi
  if [[ -e "$partial" || -e "$EVAL_ROOT/$run_name" ]]; then
    echo "incomplete evaluation requires inspection: $EVAL_ROOT/$run_name" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" -c "$REPO_ROOT" \
    "exec env CABI_CAMERA_OUTPUT_ROOT='$EVAL_ROOT' CABI_EVAL_SPLIT=test \
      CABI_CAMERA_CONFIG='$CAMERA_CONFIG' \
      CABI_EVAL_STATE_INDICES=40,42,44,47,49 \
      CABI_EVAL_EDGES=red-left,red-right,white-left,yellow_white-right \
      CABI_CAMERA_POSES=baseline,az_m60,az_p60,el_m25,el_p25,rad_0900,rad_1250 \
      CABI_EVAL_HORIZONS=3 CABI_EVAL_MAX_STEPS=320 \
      CABI_SCENE_CUE_MODE=cue_randomized CABI_EVAL_FRAME_EPISODES='$record_frames' \
      CABI_CAMERA_FRAME_POSES=baseline,az_m60,az_p60 \
      CABI_WRIST_RAY_MODE='$wrist_mode' \
      bash scripts/cabi_vla/run_libero_bind_camera_sweep.sh \
      '$model' '$run_name' '$gpu'"
  echo "started evaluation: $session gpu=$gpu wrist_ray=$wrist_mode"
}

wait_for_eval() {
  local label=$1
  local session="dualcam-eval-${label}-s${SEED}"
  local output="$EVAL_ROOT/${label}-s${SEED}-u${STEPS}/camera_sweep_test.json"
  while [[ ! -s "$output" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "evaluation stopped without a final result: $session" >&2
      exit 1
    fi
    sleep 30
  done
  echo "evaluation ready: $output"
}

jobs=(
  "dual_rgb_fla dual-rgb 0 correct 1"
  "dual_control_fla dual-control 1 correct 1"
  "external_fla external 2 correct 0"
  "wrist_fla wrist 3 correct 0"
  "dual_fla dual 4 correct 1"
  "dual_fla dual-wrist-initial 5 initial 0"
  "dual_fla dual-wrist-lagged 6 lagged 0"
)

for item in \
  "dual_rgb_fla rgb" \
  "dual_control_fla control" \
  "external_fla external" \
  "wrist_fla wrist" \
  "dual_fla dual"; do
  read -r arm label <<<"$item"
  wait_for_training "$arm" "$label"
done

mkdir -p "$EVAL_ROOT"
for job in "${jobs[@]}"; do
  read -r arm label gpu wrist_mode record_frames <<<"$job"
  start_eval "$arm" "$label" "$gpu" "$wrist_mode" "$record_frames"
done
for job in "${jobs[@]}"; do
  read -r _arm label _gpu _wrist_mode _record_frames <<<"$job"
  wait_for_eval "$label"
done

if [[ ! -s "$SUMMARY" ]]; then
  PYTHONPATH="$REPO_ROOT/scripts/cabi_vla" "$REPO_ROOT/.venv/bin/python" \
    scripts/cabi_vla/summarize_kyc_dual_camera_screen.py \
    --evaluation "dual_rgb_fla=$EVAL_ROOT/dual-rgb-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --evaluation "dual_control_fla=$EVAL_ROOT/dual-control-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --evaluation "external_fla=$EVAL_ROOT/external-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --evaluation "wrist_fla=$EVAL_ROOT/wrist-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --evaluation "dual_fla=$EVAL_ROOT/dual-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --wrist-intervention "initial=$EVAL_ROOT/dual-wrist-initial-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --wrist-intervention "lagged=$EVAL_ROOT/dual-wrist-lagged-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --output "$SUMMARY"
fi

if [[ ! -s "$FIGURE" ]]; then
  PYTHONPATH="$REPO_ROOT/scripts/cabi_vla" "$REPO_ROOT/.venv/bin/python" \
    scripts/cabi_vla/render_kyc_dual_camera_screen.py \
    --summary "$SUMMARY" \
    --evaluation "dual_rgb_fla=$EVAL_ROOT/dual-rgb-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --evaluation "dual_control_fla=$EVAL_ROOT/dual-control-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --evaluation "external_fla=$EVAL_ROOT/external-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --evaluation "wrist_fla=$EVAL_ROOT/wrist-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --evaluation "dual_fla=$EVAL_ROOT/dual-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --output "$FIGURE"
fi

for label in dual-rgb dual-control dual; do
  run_dir="$EVAL_ROOT/${label}-s${SEED}-u${STEPS}"
  if [[ ! -s "$run_dir/videos_av1/manifest.json" ]]; then
    "$REPO_ROOT/.venv/bin/python" scripts/cabi_vla/render_libero_bind_eval_frames.py \
      --evaluation "$run_dir/camera_sweep_test.json" \
      --frame-dir "$run_dir/frames" \
      --output-dir "$run_dir/videos_av1" \
      --codecs av1 --fps 20
  fi
done

paired_dir="$EVAL_ROOT/paired-dual-control-vs-dual-av1"
if [[ ! -s "$paired_dir/manifest.json" ]]; then
  "$REPO_ROOT/.venv/bin/python" scripts/cabi_vla/render_libero_bind_paired_videos.py \
    --baseline-evaluation "$EVAL_ROOT/dual-control-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --baseline-frame-dir "$EVAL_ROOT/dual-control-s${SEED}-u${STEPS}/frames" \
    --method-evaluation "$EVAL_ROOT/dual-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --method-frame-dir "$EVAL_ROOT/dual-s${SEED}-u${STEPS}/frames" \
    --output-dir "$paired_dir" \
    --baseline-name "Dual-Control" --method-name "Dual-KYC" \
    --codecs av1 --fps 20
fi

echo "dual-camera KYC screen complete: $SUMMARY"
