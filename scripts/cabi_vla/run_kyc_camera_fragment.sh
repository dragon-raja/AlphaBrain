#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${KYC_LIBERO_PYTHON:-/share/longjunyu/capt-vla/envs/libero/bin/python}
TRAINING_VIEW=${KYC_SOURCE_TRAINING_VIEW:-/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v15-decision-observed-edge-phase-loss-balanced-h20}
SUITE_ROOT=${KYC_SUITE_ROOT:-/share/longjunyu/cabi-vla/libero-bind-v0}
CAMERA_CONFIG=${KYC_CAMERA_CONFIG:-$REPO_ROOT/docs/cabi_vla/configs/camera_pose_train_random_v2.json}
FRAGMENT_ROOT=${KYC_FRAGMENT_ROOT:-/share/longjunyu/cabi-vla/camera-viewpoint-study-v2/training_fragments_v2}

EDGE=${1:?usage: run_kyc_camera_fragment.sh EDGE GPU_ID}
GPU_ID=${2:?usage: run_kyc_camera_fragment.sh EDGE GPU_ID}
OUTPUT="$FRAGMENT_ROOT/$EDGE"
LOG_DIR="$FRAGMENT_ROOT/logs"
LOG="$LOG_DIR/$EDGE.log"
KEEPALIVE_SESSION="gpu-keepalive-${GPU_ID}"
KEEPALIVE_WAS_RUNNING=0

if [[ ! -x "$PYTHON" ]]; then
  echo "missing LIBERO Python: $PYTHON" >&2
  exit 1
fi
if [[ -e "$OUTPUT" ]]; then
  echo "refusing to overwrite camera fragment: $OUTPUT" >&2
  exit 1
fi

restore_keepalive() {
  if [[ "$KEEPALIVE_WAS_RUNNING" == "1" ]]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-1}" "${AI2R_KEEPALIVE_N:-8192}" \
      "$KEEPALIVE_SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap restore_keepalive EXIT

if tmux has-session -t "$KEEPALIVE_SESSION" 2>/dev/null; then
  KEEPALIVE_WAS_RUNNING=1
  tmux kill-session -t "$KEEPALIVE_SESSION"
fi

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/scripts/cabi_vla:/share/longjunyu/capt-vla/vendor/LIBERO"
export LIBERO_CONFIG_PATH=/share/longjunyu/capt-vla/config/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU_ID"

"$PYTHON" scripts/cabi_vla/generate_libero_bind_camera_training_fragment.py \
  --training-view "$TRAINING_VIEW" \
  --suite-root "$SUITE_ROOT" \
  --camera-config "$CAMERA_CONFIG" \
  --output "$OUTPUT" \
  --edges "$EDGE" \
  --baseline-image-mae-tolerance "${KYC_BASELINE_IMAGE_MAE_TOLERANCE:-1.0}" \
  2>&1 | tee "$LOG"
