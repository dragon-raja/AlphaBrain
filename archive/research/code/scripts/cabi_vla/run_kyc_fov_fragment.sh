#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${KYC_LIBERO_PYTHON:-/share/longjunyu/capt-vla/envs/libero/bin/python}
SUITE_ROOT=${KYC_SUITE_ROOT:-/share/longjunyu/cabi-vla/libero-bind-v0}
CAMERA_CONFIG=${KYC_FOV_CONFIG:-$REPO_ROOT/docs/cabi_vla/configs/camera_pose_fov_extended_v3.json}
OUTPUT_ROOT=${KYC_FOV_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/camera-viewpoint-study-v2/fov_guard_extended_state0_v4}
SPLIT=${KYC_FOV_SPLIT:-train}
STATE_INDICES=${KYC_FOV_STATE_INDICES:-0}

EDGE=${1:?usage: run_kyc_fov_fragment.sh EDGE GPU_ID}
GPU_ID=${2:?usage: run_kyc_fov_fragment.sh EDGE GPU_ID}
OUTPUT="$OUTPUT_ROOT/$EDGE.json"
PARTIAL="$OUTPUT_ROOT/$EDGE.partial.json"
LOG="$OUTPUT_ROOT/logs/$EDGE.log"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing LIBERO Python: $PYTHON" >&2
  exit 1
fi
if [[ -e "$OUTPUT" || -e "$PARTIAL" ]]; then
  echo "refusing to overwrite FOV fragment: $OUTPUT or $PARTIAL" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/logs"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/scripts/cabi_vla:/share/longjunyu/capt-vla/vendor/LIBERO"
export LIBERO_CONFIG_PATH=/share/longjunyu/capt-vla/config/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU_ID"

"$PYTHON" scripts/cabi_vla/scan_libero_camera_visibility.py \
  --suite-root "$SUITE_ROOT" \
  --camera-config "$CAMERA_CONFIG" \
  --output "$OUTPUT" \
  --split "$SPLIT" \
  --state-indices "$STATE_INDICES" \
  --edges "$EDGE" \
  >"$LOG" 2>&1
