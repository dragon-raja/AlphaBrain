#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SIM_PYTHON=${FRESH_LIBERO_PYTHON:-/workspace/envs/fresh-libero/bin/python}
LIBERO_SOURCE=${FRESH_LIBERO_SOURCE:-/projects/openpi/third_party/libero}
GPU_ID=${FRESH_LIBERO_GPU_ID:-7}
KEEPALIVE_SESSION="gpu-keepalive-${GPU_ID}"
KEEPALIVE_WAS_RUNNING=0

if [ ! -x "$SIM_PYTHON" ]; then
  echo "missing isolated LIBERO Python: $SIM_PYTHON" >&2
  exit 1
fi
if [ ! -d "$LIBERO_SOURCE/libero" ]; then
  echo "missing LIBERO source: $LIBERO_SOURCE" >&2
  exit 1
fi

restore_keepalive() {
  if [ "$KEEPALIVE_WAS_RUNNING" = "1" ]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      4 8192 "$KEEPALIVE_SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap restore_keepalive EXIT

if tmux has-session -t "$KEEPALIVE_SESSION" 2>/dev/null; then
  KEEPALIVE_WAS_RUNNING=1
  tmux kill-session -t "$KEEPALIVE_SESSION"
fi

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/scripts/fresh_vla:$LIBERO_SOURCE${PYTHONPATH:+:$PYTHONPATH}" \
LIBERO_CONFIG_PATH="$REPO_ROOT/scripts/fresh_vla/libero_config" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
"$SIM_PYTHON" scripts/fresh_vla/generate_libero_training_data.py "$@"
