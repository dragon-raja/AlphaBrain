#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
POLICY_PYTHON=${CABI_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
SIM_PYTHON=${CABI_LIBERO_PYTHON:-/share/longjunyu/capt-vla/envs/libero/bin/python}
LIBERO_SOURCE=${CABI_LIBERO_SOURCE:-/share/longjunyu/capt-vla/vendor/LIBERO}
LIBERO_CONFIG=${CABI_LIBERO_CONFIG:-/share/longjunyu/capt-vla/config/libero}
SUITE_ROOT=${CABI_SUITE_ROOT:-/share/longjunyu/cabi-vla/libero-bind-v0}
CAMERA_CONFIG=${CABI_CAMERA_CONFIG:-$REPO_ROOT/docs/cabi_vla/configs/camera_pose_policy_gate_v6.json}
CHECKPOINT=${KYC_RAY_CHECKPOINT:-/share/longjunyu/cabi-vla/kyc-runs/kyc_kyc_h20_seed41_steps33000/final_model}
OUTPUT=${KYC_RAY_OUTPUT:-/share/longjunyu/cabi-vla/kyc-scaling-v3/diagnostics/kyc_seed41_ray_use_v1.json}
GPU_ID=${1:-0}

if [[ ! "$GPU_ID" =~ ^[0-7]$ ]]; then
  echo "GPU_ID must be in [0, 7]" >&2
  exit 2
fi
if [[ ! -s "$CHECKPOINT/model.safetensors" ]]; then
  echo "missing completed KYC checkpoint" >&2
  exit 1
fi
if [[ -e "$OUTPUT" ]]; then
  echo "refusing to overwrite ray diagnostic: $OUTPUT" >&2
  exit 1
fi

RUN_DIR=$(dirname "$OUTPUT")
SOCKET="/tmp/kyc-ray-use-$$.sock"
SERVER_LOG="$RUN_DIR/kyc_ray_policy_server.log"
KEEPALIVE_SESSION="gpu-keepalive-${GPU_ID}"
KEEPALIVE_WAS_RUNNING=0
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$SOCKET"
  if [[ "$KEEPALIVE_WAS_RUNNING" == "1" ]]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-1}" "${AI2R_KEEPALIVE_N:-8192}" \
      "$KEEPALIVE_SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap cleanup EXIT

if tmux has-session -t "$KEEPALIVE_SESSION" 2>/dev/null; then
  KEEPALIVE_WAS_RUNNING=1
  tmux kill-session -t "$KEEPALIVE_SESSION"
fi

mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
PYTHONDONTWRITEBYTECODE=1 \
"$POLICY_PYTHON" scripts/fresh_vla/pi05_policy_server.py \
  --checkpoint "$CHECKPOINT" \
  --socket "$SOCKET" \
  --device cuda:0 >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 600); do
  [[ -S "$SOCKET" ]] && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 80 "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 1
done
if [[ ! -S "$SOCKET" ]]; then
  echo "timed out waiting for Pi0.5 policy server" >&2
  exit 1
fi

PYTHONPATH="$REPO_ROOT/scripts/cabi_vla:$LIBERO_SOURCE${PYTHONPATH:+:$PYTHONPATH}" \
LIBERO_CONFIG_PATH="$LIBERO_CONFIG" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONDONTWRITEBYTECODE=1 \
"$SIM_PYTHON" scripts/cabi_vla/diagnose_kyc_ray_use.py \
  --suite-root "$SUITE_ROOT" \
  --policy-socket "$SOCKET" \
  --camera-config "$CAMERA_CONFIG" \
  --output "$OUTPUT" \
  --split test \
  --state-indices 47,48,49 \
  --edges red-left,red-right,white-left,yellow_white-right \
  --poses baseline,az_m60,el_p25,rad_1250

