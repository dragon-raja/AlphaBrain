#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${CABI_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}
ANCHORS=${CABI_RELOAD_ANCHORS:-/share/longjunyu/cabi-vla/libero-bind-v0-state0-view-v1/anchors.npz}
CHECKPOINT=${1:?usage: verify_policy_reload.sh CHECKPOINT GPU_ID}
GPU_ID=${2:?usage: verify_policy_reload.sh CHECKPOINT GPU_ID}
SESSION="gpu-keepalive-${GPU_ID}"
SOCKET="/tmp/cabi-reload-${GPU_ID}-$$.sock"
LOG="$CHECKPOINT/reload_smoke.log"
SERVER_PID=""
WAS_RUNNING=0

if [[ ! -s "$CHECKPOINT/model.safetensors" ]]; then
  echo "missing checkpoint weights: $CHECKPOINT/model.safetensors" >&2
  exit 1
fi
if [[ ! -s "$ANCHORS" ]]; then
  echo "missing reload-smoke anchors: $ANCHORS" >&2
  exit 1
fi

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$SOCKET"
  if [[ "$WAS_RUNNING" == "1" ]]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-1}" "${AI2R_KEEPALIVE_N:-8192}" \
      "$SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap cleanup EXIT

if tmux has-session -t "$SESSION" 2>/dev/null; then
  WAS_RUNNING=1
  tmux kill-session -t "$SESSION"
fi

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" PRETRAINED_MODELS_DIR="$PRETRAINED_MODELS_DIR" \
  "$PYTHON" scripts/fresh_vla/pi05_policy_server.py \
  --checkpoint "$CHECKPOINT" --socket "$SOCKET" --device cuda:0 >"$LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 600); do
  [[ -S "$SOCKET" ]] && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 80 "$LOG" >&2
    exit 1
  fi
  sleep 1
done
if [[ ! -S "$SOCKET" ]]; then
  echo "timed out waiting for CABI reload server" >&2
  exit 1
fi

"$PYTHON" - "$SOCKET" "$ANCHORS" <<'PY'
import sys
from multiprocessing.connection import Client

import numpy as np

socket, anchors_path = sys.argv[1:]
with np.load(anchors_path, allow_pickle=False) as anchors:
    prefix = "red-left__state_00__"
    target_actions = np.asarray(anchors[prefix + "action"], dtype=np.float32)
    example = {
        "image": [np.asarray(anchors[prefix + "agentview"]), np.asarray(anchors[prefix + "wrist"])],
        "lang": "put the red mug on the left plate",
        "language": "put the red mug on the left plate",
        "state": np.asarray(anchors[prefix + "state"], dtype=np.float32),
    }
connection = Client(socket, family="AF_UNIX", authkey=b"fresh-vla-local")
handshake = connection.recv()
connection.send({"op": "predict", "seed": 20260722, "example": example})
response = connection.recv()
connection.send({"op": "close"})
connection.close()
if "error" in response:
    raise RuntimeError(response["error"])
actions = np.asarray(response["actions"], dtype=np.float32)
expected = (int(handshake["horizon"]), 7)
if actions.shape != expected or not np.all(np.isfinite(actions)):
    raise ValueError(f"invalid reloaded actions: {actions.shape}")
chunk_mse = float(np.square(actions - target_actions).mean())
print(
    "reload_predict_ok",
    "shape=", actions.shape,
    "min=", round(float(actions.min()), 5),
    "max=", round(float(actions.max()), 5),
    "teacher_chunk_mse=", round(chunk_mse, 6),
    "pred_first_xyz=", np.round(actions[0, :3], 4).tolist(),
    "teacher_first_xyz=", np.round(target_actions[0, :3], 4).tolist(),
    "device=", handshake.get("device_name"),
)
PY
