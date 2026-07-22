#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${CABI_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
TRAINING_VIEW=${CABI_DATA_ROOT:-/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v5-loss-balanced}
OUTPUT_ROOT=${CABI_DIAGNOSTIC_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/diagnostics}
STATE_INDICES=${CABI_DIAGNOSTIC_STATE_INDICES:-0}
FRAME_STRIDE=${CABI_DIAGNOSTIC_FRAME_STRIDE:-20}
SEEDS=${CABI_DIAGNOSTIC_SEEDS:-20260722 20260723 20260724}
SERVER_TIMEOUT=${CABI_POLICY_SERVER_TIMEOUT:-600}

CHECKPOINT=${1:?usage: run_libero_bind_policy_diagnosis.sh CHECKPOINT RUN_NAME GPU_ID}
RUN_NAME=${2:?usage: run_libero_bind_policy_diagnosis.sh CHECKPOINT RUN_NAME GPU_ID}
GPU_ID=${3:?usage: run_libero_bind_policy_diagnosis.sh CHECKPOINT RUN_NAME GPU_ID}

if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "RUN_NAME contains unsupported characters" >&2
  exit 2
fi
if [[ ! "$FRAME_STRIDE" =~ ^[1-9][0-9]*$ ]]; then
  echo "CABI_DIAGNOSTIC_FRAME_STRIDE must be a positive integer" >&2
  exit 2
fi
if [[ ! -s "$CHECKPOINT/model.safetensors" ]]; then
  echo "missing checkpoint: $CHECKPOINT/model.safetensors" >&2
  exit 1
fi

RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
OUTPUT="$RUN_DIR/policy_diagnosis.json"
SOCKET="/tmp/cabi-diagnosis-${RUN_NAME}-$$.sock"
SESSION="gpu-keepalive-${GPU_ID}"
SERVER_PID=""
WAS_RUNNING=0
if [[ -e "$OUTPUT" ]]; then
  echo "refusing to overwrite existing output: $OUTPUT" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$SOCKET"
  if [[ "$WAS_RUNNING" == 1 ]]; then
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
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
PYTHONDONTWRITEBYTECODE=1 \
"$PYTHON" scripts/fresh_vla/pi05_policy_server.py \
  --checkpoint "$CHECKPOINT" --socket "$SOCKET" --device cuda:0 \
  >"$RUN_DIR/policy_server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 "$SERVER_TIMEOUT"); do
  [[ -S "$SOCKET" ]] && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 80 "$RUN_DIR/policy_server.log" >&2
    exit 1
  fi
  sleep 1
done
if [[ ! -S "$SOCKET" ]]; then
  echo "timed out waiting for Pi0.5 policy server" >&2
  exit 1
fi

read -r -a state_args <<<"$STATE_INDICES"
read -r -a seed_args <<<"$SEEDS"
"$PYTHON" scripts/cabi_vla/diagnose_libero_bind_policy.py \
  --training-view "$TRAINING_VIEW" \
  --policy-socket "$SOCKET" \
  --output "$OUTPUT" \
  --state-indices "${state_args[@]}" \
  --frame-stride "$FRAME_STRIDE" \
  --seeds "${seed_args[@]}"
