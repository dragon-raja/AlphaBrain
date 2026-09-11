#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
POLICY_PYTHON=${CABI_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
SIM_PYTHON=${CABI_LIBERO_PYTHON:-/share/longjunyu/capt-vla/envs/libero/bin/python}
LIBERO_SOURCE=${CABI_LIBERO_SOURCE:-/share/longjunyu/capt-vla/vendor/LIBERO}
LIBERO_CONFIG=${CABI_LIBERO_CONFIG:-/share/longjunyu/capt-vla/config/libero}
SUITE_ROOT=${CABI_SUITE_ROOT:-/share/longjunyu/cabi-vla/libero-bind-v0}
COLLECTION_ROOT=${CABI_COLLECTION_ROOT:-/share/longjunyu/cabi-vla/libero-bind-v0-train-v1}
OUTPUT_ROOT=${CABI_HANDOFF_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/target-handoff-diagnostics}
STATE_INDICES=${CABI_HANDOFF_STATE_INDICES:-0}
EXECUTION_HORIZON=${CABI_HANDOFF_K:-3}
TOTAL_BUDGET=${CABI_HANDOFF_TOTAL_BUDGET:-320}
SERVER_TIMEOUT=${CABI_POLICY_SERVER_TIMEOUT:-600}

CHECKPOINT=${1:?usage: run_libero_bind_target_handoff.sh CHECKPOINT RUN_NAME GPU_ID}
RUN_NAME=${2:?missing run name}
GPU_ID=${3:?missing GPU id}
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "RUN_NAME contains unsupported characters" >&2
  exit 2
fi
for value in "$GPU_ID" "$EXECUTION_HORIZON" "$TOTAL_BUDGET"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "GPU id, K, and budget must be non-negative integers" >&2
    exit 2
  fi
done
if [[ "$EXECUTION_HORIZON" != 1 && "$EXECUTION_HORIZON" != 2 && "$EXECUTION_HORIZON" != 3 ]]; then
  echo "CABI_HANDOFF_K must be 1, 2, or 3" >&2
  exit 2
fi
if [[ ! -s "$CHECKPOINT/model.safetensors" ]]; then
  echo "missing checkpoint: $CHECKPOINT/model.safetensors" >&2
  exit 1
fi

RUN_DIR=$OUTPUT_ROOT/$RUN_NAME
OUTPUT=$RUN_DIR/target_handoff.json
SOCKET=/tmp/cabi-handoff-${RUN_NAME}-$$.sock
SERVER_LOG=$RUN_DIR/policy_server.log
SESSION=gpu-keepalive-${GPU_ID}
SERVER_PID=""
WAS_RUNNING=0
if [[ -e "$OUTPUT" ]]; then
  echo "refusing to overwrite target handoff: $OUTPUT" >&2
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
"$POLICY_PYTHON" scripts/fresh_vla/pi05_policy_server.py \
  --checkpoint "$CHECKPOINT" --socket "$SOCKET" --device cuda:0 \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 "$SERVER_TIMEOUT"); do
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

read -r -a state_args <<<"$STATE_INDICES"
PYTHONPATH="$REPO_ROOT/scripts/cabi_vla:$LIBERO_SOURCE${PYTHONPATH:+:$PYTHONPATH}" \
LIBERO_CONFIG_PATH="$LIBERO_CONFIG" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONDONTWRITEBYTECODE=1 \
"$SIM_PYTHON" scripts/cabi_vla/evaluate_libero_bind_target_handoff.py \
  --suite-root "$SUITE_ROOT" \
  --collection-root "$COLLECTION_ROOT" \
  --policy-socket "$SOCKET" \
  --output "$OUTPUT" \
  --state-indices "${state_args[@]}" \
  --execution-horizon "$EXECUTION_HORIZON" \
  --total-action-budget "$TOTAL_BUDGET" \
  --frame-dir "$RUN_DIR/frames"
