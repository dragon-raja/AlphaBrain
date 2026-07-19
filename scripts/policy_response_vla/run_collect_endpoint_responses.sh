#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MODEL_PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
SIM_PYTHON=${FRESH_LIBERO_PYTHON:-/workspace/envs/fresh-libero/bin/python}
LIBERO_SOURCE=${FRESH_LIBERO_SOURCE:-/projects/openpi/third_party/libero}
CCV_ROOT=${POLICY_RESPONSE_CCV_ROOT:-/share/longjunyu/fresh-vla/ccv-vla/gate0-coupled-v3}
EPISODE_ROOT=${POLICY_RESPONSE_EPISODE_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-v2-128}
OUTPUT_ROOT=${POLICY_RESPONSE_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/policy-response-vla/gate-minus1-v1}
CHECKPOINT=${POLICY_RESPONSE_CHECKPOINT:-/share/longjunyu/fresh-vla/runs/baseline-repair-v1/baseline_repair_full_h_ddp8_seed41_steps13804_formal-v2/checkpoints/steps_10353}
PREREGISTRATION=$REPO_ROOT/docs/policy_response_vla/gate_minus1_preregistration.md
GPU_ID=${1:?usage: run_collect_endpoint_responses.sh GPU_ID [MAX_STATES] [STATE_OFFSET]}
MAX_STATES=${2:-}
STATE_OFFSET=${3:-0}
SOCKET=/tmp/policy-response-${GPU_ID}-$$.sock
SESSION=gpu-keepalive-${GPU_ID}
WAS_RUNNING=0
SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ "$WAS_RUNNING" = 1 ]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh 1 2048 "$SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap cleanup EXIT

if tmux has-session -t "$SESSION" 2>/dev/null; then
  WAS_RUNNING=1
  tmux kill-session -t "$SESSION"
fi
mkdir -p "$OUTPUT_ROOT/logs"
cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
  "$MODEL_PYTHON" scripts/fresh_vla/pi05_policy_server.py \
  --checkpoint "$CHECKPOINT" --socket "$SOCKET" --device cuda:0 \
  >"$OUTPUT_ROOT/logs/policy-server-g${GPU_ID}-o${STATE_OFFSET}.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 600); do
  [ -S "$SOCKET" ] && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "policy server exited" >&2; exit 1; }
  sleep 1
done
[ -S "$SOCKET" ] || { echo "policy server timeout" >&2; exit 1; }

extra=()
[ -n "$MAX_STATES" ] && extra+=(--max-states "$MAX_STATES")
PYTHONPATH="$REPO_ROOT:$REPO_ROOT/scripts/ccv_vla:$REPO_ROOT/scripts/cora_vla:$REPO_ROOT/scripts/fresh_vla:$LIBERO_SOURCE" \
LIBERO_CONFIG_PATH="$REPO_ROOT/scripts/fresh_vla/libero_config" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES="$GPU_ID" "$SIM_PYTHON" scripts/policy_response_vla/collect_endpoint_responses.py \
  --policy-socket "$SOCKET" --ccv-root "$CCV_ROOT" --episode-root "$EPISODE_ROOT" \
  --output-root "$OUTPUT_ROOT" --preregistration "$PREREGISTRATION" \
  --state-offset "$STATE_OFFSET" --resume "${extra[@]}"

