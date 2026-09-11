#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MODEL_PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
SIM_PYTHON=${FRESH_LIBERO_PYTHON:-/workspace/envs/fresh-libero/bin/python}
LIBERO_SOURCE=${FRESH_LIBERO_SOURCE:-/projects/openpi/third_party/libero}
EPISODE_ROOT=${CORA_EPISODE_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-v2-128}
OUTPUT_ROOT=${CORA_GATE1_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/cora-vla/gate1-candidate-support-v1}
PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}

SEED=${1:?usage: run_candidate_support.sh SEED GPU_ID [MAX_GROUPS]}
GPU_ID=${2:?usage: run_candidate_support.sh SEED GPU_ID [MAX_GROUPS]}
MAX_GROUPS=${3:-}
CORA_GROUP_START=${CORA_GROUP_START:-0}
CORA_GROUP_COUNT=${CORA_GROUP_COUNT:-}
CORA_OUTPUT_TAG=${CORA_OUTPUT_TAG:-}

case "$SEED" in
  41) CHECKPOINT=/share/longjunyu/fresh-vla/runs/baseline-repair-v1/baseline_repair_full_h_ddp8_seed41_steps13804_formal-v2/checkpoints/steps_10353 ;;
  42) CHECKPOINT=/share/longjunyu/fresh-vla/runs/baseline-repair-v1/baseline_repair_full_h_ddp8_seed42_steps10353_formal-budget-v2/checkpoints/steps_10353 ;;
  43) CHECKPOINT=/share/longjunyu/fresh-vla/runs/baseline-repair-v1/baseline_repair_full_h_ddp8_seed43_steps10353_formal-budget-v2/checkpoints/steps_10353 ;;
  *) echo "seed must be 41, 42, or 43" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_ROOT"
OUTPUT="$OUTPUT_ROOT/seed${SEED}${CORA_OUTPUT_TAG:+-$CORA_OUTPUT_TAG}.json"
if [ -f "$OUTPUT" ]; then
  echo "skip completed CORA Gate 1 seed=$SEED"
  exit 0
fi
if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  echo "missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if [ -n "$MAX_GROUPS" ] && [[ ! "$MAX_GROUPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_GROUPS must be a positive integer" >&2
  exit 2
fi

KEEPALIVE_SESSION="gpu-keepalive-${GPU_ID}"
KEEPALIVE_WAS_RUNNING=0
SERVER_PID=""
SOCKET_PATH="/tmp/cora-gate1-seed${SEED}-$$.sock"
cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$SOCKET_PATH"
  if [ "$KEEPALIVE_WAS_RUNNING" = 1 ]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-1}" "${AI2R_KEEPALIVE_N:-8192}" "$KEEPALIVE_SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap cleanup EXIT

if tmux has-session -t "$KEEPALIVE_SESSION" 2>/dev/null; then
  KEEPALIVE_WAS_RUNNING=1
  tmux kill-session -t "$KEEPALIVE_SESSION"
fi

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PRETRAINED_MODELS_DIR="$PRETRAINED_MODELS_DIR" \
PYTHONDONTWRITEBYTECODE=1 \
"$MODEL_PYTHON" scripts/fresh_vla/pi05_policy_server.py \
  --checkpoint "$CHECKPOINT" \
  --socket "$SOCKET_PATH" \
  --device cuda:0 >"$OUTPUT_ROOT/seed${SEED}-server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 600); do
  if [ -S "$SOCKET_PATH" ]; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Pi0.5 server exited before readiness" >&2
    exit 1
  fi
  sleep 1
done
if [ ! -S "$SOCKET_PATH" ]; then
  echo "timed out waiting for Pi0.5 server" >&2
  exit 1
fi

max_group_args=()
if [ -n "$MAX_GROUPS" ]; then
  max_group_args=(--max-groups "$MAX_GROUPS")
fi
group_args=(--group-start "$CORA_GROUP_START")
if [ -n "$CORA_GROUP_COUNT" ]; then
  group_args+=(--group-count "$CORA_GROUP_COUNT")
fi
PYTHONPATH="$REPO_ROOT/scripts/cora_vla:$REPO_ROOT/scripts/fresh_vla:$LIBERO_SOURCE${PYTHONPATH:+:$PYTHONPATH}" \
LIBERO_CONFIG_PATH="$REPO_ROOT/scripts/fresh_vla/libero_config" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PRETRAINED_MODELS_DIR="$PRETRAINED_MODELS_DIR" \
PYTHONDONTWRITEBYTECODE=1 \
"$SIM_PYTHON" scripts/cora_vla/evaluate_candidate_support.py \
  --policy-socket "$SOCKET_PATH" \
  --episode-root "$EPISODE_ROOT" \
  --output "$OUTPUT" \
  --checkpoint-seed "$SEED" \
  --split val \
  "${group_args[@]}" \
  "${max_group_args[@]}"
