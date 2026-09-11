#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MODEL_PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
SIM_PYTHON=${FRESH_LIBERO_PYTHON:-/workspace/envs/fresh-libero/bin/python}
LIBERO_SOURCE=${FRESH_LIBERO_SOURCE:-/projects/openpi/third_party/libero}
EPISODE_ROOT=${CORA_EPISODE_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-v2-128}
OUTPUT_ROOT=${CORA_ONPOLICY_ROOT:-/share/longjunyu/fresh-vla/cora-vla/onpolicy-support-v1}
SEED=${1:?usage: run_onpolicy_support.sh SEED GPU_ID [GROUP_OFFSET] [MAX_GROUPS] [TAG]}
GPU_ID=${2:?usage: run_onpolicy_support.sh SEED GPU_ID [GROUP_OFFSET] [MAX_GROUPS] [TAG]}
GROUP_OFFSET=${3:-0}
MAX_GROUPS=${4:-}
TAG=${5:-full}

case "$SEED" in
  41) CHECKPOINT=/share/longjunyu/fresh-vla/runs/baseline-repair-v1/baseline_repair_full_h_ddp8_seed41_steps13804_formal-v2/checkpoints/steps_10353 ;;
  42) CHECKPOINT=/share/longjunyu/fresh-vla/runs/baseline-repair-v1/baseline_repair_full_h_ddp8_seed42_steps10353_formal-budget-v2/checkpoints/steps_10353 ;;
  43) CHECKPOINT=/share/longjunyu/fresh-vla/runs/baseline-repair-v1/baseline_repair_full_h_ddp8_seed43_steps10353_formal-budget-v2/checkpoints/steps_10353 ;;
  *) echo "seed must be 41, 42, or 43" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_ROOT"
OUTPUT="$OUTPUT_ROOT/seed${SEED}-${TAG}.json"
SOCKET="/tmp/cora-onpolicy-${SEED}-${TAG}-$$.sock"
SESSION="gpu-keepalive-${GPU_ID}"
WAS_RUNNING=0
SERVER_PID=""
cleanup() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  [ -n "$SERVER_PID" ] && wait "$SERVER_PID" 2>/dev/null || true
  rm -f "$SOCKET"
  if [ "$WAS_RUNNING" = 1 ]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh 1 8192 "$SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap cleanup EXIT
if tmux has-session -t "$SESSION" 2>/dev/null; then
  WAS_RUNNING=1
  tmux kill-session -t "$SESSION"
fi

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" PRETRAINED_MODELS_DIR=/share/longjunyu/alphabrain/pretrained_models \
"$MODEL_PYTHON" scripts/fresh_vla/pi05_policy_server.py --checkpoint "$CHECKPOINT" --socket "$SOCKET" --device cuda:0 \
  >"$OUTPUT_ROOT/seed${SEED}-${TAG}-server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 600); do
  [ -S "$SOCKET" ] && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "policy server exited" >&2; exit 1; }
  sleep 1
done
[ -S "$SOCKET" ] || { echo "policy server timeout" >&2; exit 1; }

group_args=()
[ -n "$MAX_GROUPS" ] && group_args=(--max-groups "$MAX_GROUPS")
PYTHONPATH="$REPO_ROOT/scripts/cora_vla:$REPO_ROOT/scripts/fresh_vla:$LIBERO_SOURCE" \
LIBERO_CONFIG_PATH="$REPO_ROOT/scripts/fresh_vla/libero_config" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES="$GPU_ID" "$SIM_PYTHON" scripts/cora_vla/evaluate_onpolicy_support.py \
  --policy-socket "$SOCKET" --episode-root "$EPISODE_ROOT" --output "$OUTPUT" \
  --cache-dir "$OUTPUT_ROOT/cache/seed${SEED}-${TAG}" --seed "$SEED" --split val \
  --group-offset "$GROUP_OFFSET" "${group_args[@]}"
