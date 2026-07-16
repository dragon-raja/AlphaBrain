#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
BASELINE_VIEW_ROOT=${FRESH_BASELINE_VIEW_ROOT:-/share/longjunyu/fresh-vla/runs/baseline-repair-v1/eval_views}
EPISODE_ROOT=${FRESH_EPISODE_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-v2-128}
OUTPUT_ROOT=${FRESH_MECHANISM_DIAGNOSTIC_ROOT:-/share/longjunyu/fresh-vla/runs/mechanism-diagnostics-v1}
PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}

SEED=${1:?usage: run_counterfactual_feedback_revision_diagnostic.sh SEED GPU_ID [MAX_GROUPS]}
GPU_ID=${2:?usage: run_counterfactual_feedback_revision_diagnostic.sh SEED GPU_ID [MAX_GROUPS]}
MAX_GROUPS=${3:-}

case "$SEED" in
  41) EXPECTED_SHA256=31cc2edf0b53fa69a0e05d9ef83171e7b42cebf76a482a4111c8d672c0f76dce ;;
  42) EXPECTED_SHA256=596349bdb536b413b3853106a7ae75613528a082ce84e9063b41735ece6bb185 ;;
  43) EXPECTED_SHA256=a82def7273a8ea0abe6ff171958ed0c4b5527b0f3d7c96a77474f363f1df6253 ;;
  *) echo "seed must be one of 41, 42, or 43" >&2; exit 2 ;;
esac
if [[ ! "$GPU_ID" =~ ^[0-7]$ ]]; then
  echo "GPU_ID must be in [0,7]" >&2
  exit 2
fi
if [ -n "$MAX_GROUPS" ] && [[ ! "$MAX_GROUPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_GROUPS must be positive" >&2
  exit 2
fi

CHECKPOINT="$BASELINE_VIEW_ROOT/fresh_closed_loop_repair_step3451_seed${SEED}/final_model"
TAG="counterfactual_feedback_revision_train_seed${SEED}"
if [ -n "$MAX_GROUPS" ]; then
  TAG="${TAG}_smoke${MAX_GROUPS}"
fi
OUTPUT="$OUTPUT_ROOT/${TAG}.json"
LOG_DIR="$OUTPUT_ROOT/logs"
SOCKET_PATH="/tmp/fresh-feedback-revision-${SEED}-$$.sock"
KEEPALIVE_SESSION="gpu-keepalive-${GPU_ID}"
KEEPALIVE_WAS_RUNNING=0
SERVER_PID=""

if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  echo "missing selected repaired checkpoint: $CHECKPOINT/model.safetensors" >&2
  exit 1
fi
if [ -e "$OUTPUT" ]; then
  echo "refusing to overwrite diagnostic: $OUTPUT" >&2
  exit 1
fi

cd "$REPO_ROOT"
GIT_SHA=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "formal feedback-revision diagnostic requires a clean Git worktree" >&2
  exit 1
fi
ACTUAL_SHA256=$(sha256sum "$CHECKPOINT/model.safetensors" | awk '{print $1}')
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
  echo "checkpoint SHA256 mismatch for seed $SEED" >&2
  exit 1
fi

restore_runtime() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$SOCKET_PATH"
  if [ "$KEEPALIVE_WAS_RUNNING" = 1 ]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-1}" "${AI2R_KEEPALIVE_N:-8192}" \
      "$KEEPALIVE_SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap restore_runtime EXIT

if tmux has-session -t "$KEEPALIVE_SESSION" 2>/dev/null; then
  KEEPALIVE_WAS_RUNNING=1
  tmux kill-session -t "$KEEPALIVE_SESSION"
fi
mkdir -p "$LOG_DIR"
export PRETRAINED_MODELS_DIR
export ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1
export NO_ALBUMENTATIONS_UPDATE=1

CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONDONTWRITEBYTECODE=1 \
"$PYTHON" scripts/fresh_vla/pi05_policy_server.py \
  --checkpoint "$CHECKPOINT" \
  --socket "$SOCKET_PATH" \
  --device cuda:0 >"$LOG_DIR/${TAG}_server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 600); do
  if [ -S "$SOCKET_PATH" ]; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Pi0.5 policy server exited before becoming ready" >&2
    exit 1
  fi
  sleep 1
done
if [ ! -S "$SOCKET_PATH" ]; then
  echo "timed out waiting for Pi0.5 policy server" >&2
  exit 1
fi

max_group_args=()
if [ -n "$MAX_GROUPS" ]; then
  max_group_args=(--max-groups "$MAX_GROUPS")
fi

FRESH_GIT_SHA="$GIT_SHA" \
FRESH_GIT_DIRTY=0 \
PYTHONPATH="$REPO_ROOT/scripts/fresh_vla${PYTHONPATH:+:$PYTHONPATH}" \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONDONTWRITEBYTECODE=1 \
"$PYTHON" scripts/fresh_vla/diagnose_counterfactual_feedback_revision.py \
  --policy-socket "$SOCKET_PATH" \
  --episode-root "$EPISODE_ROOT" \
  --output "$OUTPUT" \
  --policy-seed "$SEED" \
  --split train \
  "${max_group_args[@]}" \
  2>&1 | tee "$LOG_DIR/${TAG}.log"
