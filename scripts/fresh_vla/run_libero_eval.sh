#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}
OUTPUT_ROOT=${FRESH_TRAIN_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-counterfactual-v1}
DATA_ROOT=${FRESH_DATA_ROOT:-/share/longjunyu/fresh-vla/libero-counterfactual-v1-128}

METHOD=${1:?usage: run_libero_eval.sh METHOD SEED GPU_ID}
SEED=${2:?usage: run_libero_eval.sh METHOD SEED GPU_ID}
GPU_ID=${3:?usage: run_libero_eval.sh METHOD SEED GPU_ID}
RUN_DIR="$OUTPUT_ROOT/fresh_libero_${METHOD}_seed${SEED}"
CHECKPOINT="$RUN_DIR/final_model"
OUTPUT="$RUN_DIR/offline_eval.json"
CONTROL_OUTPUT="$RUN_DIR/deterministic_control_eval.json"
KEEPALIVE_SESSION="gpu-keepalive-${GPU_ID}"
KEEPALIVE_WAS_RUNNING=0

if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  echo "missing trained checkpoint: $CHECKPOINT/model.safetensors" >&2
  exit 1
fi
if [ -e "$OUTPUT" ] && [ -e "$CONTROL_OUTPUT" ]; then
  echo "both evaluations already exist: $RUN_DIR"
  exit 0
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
run_evaluation() {
  local output=$1
  local task=$2
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PRETRAINED_MODELS_DIR="$PRETRAINED_MODELS_DIR" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" scripts/fresh_vla/evaluate_libero_offline.py \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --output "$output" \
    --split test \
    --tasks "$task" \
    --fixed-k 2 \
    --device cuda:0
}

if [ ! -e "$OUTPUT" ] && [ ! -e "$CONTROL_OUTPUT" ]; then
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PRETRAINED_MODELS_DIR="$PRETRAINED_MODELS_DIR" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" scripts/fresh_vla/evaluate_libero_offline.py \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --output "$OUTPUT" \
    --control-output "$CONTROL_OUTPUT" \
    --split test \
    --tasks grasp_slip \
    --fixed-k 2 \
    --device cuda:0
elif [ ! -e "$OUTPUT" ]; then
  run_evaluation "$OUTPUT" grasp_slip
elif [ ! -e "$CONTROL_OUTPUT" ]; then
  run_evaluation "$CONTROL_OUTPUT" deterministic_reach
fi
