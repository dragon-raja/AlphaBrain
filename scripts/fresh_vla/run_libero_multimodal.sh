#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
OUTPUT_ROOT=${FRESH_TRAIN_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-counterfactual-v1}
DATA_ROOT=${FRESH_DATA_ROOT:-/share/longjunyu/fresh-vla/libero-counterfactual-v1-128}
PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}
METHOD=${1:?usage: run_libero_multimodal.sh METHOD SEED GPU_ID}
SEED=${2:?usage: run_libero_multimodal.sh METHOD SEED GPU_ID}
GPU_ID=${3:?usage: run_libero_multimodal.sh METHOD SEED GPU_ID}

RUN_DIR="$OUTPUT_ROOT/fresh_libero_${METHOD}_seed${SEED}"
CHECKPOINT="$RUN_DIR/final_model"
OUTPUT="$RUN_DIR/multimodal_sampling.json"
SESSION="gpu-keepalive-${GPU_ID}"
WAS_RUNNING=0

if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  echo "missing checkpoint: $CHECKPOINT/model.safetensors" >&2
  exit 1
fi
if [ -e "$OUTPUT" ]; then
  echo "refusing to overwrite existing sampling result: $OUTPUT" >&2
  exit 1
fi
restore_keepalive() {
  if [ "$WAS_RUNNING" = 1 ]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh 4 8192 "$SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap restore_keepalive EXIT
if tmux has-session -t "$SESSION" 2>/dev/null; then
  WAS_RUNNING=1
  tmux kill-session -t "$SESSION"
fi

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PRETRAINED_MODELS_DIR="$PRETRAINED_MODELS_DIR" \
PYTHONDONTWRITEBYTECODE=1 \
"$PYTHON" scripts/fresh_vla/sample_libero_multimodal.py \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT" \
  --device cuda:0 \
  --seed "$((27183 + SEED))" \
  --samples-per-context 32 \
  --max-contexts 8
