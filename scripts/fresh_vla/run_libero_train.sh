#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${FRESH_TRAIN_CONFIG:-$REPO_ROOT/configs/experiments/fresh_vla_libero_training.yaml}
PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}
OUTPUT_ROOT=${FRESH_TRAIN_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-counterfactual-v1}

METHOD=${1:?usage: run_libero_train.sh METHOD SEED GPU_ID [STEPS]}
SEED=${2:?usage: run_libero_train.sh METHOD SEED GPU_ID [STEPS]}
GPU_ID=${3:?usage: run_libero_train.sh METHOD SEED GPU_ID [STEPS]}
STEPS=${4:-1200}
MODE="fresh_libero_${METHOD}"
RUN_ID="${MODE}_seed${SEED}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
KEEPALIVE_SESSION="gpu-keepalive-${GPU_ID}"
KEEPALIVE_WAS_RUNNING=0

if [ ! -x "$PYTHON" ]; then
  echo "missing AlphaBrain Python: $PYTHON" >&2
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "missing training config: $CONFIG" >&2
  exit 1
fi
if [ -e "$OUTPUT_DIR" ]; then
  echo "refusing to overwrite existing run: $OUTPUT_DIR" >&2
  exit 1
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

mkdir -p "$OUTPUT_DIR"
PORT=$((29600 + (SEED % 100) * 10 + GPU_ID))
LOG="$OUTPUT_DIR/launcher.log"

cd "$REPO_ROOT"
export PRETRAINED_MODELS_DIR
export ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export ACCELERATE_CONFIG_FILE=configs/deepspeed/accelerate_1gpu_simple.yaml

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -m accelerate.commands.launch \
  --config_file configs/deepspeed/accelerate_1gpu_simple.yaml \
  --num_processes 1 \
  --main_process_port "$PORT" \
  AlphaBrain/training/train_alphabrain.py \
  --config_yaml "$CONFIG" \
  --mode "$MODE" \
  "run_id=$RUN_ID" \
  "seed=$SEED" \
  "output_root_dir=$OUTPUT_ROOT" \
  "trainer.max_train_steps=$STEPS" \
  "trainer.save_interval=$((STEPS + 1))" \
  "trainer.eval_interval=$((STEPS + 1))" \
  2>&1 | tee "$LOG"
