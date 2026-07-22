#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${CABI_TRAIN_CONFIG:-$REPO_ROOT/configs/experiments/cabi_vla_libero_bind.yaml}
PYTHON=${CABI_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}
OUTPUT_ROOT=${CABI_TRAIN_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/runs}
DATA_ROOT=${CABI_DATA_ROOT:-/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v5-loss-balanced}

MODE=${1:?usage: run_pilot_train.sh MODE SEED GPU_ID [STEPS]}
SEED=${2:?usage: run_pilot_train.sh MODE SEED GPU_ID [STEPS]}
GPU_ID=${3:?usage: run_pilot_train.sh MODE SEED GPU_ID [STEPS]}
STEPS=${4:-100}
RUN_TAG=${CABI_RUN_TAG:-}
if [[ -n "$RUN_TAG" && ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "CABI_RUN_TAG may only contain letters, numbers, dots, underscores, and dashes" >&2
  exit 2
fi
RUN_ID="${MODE}_seed${SEED}_steps${STEPS}${RUN_TAG:+_$RUN_TAG}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
KEEPALIVE_SESSION="gpu-keepalive-${GPU_ID}"
KEEPALIVE_WAS_RUNNING=0

if [[ ! -x "$PYTHON" ]]; then
  echo "missing AlphaBrain Python: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "missing CABI config: $CONFIG" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to overwrite existing run: $OUTPUT_DIR" >&2
  exit 1
fi

restore_keepalive() {
  if [[ "$KEEPALIVE_WAS_RUNNING" == "1" ]]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-1}" "${AI2R_KEEPALIVE_N:-8192}" \
      "$KEEPALIVE_SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap restore_keepalive EXIT

if tmux has-session -t "$KEEPALIVE_SESSION" 2>/dev/null; then
  KEEPALIVE_WAS_RUNNING=1
  tmux kill-session -t "$KEEPALIVE_SESSION"
fi

mkdir -p "$OUTPUT_DIR"
PORT=$((29710 + (SEED % 100) * 10 + GPU_ID))
LOG="$OUTPUT_DIR/launcher.log"

cd "$REPO_ROOT"
export PRETRAINED_MODELS_DIR
export ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

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
  "data_root=$DATA_ROOT" \
  "datasets.vla_data.data_root_dir=$DATA_ROOT" \
  "trainer.max_train_steps=$STEPS" \
  "trainer.save_interval=$((STEPS + 1))" \
  "trainer.eval_interval=$((STEPS + 1))" \
  2>&1 | tee "$LOG"
