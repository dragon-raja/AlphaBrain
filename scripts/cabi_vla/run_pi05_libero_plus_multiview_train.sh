#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${PLUS_MV_CONFIG:-$REPO_ROOT/configs/experiments/pi05_libero_plus_multiview.yaml}
PYTHON=${PLUS_MV_PYTHON:-$REPO_ROOT/.venv/bin/python}
DATA_ROOT=${PLUS_MV_DATA_ROOT:-/share/longjunyu/alphabrain/datasets/libero-plus/views/pi05-mv-rgb-v1}
OUTPUT_ROOT=${PLUS_MV_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/libero-plus-mv-rgb-v1/runs}
BUDGET_FRACTION=${PLUS_MV_BUDGET_FRACTION:-1.0}
SKIP_FINAL_SAVE=${PLUS_MV_SKIP_FINAL_SAVE:-false}
RUN_TAG=${PLUS_MV_RUN_TAG:-}

ARM=${1:?usage: run_pi05_libero_plus_multiview_train.sh ARM SEED GPU_ID [STEPS]}
SEED=${2:?usage: run_pi05_libero_plus_multiview_train.sh ARM SEED GPU_ID [STEPS]}
GPU_ID=${3:?usage: run_pi05_libero_plus_multiview_train.sh ARM SEED GPU_ID [STEPS]}
STEPS=${4:-33000}

case "$ARM" in
  action_only)
    MODE=pi05_plus_mv_rgb_action_only
    ;;
  visual_lora)
    MODE=pi05_plus_mv_rgb_visual_lora
    ;;
  *)
    echo "unknown LIBERO-Plus multiview arm: $ARM" >&2
    exit 2
    ;;
esac

if [[ ! "$SEED" =~ ^[0-9]+$ || ! "$GPU_ID" =~ ^[0-7]$ || ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEED/STEPS must be non-negative integers and GPU_ID must be in [0,7]" >&2
  exit 2
fi
if [[ -n "$RUN_TAG" && ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "PLUS_MV_RUN_TAG contains unsafe filename characters" >&2
  exit 2
fi
if [[ "$SKIP_FINAL_SAVE" != true && "$SKIP_FINAL_SAVE" != false ]]; then
  echo "PLUS_MV_SKIP_FINAL_SAVE must be true or false" >&2
  exit 2
fi

TAG_SEGMENT=""
if [[ -n "$RUN_TAG" ]]; then
  TAG_SEGMENT="_${RUN_TAG}"
fi
RUN_ID="pi05_plus_mv_${ARM}${TAG_SEGMENT}_seed${SEED}_steps${STEPS}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
KEEPALIVE_SESSION="gpu-keepalive-${GPU_ID}"
KEEPALIVE_WAS_RUNNING=0

if [[ ! -x "$PYTHON" ]]; then
  echo "missing AlphaBrain Python: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "missing training config: $CONFIG" >&2
  exit 1
fi
if [[ ! -s "$DATA_ROOT/manifest.json" ]]; then
  echo "missing indexed LIBERO-Plus training view: $DATA_ROOT/manifest.json" >&2
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
PORT=$((30900 + (SEED % 100) * 10 + GPU_ID))
LOG="$OUTPUT_DIR/launcher.log"

cd "$REPO_ROOT"
export PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}
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
  "datasets.vla_data.data_root_dir=$DATA_ROOT" \
  "datasets.vla_data.budget_fraction=$BUDGET_FRACTION" \
  "trainer.skip_final_save=$SKIP_FINAL_SAVE" \
  "trainer.max_train_steps=$STEPS" \
  "trainer.save_interval=$((STEPS + 1))" \
  "trainer.eval_interval=$((STEPS + 1))" \
  2>&1 | tee "$LOG"
