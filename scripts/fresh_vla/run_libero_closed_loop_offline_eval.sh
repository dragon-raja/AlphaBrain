#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
OUTPUT_ROOT=${FRESH_CLOSED_LOOP_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2}
DATA_ROOT=${FRESH_WINDOW_ROOT:-/share/longjunyu/fresh-vla/libero-full-episode-windows-v2-128}
PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}

METHOD=${1:?usage: run_libero_closed_loop_offline_eval.sh METHOD SEED GPU_ID}
SEED=${2:?usage: run_libero_closed_loop_offline_eval.sh METHOD SEED GPU_ID}
GPU_ID=${3:?usage: run_libero_closed_loop_offline_eval.sh METHOD SEED GPU_ID}
RUN_DIR="$OUTPUT_ROOT/fresh_closed_loop_${METHOD}_seed${SEED}"
CHECKPOINT="$RUN_DIR/final_model"
SESSION="gpu-keepalive-${GPU_ID}"
WAS_RUNNING=0

if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  echo "missing checkpoint: $CHECKPOINT/model.safetensors" >&2
  exit 1
fi

restore_keepalive() {
  if [ "$WAS_RUNNING" = 1 ]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start.sh \
      "${AI2R_KEEPALIVE_EXTRA_GIB:-1}" "${AI2R_KEEPALIVE_N:-8192}" "$SESSION" "$GPU_ID" >/dev/null || true
  fi
}
trap restore_keepalive EXIT
if tmux has-session -t "$SESSION" 2>/dev/null; then
  WAS_RUNNING=1
  tmux kill-session -t "$SESSION"
fi

cd "$REPO_ROOT"
export PRETRAINED_MODELS_DIR ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1 NO_ALBUMENTATIONS_UPDATE=1
if [ ! -f "$RUN_DIR/offline_eval.json" ]; then
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
    scripts/fresh_vla/evaluate_libero_offline.py \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --dataset-format episode_window \
    --tasks grasp_slip_full_episode \
    --output "$RUN_DIR/offline_eval.json" \
    --max-samples "${FRESH_OFFLINE_MAX_SAMPLES:-256}" \
    --device cuda:0 \
    --seed "$((9107 + SEED))"
fi
if [ ! -f "$RUN_DIR/mode_coverage.json" ]; then
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
    scripts/fresh_vla/sample_libero_multimodal.py \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --dataset-format episode_window \
    --tasks grasp_slip_full_episode \
    --output "$RUN_DIR/mode_coverage.json" \
    --samples-per-context "${FRESH_MODE_SAMPLES:-32}" \
    --max-contexts "${FRESH_MODE_CONTEXTS:-8}" \
    --device cuda:0 \
    --seed "$((27183 + SEED))"
fi
