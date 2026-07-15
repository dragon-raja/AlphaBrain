#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${FRESH_RECOVERY_SUPPORT_CONFIG:-$REPO_ROOT/configs/experiments/fresh_vla_libero_closed_loop.yaml}
PYTHON=${FRESH_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
BASELINE_ROOT=${FRESH_BASELINE_ROOT:-/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2}
OUTPUT_ROOT=${FRESH_RECOVERY_SUPPORT_ROOT:-/share/longjunyu/fresh-vla/runs/recovery-support-v1}
PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}

ARM=${1:?usage: run_recovery_support_train.sh ARM SEED GPU_ID STEPS}
SEED=${2:?usage: run_recovery_support_train.sh ARM SEED GPU_ID STEPS}
GPU_ID=${3:?usage: run_recovery_support_train.sh ARM SEED GPU_ID STEPS}
STEPS=${4:?usage: run_recovery_support_train.sh ARM SEED GPU_ID STEPS}

if [ "$ARM" != base_continuation ]; then
  echo "only the preregistered base_continuation calibration arm is enabled" >&2
  exit 2
fi
case "$SEED" in
  41) EXPECTED_SHA256=144a3b3d3dcc8421418564a62059a1038c9a7ef3196ac157f5f9ea1997a31f30 ;;
  42) EXPECTED_SHA256=98dc52d2ed1983776d218fee7666f3131053d1a55296e93e9f521b1c088ce875 ;;
  43) EXPECTED_SHA256=5db16350d9835c1f28d01b660dd6e9234bcab3da79abbce1f092e92b08ac9149 ;;
  *) echo "seed must be one of 41, 42, or 43" >&2; exit 2 ;;
esac
if [[ ! "$GPU_ID" =~ ^[0-7]$ ]] || [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU_ID must be in [0,7] and STEPS must be positive" >&2
  exit 2
fi

CHECKPOINT="$BASELINE_ROOT/fresh_closed_loop_full_h_seed${SEED}/final_model"
RUN_ID="recovery_support_${ARM}_seed${SEED}_steps${STEPS}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
KEEPALIVE_SESSION="gpu-keepalive-${GPU_ID}"
KEEPALIVE_WAS_RUNNING=0

if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  echo "missing baseline checkpoint: $CHECKPOINT/model.safetensors" >&2
  exit 1
fi
if [ -e "$OUTPUT_DIR" ]; then
  echo "refusing to overwrite existing run: $OUTPUT_DIR" >&2
  exit 1
fi

cd "$REPO_ROOT"
GIT_SHA=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "formal recovery-support training requires a clean Git worktree" >&2
  exit 1
fi

restore_keepalive() {
  if [ "$KEEPALIVE_WAS_RUNNING" = 1 ]; then
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
"$PYTHON" - "$OUTPUT_DIR/run_identity.json" <<PY
import json, pathlib
path = pathlib.Path(__import__('sys').argv[1])
path.write_text(json.dumps({
    "schema_version": 1,
    "arm": "$ARM",
    "seed": $SEED,
    "steps": $STEPS,
    "git_sha": "$GIT_SHA",
    "git_dirty_at_launch": False,
    "initial_checkpoint": "$CHECKPOINT",
    "initial_checkpoint_sha256": "$EXPECTED_SHA256",
    "checkpoint_load_format_required": "alphabrain_native",
    "optimizer_state": "reset_equally_for_all_arms",
    "learning_rate": 1.0e-5,
    "minimum_learning_rate": 2.0e-6,
    "warmup_steps": 100,
    "dataset": "/share/longjunyu/fresh-vla/libero-full-episode-windows-v2-128",
}, indent=2, sort_keys=True) + "\n")
PY

PORT=$((30600 + (SEED % 100) * 10 + GPU_ID))
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
  --mode fresh_closed_loop_full_h \
  "run_id=$RUN_ID" \
  "seed=$SEED" \
  "output_root_dir=$OUTPUT_ROOT" \
  "trainer.pretrained_checkpoint=$CHECKPOINT" \
  "trainer.learning_rate.base=1.0e-5" \
  "trainer.learning_rate.action_model=1.0e-5" \
  "trainer.learning_rate.paligemma_vl_interface=1.0e-5" \
  "trainer.num_warmup_steps=100" \
  "trainer.scheduler_specific_kwargs.min_lr=2.0e-6" \
  "trainer.max_train_steps=$STEPS" \
  "trainer.save_interval=$((STEPS + 1))" \
  "trainer.eval_interval=$((STEPS + 1))" \
  2>&1 | tee "$OUTPUT_DIR/launcher.log"

if ! grep -q 'Source format:  alphabrain_native' "$OUTPUT_DIR/launcher.log"; then
  echo "training did not confirm native checkpoint loading" >&2
  exit 1
fi
if ! grep -q 'Matched:        827/827' "$OUTPUT_DIR/launcher.log"; then
  echo "training did not load all 827 checkpoint keys" >&2
  exit 1
fi
test -f "$OUTPUT_DIR/final_model/model.safetensors"
