#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${KYC_TRAIN_CONFIG:-$REPO_ROOT/configs/experiments/kyc_libero_bind.yaml}
PYTHON=${KYC_TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}
OUTPUT_ROOT=${KYC_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/kyc-runs}
DATA_ROOT_OVERRIDE=${KYC_DATA_ROOT_OVERRIDE:-}
RUN_TAG=${KYC_RUN_TAG:-}
WRIST_MODE=${KYC_WRIST_MODE:-on}
CABI_ANCHOR_PERIOD=${KYC_CABI_ANCHOR_PERIOD:-32}

ARM=${1:?usage: run_kyc_train.sh ARM SEED GPU_ID [STEPS]}
SEED=${2:?usage: run_kyc_train.sh ARM SEED GPU_ID [STEPS]}
GPU_ID=${3:?usage: run_kyc_train.sh ARM SEED GPU_ID [STEPS]}
STEPS=${4:-33000}

case "$ARM" in
  poseaug_rgb)
    MODE=kyc_poseaug_rgb_h20
    DATA_ROOT=/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v17-camera-random-episode-pool-h20
    ;;
  poseaug_rgb_fla)
    MODE=kyc_poseaug_rgb_fla_h20
    DATA_ROOT=/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v17-camera-random-episode-pool-h20
    ;;
  pm_fixed)
    MODE=kyc_pm_fixed_h20
    DATA_ROOT=/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v15-decision-observed-edge-phase-loss-balanced-h20
    ;;
  poseaug_control)
    MODE=kyc_poseaug_control_h20
    DATA_ROOT=/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v17-camera-random-episode-pool-h20
    ;;
  poseaug_control_fla)
    MODE=kyc_poseaug_control_fla_h20
    DATA_ROOT=/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v17-camera-random-episode-pool-h20
    ;;
  kyc)
    MODE=kyc_real_h20
    DATA_ROOT=/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v17-camera-random-episode-pool-h20
    ;;
  kyc_fla)
    MODE=kyc_real_fla_h20
    DATA_ROOT=/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v17-camera-random-episode-pool-h20
    ;;
  dual_rgb_fla)
    MODE=kyc_dual_rgb_fla_h20
    DATA_ROOT=/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v18b-dual-camera-random-episode-pool-h20
    ;;
  dual_control_fla)
    MODE=kyc_dual_control_fla_h20
    DATA_ROOT=/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v18b-dual-camera-random-episode-pool-h20
    ;;
  external_fla)
    MODE=kyc_external_fla_h20
    DATA_ROOT=/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v18b-dual-camera-random-episode-pool-h20
    ;;
  wrist_fla)
    MODE=kyc_wrist_fla_h20
    DATA_ROOT=/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v18b-dual-camera-random-episode-pool-h20
    ;;
  dual_fla)
    MODE=kyc_dual_fla_h20
    DATA_ROOT=/share/longjunyu/cabi-vla/libero-bind-v0-train-view-v18b-dual-camera-random-episode-pool-h20
    ;;
  *)
    echo "unknown KYC arm: $ARM" >&2
    exit 2
    ;;
esac

if [[ -n "$DATA_ROOT_OVERRIDE" ]]; then
  DATA_ROOT=$DATA_ROOT_OVERRIDE
fi
if [[ "$WRIST_MODE" != on && "$WRIST_MODE" != off ]]; then
  echo "KYC_WRIST_MODE must be on or off" >&2
  exit 2
fi
if [[ ! "$CABI_ANCHOR_PERIOD" =~ ^[1-9][0-9]*$ ]]; then
  echo "KYC_CABI_ANCHOR_PERIOD must be a positive integer" >&2
  exit 2
fi
if [[ -n "$RUN_TAG" && ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "KYC_RUN_TAG must contain only safe filename characters" >&2
  exit 2
fi

tag_segment=""
if [[ -n "$RUN_TAG" ]]; then
  tag_segment="_${RUN_TAG}"
fi
RUN_ID="kyc_${ARM}${tag_segment}_h20_seed${SEED}_steps${STEPS}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
KEEPALIVE_SESSION="gpu-keepalive-${GPU_ID}"
KEEPALIVE_WAS_RUNNING=0

if [[ ! -x "$PYTHON" ]]; then
  echo "missing AlphaBrain Python: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "missing KYC config: $CONFIG" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "missing KYC data root: $DATA_ROOT" >&2
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
PORT=$((29900 + (SEED % 100) * 10 + GPU_ID))
LOG="$OUTPUT_DIR/launcher.log"

cd "$REPO_ROOT"
export PRETRAINED_MODELS_DIR=${PRETRAINED_MODELS_DIR:-/share/longjunyu/alphabrain/pretrained_models}
export ALPHABRAIN_DISABLE_AUTO_DOWNLOAD=1
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

image_mask="[true,true,false]"
if [[ "$WRIST_MODE" == off ]]; then
  image_mask="[true,false,false]"
fi

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
  "datasets.vla_data.cabi_anchor_period=$CABI_ANCHOR_PERIOD" \
  "framework.paligemma.image_mask=$image_mask" \
  "trainer.max_train_steps=$STEPS" \
  "trainer.save_interval=$((STEPS + 1))" \
  "trainer.eval_interval=$((STEPS + 1))" \
  2>&1 | tee "$LOG"
