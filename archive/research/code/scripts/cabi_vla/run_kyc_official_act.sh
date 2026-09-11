#!/usr/bin/env bash
set -euo pipefail

OFFICIAL_ROOT=${KYC_OFFICIAL_ROOT:-/workspace/ai2r/vendor/CamPoseOpensource}
PYTHON=${KYC_OFFICIAL_PYTHON:-/workspace/ai2r/venvs/kyc-official/bin/python}
DATASET_DIR=${KYC_OFFICIAL_DATASET_DIR:-/share/longjunyu/kyc-official-data/drive/demos_robosuite}
OUTPUT_ROOT=${KYC_OFFICIAL_OUTPUT_ROOT:-/share/longjunyu/kyc-official-data/runs}

ARM=${1:?usage: run_kyc_official_act.sh ARM SEED GPU_ID}
SEED=${2:?usage: run_kyc_official_act.sh ARM SEED GPU_ID}
GPU_ID=${3:?usage: run_kyc_official_act.sh ARM SEED GPU_ID}

case "$ARM" in
  image)
    USE_PLUCKER=0
    ;;
  kyc)
    USE_PLUCKER=1
    ;;
  *)
    echo "ARM must be image or kyc" >&2
    exit 2
    ;;
esac
if [[ ! "$SEED" =~ ^[0-9]+$ || ! "$GPU_ID" =~ ^[0-7]$ ]]; then
  echo "SEED must be non-negative and GPU_ID must be in [0, 7]" >&2
  exit 2
fi

RUN_NAME="official_act_lift_randomized_${ARM}_seed${SEED}"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
LOG="$RUN_DIR/launcher.log"
FINAL_CHECKPOINT="$RUN_DIR/epoch_20000.pth"
KEEPALIVE_SESSION="gpu-keepalive-${GPU_ID}"
KEEPALIVE_WAS_RUNNING=0

if [[ ! -x "$PYTHON" ]]; then
  echo "missing official KYC Python: $PYTHON" >&2
  exit 1
fi
if [[ ! -s "$DATASET_DIR/liftrand_eef_delta.hdf5" ]]; then
  echo "missing official Lift demonstration file" >&2
  exit 1
fi
if [[ ! -d "$OFFICIAL_ROOT/robosuite_source/robosuite" ]]; then
  echo "missing pinned official RoboSuite submodule" >&2
  exit 1
fi
if [[ -s "$FINAL_CHECKPOINT" ]]; then
  echo "already complete: $FINAL_CHECKPOINT"
  exit 0
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

mkdir -p "$RUN_DIR"
exec > >(tee -a "$LOG") 2>&1

cd "$OFFICIAL_ROOT/policy_robosuite"
export WANDB_MODE=disabled
export WANDB_SILENT=true
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${KYC_OFFICIAL_CPU_THREADS:-8}
export MKL_NUM_THREADS=${KYC_OFFICIAL_CPU_THREADS:-8}
export OPENBLAS_NUM_THREADS=${KYC_OFFICIAL_CPU_THREADS:-8}
export NUMEXPR_NUM_THREADS=${KYC_OFFICIAL_CPU_THREADS:-8}

while [[ ! -s "$FINAL_CHECKPOINT" ]]; do
  "$PYTHON" train.py \
    --name "$RUN_NAME" \
    --dataset_dir "$DATASET_DIR" \
    --camera_poses_dir "$OFFICIAL_ROOT/policy_robosuite/camera_poses" \
    --ckpt_dir "$RUN_DIR" \
    --seed "$SEED" \
    --use_plucker "$USE_PLUCKER" \
    --num_epochs 20001 \
    --eval_start_epoch 10000 \
    --eval_max_steps 150
done

echo "complete: $FINAL_CHECKPOINT"
