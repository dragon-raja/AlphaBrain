#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${KYC_SCALING_DATA_ROOT:-/share/longjunyu/cabi-vla/kyc-scaling-v3}
RUN_ROOT=${KYC_OUTPUT_ROOT:-$DATA_ROOT/runs}
EVAL_ROOT=${KYC_SCALING_EVAL_ROOT:-$DATA_ROOT/eval/stage-b1}
STEPS=${KYC_SCALING_STEPS:-33000}

CATALOG_SIZE=${1:?usage: run_kyc_scaling_eval.sh CATALOG_SIZE ARM SEED GPU_ID}
ARM=${2:?usage: run_kyc_scaling_eval.sh CATALOG_SIZE ARM SEED GPU_ID}
SEED=${3:?usage: run_kyc_scaling_eval.sh CATALOG_SIZE ARM SEED GPU_ID}
GPU_ID=${4:?usage: run_kyc_scaling_eval.sh CATALOG_SIZE ARM SEED GPU_ID}

case "$CATALOG_SIZE" in
  10|45|215|1000) ;;
  *)
    echo "CATALOG_SIZE must be one of 10, 45, 215, or 1000" >&2
    exit 2
    ;;
esac
case "$ARM" in
  poseaug_rgb|poseaug_control|kyc) ;;
  *)
    echo "ARM must be poseaug_rgb, poseaug_control, or kyc" >&2
    exit 2
    ;;
esac

TAG="scale-n${CATALOG_SIZE}-fixed-wrist-on"
RUN_ID="kyc_${ARM}_${TAG}_h20_seed${SEED}_steps${STEPS}"
CHECKPOINT="$RUN_ROOT/$RUN_ID/final_model"
RUN_NAME="n${CATALOG_SIZE}-${ARM}-s${SEED}-fixed-wrist-on"

if [[ ! -s "$CHECKPOINT/model.safetensors" ]]; then
  echo "missing completed scaling checkpoint: $CHECKPOINT/model.safetensors" >&2
  exit 1
fi

export CABI_CAMERA_CONFIG=$REPO_ROOT/docs/cabi_vla/configs/camera_pose_policy_gate_v6.json
export CABI_CAMERA_OUTPUT_ROOT=$EVAL_ROOT/n${CATALOG_SIZE}
export CABI_EVAL_SPLIT=test
export CABI_EVAL_STATE_INDICES=40,41,42,43,44,45,46,47,48,49
export CABI_EVAL_EDGES=${CABI_EVAL_EDGES:-red-left,red-right,white-left,yellow_white-right}
export CABI_EVAL_HORIZONS=3
export CABI_EVAL_MAX_STEPS=320
export CABI_EVAL_SEED=20260722
export CABI_SCENE_CUE_MODE=fixed
export CABI_SCENE_CUE_SEED=20260728
export FRESH_TORCH_NUM_THREADS=${FRESH_TORCH_NUM_THREADS:-4}
export FRESH_TORCH_INTEROP_THREADS=${FRESH_TORCH_INTEROP_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false

exec "$REPO_ROOT/scripts/cabi_vla/run_libero_bind_camera_sweep.sh" \
  "$CHECKPOINT" "$RUN_NAME" "$GPU_ID"
