#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${KYC_SCALING_DATA_ROOT:-/share/longjunyu/cabi-vla/kyc-scaling-v3}
RUN_ROOT=${KYC_OUTPUT_ROOT:-$DATA_ROOT/runs}
EVAL_ROOT=${KYC_FACTORIAL_EVAL_ROOT:-$DATA_ROOT/eval/factorial}
STEPS=${KYC_SCALING_STEPS:-33000}

CATALOG_SIZE=${1:?usage: run_kyc_factorial_eval.sh CATALOG_SIZE TRAIN_SCENE WRIST ARM SEED EVAL_SCENE GPU_ID}
TRAIN_SCENE=${2:?usage: run_kyc_factorial_eval.sh CATALOG_SIZE TRAIN_SCENE WRIST ARM SEED EVAL_SCENE GPU_ID}
WRIST=${3:?usage: run_kyc_factorial_eval.sh CATALOG_SIZE TRAIN_SCENE WRIST ARM SEED EVAL_SCENE GPU_ID}
ARM=${4:?usage: run_kyc_factorial_eval.sh CATALOG_SIZE TRAIN_SCENE WRIST ARM SEED EVAL_SCENE GPU_ID}
SEED=${5:?usage: run_kyc_factorial_eval.sh CATALOG_SIZE TRAIN_SCENE WRIST ARM SEED EVAL_SCENE GPU_ID}
EVAL_SCENE=${6:?usage: run_kyc_factorial_eval.sh CATALOG_SIZE TRAIN_SCENE WRIST ARM SEED EVAL_SCENE GPU_ID}
GPU_ID=${7:?usage: run_kyc_factorial_eval.sh CATALOG_SIZE TRAIN_SCENE WRIST ARM SEED EVAL_SCENE GPU_ID}

if [[ ! "$CATALOG_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "CATALOG_SIZE must be a positive integer" >&2
  exit 2
fi
for scene in "$TRAIN_SCENE" "$EVAL_SCENE"; do
  if [[ "$scene" != fixed && "$scene" != cue_randomized ]]; then
    echo "train and evaluation scenes must be fixed or cue_randomized" >&2
    exit 2
  fi
done
if [[ "$WRIST" != on && "$WRIST" != off ]]; then
  echo "WRIST must be on or off" >&2
  exit 2
fi
if [[ "$ARM" != poseaug_control && "$ARM" != kyc ]]; then
  echo "ARM must be poseaug_control or kyc" >&2
  exit 2
fi
if [[ ! "$SEED" =~ ^[0-9]+$ || ! "$GPU_ID" =~ ^[0-7]$ ]]; then
  echo "SEED must be non-negative and GPU_ID must be in [0, 7]" >&2
  exit 2
fi

if [[ "$TRAIN_SCENE" == fixed && "$WRIST" == on ]]; then
  tag="scale-n${CATALOG_SIZE}-fixed-wrist-on"
else
  tag="factorial-n${CATALOG_SIZE}-${TRAIN_SCENE}-wrist-${WRIST}"
fi
run_id="kyc_${ARM}_${tag}_h20_seed${SEED}_steps${STEPS}"
checkpoint="$RUN_ROOT/$run_id/final_model"
if [[ ! -s "$checkpoint/model.safetensors" ]]; then
  echo "missing completed factorial checkpoint: $checkpoint/model.safetensors" >&2
  exit 1
fi

case "$TRAIN_SCENE" in
  fixed) train_scene_tag=fx ;;
  cue_randomized) train_scene_tag=cue ;;
esac
case "$EVAL_SCENE" in
  fixed) eval_scene_tag=fx ;;
  cue_randomized) eval_scene_tag=cue ;;
esac
case "$ARM" in
  poseaug_control) arm_tag=ctrl ;;
  kyc) arm_tag=kyc ;;
esac
run_name="n${CATALOG_SIZE}-tr${train_scene_tag}-w${WRIST}-m${arm_tag}-s${SEED}-ev${eval_scene_tag}"
export CABI_CAMERA_CONFIG=$REPO_ROOT/docs/cabi_vla/configs/camera_pose_policy_gate_v6.json
export CABI_CAMERA_OUTPUT_ROOT=$EVAL_ROOT/n${CATALOG_SIZE}
export CABI_EVAL_SPLIT=test
export CABI_EVAL_STATE_INDICES=40,41,42,43,44,45,46,47,48,49
export CABI_EVAL_EDGES=${CABI_EVAL_EDGES:-red-left,red-right,white-left,yellow_white-right}
export CABI_EVAL_HORIZONS=3
export CABI_EVAL_MAX_STEPS=320
export CABI_EVAL_SEED=20260722
export CABI_SCENE_CUE_MODE=$EVAL_SCENE
export CABI_SCENE_CUE_SEED=20260728
export FRESH_TORCH_NUM_THREADS=${FRESH_TORCH_NUM_THREADS:-4}
export FRESH_TORCH_INTEROP_THREADS=${FRESH_TORCH_INTEROP_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false

exec "$REPO_ROOT/scripts/cabi_vla/run_libero_bind_camera_sweep.sh" \
  "$checkpoint" "$run_name" "$GPU_ID"
