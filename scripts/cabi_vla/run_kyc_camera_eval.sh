#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUTPUT_BASE=${KYC_CAMERA_EVAL_ROOT:-/share/longjunyu/cabi-vla/kyc-camera-eval-v1}
KYC_RUN_ROOT=${KYC_OUTPUT_ROOT:-/share/longjunyu/cabi-vla/kyc-runs}
BASE_CHECKPOINT=${KYC_BASE_CHECKPOINT:-/share/longjunyu/cabi-vla/runs/cabi_bind_pi05_action_bridge_h20_smoke_seed41_steps33000_h20-bridge-edge-balanced-3epoch-v15/final_model}

SCOPE=${1:?usage: run_kyc_camera_eval.sh SCOPE METHOD SEED GPU_ID}
METHOD=${2:?usage: run_kyc_camera_eval.sh SCOPE METHOD SEED GPU_ID}
SEED=${3:?usage: run_kyc_camera_eval.sh SCOPE METHOD SEED GPU_ID}
GPU_ID=${4:?usage: run_kyc_camera_eval.sh SCOPE METHOD SEED GPU_ID}

case "$SCOPE" in
  dense)
    CAMERA_CONFIG=$REPO_ROOT/docs/cabi_vla/configs/camera_pose_policy_dense_v2.json
    STATE_INDICES=40
    ;;
  gate)
    CAMERA_CONFIG=$REPO_ROOT/docs/cabi_vla/configs/camera_pose_policy_gate_v6.json
    STATE_INDICES=40,41,42,43,44,45,46,47,48,49
    ;;
  *)
    echo "unknown evaluation scope: $SCOPE" >&2
    exit 2
    ;;
esac

case "$METHOD" in
  base)
    CHECKPOINT=$BASE_CHECKPOINT
    ;;
  poseaug_rgb|pm_fixed|poseaug_control|kyc)
    CHECKPOINT=$KYC_RUN_ROOT/kyc_${METHOD}_h20_seed${SEED}_steps33000/final_model
    ;;
  *)
    echo "unknown KYC method: $METHOD" >&2
    exit 2
    ;;
esac

if [[ ! -s "$CHECKPOINT/model.safetensors" ]]; then
  echo "missing completed checkpoint: $CHECKPOINT/model.safetensors" >&2
  exit 1
fi

RUN_NAME="${METHOD}_s${SEED}_${SCOPE}"
export CABI_CAMERA_CONFIG=$CAMERA_CONFIG
export CABI_CAMERA_OUTPUT_ROOT=$OUTPUT_BASE/$SCOPE
export CABI_EVAL_SPLIT=test
export CABI_EVAL_STATE_INDICES=$STATE_INDICES
export CABI_EVAL_EDGES=${CABI_EVAL_EDGES:-red-left,red-right,white-left,yellow_white-right}
export CABI_EVAL_HORIZONS=${CABI_EVAL_HORIZONS:-3}
export CABI_EVAL_MAX_STEPS=${CABI_EVAL_MAX_STEPS:-320}
export CABI_EVAL_SEED=${CABI_EVAL_SEED:-20260722}
export FRESH_TORCH_NUM_THREADS=${FRESH_TORCH_NUM_THREADS:-4}
export FRESH_TORCH_INTEROP_THREADS=${FRESH_TORCH_INTEROP_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false

exec "$REPO_ROOT/scripts/cabi_vla/run_libero_bind_camera_sweep.sh" \
  "$CHECKPOINT" "$RUN_NAME" "$GPU_ID"
