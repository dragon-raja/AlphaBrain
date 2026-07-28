#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${KYC_SCALING_DATA_ROOT:-/share/longjunyu/cabi-vla/kyc-scaling-v3}
OUTPUT_ROOT=${KYC_OUTPUT_ROOT:-$DATA_ROOT/runs}

CATALOG_SIZE=${1:?usage: run_kyc_factorial_train.sh CATALOG_SIZE SCENE WRIST ARM SEED GPU_ID [STEPS]}
SCENE=${2:?usage: run_kyc_factorial_train.sh CATALOG_SIZE SCENE WRIST ARM SEED GPU_ID [STEPS]}
WRIST=${3:?usage: run_kyc_factorial_train.sh CATALOG_SIZE SCENE WRIST ARM SEED GPU_ID [STEPS]}
ARM=${4:?usage: run_kyc_factorial_train.sh CATALOG_SIZE SCENE WRIST ARM SEED GPU_ID [STEPS]}
SEED=${5:?usage: run_kyc_factorial_train.sh CATALOG_SIZE SCENE WRIST ARM SEED GPU_ID [STEPS]}
GPU_ID=${6:?usage: run_kyc_factorial_train.sh CATALOG_SIZE SCENE WRIST ARM SEED GPU_ID [STEPS]}
STEPS=${7:-33000}

if [[ ! "$CATALOG_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "CATALOG_SIZE must be a positive integer" >&2
  exit 2
fi
case "$SCENE" in
  fixed)
    DATA_CELL="n${CATALOG_SIZE}-fixed"
    ;;
  cue_randomized)
    DATA_CELL="n${CATALOG_SIZE}-cue"
    ;;
  *)
    echo "SCENE must be fixed or cue_randomized" >&2
    exit 2
    ;;
esac
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
if [[ "$SCENE" == fixed && "$WRIST" == on ]]; then
  echo "fixed+wrist-on is the Stage B1 checkpoint and must not be retrained" >&2
  exit 2
fi

DATA_VIEW="$DATA_ROOT/views/libero-bind-kyc-${DATA_CELL}-h20"
if [[ ! -s "$DATA_VIEW/manifest.json" ]]; then
  echo "missing completed factorial data view: $DATA_VIEW" >&2
  exit 1
fi

export KYC_OUTPUT_ROOT=$OUTPUT_ROOT
export KYC_DATA_ROOT_OVERRIDE=$DATA_VIEW
export KYC_RUN_TAG="factorial-n${CATALOG_SIZE}-${SCENE}-wrist-${WRIST}"
export KYC_WRIST_MODE=$WRIST
export KYC_CABI_ANCHOR_PERIOD=1000000

exec "$REPO_ROOT/scripts/cabi_vla/run_kyc_train.sh" \
  "$ARM" "$SEED" "$GPU_ID" "$STEPS"
