#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SEED=${1:-41}
STEPS=${2:-20}
RUN_TAG=${3:-quickgate20-v1}
NUM_GPUS=${DSOL_SMOKE_NUM_GPUS:-8}
DATA_ROOT=${DSOL_PAIR_DATA_ROOT:-/share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad32_stride2}

ARMS=(
  canonical_unique
  canonical_repeat
  image_augmentation_unique
  broad_unpaired_practical
  broad_unpaired_state_matched
  broad_paired_fm
  broad_paired_consistency
)

cd "$REPO_ROOT"
for arm in "${ARMS[@]}"; do
  echo "[DSOL smoke matrix] starting arm=$arm seed=$SEED steps=$STEPS"
  DSOL_PAIR_DATA_ROOT="$DATA_ROOT" \
  DSOL_GLOBAL_EXAMPLES=32 \
  WANDB_MODE=${WANDB_MODE:-offline} \
    scripts/dsol_paper1/run_libero_pair_train.sh \
      "$arm" "$SEED" "$NUM_GPUS" "$STEPS" "$RUN_TAG"
done

echo "[DSOL smoke matrix] COMPLETE"
