#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TRAIN_RUN=${TRAIN_RUN:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000}
OFFICIAL_CHECKPOINT=${OFFICIAL_CHECKPOINT:-/share/longjunyu/alphabrain/pretrained_models/openpi/pi05_libero_pytorch}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1}
VIEW_GAP_ROOT=${VIEW_GAP_ROOT:-/share/longjunyu/alphabrain/experiments/libero-plus-view-gap-v1}

while [[ ! -s "$TRAIN_RUN/final_model/model.safetensors" ]]; do
  sleep 30
done

POLICY_BACKEND=openpi \
CHECKPOINT="$OFFICIAL_CHECKPOINT" \
OUTPUT_DIR="$EXPERIMENT_ROOT/closed_loop_smoke/official-exact-state-openpi-v1" \
GPU_COUNT=8 \
BASE_PORT=18700 \
MAX_EPISODES_PER_SHARD=1 \
RUN_ANALYSIS=0 \
  bash "$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh"

POLICY_BACKEND=openpi \
CHECKPOINT="$OFFICIAL_CHECKPOINT" \
OUTPUT_DIR="$EXPERIMENT_ROOT/closed_loop/dsol_official_exact-state-v1" \
GPU_COUNT=8 \
BASE_PORT=18720 \
  bash "$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh"

POLICY_BACKEND=alphabrain \
CHECKPOINT="$TRAIN_RUN/final_model" \
OUTPUT_DIR="$EXPERIMENT_ROOT/closed_loop/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000" \
GPU_COUNT=8 \
BASE_PORT=18740 \
  bash "$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh"

CHECKPOINT="$TRAIN_RUN/final_model" \
OUTPUT_DIR="$VIEW_GAP_ROOT/pi05-libero-broad64-quick-gate-v1" \
GPU_COUNT=8 \
BASE_PORT=18760 \
EVAL_MODES=gap \
POLICY_PYTHON=/alphabrain/.venv/bin/python \
  bash "$REPO_ROOT/scripts/cabi_vla/run_alphabrain_pi05_libero_plus_view_eval.sh"

echo "broad64_posttrain_gate_complete=$TRAIN_RUN"
