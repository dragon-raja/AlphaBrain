#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
STEPS=${DSOL_FORMAL_STEPS:?set DSOL_FORMAL_STEPS to the frozen common budget}
SEEDS_TEXT=${DSOL_FORMAL_SEEDS:-41}
RUN_TAG=${DSOL_FORMAL_RUN_TAG:-quick-gate-v1}
GPU_COUNT=${DSOL_EVAL_GPUS:-8}
TRAIN_GPUS=${DSOL_FORMAL_GPUS:-8}
TRAIN_ROOT=${DSOL_PAIR_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs}
EVAL_ROOT=${DSOL_EVAL_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/closed_loop}

[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "DSOL_FORMAL_STEPS must be positive" >&2; exit 2; }
[[ "$GPU_COUNT" =~ ^[1-8]$ ]] || { echo "DSOL_EVAL_GPUS must be in [1,8]" >&2; exit 2; }
[[ "$TRAIN_GPUS" =~ ^[1-8]$ ]] || { echo "DSOL_FORMAL_GPUS must be in [1,8]" >&2; exit 2; }
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "unsafe DSOL_FORMAL_RUN_TAG" >&2; exit 2; }

IFS=',' read -r -a seeds <<< "$SEEDS_TEXT"
arms=(
  canonical_unique
  canonical_repeat
  image_augmentation_unique
  broad_unpaired_practical
  broad_unpaired_state_matched
  broad_paired_fm
  broad_paired_consistency
)

for seed in "${seeds[@]}"; do
  [[ "$seed" =~ ^[1-9][0-9]*$ ]] || { echo "invalid seed: $seed" >&2; exit 2; }
  for arm in "${arms[@]}"; do
    run_id="dsol_${arm}_${RUN_TAG}_seed${seed}_g${TRAIN_GPUS}_gb32_steps${STEPS}"
    checkpoint="$TRAIN_ROOT/$run_id/final_model"
    output_dir="$EVAL_ROOT/$run_id"
    [[ -s "$checkpoint/model.safetensors" && -s "$checkpoint/framework_config.yaml" ]] || {
      echo "missing complete checkpoint: $checkpoint" >&2
      exit 1
    }
    if [[ -s "$output_dir/analysis/metrics.json" ]] && \
      jq -e '.status == "PASS"' "$output_dir/analysis/metrics.json" >/dev/null; then
      echo "closed_loop_matrix_skip_complete=$run_id"
      continue
    fi
    echo "closed_loop_matrix_start=$run_id"
    CHECKPOINT="$checkpoint" OUTPUT_DIR="$output_dir" GPU_COUNT="$GPU_COUNT" \
      "$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh"
    jq -e '.status == "PASS"' "$output_dir/analysis/metrics.json" >/dev/null
    echo "closed_loop_matrix_complete=$run_id"
  done
done

echo "closed_loop_matrix_all_complete=1 seeds=$SEEDS_TEXT steps=$STEPS"
