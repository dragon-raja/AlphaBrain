#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUDGET_DECISION=${DSOL_BUDGET_DECISION:-$REPO_ROOT/configs/dsol_paper1/libero_training_budget_quick_gate_v1.json}
[[ -s "$BUDGET_DECISION" ]] || { echo "missing budget decision: $BUDGET_DECISION" >&2; exit 2; }
STEPS=${DSOL_FORMAL_STEPS:-$(jq -r '.training_steps' "$BUDGET_DECISION")}
SCHEDULER_STEPS=${DSOL_FORMAL_SCHEDULER_STEPS:-$(jq -r '.scheduler_total_steps' "$BUDGET_DECISION")}
NUM_GPUS=${DSOL_FORMAL_GPUS:-8}
SEEDS_TEXT=${DSOL_FORMAL_SEEDS:-41}
RUN_TAG=${DSOL_FORMAL_RUN_TAG:-quick-gate-v1}
OUTPUT_ROOT=${DSOL_PAIR_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs}
DATA_ROOT=${DSOL_PAIR_DATA_ROOT:-/share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad32_stride2}

[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "DSOL_FORMAL_STEPS must be positive" >&2; exit 2; }
[[ "$SCHEDULER_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "DSOL_FORMAL_SCHEDULER_STEPS must be positive" >&2; exit 2; }
(( SCHEDULER_STEPS >= STEPS )) || { echo "DSOL_FORMAL_SCHEDULER_STEPS must be >= DSOL_FORMAL_STEPS" >&2; exit 2; }
[[ "$NUM_GPUS" =~ ^[1-8]$ ]] || { echo "DSOL_FORMAL_GPUS must be in [1,8]" >&2; exit 2; }
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "unsafe DSOL_FORMAL_RUN_TAG" >&2; exit 2; }
[[ -s "$DATA_ROOT/manifest.json" ]] || { echo "missing formal dataset: $DATA_ROOT" >&2; exit 2; }

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
    run_id="dsol_${arm}_${RUN_TAG}_seed${seed}_g${NUM_GPUS}_gb32_steps${STEPS}"
    output_dir="$OUTPUT_ROOT/$run_id"
    if [[ -s "$output_dir/final_model/model.safetensors" && -s "$output_dir/final_model/framework_config.yaml" ]]; then
      echo "formal_matrix_skip_complete=$run_id"
      continue
    fi
    if [[ -e "$output_dir" ]]; then
      echo "formal_matrix_refusing_partial=$output_dir" >&2
      exit 1
    fi
    echo "formal_matrix_start=$run_id"
    DSOL_PAIR_DATA_ROOT="$DATA_ROOT" \
    DSOL_PAIR_OUTPUT_ROOT="$OUTPUT_ROOT" \
    DSOL_GLOBAL_EXAMPLES=32 \
    DSOL_CALIBRATION=0 \
    DSOL_SKIP_FINAL_SAVE=0 \
    DSOL_BUDGET_DECISION="$BUDGET_DECISION" \
    DSOL_SCHEDULER_STEPS="$SCHEDULER_STEPS" \
    WANDB_MODE=${WANDB_MODE:-offline} \
      "$REPO_ROOT/scripts/dsol_paper1/run_libero_pair_train.sh" \
        "$arm" "$seed" "$NUM_GPUS" "$STEPS" "$RUN_TAG"
    [[ -s "$output_dir/final_model/model.safetensors" ]] || {
      echo "formal_matrix_missing_final_model=$output_dir" >&2
      exit 1
    }
    echo "formal_matrix_complete=$run_id"
  done
done

echo "formal_matrix_all_complete=1 seeds=$SEEDS_TEXT steps=$STEPS scheduler_steps=$SCHEDULER_STEPS"
