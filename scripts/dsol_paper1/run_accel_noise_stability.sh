#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TRAIN_ROOT=${DSOL_PAIR_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs}
REFERENCE_ROOT=${DSOL_ACCEL_REFERENCE_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-accel-constructed-v2}
RUN_ROOT=${DSOL_ACCEL_NOISE_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-accel-noise-stability-v1/runs}
ANALYSIS_ROOT=${DSOL_ACCEL_NOISE_ANALYSIS_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-accel-noise-stability-v1/analysis}
RENDER_SOURCE=${DSOL_ACCEL_RENDER_SOURCE:-$REFERENCE_ROOT/broad64-practical-seed41-full}
REFERENCE_SEED=${DSOL_ACCEL_NOISE_REFERENCE_SEED:-20260820}
RUN_SEEDS=${DSOL_ACCEL_NOISE_SEEDS:-"20260821 20260822 20260823 20260824 20260825 20260826 20260827"}
ANALYSIS_SEEDS=${DSOL_ACCEL_NOISE_ANALYSIS_SEEDS:-"$REFERENCE_SEED $RUN_SEEDS"}
RUNNER=$REPO_ROOT/scripts/dsol_paper1/run_constructed_accel_bank.sh
ANALYZER=$REPO_ROOT/scripts/dsol_paper1/analyze_accel_noise_stability.py

[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
  echo "refusing scientific run from a dirty worktree" >&2
  exit 2
}
[[ -s "$RENDER_SOURCE/render_summary.json" ]] || {
  echo "missing shared render bank: $RENDER_SOURCE" >&2
  exit 2
}

declare -a names=(
  broad64-practical
  broad64-state-matched
  broad64-paired-fm
  broad64-paired-consistency
)
declare -a checkpoints=(
  "$TRAIN_ROOT/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000/final_model"
  "$TRAIN_ROOT/dsol_broad_unpaired_state_matched_broad64-parallel-formal-v1_seed41_g2_gb32_steps2000/final_model"
  "$TRAIN_ROOT/dsol_broad_paired_fm_broad64-parallel-formal-v1_seed41_g2_gb32_steps2000/final_model"
  "$TRAIN_ROOT/dsol_broad_paired_consistency_broad64-parallel-formal-v1_seed41_g2_gb32_steps2000/final_model"
)
declare -a devices=(0 1 2 3)

mkdir -p "$RUN_ROOT/logs" "$ANALYSIS_ROOT"
exec > >(tee -a "$RUN_ROOT/controller.log") 2>&1
printf 'accel_noise_stability_start=%s git_commit=%s run_seeds=%s analysis_seeds=%s\n' \
  "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)" \
  "$RUN_SEEDS" "$ANALYSIS_SEEDS"

declare -a pids=()
for index in "${!names[@]}"; do
  name=${names[$index]}
  checkpoint=${checkpoints[$index]}
  device=${devices[$index]}
  [[ -s "$checkpoint/model.safetensors" ]] || {
    echo "missing checkpoint: $checkpoint" >&2
    exit 2
  }
  (
    for seed in $RUN_SEEDS; do
      output="$RUN_ROOT/$name-flow-seed$seed"
      DSOL_ACCEL_CHECKPOINT="$checkpoint" \
      DSOL_ACCEL_OUTPUT_DIR="$output" \
      DSOL_ACCEL_RENDER_SOURCE_DIR="$RENDER_SOURCE" \
      DSOL_ACCEL_DEVICE="cuda:$device" \
      DSOL_ACCEL_RENDER_GPU="$device" \
      DSOL_ACCEL_SEED="$seed" \
        "$RUNNER" > "$RUN_ROOT/logs/$name-flow-seed$seed.log" 2>&1
    done
  ) &
  pids+=("$!")
  printf 'launched_model=%s device=%s pid=%s\n' "$name" "$device" "$!"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
[[ "$failed" == 0 ]] || { echo "one or more Accel noise workers failed" >&2; exit 1; }

seed_args=()
for seed in $ANALYSIS_SEEDS; do seed_args+=("$seed"); done
/alphabrain/.venv/bin/python "$ANALYZER" \
  --reference-root "$REFERENCE_ROOT" \
  --run-root "$RUN_ROOT" \
  --output-dir "$ANALYSIS_ROOT" \
  --reference-seed "$REFERENCE_SEED" \
  --seeds "${seed_args[@]}"

printf 'accel_noise_stability_complete=%s summary=%s\n' \
  "$(date -u +%FT%TZ)" "$ANALYSIS_ROOT/summary.json"
