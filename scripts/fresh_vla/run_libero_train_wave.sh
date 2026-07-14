#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SEED=${1:?usage: run_libero_train_wave.sh SEED [STEPS]}
STEPS=${2:-1200}
OUTPUT_ROOT=${FRESH_TRAIN_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-counterfactual-v1}
METHODS=(
  full_h
  random_soft010
  shuffled_oracle_soft010
  early_oracle_soft010
  late_oracle_soft010
  gripper_soft010
  oracle_soft010
  short_h
)

mkdir -p "$OUTPUT_ROOT/wave_logs"
pids=()
names=()

for gpu in "${!METHODS[@]}"; do
  method=${METHODS[$gpu]}
  final_model="$OUTPUT_ROOT/fresh_libero_${method}_seed${SEED}/final_model/model.safetensors"
  if [ -f "$final_model" ]; then
    echo "skip completed method=$method seed=$SEED"
    continue
  fi
  log="$OUTPUT_ROOT/wave_logs/${method}_seed${SEED}.log"
  bash "$REPO_ROOT/scripts/fresh_vla/run_libero_train.sh" \
    "$method" "$SEED" "$gpu" "$STEPS" >"$log" 2>&1 &
  pids+=("$!")
  names+=("$method")
  echo "started method=$method seed=$SEED gpu=$gpu pid=$!"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "completed method=${names[$index]} seed=$SEED"
  else
    echo "failed method=${names[$index]} seed=$SEED" >&2
    failed=1
  fi
done
exit "$failed"
