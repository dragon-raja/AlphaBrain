#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUTPUT_ROOT=${FRESH_TRAIN_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-counterfactual-v1}
METHODS=(full_h full_h full_h oracle_soft010 oracle_soft010 oracle_soft010)
SEEDS=(41 42 43 41 42 43)
pids=()
names=()
mkdir -p "$OUTPUT_ROOT/multimodal_wave_logs"
for gpu in "${!METHODS[@]}"; do
  method=${METHODS[$gpu]}
  seed=${SEEDS[$gpu]}
  output="$OUTPUT_ROOT/fresh_libero_${method}_seed${seed}/multimodal_sampling.json"
  if [ -f "$output" ]; then
    echo "skip completed sampling method=$method seed=$seed"
    continue
  fi
  log="$OUTPUT_ROOT/multimodal_wave_logs/${method}_seed${seed}.log"
  bash "$REPO_ROOT/scripts/fresh_vla/run_libero_multimodal.sh" "$method" "$seed" "$gpu" >"$log" 2>&1 &
  pids+=("$!")
  names+=("${method}_seed${seed}")
  echo "started sampling method=$method seed=$seed gpu=$gpu pid=$!"
done
failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "completed sampling ${names[$index]}"
  else
    echo "failed sampling ${names[$index]}" >&2
    failed=1
  fi
done
exit "$failed"
