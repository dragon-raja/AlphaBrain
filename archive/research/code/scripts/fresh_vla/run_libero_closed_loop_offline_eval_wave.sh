#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SEED=${1:?usage: run_libero_closed_loop_offline_eval_wave.sh SEED}
OUTPUT_ROOT=${FRESH_CLOSED_LOOP_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2}
METHODS=(full_h random_soft010 shuffled_oracle_soft010 gripper_soft010 oracle_soft010 short_h)

mkdir -p "$OUTPUT_ROOT/offline_wave_logs"
pids=()
names=()
for gpu in "${!METHODS[@]}"; do
  method=${METHODS[$gpu]}
  run_dir="$OUTPUT_ROOT/fresh_closed_loop_${method}_seed${SEED}"
  if [ -f "$run_dir/offline_eval.json" ] && [ -f "$run_dir/mode_coverage.json" ]; then
    echo "skip completed method=$method seed=$SEED"
    continue
  fi
  bash "$REPO_ROOT/scripts/fresh_vla/run_libero_closed_loop_offline_eval.sh" \
    "$method" "$SEED" "$gpu" >"$OUTPUT_ROOT/offline_wave_logs/${method}_seed${SEED}.log" 2>&1 &
  pids+=("$!")
  names+=("$method")
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "completed offline evaluation method=${names[$index]} seed=$SEED"
  else
    echo "failed offline evaluation method=${names[$index]} seed=$SEED" >&2
    failed=1
  fi
done
exit "$failed"
