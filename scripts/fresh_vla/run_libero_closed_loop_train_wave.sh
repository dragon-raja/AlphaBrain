#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SEED=${1:?usage: run_libero_closed_loop_train_wave.sh SEED [STEPS]}
STEPS=${2:-2400}
OUTPUT_ROOT=${FRESH_CLOSED_LOOP_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2}
METHODS=(
  full_h
  random_soft010
  shuffled_oracle_soft010
  gripper_soft010
  oracle_soft010
  short_h
)

mkdir -p "$OUTPUT_ROOT/wave_logs"
pids=()
names=()
for gpu in "${!METHODS[@]}"; do
  method=${METHODS[$gpu]}
  checkpoint="$OUTPUT_ROOT/fresh_closed_loop_${method}_seed${SEED}/final_model/model.safetensors"
  if [ -f "$checkpoint" ]; then
    echo "skip completed method=$method seed=$SEED"
    continue
  fi
  log="$OUTPUT_ROOT/wave_logs/${method}_seed${SEED}.log"
  bash "$REPO_ROOT/scripts/fresh_vla/run_libero_closed_loop_train.sh" \
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
