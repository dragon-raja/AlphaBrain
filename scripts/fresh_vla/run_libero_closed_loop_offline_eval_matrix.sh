#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUTPUT_ROOT=${FRESH_CLOSED_LOOP_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2}
GPU_COUNT=${FRESH_GPU_COUNT:-8}
METHODS=(full_h random_soft010 shuffled_oracle_soft010 gripper_soft010 oracle_soft010 short_h)
SEEDS=(41 42 43)

if [[ ! "$GPU_COUNT" =~ ^[1-8]$ ]]; then
  echo "FRESH_GPU_COUNT must be in [1, 8]" >&2
  exit 2
fi

pending=()
for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_dir="$OUTPUT_ROOT/fresh_closed_loop_${method}_seed${seed}"
    if [ ! -f "$run_dir/final_model/model.safetensors" ]; then
      echo "missing checkpoint: $run_dir/final_model/model.safetensors" >&2
      exit 1
    fi
    if [ -f "$run_dir/offline_eval.json" ] && [ -f "$run_dir/mode_coverage.json" ]; then
      echo "skip completed method=$method seed=$seed"
    else
      pending+=("$method:$seed")
    fi
  done
done
if [ "${#pending[@]}" = 0 ]; then
  echo "all matrix offline evaluations are complete"
  exit 0
fi

mkdir -p "$OUTPUT_ROOT/offline_matrix_logs"
run_worker() {
  local gpu=$1
  local index=$gpu
  local failed=0
  while [ "$index" -lt "${#pending[@]}" ]; do
    local method seed
    IFS=: read -r method seed <<<"${pending[$index]}"
    echo "start offline evaluation method=$method seed=$seed gpu=$gpu"
    if bash "$REPO_ROOT/scripts/fresh_vla/run_libero_closed_loop_offline_eval.sh" "$method" "$seed" "$gpu"; then
      echo "complete offline evaluation method=$method seed=$seed gpu=$gpu"
    else
      echo "failed offline evaluation method=$method seed=$seed gpu=$gpu" >&2
      failed=1
    fi
    index=$((index + GPU_COUNT))
  done
  return "$failed"
}

pids=()
workers=()
for ((gpu = 0; gpu < GPU_COUNT && gpu < ${#pending[@]}; gpu++)); do
  run_worker "$gpu" >"$OUTPUT_ROOT/offline_matrix_logs/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
  workers+=("$gpu")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "offline evaluation worker failed gpu=${workers[$index]}" >&2
    failed=1
  fi
done
exit "$failed"
