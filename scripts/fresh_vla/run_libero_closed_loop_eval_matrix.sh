#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUTPUT_ROOT=${FRESH_CLOSED_LOOP_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2}
GPU_COUNT=${FRESH_GPU_COUNT:-8}
EVAL_ONLY=${FRESH_EVAL_ONLY:-all}
OUTPUT_TAG=${FRESH_EVAL_OUTPUT_TAG:-}
METHODS=(full_h random_soft010 shuffled_oracle_soft010 gripper_soft010 oracle_soft010 short_h)
SEEDS=(41 42 43)

if [[ ! "$GPU_COUNT" =~ ^[1-8]$ ]]; then
  echo "FRESH_GPU_COUNT must be in [1, 8]" >&2
  exit 2
fi
case "$EVAL_ONLY" in
  all|closed_loop|isolated|end_to_end|reach) ;;
  *) echo "FRESH_EVAL_ONLY must be one of: all, closed_loop, isolated, end_to_end, reach" >&2; exit 2 ;;
esac
suffix=""
if [ -n "$OUTPUT_TAG" ]; then
  suffix="_$OUTPUT_TAG"
fi

is_complete() {
  local run_dir=$1
  case "$EVAL_ONLY" in
    all)
      [ -f "$run_dir/closed_loop_isolated${suffix}.json" ] \
        && [ -f "$run_dir/closed_loop_end_to_end${suffix}.json" ] \
        && [ -f "$run_dir/deterministic_reach${suffix}.json" ]
      ;;
    closed_loop)
      [ -f "$run_dir/closed_loop_isolated${suffix}.json" ] \
        && [ -f "$run_dir/closed_loop_end_to_end${suffix}.json" ]
      ;;
    isolated) [ -f "$run_dir/closed_loop_isolated${suffix}.json" ] ;;
    end_to_end) [ -f "$run_dir/closed_loop_end_to_end${suffix}.json" ] ;;
    reach) [ -f "$run_dir/deterministic_reach${suffix}.json" ] ;;
  esac
}

pending=()
for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_dir="$OUTPUT_ROOT/fresh_closed_loop_${method}_seed${seed}"
    if [ ! -f "$run_dir/final_model/model.safetensors" ]; then
      echo "missing checkpoint: $run_dir/final_model/model.safetensors" >&2
      exit 1
    fi
    if is_complete "$run_dir"; then
      echo "skip completed method=$method seed=$seed"
    else
      pending+=("$method:$seed")
    fi
  done
done
if [ "${#pending[@]}" = 0 ]; then
  echo "all matrix evaluations are complete"
  exit 0
fi

log_dir="$OUTPUT_ROOT/eval_matrix_logs${suffix}"
mkdir -p "$log_dir"
run_worker() {
  local gpu=$1
  local index=$gpu
  local failed=0
  while [ "$index" -lt "${#pending[@]}" ]; do
    local method seed
    IFS=: read -r method seed <<<"${pending[$index]}"
    echo "start evaluation method=$method seed=$seed gpu=$gpu mode=$EVAL_ONLY"
    if bash "$REPO_ROOT/scripts/fresh_vla/run_libero_closed_loop_eval.sh" "$method" "$seed" "$gpu"; then
      echo "complete evaluation method=$method seed=$seed gpu=$gpu"
    else
      echo "failed evaluation method=$method seed=$seed gpu=$gpu" >&2
      failed=1
    fi
    index=$((index + GPU_COUNT))
  done
  return "$failed"
}

pids=()
workers=()
for ((gpu = 0; gpu < GPU_COUNT && gpu < ${#pending[@]}; gpu++)); do
  run_worker "$gpu" >"$log_dir/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
  workers+=("$gpu")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "evaluation worker failed gpu=${workers[$index]}" >&2
    failed=1
  fi
done
exit "$failed"
