#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUTPUT_ROOT=${FRESH_ORACLE_COMMIT_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-oracle-commit-final-v1}
NON_SELF_METHODS=(
  oracle_branch_safe_commit
  oracle_feedback_reveal_commit
  gripper_commit
  random_matched_commit
)
SEEDS=(41 42 43)

mkdir -p "$OUTPUT_ROOT/logs"
if ! command -v flock >/dev/null; then
  echo "required command missing: flock" >&2
  exit 1
fi
exec 9>"$OUTPUT_ROOT/.matrix.lock"
if ! flock -n 9; then
  echo "Oracle commit matrix is already running for $OUTPUT_ROOT" >&2
  exit 1
fi

run_phase() {
  local phase=$1
  shift
  local tasks=("$@")
  local pids=()
  local status=0
  for gpu in 0 1 2 3 4 5 6 7; do
    (
      for ((index=gpu; index<${#tasks[@]}; index+=8)); do
        task=${tasks[$index]}
        method=${task%%:*}
        remainder=${task#*:}
        seed=${remainder%%:*}
        mode=${remainder##*:}
        echo "start phase=$phase method=$method seed=$seed mode=$mode gpu=$gpu"
        FRESH_ORACLE_EVAL_ONLY="$mode" \
          bash "$REPO_ROOT/scripts/fresh_vla/run_libero_oracle_commit_eval.sh" "$method" "$seed" "$gpu"
        echo "complete phase=$phase method=$method seed=$seed mode=$mode gpu=$gpu"
      done
    ) >"$OUTPUT_ROOT/logs/${phase}-gpu-${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

non_self_tasks=()
for method in "${NON_SELF_METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    non_self_tasks+=("$method:$seed:all")
  done
done
run_phase non-self "${non_self_tasks[@]}"

self_tasks=()
for seed in "${SEEDS[@]}"; do
  self_tasks+=("self_consistency_commit:$seed:isolated_reach")
  self_tasks+=("self_consistency_commit:$seed:end_to_end")
done
run_phase self-consistency "${self_tasks[@]}"
