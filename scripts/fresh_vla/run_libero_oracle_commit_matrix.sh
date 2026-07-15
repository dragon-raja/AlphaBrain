#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUTPUT_ROOT=${FRESH_ORACLE_COMMIT_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-oracle-commit-final-v1}
BASELINE_ROOT=${FRESH_CLOSED_LOOP_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2}
NON_SELF_METHODS=(
  oracle_branch_safe_commit
  oracle_feedback_reveal_commit
  gripper_commit
  random_matched_commit
)
SEEDS=(41 42 43)

mkdir -p "$OUTPUT_ROOT/logs"
if ! command -v flock >/dev/null || ! command -v sha256sum >/dev/null; then
  echo "required command missing: flock and sha256sum are mandatory" >&2
  exit 1
fi
exec 9>"$OUTPUT_ROOT/.matrix.lock"
if ! flock -n 9; then
  echo "Oracle commit matrix is already running for $OUTPUT_ROOT" >&2
  exit 1
fi

for seed in 41 42 43; do
  case "$seed" in
    41) expected=144a3b3d3dcc8421418564a62059a1038c9a7ef3196ac157f5f9ea1997a31f30 ;;
    42) expected=98dc52d2ed1983776d218fee7666f3131053d1a55296e93e9f521b1c088ce875 ;;
    43) expected=5db16350d9835c1f28d01b660dd6e9234bcab3da79abbce1f092e92b08ac9149 ;;
  esac
  checkpoint="$BASELINE_ROOT/fresh_closed_loop_full_h_seed${seed}/final_model/model.safetensors"
  actual=$(sha256sum "$checkpoint" | awk '{print $1}')
  if [ "$actual" != "$expected" ]; then
    echo "frozen checkpoint SHA256 mismatch for seed $seed" >&2
    exit 1
  fi
  export "FRESH_PREVERIFIED_CHECKPOINT_SHA256_${seed}=$actual"
done

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
