#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUNNER="$REPO_ROOT/scripts/cabi_vla/run_pi05_libero_plus_multiview_train.sh"
OUTPUT_ROOT=${PLUS_MV_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/libero-plus-mv-rgb-v1/runs}
SEED=${PLUS_MV_WAVE_SEED:-41}
STEPS=${PLUS_MV_WAVE_STEPS:-33000}
COMMAND=${1:-status}

JOBS=(
  "plus-mv-a100-s${SEED}|action_only|1.0|0|gate-v1-b100"
  "plus-mv-a025-s${SEED}|action_only|0.25|1|gate-v1-b025"
  "plus-mv-v100-s${SEED}|visual_lora|1.0|2|gate-v1-b100"
  "plus-mv-v025-s${SEED}|visual_lora|0.25|3|gate-v1-b025"
)

run_id() {
  local arm=$1 tag=$2
  printf 'pi05_plus_mv_%s_%s_seed%s_steps%s' "$arm" "$tag" "$SEED" "$STEPS"
}

job_status() {
  local session=$1 arm=$2 tag=$3
  local output_dir="$OUTPUT_ROOT/$(run_id "$arm" "$tag")"
  if tmux has-session -t "$session" 2>/dev/null; then
    printf 'RUNNING'
  elif [[ -s "$output_dir/final_model/model.safetensors" ]]; then
    printf 'COMPLETE'
  elif [[ -e "$output_dir" ]]; then
    printf 'STOPPED_OR_FAILED'
  else
    printf 'NOT_STARTED'
  fi
}

start_job() {
  local session=$1 arm=$2 fraction=$3 gpu=$4 tag=$5
  local output_dir="$OUTPUT_ROOT/$(run_id "$arm" "$tag")"
  local status
  status=$(job_status "$session" "$arm" "$tag")
  if [[ "$status" != "NOT_STARTED" ]]; then
    printf '%-24s %-18s %s\n' "$session" "$status" "$output_dir"
    return
  fi

  tmux new-session -d -s "$session" \
    env PLUS_MV_OUTPUT_ROOT="$OUTPUT_ROOT" \
      PLUS_MV_BUDGET_FRACTION="$fraction" \
      PLUS_MV_RUN_TAG="$tag" \
      "$RUNNER" "$arm" "$SEED" "$gpu" "$STEPS"
  printf '%-24s %-18s %s\n' "$session" "STARTED_GPU_${gpu}" "$output_dir"
}

case "$COMMAND" in
  start)
    for job in "${JOBS[@]}"; do
      IFS='|' read -r session arm fraction gpu tag <<<"$job"
      start_job "$session" "$arm" "$fraction" "$gpu" "$tag"
    done
    ;;
  status)
    for job in "${JOBS[@]}"; do
      IFS='|' read -r session arm fraction gpu tag <<<"$job"
      output_dir="$OUTPUT_ROOT/$(run_id "$arm" "$tag")"
      printf '%-24s %-18s gpu=%s budget=%s %s\n' \
        "$session" "$(job_status "$session" "$arm" "$tag")" "$gpu" "$fraction" "$output_dir"
    done
    ;;
  *)
    echo "usage: $0 {start|status}" >&2
    exit 2
    ;;
esac
