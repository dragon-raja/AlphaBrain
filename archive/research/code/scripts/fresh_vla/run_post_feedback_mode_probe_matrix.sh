#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN_ROOT=${FRESH_RESEARCH_RESET_ROOT:-/share/longjunyu/fresh-vla/research-reset}
mkdir -p "$RUN_ROOT/logs"

pids=()
index=0
for seed in 41 42 43; do
  gpu=$index
  bash "$REPO_ROOT/scripts/fresh_vla/run_post_feedback_mode_probe.sh" "$seed" "$gpu" \
    >"$RUN_ROOT/logs/post-feedback-modes-seed${seed}.log" 2>&1 &
  pids+=("$!")
  index=$((index + 1))
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [ "$status" -ne 0 ]; then
  exit "$status"
fi

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/scripts/fresh_vla" \
"$REPO_ROOT/.venv/bin/python" scripts/fresh_vla/summarize_post_feedback_modes.py \
  --inputs \
    "$RUN_ROOT/post_feedback_modes_seed41.json" \
    "$RUN_ROOT/post_feedback_modes_seed42.json" \
    "$RUN_ROOT/post_feedback_modes_seed43.json" \
  --output "$RUN_ROOT/post_feedback_modes_summary.json"
