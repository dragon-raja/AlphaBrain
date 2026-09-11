#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RESET_ROOT=${FRESH_RESEARCH_RESET_ROOT:-/share/longjunyu/fresh-vla/research-reset}
RUN_ROOT="$RESET_ROOT/recovery_prompt"
mkdir -p "$RUN_ROOT/logs"

tasks=(
  explicit_recovery:41:0
  explicit_recovery:42:1
  explicit_recovery:43:2
  false_success_assumption:41:3
  false_success_assumption:42:4
  false_success_assumption:43:5
)
pids=()
for task in "${tasks[@]}"; do
  variant=${task%%:*}
  remainder=${task#*:}
  seed=${remainder%%:*}
  gpu=${remainder##*:}
  bash "$REPO_ROOT/scripts/fresh_vla/run_recovery_prompt_eval.sh" "$variant" "$seed" "$gpu" \
    >"$RUN_ROOT/logs/${variant}-seed${seed}.log" 2>&1 &
  pids+=("$!")
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
"$REPO_ROOT/.venv/bin/python" scripts/fresh_vla/summarize_recovery_prompt.py \
  --runs-root "$RUN_ROOT" \
  --output "$RESET_ROOT/recovery_prompt_summary.json"
