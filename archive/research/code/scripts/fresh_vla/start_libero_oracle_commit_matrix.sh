#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUTPUT_ROOT=${FRESH_ORACLE_COMMIT_OUTPUT_ROOT:-/share/longjunyu/fresh-vla/runs/libero-oracle-commit-final-v1}
SESSION=${FRESH_ORACLE_COMMIT_SESSION:-fresh-oracle-final}
LOG=${FRESH_ORACLE_COMMIT_LOG:-/share/longjunyu/fresh-vla/oracle_commit_final_matrix.log}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT" "$(dirname "$LOG")"
tmux new-session -d -s "$SESSION" \
  "cd '$REPO_ROOT' && FRESH_ORACLE_COMMIT_OUTPUT_ROOT='$OUTPUT_ROOT' exec bash scripts/fresh_vla/run_libero_oracle_commit_matrix.sh >'$LOG' 2>&1"
echo "started session=$SESSION log=$LOG output=$OUTPUT_ROOT"
