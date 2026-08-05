#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SCRIPT=$(realpath "$0")
COMMAND=${1:-status}
SESSION=${PLUS_GOAL_AUDIT_SESSION:-plus-goal-archive-audit}
ROOT=${PLUS_GOAL_ROOT:-/share/longjunyu/alphabrain/datasets/libero-plus/verified/goal-suite}
ARCHIVE=${PLUS_GOAL_ARCHIVE:-$ROOT/libero_goal.zip}
PARTIAL=${ARCHIVE}.aria2
DOWNLOAD_SESSION=${PLUS_GOAL_DOWNLOAD_SESSION:-plus-goal-download}
DATA_PYTHON=${PLUS_DATA_PYTHON:-/share/longjunyu/alphabrain/envs/libero-plus-data-v1/bin/python}
SAMPLE_ROOT=${PLUS_GOAL_SAMPLE_ROOT:-$ROOT/schema-sample}
OUTPUT=${PLUS_GOAL_AUDIT_OUTPUT:-$ROOT/archive_factor_audit.json}
LOG=$ROOT/archive-audit.log
WAIT_SECONDS=${PLUS_GOAL_AUDIT_WAIT_SECONDS:-30}

print_status() {
  if [[ -s "$OUTPUT" ]]; then
    OUTPUT="$OUTPUT" "$REPO_ROOT/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

report = json.loads(Path(os.environ["OUTPUT"]).read_text())
print("audit=COMPLETE")
print("decision=" + report["strict_composition_gate"]["decision"])
print("tfrecord_shards=" + str(report["inventory"]["tfrecord_shard_count"]))
PY
  elif tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "audit=WAITING_OR_RUNNING session=$SESSION"
  else
    echo "audit=NOT_RUNNING"
  fi
  if [[ -e "$PARTIAL" ]]; then
    echo "download=IN_PROGRESS"
  elif [[ -s "$ARCHIVE" ]]; then
    echo "download=COMPLETE"
  else
    echo "download=MISSING"
  fi
}

run_worker() {
  mkdir -p "$ROOT"
  exec > >(tee -a "$LOG") 2>&1
  echo "goal_archive_audit_started=$(date -u +%FT%TZ) commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  while [[ -e "$PARTIAL" ]]; do
    if ! tmux has-session -t "$DOWNLOAD_SESSION" 2>/dev/null; then
      echo "download session ended while partial marker remains: $PARTIAL" >&2
      exit 1
    fi
    echo "waiting_for_download=$(date -u +%FT%TZ)"
    sleep "$WAIT_SECONDS"
  done
  if [[ ! -s "$ARCHIVE" ]]; then
    echo "download completed without archive: $ARCHIVE" >&2
    exit 1
  fi
  if [[ -s "$OUTPUT" ]]; then
    echo "audit already complete: $OUTPUT"
    exit 0
  fi
  "$DATA_PYTHON" "$REPO_ROOT/scripts/cabi_vla/audit_libero_plus_goal_archive.py" \
    --archive "$ARCHIVE" \
    --sample-root "$SAMPLE_ROOT" \
    --output "$OUTPUT"
  echo "goal_archive_audit_complete=$(date -u +%FT%TZ) output=$OUTPUT"
}

case "$COMMAND" in
  start)
    mkdir -p "$ROOT"
    if [[ -s "$OUTPUT" ]]; then
      echo "goal archive audit already complete: $OUTPUT"
      exit 0
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "goal archive audit already running: $SESSION"
      exit 0
    fi
    tmux new-session -d -s "$SESSION" "$SCRIPT" worker
    echo "goal archive audit started: $SESSION log=$LOG"
    ;;
  status)
    print_status
    ;;
  worker)
    run_worker
    ;;
  *)
    echo "usage: $0 {start|status}" >&2
    exit 2
    ;;
esac
