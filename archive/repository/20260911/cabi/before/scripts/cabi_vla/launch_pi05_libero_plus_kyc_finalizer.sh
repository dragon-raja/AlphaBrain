#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SCRIPT=$(realpath "$0")
COMMAND=${1:-status}
SESSION=${PLUS_KYC_FINALIZER_SESSION:-plus-kyc-study-finalizer}
MATCHED=${PLUS_KYC_MATCHED_RESULT:-/share/longjunyu/alphabrain/experiments/libero-plus-kyc-matched-v1/final/metrics.json}
FACTOR=${PLUS_KYC_FACTOR_RESULT:-/share/longjunyu/alphabrain/experiments/libero-plus-kyc-factor-separated-v1/final/metrics.json}
OUTPUT_ROOT=${PLUS_KYC_FINAL_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/libero-plus-kyc-final-study-v1}
WAIT_SECONDS=${PLUS_KYC_FINALIZER_WAIT_SECONDS:-60}
LOG=$OUTPUT_ROOT/finalizer.log

run_worker() {
  mkdir -p "$OUTPUT_ROOT"
  exec > >(tee -a "$LOG") 2>&1
  echo "finalizer_started=$(date -u +%FT%TZ) commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  while [[ ! -s "$MATCHED" || ! -s "$FACTOR" ]]; do
    echo "waiting_for_results=$(date -u +%FT%TZ) matched=$([[ -s "$MATCHED" ]] && echo ready || echo waiting) factor=$([[ -s "$FACTOR" ]] && echo ready || echo waiting)"
    sleep "$WAIT_SECONDS"
  done
  "$REPO_ROOT/.venv/bin/python" \
    "$REPO_ROOT/scripts/cabi_vla/finalize_pi05_libero_plus_kyc_study.py" \
    --official-act /share/longjunyu/kyc-official-data/runs/analysis/official_act_summary.json \
    --joint-ood /share/longjunyu/alphabrain/experiments/libero-plus-camera-background-v1/final-joint-ood-v2/metrics.json \
    --matched "$MATCHED" \
    --factor-separated "$FACTOR" \
    --factorial /share/longjunyu/cabi-vla/kyc-scaling-v3/eval/factorial/n10/analysis/confirmed/summary.json \
    --ray-alignment /share/longjunyu/alphabrain/experiments/libero-plus-kyc-matched-v1/diagnostics/ray_alignment_v1.json \
    --output-json "$OUTPUT_ROOT/final_decision.json" \
    --output-report "$OUTPUT_ROOT/report_zh.md" \
    --output-figure "$OUTPUT_ROOT/summary.png"
  echo "finalizer_complete=$(date -u +%FT%TZ) output=$OUTPUT_ROOT"
}

case "$COMMAND" in
  start)
    mkdir -p "$OUTPUT_ROOT"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "finalizer already running: $SESSION"
      exit 0
    fi
    if [[ -s "$OUTPUT_ROOT/final_decision.json" ]]; then
      echo "finalizer already complete: $OUTPUT_ROOT"
      exit 0
    fi
    tmux new-session -d -s "$SESSION" "$SCRIPT" worker
    echo "finalizer started: $SESSION log=$LOG"
    ;;
  status)
    echo "matched=$([[ -s "$MATCHED" ]] && echo READY || echo WAITING)"
    echo "factor_separated=$([[ -s "$FACTOR" ]] && echo READY || echo WAITING)"
    if [[ -s "$OUTPUT_ROOT/final_decision.json" ]]; then
      echo "finalizer=COMPLETE output=$OUTPUT_ROOT"
    elif tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "finalizer=RUNNING session=$SESSION"
    else
      echo "finalizer=NOT_RUNNING"
    fi
    ;;
  worker)
    run_worker
    ;;
  *)
    echo "usage: $0 {start|status}" >&2
    exit 2
    ;;
esac
