#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WAIT_SESSION=${DSOL_ACCEL_WAIT_SESSION:-dsol-constructed-m0-after-retention}
POLL_SECONDS=${DSOL_WAIT_POLL_SECONDS:-60}
PAIR_ROOT=${DSOL_ACCEL_PAIR_ROOT:-/share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad64_stride2}
CHECKPOINT=${DSOL_ACCEL_CHECKPOINT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000/final_model}
OUTPUT_ROOT=${DSOL_ACCEL_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-accel-fixed-state-v1/broad64-seed41-engineering-smoke}
DEVICE=${DSOL_ACCEL_DEVICE:-cuda:7}

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid DSOL_WAIT_POLL_SECONDS" >&2; exit 2; }
[[ -s "$PAIR_ROOT/manifest.json" ]] || { echo "missing pair data: $PAIR_ROOT" >&2; exit 1; }
[[ -s "$CHECKPOINT/model.safetensors" ]] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$OUTPUT_ROOT/controller.log") 2>&1
printf 'controller_start=%s git_commit=%s wait_session=%s\n' \
  "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)" "$WAIT_SESSION"

while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  printf 'waiting_for_m0=%s at=%s\n' "$WAIT_SESSION" "$(date -u +%FT%TZ)"
  sleep "$POLL_SECONDS"
done
while [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; do
  printf 'waiting_for_clean_worktree=%s at=%s\n' "$REPO_ROOT" "$(date -u +%FT%TZ)"
  sleep "$POLL_SECONDS"
done

if [[ -s "$OUTPUT_ROOT/manifest.json" ]]; then
  echo "existing Accel smoke manifest found; refusing to overwrite: $OUTPUT_ROOT/manifest.json"
  exit 0
fi

if [[ "$DEVICE" =~ ^cuda:([0-9]+)$ ]]; then
  KEEPALIVE_SESSION="gpu-keepalive-${BASH_REMATCH[1]}"
  tmux kill-session -t "$KEEPALIVE_SESSION" 2>/dev/null || true
fi

PYTHONPATH="$REPO_ROOT" \
  /alphabrain/.venv/bin/python \
  "$REPO_ROOT/scripts/dsol_paper1/run_accel_checkpoint_smoke.py" \
  --pair-root "$PAIR_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_ROOT" \
  --device "$DEVICE" \
  --split test \
  --record-index 0 \
  --seed 20260820

printf 'controller_complete=%s\n' "$(date -u +%FT%TZ)"
