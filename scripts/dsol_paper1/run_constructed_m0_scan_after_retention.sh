#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RETENTION_SESSION=${DSOL_RETENTION_SESSION:-dsol-original-retention-s41-v3}
POLL_SECONDS=${DSOL_WAIT_POLL_SECONDS:-60}
PLAN=${DSOL_CONSTRUCTED_M0_PLAN:-$REPO_ROOT/configs/dsol_paper1/libero_constructed_blind_reveal_scan_plan_v1.json}
CATALOG=${DSOL_CONSTRUCTED_M0_CATALOG:-$REPO_ROOT/configs/dsol_paper1/libero_view_catalog_v2_m1.json}
OUTPUT_ROOT=${DSOL_CONSTRUCTED_M0_OUTPUT_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m0-v1/operational-three-task-scan}

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid DSOL_WAIT_POLL_SECONDS" >&2; exit 2; }
[[ -s "$PLAN" ]] || { echo "missing constructed M0 plan: $PLAN" >&2; exit 1; }
[[ -s "$CATALOG" ]] || { echo "missing M0 catalog: $CATALOG" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$OUTPUT_ROOT/controller.log") 2>&1
printf 'controller_start=%s git_commit=%s\n' \
  "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)"

while tmux has-session -t "$RETENTION_SESSION" 2>/dev/null; do
  printf 'waiting_for_retention=%s at=%s\n' "$RETENTION_SESSION" "$(date -u +%FT%TZ)"
  sleep "$POLL_SECONDS"
done
while [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; do
  printf 'waiting_for_clean_worktree=%s at=%s\n' "$REPO_ROOT" "$(date -u +%FT%TZ)"
  sleep "$POLL_SECONDS"
done

PLAN="$PLAN" \
CATALOG="$CATALOG" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
GPU_COUNT=2 \
DSOL_GPU_DEVICES=6,7 \
CANDIDATE_GROUPS=canonical,broad_heldout_32,wide_extrapolation_24,diagnostic_extreme_orbit,diagnostic_crossed_orbit,diagnostic_look_away,sensor_controls \
  "$REPO_ROOT/scripts/dsol_paper1/run_libero_visibility_scan_matrix.sh"

printf 'controller_complete=%s\n' "$(date -u +%FT%TZ)"
