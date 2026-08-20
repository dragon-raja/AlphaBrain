#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WAIT_SESSION=${DSOL_M0_AUDIT_WAIT_SESSION:-dsol-constructed-m0-selection}
POLL_SECONDS=${DSOL_WAIT_POLL_SECONDS:-30}
SCAN_ROOT=${DSOL_CONSTRUCTED_M0_SCAN_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m0-v1/operational-three-task-scan}
SELECTION=${DSOL_CONSTRUCTED_M0_SELECTION:-$SCAN_ROOT/selection_v1.json}
PLAN=${DSOL_CONSTRUCTED_M0_PLAN:-$REPO_ROOT/configs/dsol_paper1/libero_constructed_blind_reveal_scan_plan_v1.json}
CATALOG=${DSOL_CONSTRUCTED_M0_CATALOG:-$REPO_ROOT/configs/dsol_paper1/libero_view_catalog_v2_m1.json}
RUNTIME=${DSOL_LIBERO_RUNTIME:-/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus}
CONFIG_ROOT=${DSOL_LIBERO_CONFIG_ROOT:-/share/longjunyu/alphabrain/envs/libero-plus-runtime-config-v1}
OUTPUT_ROOT=${DSOL_CONSTRUCTED_M0_AUDIT_ROOT:-$SCAN_ROOT/manual_audit_renders_v1}
PER_TASK=${DSOL_CONSTRUCTED_M0_AUDIT_PER_TASK:-7}
TARGET_TOTAL=${DSOL_CONSTRUCTED_M0_AUDIT_TARGET_TOTAL:-21}
RENDER_GPU=${DSOL_CONSTRUCTED_M0_AUDIT_GPU:-7}

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid DSOL_WAIT_POLL_SECONDS" >&2; exit 2; }
[[ "$PER_TASK" =~ ^[1-9][0-9]*$ ]] || { echo "invalid DSOL_CONSTRUCTED_M0_AUDIT_PER_TASK" >&2; exit 2; }
[[ "$TARGET_TOTAL" =~ ^[1-9][0-9]*$ ]] || { echo "invalid DSOL_CONSTRUCTED_M0_AUDIT_TARGET_TOTAL" >&2; exit 2; }
[[ "$RENDER_GPU" =~ ^[0-9]+$ ]] || { echo "invalid DSOL_CONSTRUCTED_M0_AUDIT_GPU" >&2; exit 2; }

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$OUTPUT_ROOT/audit_render_controller.log") 2>&1
printf 'audit_render_controller_start=%s git_commit=%s wait_session=%s\n' \
  "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)" "$WAIT_SESSION"

while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  printf 'waiting_for_selection=%s at=%s\n' "$WAIT_SESSION" "$(date -u +%FT%TZ)"
  sleep "$POLL_SECONDS"
done
while [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; do
  printf 'waiting_for_clean_worktree=%s at=%s\n' "$REPO_ROOT" "$(date -u +%FT%TZ)"
  sleep "$POLL_SECONDS"
done

[[ -s "$SELECTION" ]] || { echo "missing selection: $SELECTION" >&2; exit 1; }
status=$(jq -r '.status // "UNKNOWN"' "$SELECTION")
if [[ "$status" != PASS ]]; then
  printf 'audit_render_skipped=%s selection_status=%s\n' "$(date -u +%FT%TZ)" "$status"
  exit 0
fi

for required in "$PLAN" "$CATALOG"; do
  [[ -s "$required" ]] || { echo "missing required file: $required" >&2; exit 1; }
done

PYTHONPATH="$REPO_ROOT:$REPO_ROOT/scripts/dsol_paper1" \
  /workspace/envs/fresh-libero/bin/python \
  "$REPO_ROOT/scripts/dsol_paper1/render_constructed_m0_audit_montages.py" \
  --selection "$SELECTION" \
  --scan-plan "$PLAN" \
  --catalog "$CATALOG" \
  --runtime "$RUNTIME" \
  --config-root "$CONFIG_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --per-task "$PER_TASK" \
  --target-total "$TARGET_TOTAL" \
  --render-gpu "$RENDER_GPU"

jq -e '
  .schema == "dsol_constructed_m0_manual_audit_render_v1"
  and .status == "RENDERED_PENDING_MANUAL_AUDIT"
  and .manual_audit_status == "PENDING"
  and .automatically_promoted == false
' "$OUTPUT_ROOT/audit_render_manifest.json" >/dev/null
printf 'audit_render_controller_complete=%s output=%s\n' \
  "$(date -u +%FT%TZ)" "$OUTPUT_ROOT/audit_render_manifest.json"
