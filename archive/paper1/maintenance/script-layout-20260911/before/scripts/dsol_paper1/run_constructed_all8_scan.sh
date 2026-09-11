#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${DSOL_CONSTRUCTED_ALL8_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-all8-v1}
PLAN=${DSOL_CONSTRUCTED_ALL8_PLAN:-$ROOT/scan-plan.json}
SMOKE_PLAN=${DSOL_CONSTRUCTED_ALL8_SMOKE_PLAN:-$ROOT/smoke-plan.json}
CATALOG=$REPO_ROOT/configs/dsol_paper1/libero_view_catalog_v2_m1.json
SCANNER=$REPO_ROOT/scripts/dsol_paper1/run_libero_visibility_scan_matrix.sh
FREEZER=$REPO_ROOT/scripts/dsol_paper1/freeze_constructed_task_pairs.py
DENSE_BUILDER=$REPO_ROOT/scripts/dsol_paper1/build_constructed_view_oracle_protocol.py

mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/controller.log") 2>&1

jq -e '.status == "PASS" and .record_count == 260' "$PLAN" >/dev/null
jq -e '.record_count == 8 and (.records | map(.task_id) | unique | length) == 8' \
  "$SMOKE_PLAN" >/dev/null

run_scan() {
  local plan=$1
  local output=$2
  PLAN="$plan" \
  CATALOG="$CATALOG" \
  OUTPUT_ROOT="$output" \
  CANDIDATE_GROUPS=canonical \
  GPU_COUNT=8 \
  DSOL_GPU_DEVICES=0,1,2,3,4,5,6,7 \
    "$SCANNER"
}

printf 'constructed_all8_start=%s git_commit=%s\n' \
  "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)"

if [[ ! -s "$ROOT/smoke-scan/analysis/summary.json" ]]; then
  run_scan "$SMOKE_PLAN" "$ROOT/smoke-scan"
fi
jq -e '.scan_count == 8 and .status == "MEASUREMENT_COMPLETE_THRESHOLD_UNFROZEN"' \
  "$ROOT/smoke-scan/analysis/summary.json" >/dev/null

if [[ ! -s "$ROOT/scan/analysis/summary.json" ]]; then
  run_scan "$PLAN" "$ROOT/scan"
fi
jq -e '.scan_count == 260 and .status == "MEASUREMENT_COMPLETE_THRESHOLD_UNFROZEN"' \
  "$ROOT/scan/analysis/summary.json" >/dev/null

python3 "$FREEZER" "$ROOT"/scan/shard-*.jsonl \
  --minimum-strong-delta 0.005 \
  --maximum-control-abs-delta 0.005 \
  --output "$ROOT/frozen_task_pairs.json"

python3 "$DENSE_BUILDER" "$ROOT"/scan/shard-*.jsonl \
  --catalog "$CATALOG" \
  --split val \
  --stage-targets 0.20,0.55 \
  --output "$ROOT/dense-discovery-protocol.json"

jq -n \
  --arg completed_at "$(date -u +%FT%TZ)" \
  --arg git_commit "$(git -C "$REPO_ROOT" rev-parse HEAD)" \
  --argjson frozen_tasks "$(jq '.tasks | length' "$ROOT/frozen_task_pairs.json")" \
  --argjson passing_tasks "$(jq '[.tasks[] | select(.status == "PASS")] | length' "$ROOT/frozen_task_pairs.json")" \
  --argjson discovery_states "$(jq '.selected_state_count' "$ROOT/dense-discovery-protocol.json")" \
  '{schema:"dsol_constructed_all8_scan_completion_v1",status:"COMPLETE",completed_at:$completed_at,git_commit:$git_commit,frozen_tasks:$frozen_tasks,passing_tasks:$passing_tasks,dense_discovery_states:$discovery_states}' \
  > "$ROOT/completion.json"

printf 'constructed_all8_complete=%s\n' "$(date -u +%FT%TZ)"
