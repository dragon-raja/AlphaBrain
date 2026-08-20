#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WAIT_SESSION=${DSOL_M0_SELECTION_WAIT_SESSION:-dsol-constructed-m0-m-a-v3}
POLL_SECONDS=${DSOL_WAIT_POLL_SECONDS:-30}
SCAN_ROOT=${DSOL_CONSTRUCTED_M0_SCAN_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m0-v1/operational-three-task-scan}
PLAN=${DSOL_CONSTRUCTED_M0_PLAN:-$REPO_ROOT/configs/dsol_paper1/libero_constructed_blind_reveal_scan_plan_v1.json}
OUTPUT=${DSOL_CONSTRUCTED_M0_SELECTION_OUTPUT:-$SCAN_ROOT/selection_v1.json}

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid DSOL_WAIT_POLL_SECONDS" >&2; exit 2; }
[[ -s "$PLAN" ]] || { echo "missing scan plan: $PLAN" >&2; exit 1; }
mkdir -p "$SCAN_ROOT"
exec > >(tee -a "$SCAN_ROOT/selection_controller.log") 2>&1
printf 'selection_controller_start=%s git_commit=%s wait_session=%s\n' \
  "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)" "$WAIT_SESSION"

while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  printf 'waiting_for_scan=%s at=%s\n' "$WAIT_SESSION" "$(date -u +%FT%TZ)"
  sleep "$POLL_SECONDS"
done
while [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; do
  printf 'waiting_for_clean_worktree=%s at=%s\n' "$REPO_ROOT" "$(date -u +%FT%TZ)"
  sleep "$POLL_SECONDS"
done

[[ -s "$SCAN_ROOT/analysis/summary.json" ]] || {
  echo "scan session ended without analysis summary: $SCAN_ROOT/analysis/summary.json" >&2
  exit 1
}
mapfile -t ledgers < <(find "$SCAN_ROOT" -maxdepth 1 -type f -name 'shard-*.jsonl' | sort)
[[ "${#ledgers[@]}" -gt 0 ]] || { echo "no M0 shard ledgers found" >&2; exit 1; }

command=(
  /alphabrain/.venv/bin/python
  "$REPO_ROOT/scripts/dsol_paper1/select_constructed_m0_candidates.py"
  --scan-plan "$PLAN"
  --output "$OUTPUT"
)
for ledger in "${ledgers[@]}"; do command+=(--ledger "$ledger"); done
PYTHONPATH="$REPO_ROOT:$REPO_ROOT/scripts/dsol_paper1" "${command[@]}"

jq -e '.schema == "dsol_constructed_m0_candidate_selection_v1" and (.status == "PASS" or .status == "HOLD")' \
  "$OUTPUT" >/dev/null
printf 'selection_controller_complete=%s output=%s status=%s\n' \
  "$(date -u +%FT%TZ)" "$OUTPUT" "$(jq -r .status "$OUTPUT")"
