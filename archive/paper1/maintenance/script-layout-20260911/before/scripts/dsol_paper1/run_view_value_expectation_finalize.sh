#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${DSOL_VIEW_EXPECTATION_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-view-value-expectation-v1}
POST_SESSION=${POST_SESSION:-dsol-view-expectation-post-v1}
OUTPUT=$ROOT/final-report
mkdir -p "$ROOT/logs" "$OUTPUT"
exec > >(tee -a "$ROOT/logs/finalize.log") 2>&1

while tmux has-session -t "$POST_SESSION" 2>/dev/null; do
  printf 'waiting_for_post_calibration time=%s\n' "$(date -u +%FT%TZ)"
  sleep 60
done

rg -q '^view_value_expectation_post_calibration_complete=' \
  "$ROOT/logs/post-calibration.log" || {
  echo "post-calibration controller ended without a completion marker" >&2
  exit 2
}

analysis=$ROOT/heldout/analysis-primary/primary-analysis.json
if [[ -s "$ROOT/heldout/analysis-primary/reserve-decision.json" ]] && \
  jq -e '.activate_bank_F == true' \
    "$ROOT/heldout/analysis-primary/reserve-decision.json" >/dev/null; then
  [[ -s "$ROOT/heldout/analysis-final/primary-analysis.json" ]] || {
    echo "bank F was required but final 64-repeat analysis is missing" >&2
    exit 2
  }
  analysis=$ROOT/heldout/analysis-final/primary-analysis.json
fi
[[ -s "$ROOT/calibration/analysis/analysis.json" && -s "$analysis" ]] || {
  echo "formal pipeline ended without complete calibration and held-out analyses" >&2
  exit 2
}

PYTHONPATH="$REPO_ROOT/scripts/dsol_paper1:$REPO_ROOT" \
  /alphabrain/.venv/bin/python \
    "$REPO_ROOT/scripts/dsol_paper1/build_view_value_expectation_final_report.py" \
      --root "$ROOT" --output-dir "$OUTPUT"

WANDB_MODE=${WANDB_MODE:-offline} \
  /alphabrain/.venv/bin/python \
    "$REPO_ROOT/scripts/dsol_paper1/log_view_value_expectation_wandb.py" \
      --analysis "$analysis" --decision "$OUTPUT/final_decision.json" \
      --run-dir "$ROOT/wandb" --repo "$REPO_ROOT"

printf 'view_value_expectation_finalize_complete=%s\n' "$(date -u +%FT%TZ)"
