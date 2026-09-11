#!/usr/bin/env bash
# CPU-only report generation after every static segment has passed its audit.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
LANDSCAPE_ROOT=/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/alphabrain/.venv/bin/python - "$LANDSCAPE_ROOT/controller-status.json" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    status = json.load(stream)
if status.get('status') != 'PASS_COMPLETE_STATIC_DIAGNOSTIC' or status.get('completed_unique_episodes') != 24832:
    raise SystemExit('closed-loop matrix is incomplete; no final paired report')
PY
/alphabrain/.venv/bin/python "$REPO_ROOT/reports/paper1/historical/build_view_landscape_report_v1.py" \
  --accel-root "$LANDSCAPE_ROOT/canonical-accel" \
  --dense-dir "$LANDSCAPE_ROOT/closed-loop/canonical/O" \
  --noise-bank-manifest /share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2/noise-banks/bank_O.manifest.json \
  --selection-manifest "$LANDSCAPE_ROOT/protocols/selection.json" \
  --checkpoint /share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_canonical_unique_mb-matched-v1_seed41_g2_gb32_steps2000/final_model \
  --expected-checkpoint-sha256 7e510b752143bb6fd4987988dfab5e94f7db1de5f7e97579f892a58eed22cc68 \
  --model-label 'Canonical matched / seed41' --training-support canonical \
  --output-dir "$LANDSCAPE_ROOT/canonical-report"
/alphabrain/.venv/bin/python "$REPO_ROOT/scripts/dsol_paper1/analysis/compare_matched_view_landscapes_v1.py" \
  --broad-root "$LANDSCAPE_ROOT/broad-existing" \
  --canonical-root "$LANDSCAPE_ROOT/canonical-report" \
  --selection-manifest "$LANDSCAPE_ROOT/protocols/selection.json" \
  --output-dir "$LANDSCAPE_ROOT/matched-comparison"
/alphabrain/.venv/bin/python "$REPO_ROOT/scripts/dsol_paper1/analysis/analyze_matched_view_metric_rules_v1.py" \
  --broad-root "$LANDSCAPE_ROOT/broad-existing" \
  --canonical-root "$LANDSCAPE_ROOT/canonical-report" \
  --selection-manifest "$LANDSCAPE_ROOT/protocols/selection.json" \
  --output-dir "$LANDSCAPE_ROOT/metric-rules-matched-v1"
