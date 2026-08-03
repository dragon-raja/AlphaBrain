#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SEED=${1:-41}
STEPS=${2:-2000}
RUN_TAG=${KYC_DUAL_RUN_TAG:-dualcam-screen-v1}
RUN_ROOT=${KYC_DUAL_RUN_ROOT:-/share/longjunyu/cabi-vla/kyc-runs}
EVAL_ROOT=${KYC_DUAL_EVAL_ROOT:-/share/longjunyu/cabi-vla/dual-camera-kyc-screen-v1}
SUMMARY="$EVAL_ROOT/summary.json"
DIAGNOSTICS="$EVAL_ROOT/diagnostics.json"
FIGURE="$EVAL_ROOT/dual_camera_diagnostics.png"

wait_for_file() {
  local path=$1
  while [[ ! -s "$path" ]]; do
    sleep 30
  done
}

run_id() {
  local arm=$1
  printf 'kyc_%s_%s_h20_seed%s_steps%s' "$arm" "$RUN_TAG" "$SEED" "$STEPS"
}

wait_for_file "$SUMMARY"
for label in dual-rgb dual-control external wrist dual dual-wrist-initial dual-wrist-lagged; do
  wait_for_file "$EVAL_ROOT/${label}-s${SEED}-u${STEPS}/camera_sweep_test.json"
done

if [[ ! -s "$DIAGNOSTICS" ]]; then
  PYTHONPATH="$REPO_ROOT/scripts/cabi_vla" "$REPO_ROOT/.venv/bin/python" \
    "$REPO_ROOT/scripts/cabi_vla/summarize_kyc_dual_camera_diagnostics.py" \
    --evaluation "dual_rgb_fla=$EVAL_ROOT/dual-rgb-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --evaluation "dual_control_fla=$EVAL_ROOT/dual-control-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --evaluation "external_fla=$EVAL_ROOT/external-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --evaluation "wrist_fla=$EVAL_ROOT/wrist-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --evaluation "dual_fla=$EVAL_ROOT/dual-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --wrist-intervention "initial=$EVAL_ROOT/dual-wrist-initial-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --wrist-intervention "lagged=$EVAL_ROOT/dual-wrist-lagged-s${SEED}-u${STEPS}/camera_sweep_test.json" \
    --training-metrics "dual_rgb_fla=$RUN_ROOT/$(run_id dual_rgb_fla)/metrics.jsonl" \
    --training-metrics "dual_control_fla=$RUN_ROOT/$(run_id dual_control_fla)/metrics.jsonl" \
    --training-metrics "external_fla=$RUN_ROOT/$(run_id external_fla)/metrics.jsonl" \
    --training-metrics "wrist_fla=$RUN_ROOT/$(run_id wrist_fla)/metrics.jsonl" \
    --training-metrics "dual_fla=$RUN_ROOT/$(run_id dual_fla)/metrics.jsonl" \
    --output "$DIAGNOSTICS"
fi

if [[ ! -s "$FIGURE" ]]; then
  "$REPO_ROOT/.venv/bin/python" \
    "$REPO_ROOT/scripts/cabi_vla/render_kyc_dual_camera_diagnostics.py" \
    --diagnostics "$DIAGNOSTICS" \
    --output "$FIGURE"
fi

echo "dual-camera diagnostic closure complete: $DIAGNOSTICS"
