#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CALIBRATION_SESSION=${CALIBRATION_SESSION:-dsol-canonical-calibration-3000-v1}
CALIBRATION_RUN=${CALIBRATION_RUN:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_canonical_unique_calibration3000-v1_seed41_g8_gb32_steps3000}
PAIR_DATA=${DSOL_PAIR_DATA_ROOT:-/share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/quick_gate_seed41_broad32_stride2}
SMOKE_CHECKPOINT=${SMOKE_CHECKPOINT:-/share/longjunyu/alphabrain/experiments/libero-plus-mv-rgb-v1/runs/pi05_plus_mv_visual_lora_gate-v1-b100_seed41_steps33000/final_model}
EVAL_SMOKE_ROOT=${EVAL_SMOKE_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/closed_loop_smoke/exact-hdf5-v1}
M0_SMOKE_ROOT=${M0_SMOKE_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/m0_visibility_smoke/catalog-v2-eight-states}
PAIR_SMOKE_RUN=/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_state_matched_flow-control-smoke-v2_seed41_g8_gb32_steps20

while tmux has-session -t "$CALIBRATION_SESSION" 2>/dev/null; do
  sleep 30
done

"/alphabrain/.venv/bin/python" "$REPO_ROOT/scripts/dsol_paper1/summarize_dsol_calibration.py" \
  "$CALIBRATION_RUN" --output-dir "$CALIBRATION_RUN/calibration_analysis"
jq -e '.status == "COMPLETE" and .last_train_step >= 3000' \
  "$CALIBRATION_RUN/calibration_analysis/calibration_summary.json" >/dev/null

if [[ ! -s "$PAIR_SMOKE_RUN/metrics.jsonl" ]] || \
  ! jq -e 'select(.step == 20)' "$PAIR_SMOKE_RUN/metrics.jsonl" >/dev/null; then
  DSOL_PAIR_DATA_ROOT="$PAIR_DATA" \
  DSOL_CALIBRATION=0 DSOL_SKIP_FINAL_SAVE=1 WANDB_MODE=offline \
    "$REPO_ROOT/scripts/dsol_paper1/run_libero_pair_train.sh" \
      broad_unpaired_state_matched 41 8 20 flow-control-smoke-v2
else
  echo "post_calibration_skip_complete=flow-control-smoke-v2"
fi

CHECKPOINT="$SMOKE_CHECKPOINT" OUTPUT_DIR="$EVAL_SMOKE_ROOT" \
GPU_COUNT=1 MAX_EPISODES_PER_SHARD=1 RUN_ANALYSIS=0 VIDEO_EPISODES=1 \
  "$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh"

OUTPUT_ROOT="$M0_SMOKE_ROOT" GPU_COUNT=8 MAX_STATES_PER_SHARD=1 \
  "$REPO_ROOT/scripts/dsol_paper1/run_libero_visibility_scan_matrix.sh"

echo "post_calibration_gates_complete=1"
