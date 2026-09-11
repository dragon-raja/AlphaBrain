#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUTPUT_ROOT=${DSOL_CONSTRUCTED_VIEW_ORACLE_ROOT:-/share/longjunyu/alphabrain/experiments/dsol-view-value-discovery-v1/constructed-val-pilot}
PROTOCOL=${DSOL_CONSTRUCTED_VIEW_ORACLE_PROTOCOL:-$OUTPUT_ROOT/protocol.json}
CHECKPOINT=${DSOL_CONSTRUCTED_VIEW_ORACLE_CHECKPOINT:-/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_broad64-quick-gate-v1_seed41_g8_gb32_steps2000/final_model}
EVALUATOR=$REPO_ROOT/scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh
ANALYZER=$REPO_ROOT/scripts/dsol_paper1/analyze_view_value_discovery.py
RUN_ROOT=$OUTPUT_ROOT/closed_loop_broad64_practical

restore_keepalive() {
  if [[ -f /workspace/ai2r/gpu_compute_keepalive/start_all.sh ]]; then
    bash /workspace/ai2r/gpu_compute_keepalive/start_all.sh 0 4096 gpu-keepalive || true
  fi
}
trap restore_keepalive EXIT

mkdir -p "$OUTPUT_ROOT/logs"
exec > >(tee -a "$OUTPUT_ROOT/logs/controller.log") 2>&1

jq -e '.status == "PASS" and .selected_state_count == 8 and .candidate_count == 97 and .episode_count == 776' "$PROTOCOL" >/dev/null
[[ -s "$CHECKPOINT/model.safetensors" ]]

printf 'constructed_view_oracle_start=%s git_commit=%s\n' "$(date -u +%FT%TZ)" "$(git -C "$REPO_ROOT" rev-parse HEAD)"

CHECKPOINT="$CHECKPOINT" \
OUTPUT_DIR="$RUN_ROOT" \
PROTOCOL="$PROTOCOL" \
POLICY_BACKEND=alphabrain \
GPU_COUNT=8 \
DSOL_GPU_DEVICES=0,1,2,3,4,5,6,7 \
BASE_PORT=20100 \
REPLAN_STEPS=5 \
WAIT_STEPS=0 \
EVAL_SEED=20260827 \
VIDEO_EPISODES=16 \
RUN_ANALYSIS=0 \
  "$EVALUATOR"

python3 "$ANALYZER" "$RUN_ROOT"/episodes-shard-*.jsonl \
  --output-dir "$RUN_ROOT/analysis_discovery"

jq -n \
  --arg completed_at "$(date -u +%FT%TZ)" \
  --arg git_commit "$(git -C "$REPO_ROOT" rev-parse HEAD)" \
  '{schema:"dsol_constructed_view_oracle_completion_v1",status:"COMPLETE",completed_at:$completed_at,git_commit:$git_commit}' \
  > "$OUTPUT_ROOT/completion.json"

printf 'constructed_view_oracle_complete=%s\n' "$(date -u +%FT%TZ)"
